import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import requests
from datetime import datetime
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="多智能體交易 AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg: #0a0a0f; --surface: #111118; --surface2: #1a1a24;
    --border: #2a2a3a; --accent: #00ff88; --accent2: #7c3aed;
    --danger: #ff4444; --warn: #ffaa00; --text: #FFD700; --muted: #b8972a;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); }
.stApp { background: var(--bg); }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
.agent-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.agent-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.agent-card.technical::before { background: #00bfff; }
.agent-card.sentiment::before { background: #ff6b6b; }
.agent-card.fundamental::before { background: #ffd93d; }
.agent-card.risk::before { background: var(--accent); }
.agent-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
    font-family: 'Space Mono', monospace; font-size: 13px; letter-spacing: 0.05em; }
.agent-icon { font-size: 20px; }
.agent-name { color: var(--muted); text-transform: uppercase; }
.agent-content { font-size: 14px; line-height: 1.7; color: var(--text); }
.verdict-box {
    background: linear-gradient(135deg, #0d1f0d, #0a0a1a);
    border: 1px solid var(--accent); border-radius: 16px;
    padding: 28px; text-align: center; margin-top: 20px;
}
.verdict-label { font-family: 'Space Mono', monospace; font-size: 11px; color: var(--muted);
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px; }
.verdict-text { font-family: 'Space Mono', monospace; font-size: 36px; font-weight: 700; margin-bottom: 4px; }
.verdict-text.buy { color: var(--accent); }
.verdict-text.sell { color: var(--danger); }
.verdict-text.hold { color: var(--warn); }
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-item { background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 18px; flex: 1; min-width: 100px; }
.metric-label { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.1em; font-family: 'Space Mono', monospace; }
.metric-value { font-size: 20px; font-weight: 600; margin-top: 4px; font-family: 'Space Mono', monospace; }
.metric-value.up { color: var(--accent); }
.metric-value.down { color: var(--danger); }
.source-badge {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 4px 12px; font-size: 11px;
    font-family: 'Space Mono', monospace; margin-bottom: 12px;
}
.source-av { background: rgba(0,191,255,0.1); border: 1px solid rgba(0,191,255,0.3); color: #00bfff; }
.source-td { background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3); color: #a78bfa; }
.section-title { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.15em; margin: 24px 0 12px 0;
    border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.stButton > button {
    background: var(--accent) !important; color: #000 !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 32px !important; letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
.news-item { background: var(--surface2); border-left: 3px solid var(--accent2);
    padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0;
    font-size: 13px; color: var(--muted); }
.stProgress > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — Alpha Vantage (主) + Twelve Data (備份)
# ══════════════════════════════════════════════════════════════════════════════

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingAI/1.0)"}


# ── Alpha Vantage helpers ─────────────────────────────────────────────────────

def _av_call(function: str, params: dict, av_key: str) -> dict:
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": function, "apikey": av_key, **params},
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if "Note" in data or "Information" in data:
        raise RuntimeError("Alpha Vantage 達到免費額度限制，自動切換至 Twelve Data…")
    return data


@st.cache_data(ttl=300, show_spinner=False)
def av_ohlcv(symbol: str, av_key: str) -> pd.DataFrame:
    data = _av_call("TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": "compact"}, av_key)
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise ValueError(f"Alpha Vantage: 找不到 {symbol} 數據")
    rows = [{"Date": pd.to_datetime(d),
             "Open": float(v["1. open"]), "High": float(v["2. high"]),
             "Low":  float(v["3. low"]),  "Close": float(v["4. close"]),
             "Volume": float(v["5. volume"])} for d, v in ts.items()]
    return pd.DataFrame(rows).set_index("Date").sort_index()


@st.cache_data(ttl=300, show_spinner=False)
def av_overview(symbol: str, av_key: str) -> dict:
    return _av_call("OVERVIEW", {"symbol": symbol}, av_key)


@st.cache_data(ttl=300, show_spinner=False)
def av_news(symbol: str, av_key: str) -> list:
    try:
        data = _av_call("NEWS_SENTIMENT", {"tickers": symbol, "limit": "8"}, av_key)
        return [i.get("title", "") for i in data.get("feed", []) if i.get("title")]
    except Exception:
        return []


def _av_fetch_all(symbol: str, av_key: str):
    df  = av_ohlcv(symbol, av_key)
    ov  = av_overview(symbol, av_key)
    nws = av_news(symbol, av_key)

    def s(k, *alt):
        for key in [k, *alt]:
            v = ov.get(key)
            if v and v != "None":
                try: return float(v)
                except: return v
        return "N/A"

    fund = {
        "market_cap":    s("MarketCapitalization"),
        "pe_ratio":      s("TrailingPE"),
        "forward_pe":    s("ForwardPE"),
        "revenue":       s("RevenueTTM"),
        "profit_margin": s("ProfitMargin"),
        "debt_to_equity":s("DebtToEquityRatio", "DebtEquityRatio"),
        "roe":           s("ReturnOnEquityTTM"),
        "eps":           s("EPS"),
        "beta":          s("Beta"),
        "short_ratio":   s("ShortRatio"),
        "analyst_target":s("AnalystTargetPrice"),
        "recommendation":ov.get("RecommendationKey", "N/A"),
        "sector":        ov.get("Sector", "N/A"),
        "industry":      ov.get("Industry", "N/A"),
        "dividend_yield":s("DividendYield"),
    }
    return df, fund, nws


# ── Twelve Data helpers ───────────────────────────────────────────────────────

def _td_call(endpoint: str, params: dict, td_key: str) -> dict:
    r = requests.get(
        f"https://api.twelvedata.com/{endpoint}",
        params={"apikey": td_key, **params},
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise ValueError("Twelve Data: " + data.get("message", "unknown error"))
    return data


@st.cache_data(ttl=300, show_spinner=False)
def td_ohlcv(symbol: str, td_key: str) -> pd.DataFrame:
    data = _td_call("time_series", {"symbol": symbol, "interval": "1day",
                                    "outputsize": 130, "order": "ASC"}, td_key)
    bars = data.get("values", [])
    if not bars:
        raise ValueError(f"Twelve Data: 找不到 {symbol} 數據")
    rows = [{"Date": pd.to_datetime(b["datetime"]),
             "Open": float(b["open"]), "High": float(b["high"]),
             "Low":  float(b["low"]),  "Close": float(b["close"]),
             "Volume": float(b["volume"])} for b in bars]
    return pd.DataFrame(rows).set_index("Date").sort_index()


@st.cache_data(ttl=300, show_spinner=False)
def td_fundamentals(symbol: str, td_key: str) -> dict:
    try:
        data  = _td_call("statistics", {"symbol": symbol}, td_key)
        stats = data.get("statistics", {})
        val   = stats.get("valuations_metrics", {})
        fin   = stats.get("financials", {})
        inc   = fin.get("income_statement", {})
        bal   = fin.get("balance_sheet", {})
        tec   = stats.get("stock_statistics", {})
        return {
            "market_cap":    val.get("market_capitalization", "N/A"),
            "pe_ratio":      val.get("trailing_pe",           "N/A"),
            "forward_pe":    val.get("forward_pe",            "N/A"),
            "revenue":       inc.get("total_revenue",         "N/A"),
            "profit_margin": inc.get("profit_margin",         "N/A"),
            "debt_to_equity":bal.get("total_debt_to_equity",  "N/A"),
            "roe":           inc.get("return_on_equity",      "N/A"),
            "eps":           val.get("trailing_eps",           "N/A"),
            "beta":          tec.get("beta",                   "N/A"),
            "short_ratio":   tec.get("short_ratio",            "N/A"),
            "analyst_target":"N/A",
            "recommendation":"N/A",
            "sector":        data.get("sector",                "N/A"),
            "industry":      data.get("industry",              "N/A"),
            "dividend_yield":"N/A",
        }
    except Exception:
        return {k: "N/A" for k in ["market_cap","pe_ratio","forward_pe","revenue",
            "profit_margin","debt_to_equity","roe","eps","beta","short_ratio",
            "analyst_target","recommendation","sector","industry","dividend_yield"]}


@st.cache_data(ttl=300, show_spinner=False)
def td_news(symbol: str, td_key: str) -> list:
    try:
        data  = _td_call("news", {"symbol": symbol, "outputsize": 8}, td_key)
        items = data if isinstance(data, list) else data.get("data", [])
        return [i.get("title", "") for i in items if i.get("title")]
    except Exception:
        return []


def _td_fetch_all(symbol: str, td_key: str):
    df   = td_ohlcv(symbol, td_key)
    fund = td_fundamentals(symbol, td_key)
    nws  = td_news(symbol, td_key)
    return df, fund, nws


# ── Unified fetcher with auto-fallback ────────────────────────────────────────

def fetch_all(symbol: str, av_key: str, td_key: str):
    """Try Alpha Vantage → Twelve Data.  Returns (df, fund, news, source)."""
    errors = []

    if av_key:
        try:
            df, fund, nws = _av_fetch_all(symbol, av_key)
            return df, fund, nws, "Alpha Vantage"
        except Exception as e:
            errors.append(f"Alpha Vantage ❌ {e}")

    if td_key:
        try:
            df, fund, nws = _td_fetch_all(symbol, td_key)
            return df, fund, nws, "Twelve Data"
        except Exception as e:
            errors.append(f"Twelve Data ❌ {e}")

    raise RuntimeError("兩個數據源均失敗：\n" + "\n".join(errors))


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_technicals(df: pd.DataFrame) -> dict:
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    delta    = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    latest = close.iloc[-1]
    prev   = close.iloc[-2] if len(close) > 1 else latest
    w252   = min(252, len(df))

    return {
        "price":       round(float(latest), 2),
        "change_pct":  round((latest - prev) / prev * 100, 2),
        "rsi":         round(float(rsi.iloc[-1]), 1),
        "macd":        round(float(macd_line.iloc[-1]), 3),
        "macd_signal": round(float(signal_line.iloc[-1]), 3),
        "macd_hist":   round(float(macd_hist.iloc[-1]), 3),
        "ema20":       round(float(ema20.iloc[-1]), 2),
        "ema50":       round(float(ema50.iloc[-1]), 2),
        "atr":         round(float(atr.iloc[-1]), 2),
        "bb_upper":    round(float(bb_upper.iloc[-1]), 2),
        "bb_lower":    round(float(bb_lower.iloc[-1]), 2),
        "volume":      int(volume.iloc[-1]),
        "avg_volume":  int(volume.rolling(20).mean().iloc[-1]),
        "high_52w":    round(float(high.iloc[-w252:].max()), 2),
        "low_52w":     round(float(low.iloc[-w252:].min()), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ AGENTS
# ══════════════════════════════════════════════════════════════════════════════

TECHNICAL_SYSTEM = """You are a senior technical analyst with 20 years of experience.
Analyse the provided technical indicators and give a concise, actionable assessment.
Be specific about key levels, momentum, and trend direction.
Keep response under 180 words. End with a one-line SIGNAL: BUY / SELL / HOLD."""

SENTIMENT_SYSTEM = """You are a market sentiment analyst specialising in news-driven trading.
Analyse the provided news headlines for sentiment, tone, and potential market impact.
Identify the dominant narrative and any catalysts.
Keep response under 180 words. End with a one-line SIGNAL: BULLISH / BEARISH / NEUTRAL."""

FUNDAMENTAL_SYSTEM = """You are a fundamental analyst at a top hedge fund.
Analyse the provided financial metrics. Focus on valuation, profitability, and risk.
Compare PE, margins, and growth signals. Be direct and quantitative.
Keep response under 180 words. End with a one-line SIGNAL: UNDERVALUED / OVERVALUED / FAIRLY VALUED."""

RISK_SYSTEM = """You are the Chief Risk Officer and final decision maker.
You receive reports from three analysts: Technical, Sentiment, and Fundamental.
Synthesise all three perspectives, weigh conflicts, and make a final trading decision.
State your confidence level (Low / Medium / High) and suggest position sizing (Small / Medium / Full).
Keep response under 220 words. End with VERDICT: BUY / SELL / HOLD and a brief rationale."""


def call_agent(client: Groq, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.3, max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:20px 0 10px 0;'>
    <div style='font-family:Space Mono,monospace;font-size:11px;color:#b8972a;
         letter-spacing:0.2em;text-transform:uppercase;margin-bottom:6px;'>受 TAURIC 啟發</div>
    <h1 style='margin:0;font-size:28px;font-family:Space Mono,monospace;color:#FFD700;'>多智能體交易 AI</h1>
    <div style='color:#b8972a;font-size:14px;margin-top:6px;'>
        Groq · llama-3.3-70b · Alpha Vantage + Twelve Data 雙源備份
    </div>
</div>
""", unsafe_allow_html=True)

# Secrets
groq_key = st.secrets.get("GROQ_API_KEY", "")
av_key   = st.secrets.get("AV_API_KEY",   "")
td_key   = st.secrets.get("TD_API_KEY",   "")

with st.sidebar:
    st.markdown("<div class='section-title'>API 金鑰</div>", unsafe_allow_html=True)

    if groq_key:
        st.markdown("<div style='font-size:12px;color:#00ff88;font-family:Space Mono,monospace;margin-bottom:4px;'>✅ GROQ_API_KEY 已載入</div>", unsafe_allow_html=True)
    else:
        groq_key = st.text_input("Groq API 金鑰", type="password", placeholder="gsk_...")

    if av_key:
        st.markdown("<div style='font-size:12px;color:#00bfff;font-family:Space Mono,monospace;margin-bottom:4px;'>✅ AV_API_KEY 已載入</div>", unsafe_allow_html=True)
    else:
        av_key = st.text_input("Alpha Vantage Key", type="password", placeholder="alphavantage.co 免費申請")

    if td_key:
        st.markdown("<div style='font-size:12px;color:#a78bfa;font-family:Space Mono,monospace;margin-bottom:8px;'>✅ TD_API_KEY 已載入</div>", unsafe_allow_html=True)
    else:
        td_key = st.text_input("Twelve Data Key", type="password", placeholder="twelvedata.com 免費申請")

    st.markdown("""
    <div style='font-size:11px;color:#b8972a;margin-bottom:12px;line-height:2;'>
    💡 Streamlit Secrets 格式：<br>
    <code>GROQ_API_KEY = "gsk_..."</code><br>
    <code>AV_API_KEY = "..."</code><br>
    <code>TD_API_KEY = "..."</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>設定</div>", unsafe_allow_html=True)
    ticker_input = st.text_input("股票代號", value="TSLA", placeholder="TSLA / NVDA / AAPL")

    st.markdown("<div class='section-title'>數據源</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;color:#b8972a;line-height:2;'>
    1️⃣ Alpha Vantage（主）<br>2️⃣ Twelve Data（備份）<br>⚡ 失敗自動切換
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>智能體</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:13px;color:#b8972a;line-height:2;'>
    📊 技術分析師<br>📰 情緒分析師<br>💰 基本面分析師<br>🎯 風險管理員
    </div>""", unsafe_allow_html=True)

    run_btn = st.button("🚀 開始分析")

# ── Idle state ────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;color:#b8972a;'>
        <div style='font-size:48px;margin-bottom:16px;'>🤖</div>
        <div style='font-family:Space Mono,monospace;font-size:14px;margin-bottom:8px;color:#FFD700;'>
            4 個 AI 智能體待命中
        </div>
        <div style='font-size:13px;'>輸入股票代號 → 點擊「開始分析」</div>
        <div style='font-size:12px;margin-top:16px;color:#555;'>
            數據：Alpha Vantage + Twelve Data · 免費使用 · 無需 yfinance
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not groq_key:
    st.error("⚠️ 請設定 GROQ_API_KEY")
    st.stop()

if not av_key and not td_key:
    st.error("⚠️ 請至少設定一個數據源 Key：AV_API_KEY 或 TD_API_KEY")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────────────────────
symbol = ticker_input.upper().strip()

with st.spinner(f"正在獲取 {symbol} 市場數據…"):
    try:
        df, fundamentals, news_headlines, source = fetch_all(symbol, av_key, td_key)
    except Exception as e:
        st.error(f"❌ 數據獲取失敗：{e}")
        st.info("請確認：① 股票代號正確（美股：TSLA，港股：9988.HK）② API Key 有效 ③ 未超出免費配額")
        st.stop()

technicals = compute_technicals(df)

# Source badge
src_cls  = "source-av" if source == "Alpha Vantage" else "source-td"
src_icon = "🔵" if source == "Alpha Vantage" else "🟣"
st.markdown(f"<div class='source-badge {src_cls}'>{src_icon} 數據來源：{source}</div>",
            unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
change = technicals["change_pct"]
c_cls  = "up" if change >= 0 else "down"
c_arr  = "▲"  if change >= 0 else "▼"

st.markdown(f"""
<div class='metric-row'>
    <div class='metric-item'><div class='metric-label'>股價</div>
        <div class='metric-value'>${technicals['price']}</div></div>
    <div class='metric-item'><div class='metric-label'>漲跌幅</div>
        <div class='metric-value {c_cls}'>{c_arr} {abs(change)}%</div></div>
    <div class='metric-item'><div class='metric-label'>RSI（14）</div>
        <div class='metric-value {"up" if technicals["rsi"] < 70 else "down"}'>{technicals['rsi']}</div></div>
    <div class='metric-item'><div class='metric-label'>ATR</div>
        <div class='metric-value'>{technicals['atr']}</div></div>
    <div class='metric-item'><div class='metric-label'>52週高位</div>
        <div class='metric-value'>${technicals['high_52w']}</div></div>
    <div class='metric-item'><div class='metric-label'>52週低位</div>
        <div class='metric-value'>${technicals['low_52w']}</div></div>
</div>""", unsafe_allow_html=True)

# ── News ──────────────────────────────────────────────────────────────────────
if news_headlines:
    st.markdown("<div class='section-title'>最新新聞標題</div>", unsafe_allow_html=True)
    for h in news_headlines[:4]:
        st.markdown(f"<div class='news-item'>📌 {h}</div>", unsafe_allow_html=True)

# ── Agents ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>智能體分析</div>", unsafe_allow_html=True)
client   = Groq(api_key=groq_key)
progress = st.progress(0, text="正在初始化智能體...")
R        = {}

def render_agent(cls, icon, name_zh, key):
    st.markdown(f"""
    <div class='agent-card {cls}'>
        <div class='agent-header'><span class='agent-icon'>{icon}</span>
            <span class='agent-name'>{name_zh}</span></div>
        <div class='agent-content'>{R[key].replace(chr(10),'<br>')}</div>
    </div>""", unsafe_allow_html=True)

# Agent 1 — Technical
progress.progress(10, text="📊 技術分析師分析中...")
try:
    R["tech"] = call_agent(client, TECHNICAL_SYSTEM, f"""
Stock: {symbol}
Price: ${technicals['price']} ({'+' if change>=0 else ''}{change}% today)
RSI(14): {technicals['rsi']}
MACD: {technicals['macd']} | Signal: {technicals['macd_signal']} | Hist: {technicals['macd_hist']}
EMA20: {technicals['ema20']} | EMA50: {technicals['ema50']}
ATR(14): {technicals['atr']}
Bollinger: {technicals['bb_upper']} / {technicals['bb_lower']}
Volume: {technicals['volume']:,} (Avg20: {technicals['avg_volume']:,})
52W: ${technicals['low_52w']} – ${technicals['high_52w']}
""")
except Exception as e:
    R["tech"] = f"Error: {e}"
render_agent("technical", "📊", "技術分析師", "tech")

# Agent 2 — Sentiment
progress.progress(35, text="📰 情緒分析師分析中...")
headlines_text = "\n".join(f"- {h}" for h in news_headlines) if news_headlines else "No recent headlines."
try:
    R["sent"] = call_agent(client, SENTIMENT_SYSTEM, f"""
Stock: {symbol}
News:\n{headlines_text}
Price change today: {'+' if change>=0 else ''}{change}%
""")
except Exception as e:
    R["sent"] = f"Error: {e}"
render_agent("sentiment", "📰", "情緒分析師", "sent")

# Agent 3 — Fundamental
progress.progress(60, text="💰 基本面分析師分析中...")
def fmt(v):
    if v == "N/A" or v is None: return "N/A"
    if isinstance(v, float):
        return f"${v/1e9:.1f}B" if v > 1e9 else (f"{v:.4f}" if v < 1 else f"{v:.2f}")
    if isinstance(v, int) and v > 1_000_000: return f"${v/1e9:.1f}B"
    return str(v)
try:
    R["fund"] = call_agent(client, FUNDAMENTAL_SYSTEM, f"""
Stock: {symbol} | Sector: {fundamentals['sector']} | Industry: {fundamentals['industry']}
Market Cap: {fmt(fundamentals['market_cap'])}
Trailing PE: {fmt(fundamentals['pe_ratio'])} | Forward PE: {fmt(fundamentals['forward_pe'])}
Revenue: {fmt(fundamentals['revenue'])} | Profit Margin: {fmt(fundamentals['profit_margin'])}
Debt/Equity: {fmt(fundamentals['debt_to_equity'])} | ROE: {fmt(fundamentals['roe'])}
EPS: {fmt(fundamentals['eps'])} | Beta: {fmt(fundamentals['beta'])}
Short Ratio: {fmt(fundamentals['short_ratio'])} | Analyst Target: {fmt(fundamentals['analyst_target'])}
Consensus: {fundamentals['recommendation']}
""")
except Exception as e:
    R["fund"] = f"Error: {e}"
render_agent("fundamental", "💰", "基本面分析師", "fund")

# Agent 4 — Risk Manager
progress.progress(80, text="🎯 風險管理員綜合評估中...")
try:
    R["risk"] = call_agent(client, RISK_SYSTEM, f"""
Stock: {symbol} @ ${technicals['price']}
TECHNICAL: {R['tech']}
SENTIMENT: {R['sent']}
FUNDAMENTAL: {R['fund']}
Make your final decision.
""")
except Exception as e:
    R["risk"] = f"Error: {e}"
render_agent("risk", "🎯", "風險管理員 · 最終決策", "risk")

progress.progress(100, text="✅ 分析完成")
time.sleep(0.4)
progress.empty()

# ── Verdict ───────────────────────────────────────────────────────────────────
verdict = "HOLD"
risk_up = R.get("risk", "").upper()
for w in ["BUY", "SELL", "HOLD"]:
    if f"VERDICT: {w}" in risk_up or risk_up.strip().endswith(w):
        verdict = w; break

v_class = verdict.lower()
v_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(verdict, "⚪")
v_zh    = {"BUY": "買入", "SELL": "賣出", "HOLD": "持有"}.get(verdict, verdict)

st.markdown(f"""
<div class='verdict-box'>
    <div class='verdict-label'>最終裁決 · {symbol} · {source}</div>
    <div class='verdict-text {v_class}'>{v_emoji} {v_zh}</div>
    <div style='font-size:12px;color:#b8972a;margin-top:8px;font-family:Space Mono,monospace;'>
        {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC · llama-3.3-70b-versatile
    </div>
</div>
<div style='margin-top:20px;padding:12px 16px;background:#1a1a24;border-radius:8px;border:1px solid #2a2a3a;'>
    <span style='font-size:11px;color:#b8972a;'>
    ⚠️ <b>免責聲明：</b>本分析由 AI 生成，僅供學習及研究用途，並非投資建議。交易涉及風險，請自行做好研究。
    </span>
</div>
""", unsafe_allow_html=True)
