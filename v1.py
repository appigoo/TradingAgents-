import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from groq import Groq
import json
from datetime import datetime, timedelta
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Trading AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --border: #2a2a3a;
    --accent: #00ff88;
    --accent2: #7c3aed;
    --danger: #ff4444;
    --warn: #ffaa00;
    --text: #e8e8f0;
    --muted: #6b6b8a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Agent cards */
.agent-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.agent-card.technical::before { background: #00bfff; }
.agent-card.sentiment::before { background: #ff6b6b; }
.agent-card.fundamental::before { background: #ffd93d; }
.agent-card.risk::before { background: var(--accent); }

.agent-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.05em;
}
.agent-icon { font-size: 20px; }
.agent-name { color: var(--muted); text-transform: uppercase; }

.agent-content {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text);
}

/* Verdict */
.verdict-box {
    background: linear-gradient(135deg, #0d1f0d, #0a0a1a);
    border: 1px solid var(--accent);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.verdict-text {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 4px;
}
.verdict-text.buy { color: var(--accent); }
.verdict-text.sell { color: var(--danger); }
.verdict-text.hold { color: var(--warn); }

/* Metrics row */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    flex: 1;
    min-width: 100px;
}
.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 20px;
    font-weight: 600;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}
.metric-value.up { color: var(--accent); }
.metric-value.down { color: var(--danger); }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
}

/* Divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Input */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}

/* News items */
.news-item {
    background: var(--surface2);
    border-left: 3px solid var(--accent2);
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: var(--muted);
}

/* Spinner override */
.stSpinner { color: var(--accent) !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def compute_technicals(df: pd.DataFrame) -> dict:
    """Calculate key technical indicators from OHLCV data."""
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # EMA
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    latest = close.iloc[-1]
    prev = close.iloc[-2] if len(close) > 1 else latest

    return {
        "price": round(float(latest), 2),
        "prev_close": round(float(prev), 2),
        "change_pct": round((latest - prev) / prev * 100, 2),
        "rsi": round(float(rsi.iloc[-1]), 1),
        "macd": round(float(macd_line.iloc[-1]), 3),
        "macd_signal": round(float(signal_line.iloc[-1]), 3),
        "macd_hist": round(float(macd_hist.iloc[-1]), 3),
        "ema20": round(float(ema20.iloc[-1]), 2),
        "ema50": round(float(ema50.iloc[-1]), 2),
        "atr": round(float(atr.iloc[-1]), 2),
        "bb_upper": round(float(bb_upper.iloc[-1]), 2),
        "bb_lower": round(float(bb_lower.iloc[-1]), 2),
        "volume": int(volume.iloc[-1]),
        "avg_volume": int(volume.rolling(20).mean().iloc[-1]),
        "high_52w": round(float(high.rolling(252).max().iloc[-1]), 2),
        "low_52w": round(float(low.rolling(252).min().iloc[-1]), 2),
    }


def get_fundamentals(ticker_obj) -> dict:
    info = ticker_obj.info
    return {
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "revenue": info.get("totalRevenue", "N/A"),
        "profit_margin": info.get("profitMargins", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "roe": info.get("returnOnEquity", "N/A"),
        "eps": info.get("trailingEps", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
        "beta": info.get("beta", "N/A"),
        "short_ratio": info.get("shortRatio", "N/A"),
        "analyst_target": info.get("targetMeanPrice", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
    }


def get_news(ticker_obj, max_items: int = 8) -> list[str]:
    try:
        news = ticker_obj.news or []
        headlines = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else ""
            if not title:
                title = item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def call_agent(client: Groq, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── Agent Prompts ──────────────────────────────────────────────────────────────

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


# ── Main App ───────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='padding: 20px 0 10px 0;'>
    <div style='font-family: Space Mono, monospace; font-size: 11px; color: #6b6b8a; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;'>TAURIC-INSPIRED</div>
    <h1 style='margin: 0; font-size: 28px; font-family: Space Mono, monospace;'>Multi-Agent Trading AI</h1>
    <div style='color: #6b6b8a; font-size: 14px; margin-top: 6px;'>Powered by Groq · llama-3.3-70b-versatile</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    ticker_input = st.text_input("Stock Ticker", value="TSLA", placeholder="TSLA, NVDA, AAPL...")
    period = st.selectbox("Historical Period", ["3mo", "6mo", "1y", "2y"], index=1)

    st.markdown("<div class='section-title'>Agents</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:13px; color:#6b6b8a; line-height:2;'>
    📊 Technical Analyst<br>
    📰 Sentiment Analyst<br>
    💰 Fundamental Analyst<br>
    🎯 Risk Manager (Final)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#6b6b8a; line-height:1.8;'>
    Inspired by <b>TradingAgents</b> (Tauric Research).<br>
    Multi-agent LLM framework for stock analysis.<br><br>
    Data: yfinance · Free tier
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Analysis")

# Main content
if not run_btn:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px; color: #6b6b8a;'>
        <div style='font-size: 48px; margin-bottom: 16px;'>🤖</div>
        <div style='font-family: Space Mono, monospace; font-size: 14px; margin-bottom: 8px;'>4 AI Agents Ready</div>
        <div style='font-size: 13px;'>Enter your Groq API key and ticker, then click Run Analysis</div>
    </div>
    """, unsafe_allow_html=True)

elif not api_key:
    st.error("⚠️ Please enter your Groq API key in the sidebar.")

else:
    ticker_symbol = ticker_input.upper().strip()

    # Fetch data
    with st.spinner(f"Fetching {ticker_symbol} data..."):
        try:
            tk = yf.Ticker(ticker_symbol)
            df = tk.history(period=period)
            if df.empty:
                st.error(f"No data found for {ticker_symbol}. Check the ticker symbol.")
                st.stop()
            technicals = compute_technicals(df)
            fundamentals = get_fundamentals(tk)
            news_headlines = get_news(tk)
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            st.stop()

    # Price metrics row
    change = technicals["change_pct"]
    change_color = "up" if change >= 0 else "down"
    change_arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-item'>
            <div class='metric-label'>Price</div>
            <div class='metric-value'>${technicals['price']}</div>
        </div>
        <div class='metric-item'>
            <div class='metric-label'>Change</div>
            <div class='metric-value {change_color}'>{change_arrow} {abs(change)}%</div>
        </div>
        <div class='metric-item'>
            <div class='metric-label'>RSI (14)</div>
            <div class='metric-value {"up" if technicals["rsi"] < 70 else "down"}'>{technicals['rsi']}</div>
        </div>
        <div class='metric-item'>
            <div class='metric-label'>ATR</div>
            <div class='metric-value'>{technicals['atr']}</div>
        </div>
        <div class='metric-item'>
            <div class='metric-label'>52W High</div>
            <div class='metric-value'>${technicals['high_52w']}</div>
        </div>
        <div class='metric-item'>
            <div class='metric-label'>52W Low</div>
            <div class='metric-value'>${technicals['low_52w']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # News preview
    if news_headlines:
        st.markdown("<div class='section-title'>Recent Headlines</div>", unsafe_allow_html=True)
        for h in news_headlines[:4]:
            st.markdown(f"<div class='news-item'>📌 {h}</div>", unsafe_allow_html=True)

    # Run agents
    st.markdown("<div class='section-title'>Agent Analysis</div>", unsafe_allow_html=True)

    client = get_groq_client(api_key)
    progress = st.progress(0, text="Initialising agents...")

    agent_results = {}

    # ── Agent 1: Technical ────────────────────────────────────────────────────
    progress.progress(10, text="📊 Technical Analyst is working...")
    tech_prompt = f"""
Stock: {ticker_symbol}
Current Price: ${technicals['price']} ({'+' if change >= 0 else ''}{change}% today)
RSI(14): {technicals['rsi']}
MACD: {technicals['macd']} | Signal: {technicals['macd_signal']} | Histogram: {technicals['macd_hist']}
EMA20: {technicals['ema20']} | EMA50: {technicals['ema50']}
ATR(14): {technicals['atr']}
Bollinger Upper: {technicals['bb_upper']} | Lower: {technicals['bb_lower']}
Volume: {technicals['volume']:,} (Avg: {technicals['avg_volume']:,})
52W Range: ${technicals['low_52w']} - ${technicals['high_52w']}
"""
    try:
        agent_results["technical"] = call_agent(client, TECHNICAL_SYSTEM, tech_prompt)
    except Exception as e:
        agent_results["technical"] = f"Error: {e}"

    st.markdown(f"""
    <div class='agent-card technical'>
        <div class='agent-header'>
            <span class='agent-icon'>📊</span>
            <span class='agent-name'>Technical Analyst</span>
        </div>
        <div class='agent-content'>{agent_results['technical'].replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Agent 2: Sentiment ────────────────────────────────────────────────────
    progress.progress(35, text="📰 Sentiment Analyst is working...")
    headlines_text = "\n".join(f"- {h}" for h in news_headlines) if news_headlines else "No recent headlines available."
    sentiment_prompt = f"""
Stock: {ticker_symbol}
Recent News Headlines:
{headlines_text}

Also note: Today's price change is {'+' if change >= 0 else ''}{change}%, which may reflect market reaction to news.
"""
    try:
        agent_results["sentiment"] = call_agent(client, SENTIMENT_SYSTEM, sentiment_prompt)
    except Exception as e:
        agent_results["sentiment"] = f"Error: {e}"

    st.markdown(f"""
    <div class='agent-card sentiment'>
        <div class='agent-header'>
            <span class='agent-icon'>📰</span>
            <span class='agent-name'>Sentiment Analyst</span>
        </div>
        <div class='agent-content'>{agent_results['sentiment'].replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Agent 3: Fundamental ──────────────────────────────────────────────────
    progress.progress(60, text="💰 Fundamental Analyst is working...")

    def fmt(v):
        if v == "N/A" or v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        if isinstance(v, int) and v > 1_000_000:
            return f"${v/1e9:.1f}B"
        return str(v)

    fund_prompt = f"""
Stock: {ticker_symbol} | Sector: {fundamentals['sector']} | Industry: {fundamentals['industry']}
Market Cap: {fmt(fundamentals['market_cap'])}
Trailing PE: {fmt(fundamentals['pe_ratio'])} | Forward PE: {fmt(fundamentals['forward_pe'])}
Revenue: {fmt(fundamentals['revenue'])}
Profit Margin: {fmt(fundamentals['profit_margin'])}
Debt/Equity: {fmt(fundamentals['debt_to_equity'])}
ROE: {fmt(fundamentals['roe'])}
EPS: {fmt(fundamentals['eps'])}
Beta: {fmt(fundamentals['beta'])}
Short Ratio: {fmt(fundamentals['short_ratio'])}
Analyst Target Price: {fmt(fundamentals['analyst_target'])}
Analyst Consensus: {fundamentals['recommendation']}
"""
    try:
        agent_results["fundamental"] = call_agent(client, FUNDAMENTAL_SYSTEM, fund_prompt)
    except Exception as e:
        agent_results["fundamental"] = f"Error: {e}"

    st.markdown(f"""
    <div class='agent-card fundamental'>
        <div class='agent-header'>
            <span class='agent-icon'>💰</span>
            <span class='agent-name'>Fundamental Analyst</span>
        </div>
        <div class='agent-content'>{agent_results['fundamental'].replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Agent 4: Risk Manager (Final) ─────────────────────────────────────────
    progress.progress(80, text="🎯 Risk Manager synthesising...")
    risk_prompt = f"""
Stock under review: {ticker_symbol} @ ${technicals['price']}

TECHNICAL ANALYST REPORT:
{agent_results['technical']}

SENTIMENT ANALYST REPORT:
{agent_results['sentiment']}

FUNDAMENTAL ANALYST REPORT:
{agent_results['fundamental']}

Make your final decision.
"""
    try:
        agent_results["risk"] = call_agent(client, RISK_SYSTEM, risk_prompt)
    except Exception as e:
        agent_results["risk"] = f"Error: {e}"

    st.markdown(f"""
    <div class='agent-card risk'>
        <div class='agent-header'>
            <span class='agent-icon'>🎯</span>
            <span class='agent-name'>Risk Manager · Final Decision</span>
        </div>
        <div class='agent-content'>{agent_results['risk'].replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    progress.progress(100, text="✅ Analysis complete")
    time.sleep(0.5)
    progress.empty()

    # ── Verdict Box ───────────────────────────────────────────────────────────
    verdict = "HOLD"
    risk_text = agent_results.get("risk", "")
    for word in ["BUY", "SELL", "HOLD"]:
        if f"VERDICT: {word}" in risk_text.upper() or risk_text.upper().endswith(word):
            verdict = word
            break

    verdict_class = verdict.lower()
    verdict_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(verdict, "⚪")

    st.markdown(f"""
    <div class='verdict-box'>
        <div class='verdict-label'>Final Verdict · {ticker_symbol}</div>
        <div class='verdict-text {verdict_class}'>{verdict_emoji} {verdict}</div>
        <div style='font-size:12px; color:#6b6b8a; margin-top:8px; font-family: Space Mono, monospace;'>
            {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC · llama-3.3-70b-versatile
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px; padding:12px 16px; background:#1a1a24; border-radius:8px; border:1px solid #2a2a3a;'>
        <span style='font-size:11px; color:#6b6b8a;'>⚠️ <b>Disclaimer:</b> This is an AI-generated analysis for educational purposes only. 
        Not financial advice. Always do your own research before trading.</span>
    </div>
    """, unsafe_allow_html=True)
