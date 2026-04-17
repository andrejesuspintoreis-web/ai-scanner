#!/usr/bin/env python3
"""
AI Market Scanner Backend — Dados Reais Grátis
Fontes: Yahoo Finance (yfinance) + análise técnica (pandas-ta) + IA Groq (opcional, gratuita)
Gera scanner_data.json para o dashboard HTML

Instalar: pip install yfinance pandas pandas-ta groq requests
Correr:   python scanner_backend.py
"""

import json, time, math, sys
from datetime import datetime, timezone

# ── Dependências ───────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
except ImportError:
    print("Instalar dependências: pip install yfinance pandas pandas-ta")
    sys.exit(1)

# ── Configuração ──────────────────────────────────────────────────────────────
GROQ_API_KEY = ""          # Opcional: chave gratuita em console.groq.com
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Modelo gratuito Groq

# Ativos a analisar — totalmente personalizável
WATCHLIST = {
    "Tech": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","ASML"],
    "ETFs": ["SPY","QQQ","VTI","ARKK","SOXX","GLD","TLT"],
    "Cripto": ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"],
    "Europa": ["SAP.DE","ASML.AS","MC.PA","SAN.MC"],
}

FLAT_LIST = [t for lst in WATCHLIST.values() for t in lst]

NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","GOOGL":"Alphabet",
    "META":"Meta","AMZN":"Amazon","TSLA":"Tesla","ASML":"ASML",
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","VTI":"US Total Market",
    "ARKK":"ARK Innovation","SOXX":"Semiconductor ETF","GLD":"Gold ETF","TLT":"20yr Treasury ETF",
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana","BNB-USD":"BNB",
    "SAP.DE":"SAP","ASML.AS":"ASML (AMS)","MC.PA":"LVMH","SAN.MC":"Santander",
}

# ── Yahoo Finance — Dados Reais ───────────────────────────────────────────────
def fetch_data(ticker: str):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        return df
    except Exception as e:
        print(f"  Erro ao buscar {ticker}: {e}")
        return None

# ── Indicadores Técnicos ──────────────────────────────────────────────────────
def calc_indicators(df):
    close = df["close"]
    # RSI
    rsi_s = ta.rsi(close, length=14)
    rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
    # SMA
    sma20 = float(ta.sma(close, length=20).iloc[-1])
    sma50 = float(ta.sma(close, length=50).iloc[-1])
    # MACD
    macd_df = ta.macd(close)
    macd  = float(macd_df.iloc[-1, 0]) if macd_df is not None else 0.0
    signal= float(macd_df.iloc[-1, 1]) if macd_df is not None else 0.0
    # Bollinger
    bb = ta.bbands(close, length=20)
    bb_low  = float(bb.iloc[-1, 0]) if bb is not None else 0.0
    bb_mid  = float(bb.iloc[-1, 2]) if bb is not None else 0.0
    bb_hi   = float(bb.iloc[-1, 4]) if bb is not None else 0.0
    # Volume médio
    vol_avg = float(df["volume"].rolling(20).mean().iloc[-1])
    vol_cur = float(df["volume"].iloc[-1])
    # Preço atual e variação
    price   = float(close.iloc[-1])
    price_y = float(close.iloc[-2]) if len(close) > 1 else price
    change  = (price - price_y) / price_y * 100
    return dict(
        price=round(price,4), change=round(change,3), rsi=round(rsi,2),
        sma20=round(sma20,4), sma50=round(sma50,4), macd=round(macd,5),
        macd_signal=round(signal,5), bb_low=round(bb_low,4),
        bb_mid=round(bb_mid,4), bb_hi=round(bb_hi,4),
        vol_ratio=round(vol_cur/vol_avg,2) if vol_avg > 0 else 1.0
    )

# ── Lógica de Sinal (sem IA) ─────────────────────────────────────────────────
def rule_based_signal(ind):
    score = 0
    reasons = []

    # RSI
    if ind["rsi"] < 32:
        score += 2; reasons.append(f"RSI={ind['rsi']} — sobrevendido")
    elif ind["rsi"] < 45:
        score += 1; reasons.append(f"RSI={ind['rsi']} — zona de compra")
    elif ind["rsi"] > 68:
        score -= 2; reasons.append(f"RSI={ind['rsi']} — sobrecomprado")
    elif ind["rsi"] > 55:
        score -= 1; reasons.append(f"RSI={ind['rsi']} — pressão de venda")

    # Preço vs SMAs
    if ind["price"] > ind["sma20"] > ind["sma50"]:
        score += 1; reasons.append("Preço acima SMA20 e SMA50 — tendência alta")
    elif ind["price"] < ind["sma20"] < ind["sma50"]:
        score -= 1; reasons.append("Preço abaixo SMAs — tendência baixista")

    # MACD
    if ind["macd"] > ind["macd_signal"] and ind["macd"] > 0:
        score += 1; reasons.append("MACD bullish — momentum positivo")
    elif ind["macd"] < ind["macd_signal"] and ind["macd"] < 0:
        score -= 1; reasons.append("MACD bearish — momentum negativo")

    # Bollinger
    if ind["price"] < ind["bb_low"]:
        score += 1; reasons.append("Abaixo Bollinger inferior — possível recuperação")
    elif ind["price"] > ind["bb_hi"]:
        score -= 1; reasons.append("Acima Bollinger superior — possível correção")

    # Volume
    if ind["vol_ratio"] > 1.5:
        if score > 0:
            score += 1; reasons.append(f"Volume {ind['vol_ratio']}× acima da média — confirma impulso")
        else:
            score -= 1; reasons.append(f"Volume {ind['vol_ratio']}× — aumenta pressão de venda")

    # Decisão
    if score >= 2:
        signal = "BUY"; confidence = min(50 + score*8, 92)
    elif score <= -2:
        signal = "SELL"; confidence = min(50 + abs(score)*8, 88)
    else:
        signal = "HOLD"; confidence = 45 + abs(score)*5

    return signal, confidence, reasons[:3]

# ── Groq IA (opcional, gratuita) ─────────────────────────────────────────────
def groq_signal(ticker, ind, base_signal, base_conf, base_reasons):
    if not GROQ_API_KEY:
        return base_signal, base_conf, base_reasons
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analisa o ativo {ticker} com estes dados técnicos:
Preço: ${ind['price']} | Variação hoje: {ind['change']}% | RSI(14): {ind['rsi']}
SMA20: {ind['sma20']} | SMA50: {ind['sma50']} | MACD: {ind['macd']} | Signal: {ind['macd_signal']}
Bollinger: [{ind['bb_low']}, {ind['bb_mid']}, {ind['bb_hi']}] | Volume ratio: {ind['vol_ratio']}x

Responde APENAS em JSON (sem explicação fora do JSON):
{{"signal":"BUY/SELL/HOLD","confidence":0-100,"reasons":["razão 1","razão 2","razão 3"]}}
Máximo 12 palavras por razão. Em português."""

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        # extrair JSON
        start = content.find("{"); end = content.rfind("}")+1
        parsed = json.loads(content[start:end])
        return parsed["signal"], int(parsed["confidence"]), parsed["reasons"]
    except Exception as e:
        print(f"  Groq fallback para {ticker}: {e}")
        return base_signal, base_conf, base_reasons

# ── Calcular Stop / Target ─────────────────────────────────────────────────────
def calc_levels(price, signal, rsi, bb_low, bb_hi, sma20):
    if signal == "BUY":
        stop   = round(price * 0.96, 2)   # -4%
        target = round(price * 1.10, 2)   # +10%
    elif signal == "SELL":
        stop   = round(price * 1.04, 2)
        target = round(price * 0.92, 2)
    else:
        stop   = round(price * 0.95, 2)
        target = round(price * 1.07, 2)
    return stop, target

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  AI Market Scanner — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"  Analisando {len(FLAT_LIST)} ativos com dados reais...")
    if GROQ_API_KEY:
        print(f"  IA Groq ativa: {GROQ_MODEL}")
    else:
        print(f"  Modo: análise técnica (sem chave Groq)")
    print(f"{'='*60}\n")

    signals = []
    for i, ticker in enumerate(FLAT_LIST):
        print(f"  [{i+1:02d}/{len(FLAT_LIST)}] {ticker}...", end=" ")
        df = fetch_data(ticker)
        if df is None:
            print("ERRO — a saltar")
            continue
        try:
            ind = calc_indicators(df)
            base_sig, base_conf, base_reasons = rule_based_signal(ind)
            signal, conf, reasons = groq_signal(ticker, ind, base_sig, base_conf, base_reasons)
            stop, target = calc_levels(ind["price"], signal, ind["rsi"], ind["bb_low"], ind["bb_hi"], ind["sma20"])
            entry = {
                "ticker":     ticker,
                "name":       NAMES.get(ticker, ticker),
                "price":      ind["price"],
                "change":     ind["change"],
                "rsi":        ind["rsi"],
                "sma20":      ind["sma20"],
                "macd":       ind["macd"],
                "signal":     signal,
                "confidence": conf,
                "reasons":    reasons,
                "stop":       stop,
                "target":     target,
            }
            signals.append(entry)
            print(f"{signal} ({conf}%) | P:${ind['price']} RSI:{ind['rsi']}")
        except Exception as e:
            print(f"ERRO: {e}")
        time.sleep(0.4)   # respeitar rate limits

    # Estatísticas
    buy_c  = sum(1 for s in signals if s["signal"]=="BUY")
    sell_c = sum(1 for s in signals if s["signal"]=="SELL")
    avg_c  = round(sum(s["confidence"] for s in signals)/len(signals),1) if signals else 0

    output = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "total":      len(signals),
        "buy_count":  buy_c,
        "sell_count": sell_c,
        "avg_conf":   avg_c,
        "signals":    sorted(signals, key=lambda x: x["confidence"], reverse=True)
    }

    with open("scanner_data.json","w",encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  PRONTO! {len(signals)} ativos analisados")
    print(f"  BUY: {buy_c}  SELL: {sell_c}  HOLD: {len(signals)-buy_c-sell_c}")
    print(f"  Confiança média: {avg_c}%")
    print(f"  Ficheiro: scanner_data.json")
    print(f"\n  Abre ai-market-scanner.html no browser!")
    print(f"  Para auto-refresh contínuo, adiciona ao cron / task scheduler.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
