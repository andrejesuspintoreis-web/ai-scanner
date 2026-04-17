# AI Market Scanner — Dados Reais Grátis

## Instalação (1 vez)
```bash
pip install yfinance pandas pandas-ta groq
```

## Uso
```bash
python scanner_backend.py
```
Depois abre `ai-market-scanner.html` no browser.

## IA Groq (opcional, 100% grátis)
1. Cria conta em https://console.groq.com
2. Gera uma API key (começa por `gsk_`)
3. Cola em `scanner_backend.py` na linha: `GROQ_API_KEY = "gsk_..."`

## Fontes de dados
- **Preços / histórico**: Yahoo Finance via yfinance (gratuito, sem registo)
- **Indicadores**: pandas-ta (local, sem API)
- **Análise IA**: Groq llama-3.3-70b (opcional, tier gratuito)

## Automação
- Linux/Mac: `crontab -e` → `*/15 * * * * cd /pasta && python scanner_backend.py`
- Windows: Task Scheduler → executar a cada 15 min
