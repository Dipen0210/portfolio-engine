# 🧠 smallQ — portfolio-engine

> **Automated Quant Portfolio Optimization & Risk Management System**  
> Build, optimize, and monitor multi-strategy portfolios with real-time rebalancing and risk control.

---

## 🌍 Overview

**smallQ** is a modular, AI-powered portfolio management engine that combines hedge-fund-level analytics with retail accessibility.

It integrates:
- 📊 **Automated stock selection** (based on chosen strategy)
- ⚙️ **Quantitative optimization** (Markowitz MVO, hybrid allocation)
- ⚖️ **Dynamic risk management** (VaR, CVaR, stress testing)
- 🔁 **Weekly or monthly rebalancing**
- 📈 **Performance & benchmark tracking** (vs S&P 500)
- 💡 **Streamlit dashboard** for user interaction and visualization

---

## 🏗️ Architecture


---

## ⚙️ Core Layers

| Stage | Layer / Module | Purpose / Task | Input | Output | Models / Techniques Used |
|-------|----------------|----------------|--------|---------|--------------------------|---------|
| 1️⃣ | **User Input Layer** | Collect user preferences | Sectors, capital, (optional) risk level | Structured user profile | 
| 2️⃣ | **Data Layer** | Fetch & preprocess OHLCV | Tickers, company.csv | Clean price data | yfinance API |
| 3️⃣ | **Hard Filter Layer** | Reduce universe (sector, market cap) | User input | Filtered stock universe | Basic filters | ✅ Done |
| 4️⃣ | **Strategy Layer** | Rank & select best stocks | Filtered universe | Ranked list | Momentum, Value, etc. | ✅ Done |
| 5️⃣ | **Forecasting Layer** | Estimate expected returns | Price history | μ (expected returns) | Rolling mean, CAPM | ✅ Done |
| 6️⃣ | **Portfolio Construction** | Build base portfolios | Ranked stocks + μ + Σ | Candidate portfolios | MVO, Utility Theory | ✅ Done |
| 7️⃣ | **Optimization Layer** | Find optimal weights | μ, Σ, constraints | Optimal weights | Markowitz, Convex Opt | ✅ Done |
| 8️⃣ | **Risk Management** | Evaluate & control risk | Portfolio weights | Risk metrics | VaR, CVaR, Beta, Vol | ✅ Done |
| 9️⃣ | **Stress Testing** | Test portfolio robustness | Portfolio + shocks | Stress results | Historical + Monte Carlo | ✅ Done |
| 🔟 | **Backtesting** | Evaluate historical performance | Portfolio weights | Equity curve | Rolling simulation | 🔜 Next |
| 11️⃣ | **Signal Generation** | Generate buy/sell signals | Optimized portfolio | Trade signals | Rebalance-driven logic | 🔜 Planned |
| 12️⃣ | **Execution** | Simulate or execute trades | Signals | Trade log | Paper trading / APIs | ⚙️ Optional |
| 13️⃣ | **Performance Layer** | Track and explain results | Portfolio history | Sharpe, Sortino, Alpha | Visualization | 🔜 Next |
| 14️⃣ | **Rebalancing** | Maintain target weights | Current vs target | Updated portfolio | Automated loop | ✅ Done |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/smallQ.git
cd smallQ

