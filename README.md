# Adaptive-FinMem-LLM-StockTrading

> **Market-Regime Adaptive Memory Management for LLM-Based Stock Trading Agents**

An improved **FinMem** framework that introduces market-state awareness into memory retrieval, feedback reinforcement, and reflection, enabling an LLM-based trading agent to adapt its memory management to changing market conditions.

---

## Overview

**Adaptive-FinMem** is an improved version of the open-source **FinMem** framework for LLM-based financial trading agents.

The project focuses on a key limitation of static memory management in financial environments:

> **Historical trading experience should not be treated equally under different market conditions.**

Based on FinMem's original **Perception → Memory → Decision → Feedback** closed-loop architecture and hierarchical memory system, this project introduces three market-state-adaptive mechanisms:

* **RGR — Regime-Gated Retrieval**
  Dynamically adjusts memory retrieval weights according to the current market regime identified from rolling volatility and return trends.

* **CFR — Coefficient-based Feedback Reinforcement**
  Replaces the original discrete feedback signal with continuous risk-adjusted feedback to enable finer-grained memory importance updates.

* **ETR — Event-Triggered Reflection**
  Dynamically triggers additional reflection when risk events such as consecutive losses, large drawdowns, or abnormal daily fluctuations occur.

---

## System Diagram
Together, these mechanisms form an adaptive memory loop:


**Figure 1 — Overall architecture of Adaptive-FinMem**
![Adaptive-FinMem Architecture](docs/images/system_architecture.png)

The figure should show the original FinMem **Perception → Memory → Decision → Feedback** loop and clearly highlight where **RGR, CFR, and ETR** are introduced.

### RGR: Regime-Gated Retrieval

RGR introduces a market-state-dependent gating layer into FinMem's memory retrieval process.

The current market state is identified using:

* Rolling volatility
* Return trend

The retrieval weights of:

* Semantic similarity
* Recency
* Memory importance

are then dynamically adjusted according to the detected regime.

Therefore, the same query can retrieve different historical experiences under different market conditions.

![Adaptive-FinMem Architecture](docs/images/rgr.png)



### CFR: Coefficient-based Feedback Reinforcement

The original FinMem uses a discrete feedback signal of:

```text
{-1, 0, +1}
```

CFR converts this into a continuous risk-adjusted feedback signal based on trading returns and volatility:

```text
Feedback = tanh(Return / Volatility)
```

The resulting feedback strength is used to update memory importance, allowing the system to distinguish between trading experiences with different levels of effectiveness and risk.

![Adaptive-FinMem Architecture](docs/images/cfr.png)



### ETR: Event-Triggered Reflection

ETR supplements the regular reflection process with event-driven reflection.

Additional reflection is triggered when predefined risk events occur, including:

* Consecutive losses
* Significant drawdown
* Abnormal daily price fluctuations

This allows the agent to perform additional experience consolidation during high-risk market conditions without continuously increasing reflection frequency.

![Adaptive-FinMem Architecture](docs/images/etr.png)


## Experiment

The framework is evaluated on historical **Apple Inc. (AAPL)** financial data.

The experimental environment integrates:

* Historical price data
* Financial news
* Financial reports
* Hierarchical memory
* FAISS-based semantic retrieval
* LLM-based trading decisions

The test period covers **2024–2025**.

The market is further divided into four representative stages:

| Stage                  | Period                       |
| ---------------------- | ---------------------------- |
| Bull Market            | Jan. 1, 2024 – Aug. 2, 2024  |
| High-Volatility Market | Aug. 3, 2024 – Dec. 31, 2024 |
| Range-bound Market     | Jan. 1, 2025 – Jun. 30, 2025 |
| Bear Market            | Jul. 1, 2025 – Dec. 31, 2025 |

These periods are used to examine the adaptability of the proposed memory mechanism under different market conditions.

---

## Baselines

The proposed **Full** model is compared with:

* Buy & Hold
* DQN
* PPO
* A2C
* FinGPT
* Original FinMem

Ablation variants are also evaluated:

* Full(RGR)
* Full(CFR)
* Full(ETR)
* Full = RGR + CFR + ETR

---

## Results

### Overall Performance

The main evaluation uses:

* **CR** — Cumulative Return
* **SR** — Sharpe Ratio
* **DailySTD** — Daily Return Volatility
* **AV** — Annualized Volatility
* **MDD** — Maximum Drawdown

| Method            |         CR |        SR |    DailySTD |           AV |        MDD |
| ----------------- | ---------: | --------: | ----------: | -----------: | ---------: |
| Buy & Hold        |      7.45% |     0.144 |     1.5774% |     25.0412% |     44.94% |
| DQN               |     50.61% |     1.147 |     1.3415% |     21.2962% |     30.16% |
| PPO               |      2.70% |     0.057 |     1.4496% |     23.0122% |     37.72% |
| A2C               |     -0.06% |    -0.001 |     1.5207% |     24.1399% |     28.51% |
| FinGPT            |     57.62% |     1.144 |     1.5321% |     24.3216% |     21.88% |
| FinMem (Original) |     46.38% |     0.965 |     1.4612% |     23.1955% |     17.71% |
| Full(CFR)         |     70.30% |     1.449 |     1.4755% |     23.4222% |     15.43% |
| Full(ETR)         |     60.80% |     1.326 |     1.3945% |     22.1364% |     15.17% |
| Full(RGR)         |     66.28% |     1.425 |     1.4147% |     22.4583% |     10.45% |
| **Full**          | **79.73%** | **1.619** | **1.4076%** | **22.7741%** | **11.39%** |

The complete model achieves a **79.73% cumulative return** and a **1.619 Sharpe ratio**, while reducing maximum drawdown to **11.39%**, compared with **46.38% / 0.965 / 17.71%** for the original FinMem.

### Performance Comparison

> **Figure 2 — Cumulative return curves**


![Cumulative Return Comparison](docs/images/cumulative_returns.png)




The figure should contain the cumulative net-worth curves from January 2024 to December 2025 for the baseline methods and the proposed variants. The thesis reports that the complete **Full** model achieves the strongest overall cumulative-return performance, while RGR and CFR provide particularly noticeable improvements among the individual modules.

---

## Ablation Study

The three mechanisms contribute to different aspects of performance.

* **CFR** provides the most pronounced contribution to return performance.
* **RGR** mainly contributes to downside-risk control and market adaptability.
* **ETR** provides additional support under high-risk and volatile conditions.
* Combining the three mechanisms produces the strongest overall performance.

The statistical analysis further supports these observations. For example, compared with original FinMem, the Full model achieves:

* CR: **p = 0.0000**
* SR: **p = 0.0000**
* MDD: **p = 0.1596**

Compared with FinGPT:

* CR: **p = 0.0055**
* SR: **p = 0.0024**
* MDD: **p = 0.0000**

The module-level analysis reports significant improvements for CFR in CR and SR, while RGR shows its strongest statistical effect on MDD.

> **Figure 3 — Ablation study**


![Ablation Study](docs/images/ablation.png)

The figure directly compare **Original FinMem, Full(RGR), Full(CFR), Full(ETR), and Full**, preferably using cumulative-return curves or a compact metric comparison.

---

## Market-Regime Analysis

To evaluate adaptation under changing market conditions, the test period is divided into:

```text
Bull
  ↓
High Volatility
  ↓
Range-bound
  ↓
Bear
```

> **Figure 4 — Performance across market regimes**

![Market Regime Analysis](docs/images/regime_analysis.png)

This figure shows the performance of the proposed Full model and representative baselines across the four market stages, emphasizing how the adaptive memory mechanism behaves when the market regime changes.


# 🚀 Usage

The implementation follows the original **FinMem** workflow.

The proposed mechanisms are integrated into the existing FinMem framework, so the overall usage process remains essentially unchanged:

## Disclaimer

This project is intended for **research and educational purposes only**.

The reported results are historical backtesting results and should not be interpreted as financial advice or as a guarantee of future trading performance.
