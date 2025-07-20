#  Stock Return Forecasting Using LSTM & Machine Learning

This project explores how machine learning — especially **Long Short-Term Memory (LSTM)** networks — can be applied to forecast short-term stock returns using historical lag features. It also compares traditional models like **Random Forest** and **XGBoost**, offering practical insights into their strengths and limitations.

>  Blends concepts from AI, time-series forecasting, and financial return behavior  
>  Built to deepen practical understanding of sequential modeling in finance  
>  Reflects the intersection of data science and modern quantitative analysis

---

##  Problem Statement

Can we forecast short-term **stock return movements** using only past return data (lagged features)?  
This simulates what many **quantitative models** in trading and portfolio construction attempt.

---

##  Goals

- Predict next-day returns based on past patterns
- Compare performance of **tree-based models** vs. **deep learning**
- Build a reusable ML pipeline (scaling, saving, evaluation)
- Explore sequential modeling with LSTM for financial time series

---

##  Project Structure

| Folder        | Contents                                     |
|---------------|----------------------------------------------|
| `data/`       | Preprocessed stock data, predictions, scalers|
| `models/`     | Trained ML models (LSTM `.h5`, RF/XGB `.pkl`)|
| `notebooks/`  | Jupyter notebooks for each modeling approach |
| `docs/`       | Explanations with diagrams and code walkthroughs |

---

##  Models Used

| Model          | Highlights                                     |
|----------------|-----------------------------------------------|
| **Random Forest** | Fast, stable, and interpretable on tabular data |
| **XGBoost**        | Regularized boosting with strong tabular performance |
| **LSTM**           | Captures time dependencies, useful for sequence patterns |

```
               ┌──────────────────────────┐
               │  Preprocessed CSV File   │
               │  (Lag_1, Lag_2, Lag_3)   │
               └────────────┬─────────────┘
                            ↓
                 ┌────────────────────┐
                 │ MinMaxScaler (X, y)│
                 └────────────┬───────┘
                            ↓
          ┌────────────────────────────────┐
          │  Time Series Sequence Creation │                                                 >> LSTM MODEL SHOWCASE <<
          │  [samples, 5 days, 3 features] │
          └────────────────┬───────────────┘
                           ↓
                 ┌────────────────────┐                                                       
                 │    LSTM Model      │
                 │(64 Units + Dropout)│                                              
                 └────────────┬───────┘
                            ↓
                     ┌────────────┐
                     │ Dense(1)   │
                     └────┬───────┘
                          ↓
                ┌────────────────────┐
                │ Inverse Scaler (y) │
                └────────────┬───────┘
                            ↓
             ┌────────────────────────────┐
             │  Predicted Return (Output) │
             └────────────────────────────┘
```


---

##  Performance Summary (on Tesla Returns)

| Model        | MAE      | RMSE     | R² Score  |
|--------------|----------|----------|-----------|
| Random Forest| 0.03165  | 0.00191  | -0.0201   |
| XGBoost      | 0.03291  | 0.00208  | -0.1084   |
| LSTM         | 0.03226  | 0.04449  | -0.0545   |

>  **Insight**:  
> While LSTM is built for sequences, simpler tree-based models like Random Forest outperformed it here — possibly due to the limited input features (no volume, no technical indicators).
This shows the importance of feature richness in deep learning for finance.

---

##  Why This Project is Finance-Relevant

- **Return Prediction** is central to:
  - Algorithmic Trading
  - Portfolio Optimization
  - Quantitative Research

- **LSTM** mimics trader behavior by using **memory across days**, unlike static models

- Reflects real industry practices:
  - Avoids data leakage by preserving sequence
  - Includes model saving & scaling pipelines
  - Evaluates models on unseen data

> This project demonstrates practical awareness of **modeling returns realistically**, not just achieving the lowest error.

---

##  How to Reproduce

1. Clone this repo
2. Use provided Tesla dataset or plug in your own return data
3. Run:
    - `random_forest_model.ipynb`
    - `xgboost_model.ipynb`
    - `lstm_model.ipynb`
4. Check predictions & visualizations

---

##  Documentation

All model decisions, diagrams, and preprocessing logic are fully explained in the `docs/` folder:

- LSTM input structure (3D)
- MinMaxScaler rationale
- Model saving with `joblib`
- Sequential data creation logic

---

## ✅ Summary

This project investigates **how machine learning can be applied to return forecasting**, a core problem in quantitative finance. It explores the pros and cons of different modeling approaches and builds a **clean, reproducible ML pipeline** for future use or extension.

---


