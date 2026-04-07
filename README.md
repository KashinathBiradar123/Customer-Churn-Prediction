# 📉 Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn (leave the service), using the IBM Telco Customer Churn dataset augmented with 50,000 synthetic samples.

---

## 📌 Problem Statement

Customer churn is a major challenge for telecom companies. Retaining existing customers is far cheaper than acquiring new ones. This project builds a binary classifier to identify customers likely to churn, enabling proactive retention strategies.

---

## 📁 Project Structure
Customer-Churn-Prediction/
│
├── CustomerCh.ipynb                    # Main Jupyter Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original IBM Telco dataset (~7K rows)
├── churn_50k_synthetic.csv             # Synthetic augmented dataset (50K rows)
└── README.md
---

## 🔄 Workflow

1. **Load & Clean** — Read the original Telco CSV, coerce `TotalCharges` to numeric, and fill missing values.
2. **Synthetic Data Generation** — Expand the dataset to 50,000 rows by adding Gaussian noise to numeric features and occasional random swaps of categorical values.
3. **Preprocessing** — `StandardScaler` for numeric features; `OneHotEncoder` for categorical features via `ColumnTransformer`.
4. **Modeling** — `RandomForestClassifier` wrapped in a `sklearn Pipeline`.
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score on a held-out test set.

---

## 📊 Model Performance

| Metric      | Class 0 (No Churn) | Class 1 (Churn) | Overall |
|-------------|-------------------|-----------------|---------|
| Precision   | 0.92              | 0.84            | —       |
| Recall      | 0.95              | 0.77            | —       |
| F1-Score    | 0.93              | 0.81            | —       |
| **Accuracy**|                   |                 | **90%** |

> Trained on 22,500 samples, tested on 7,500 samples (stratified split).

---

## 🧰 Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn
- **Environment:** Jupyter Notebook

---

## 📦 Installation
```bash
git clone https://github.com/KashinathBiradar123/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install pandas numpy scikit-learn
```

---

## 🚀 Usage

1. Open `CustomerCh.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the `INPUT` path to point to your local copy of `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Run all cells top to bottom.

---

## 📂 Dataset

- **Source:** [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Original size:** ~7,043 rows
- **Synthetic size:** 50,000 rows (generated in-notebook)

**Key features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `InternetService`, and more.

---

## ⚠️ Known Issues / Notes

- A `FutureWarning` from pandas is raised on the `fillna(inplace=True)` line — this can be silenced by replacing it with:
```python
  df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
```
- Synthetic data is generated for experimentation only and should not be used for production models without caution.

---

## 🙋 Author

**Kashinath Biradar**  
[GitHub Profile](https://github.com/KashinathBiradar123)
[Linkedin Profile](https://www.linkedin.com/in/kashinath-biradar-23b0a0274/)
