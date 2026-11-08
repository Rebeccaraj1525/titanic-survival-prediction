# 🚢 Titanic Survival Prediction

My second Machine Learning project! Predicts whether a Titanic
passenger survived using Random Forest Classification with Scikit-learn.

## 📌 Key Findings
- Model achieved high accuracy using Random Forest Classifier
- Gender and Fare are the most important survival predictors
- Women had significantly higher survival rates than men
- First class passengers survived more than third class

## 🤖 Machine Learning Model
- Algorithm: Random Forest Classifier (100 trees)
- Library: Scikit-learn
- Train/Test Split: 80% / 20%
- Evaluation: Accuracy, Confusion Matrix, Classification Report

## 🖥️ Interactive Dashboard
Run the Streamlit dashboard with live survival predictor:
```bash
streamlit run dashboard.py
```

## 🛠️ Tools Used
- Python 3, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, Streamlit, Plotly

## ▶️ How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn streamlit plotly
python analysis.py
streamlit run dashboard.py
```

## 📂 Dataset
[Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)