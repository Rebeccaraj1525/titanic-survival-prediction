import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ── Load & Train ──────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Titanic-Dataset.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"],
            inplace=True)
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return df, model, X, X_test, y_test, y_pred

df, model, X, X_test, y_test, y_pred = load_and_train()
accuracy = accuracy_score(y_test, y_pred)

# ── Title ─────────────────────────────────────────
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("Interactive ML dashboard using Random Forest to predict Titanic passenger survival.")
st.markdown("---")

# ── KPI Metrics ───────────────────────────────────
survived     = df["Survived"].sum()
not_survived = len(df) - survived
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Passengers", f"{len(df):,}")
col2.metric("✅ Survived", f"{survived:,}")
col3.metric("❌ Did Not Survive", f"{not_survived:,}")
col4.metric("🤖 Model Accuracy", f"{accuracy*100:.2f}%")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis",
                              "🤖 ML Model",
                              "🔮 Survival Predictor"])

# ── Tab 1: Data Analysis ──────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Survival Count")
        survival_df = df["Survived"].value_counts().reset_index()
        survival_df.columns = ["Survived", "Count"]
        survival_df["Status"] = survival_df["Survived"].map(
            {0: "Did Not Survive", 1: "Survived"})
        fig1 = px.bar(survival_df, x="Status", y="Count",
                      color="Status",
                      color_discrete_map={
                          "Survived": "#4CAF50",
                          "Did Not Survive": "#FF7043"},
                      title="Survival Count")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Survival by Gender")
        gender_df = df.groupby(["Sex", "Survived"]).size()\
                      .reset_index(name="Count")
        gender_df["Gender"] = gender_df["Sex"].map(
            {0: "Female", 1: "Male"})
        gender_df["Status"] = gender_df["Survived"].map(
            {0: "Did Not Survive", 1: "Survived"})
        fig2 = px.bar(gender_df, x="Gender", y="Count",
                      color="Status",
                      barmode="group",
                      color_discrete_map={
                          "Survived": "#4CAF50",
                          "Did Not Survive": "#FF7043"},
                      title="Survival by Gender")
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Survival by Passenger Class")
        class_df = df.groupby(["Pclass", "Survived"]).size()\
                     .reset_index(name="Count")
        class_df["Status"] = class_df["Survived"].map(
            {0: "Did Not Survive", 1: "Survived"})
        fig3 = px.bar(class_df, x="Pclass", y="Count",
                      color="Status", barmode="group",
                      color_discrete_map={
                          "Survived": "#4CAF50",
                          "Did Not Survive": "#FF7043"},
                      title="Survival by Class",
                      labels={"Pclass": "Passenger Class"})
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Age Distribution by Survival")
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=df[df["Survived"] == 1]["Age"],
            name="Survived", opacity=0.7,
            marker_color="#4CAF50", nbinsx=20))
        fig4.add_trace(go.Histogram(
            x=df[df["Survived"] == 0]["Age"],
            name="Did Not Survive", opacity=0.7,
            marker_color="#FF7043", nbinsx=20))
        fig4.update_layout(
            barmode="overlay",
            title="Age Distribution by Survival",
            xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

# ── Tab 2: ML Model ───────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig5 = px.imshow(cm,
                         labels=dict(x="Predicted", y="Actual"),
                         x=["Did Not Survive", "Survived"],
                         y=["Did Not Survive", "Survived"],
                         text_auto=True,
                         color_continuous_scale="Blues",
                         title="Confusion Matrix")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig6 = px.bar(feat_imp, x="Importance", y="Feature",
                      orientation="h",
                      color="Importance",
                      color_continuous_scale="Blues",
                      title="Feature Importance")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### 📊 Model Performance")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Accuracy", f"{accuracy*100:.2f}%")
    mc2.metric("Algorithm", "Random Forest")
    mc3.metric("Trees", "100 estimators")

# ── Tab 3: Survival Predictor ─────────────────────
with tab3:
    st.subheader("🔮 Will This Passenger Survive?")
    st.markdown("Fill in the passenger details and find out!")

    col1, col2, col3 = st.columns(3)
    with col1:
        pclass = st.selectbox("Passenger Class",
                              [1, 2, 3],
                              format_func=lambda x:
                              f"Class {x} ({'First' if x==1 else 'Second' if x==2 else 'Third'})")
        sex    = st.selectbox("Gender", ["female", "male"])
        age    = st.slider("Age", 1, 80, 30)
    with col2:
        sibsp  = st.slider("Siblings / Spouse aboard", 0, 8, 0)
        parch  = st.slider("Parents / Children aboard", 0, 6, 0)
        fare   = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
    with col3:
        embarked = st.selectbox("Port of Embarkation",
                                ["C", "Q", "S"],
                                format_func=lambda x:
                                {"C": "Cherbourg (C)",
                                 "Q": "Queenstown (Q)",
                                 "S": "Southampton (S)"}[x])

    sex_enc      = 0 if sex == "female" else 1
    embarked_enc = {"C": 0, "Q": 1, "S": 2}[embarked]

    input_data = pd.DataFrame([[pclass, sex_enc, age,
                                 sibsp, parch, fare,
                                 embarked_enc]],
                               columns=X.columns)

    if st.button("🔮 Predict Survival", type="primary"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success(f"### ✅ This passenger SURVIVED!")
            st.success(f"Survival probability: {probability[1]*100:.1f}%")
        else:
            st.error(f"### ❌ This passenger did NOT survive.")
            st.error(f"Survival probability: {probability[1]*100:.1f}%")

        st.info(f"Model confidence: {max(probability)*100:.1f}%")

# ── Footer ────────────────────────────────────────
st.markdown("---")
st.markdown("**Built by Rebecca Raj A** | Dataset: Titanic (Kaggle) | Tools: Python, Streamlit, Scikit-learn, Plotly")