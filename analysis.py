import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

# ── Load Data ─────────────────────────────────────
df = pd.read_csv("Titanic-Dataset.csv")

print("Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# ── Data Cleaning ─────────────────────────────────
# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())
# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Drop Cabin — too many missing values
df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"],
        inplace=True)

print("\nAfter cleaning — Missing Values:")
print(df.isnull().sum())

# ── Encode Categorical Columns ────────────────────
le = LabelEncoder()
df["Sex"]      = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ── Chart 1: Survival Count ───────────────────────
plt.figure(figsize=(7, 5))
survival_counts = df["Survived"].value_counts()
colors = ["#FF7043", "#4CAF50"]
plt.bar(["Did Not Survive", "Survived"],
        survival_counts.values, color=colors, edgecolor="white")
for i, val in enumerate(survival_counts.values):
    plt.text(i, val + 5, str(val), ha="center",
             fontsize=13, fontweight="bold")
plt.title("Survival Count", fontsize=16, fontweight="bold")
plt.ylabel("Number of Passengers", fontsize=13)
plt.tight_layout()
plt.savefig("chart1_survival_count.png", dpi=150)
plt.show()
print("Chart 1 done ✅")

# ── Chart 2: Survival by Gender ───────────────────
plt.figure(figsize=(8, 5))
gender_survival = df.groupby(["Sex", "Survived"]).size().unstack()
gender_survival.index = ["Female", "Male"]
gender_survival.columns = ["Did Not Survive", "Survived"]
gender_survival.plot(kind="bar", color=["#FF7043", "#4CAF50"],
                     edgecolor="white", rot=0)
plt.title("Survival by Gender", fontsize=16, fontweight="bold")
plt.xlabel("Gender", fontsize=13)
plt.ylabel("Number of Passengers", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("chart2_survival_by_gender.png", dpi=150)
plt.show()
print("Chart 2 done ✅")

# ── Chart 3: Survival by Passenger Class ──────────
plt.figure(figsize=(8, 5))
class_survival = df.groupby(["Pclass", "Survived"]).size().unstack()
class_survival.columns = ["Did Not Survive", "Survived"]
class_survival.plot(kind="bar", color=["#FF7043", "#4CAF50"],
                    edgecolor="white", rot=0)
plt.title("Survival by Passenger Class",
          fontsize=16, fontweight="bold")
plt.xlabel("Passenger Class (1=First, 2=Second, 3=Third)",
           fontsize=13)
plt.ylabel("Number of Passengers", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("chart3_survival_by_class.png", dpi=150)
plt.show()
print("Chart 3 done ✅")

# ── Chart 4: Age Distribution by Survival ─────────
plt.figure(figsize=(10, 5))
df[df["Survived"] == 1]["Age"].hist(
    bins=20, alpha=0.7, color="#4CAF50",
    edgecolor="white", label="Survived")
df[df["Survived"] == 0]["Age"].hist(
    bins=20, alpha=0.7, color="#FF7043",
    edgecolor="white", label="Did Not Survive")
plt.title("Age Distribution by Survival",
          fontsize=16, fontweight="bold")
plt.xlabel("Age", fontsize=13)
plt.ylabel("Number of Passengers", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("chart4_age_distribution.png", dpi=150)
plt.show()
print("Chart 4 done ✅")

# ── Prepare ML Data ───────────────────────────────
print("\nPreparing data for Machine Learning...")
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ── Train Random Forest Model ─────────────────────
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully ✅")

# ── Predictions & Evaluation ──────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n── Model Performance ──────────────────────")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Did Not Survive", "Survived"]))

# ── Chart 5: Confusion Matrix ─────────────────────
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"])
disp.plot(cmap="Blues", ax=plt.gca())
plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("chart5_confusion_matrix.png", dpi=150)
plt.show()
print("Chart 5 done ✅")

# ── Chart 6: Feature Importance ───────────────────
plt.figure(figsize=(10, 6))
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

sns.barplot(x="Importance", y="Feature",
            data=feat_imp, palette="Blues_r")
plt.title("Feature Importance — Random Forest",
          fontsize=16, fontweight="bold")
plt.xlabel("Importance Score", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.tight_layout()
plt.savefig("chart6_feature_importance.png", dpi=150)
plt.show()
print("Chart 6 done ✅")

print(f"\n🎉 All done! Model Accuracy: {accuracy*100:.2f}%")