
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Project paths
DATA_PATH = "data/Student_Performance_DT - Student_Performance_DT.csv"
MODEL_PATH = "models/student_performance_model.pkl"
METRICS_PATH = "reports/metrics.json"

print("=" * 55)
print("STUDENT PERFORMANCE - ML TRAINING PIPELINE")
print("=" * 55)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)

# Separate features and target
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Create ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Model...")

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, pos_label="Pass", zero_division=0), 4),
    "recall": round(recall_score(y_test, y_pred, pos_label="Pass", zero_division=0), 4),
    "f1_score": round(f1_score(y_test, y_pred, pos_label="Pass", zero_division=0), 4)
}

# Save model
joblib.dump(model, MODEL_PATH)

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel Training Completed!")
print("\nModel Evaluation:")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nModel saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)

print("\nTraining Pipeline Completed Successfully!")
