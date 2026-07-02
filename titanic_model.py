import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ============= LOAD DATA =============
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Make column names consistent with the files in this folder
train.columns = [col.strip().lower() for col in train.columns]
test.columns = [col.strip().lower() for col in test.columns]

# Ensure the test set has a PassengerId column
if "passengerid" not in test.columns:
    test.insert(0, "passengerid", range(1, len(test) + 1))

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}\n")

# ============= FEATURE ENGINEERING =============
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Combine train and test for preprocessing
combined = pd.concat([train.drop(columns=["survived"]), test], ignore_index=True)

# Fill missing values
combined["sex"] = combined["sex"].fillna("female").astype(str).str.lower().map({"male": 1, "female": 0}).fillna(0).astype(int)
combined["embarked"] = combined["embarked"].fillna(combined["embarked"].mode()[0]).astype(str).str.lower().map({"s": 0, "c": 1, "q": 2}).fillna(0).astype(int)
combined["age"] = combined["age"].fillna(combined["age"].median())
combined["fare"] = combined["fare"].fillna(combined["fare"].median())

# Create simple family-based features
combined["family_size"] = combined["sibsp"] + combined["parch"] + 1
combined["is_alone"] = (combined["family_size"] == 1).astype(int)

# Create age and fare bins
combined["age_bin"] = pd.cut(combined["age"], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)
combined["fare_bin"] = pd.qcut(combined["fare"], q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

print("OK: Missing values handled")
print("OK: Family and bin features created")

# ============= SELECT FEATURES =============
features_to_use = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "family_size", "is_alone", "age_bin", "fare_bin"]

X_train = combined.iloc[:len(train)][features_to_use].copy()
X_test = combined.iloc[len(train):][features_to_use].copy()
y_train = train["survived"].astype(int).copy()

print(f"\nFeatures selected: {len(features_to_use)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============= SCALE FEATURES =============
print("\n" + "=" * 60)
print("SCALING FEATURES")
print("=" * 60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("OK: Features scaled using StandardScaler")

# ============= TRAIN TEST SPLIT =============
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)
print(f"OK: Train split: {X_tr.shape[0]}, Validation split: {X_val.shape[0]}")

# ============= TRAIN MODELS =============
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# ============= SELECT BEST MODEL =============
print("\n" + "=" * 60)
print("MODEL SELECTION")
print("=" * 60)
best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_model_name]["model"]
print(f"OK: Best model selected: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

# ============= PREDICTIONS ON TEST SET =============
print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)
y_test_pred = best_model.predict(X_test_scaled)
print(f"OK: Predictions made for {len(y_test_pred)} test samples")
print(f"  Survivors predicted: {y_test_pred.sum()}")
print(f"  Non-survivors predicted: {(1 - y_test_pred).sum()}")

# ============= CREATE SUBMISSION =============
print("\n" + "=" * 60)
print("CREATING SUBMISSION")
print("=" * 60)
submission = pd.DataFrame({
    "PassengerId": test["passengerid"],
    "Survived": y_test_pred.astype(int),
})

submission.to_csv("submission.csv", index=False)
print("OK: Submission file created: submission.csv")
print("\nFirst few rows:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")
print("Submission file ready to submit!")

# ============= SUMMARY =============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Validation Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Validation F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"Test Predictions: {y_test_pred.sum()} survivors out of {len(y_test_pred)}")
print("\nFiles created:")
print("  - submission.csv (ready to submit)")
