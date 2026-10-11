import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/bug_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "loc",
    "functions",
    "loops",
    "conditions",
    "variables",
    "comment_ratio",
    "nesting"
]

X = data[features]
y = data["bug"]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE ML MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print("Best Model:", best_model_name)
# Save model comparison results
results_df.to_csv("models/model_comparison.csv", index=False)

print("Model comparison saved to: models/model_comparison.csv")


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

with open("models/best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model saved successfully!")
print("Location: models/best_model.pkl")