
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bug_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = [
    "loc",
    "functions",
    "loops",
    "conditions",
    "variables",
    "comment_ratio",
    "nesting",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run python generate_dataset.py first."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["bug"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["bug"], errors="raise").astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("The dataset must contain both bug classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "F1 Score", ascending=False
    ).reset_index(drop=True)

    best_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False,
    )

    # Save a bundle containing the model and its expected feature order.
    bundle = {
        "model": best_model,
        "feature_names": FEATURES,
        "best_model_name": best_name,
        "dataset_type": "synthetic",
    }

    with open(MODEL_DIR / "best_model.pkl", "wb") as file:
        pickle.dump(bundle, file)

    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {FEATURES}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
    print(f"\nSelected model by F1 score: {best_name}")
    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["Not Bug-Prone", "Bug-Prone"],
        zero_division=0,
    ))
    print("\nSaved models/best_model.pkl")
    print("Saved models/model_comparison.csv")
    print(
        "\nNOTE: These metrics evaluate labels generated from a synthetic "
        "complexity formula. They do not establish real-world bug "
        "prediction accuracy."
    )


if __name__ == "__main__":
    main()
