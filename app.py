

import streamlit as st
import pandas as pd
import pickle
import re


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bug-Prone Code Predictor",
    page_icon="🐞",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()
comparison = pd.read_csv("models/model_comparison.csv")


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(code):

    lines = [
        line for line in code.splitlines()
        if line.strip()
    ]

    loc = len(lines)

    # Approximate function/method count for common syntax
    functions = len(re.findall(
        r"\b(def\s+\w+|function\s+\w+|"
        r"(?:public\s+|private\s+|protected\s+|static\s+)*"
        r"(?:void|int|float|double|String|char|boolean)\s+\w+)"
        r"\s*\(",
        code,
        re.IGNORECASE
    ))

    loops = len(re.findall(
        r"\b(for|while|do)\b",
        code,
        re.IGNORECASE
    ))

    conditions = len(re.findall(
        r"\b(if|else\s+if|switch|case)\b",
        code,
        re.IGNORECASE
    ))

    variables = len(re.findall(
        r"\b(int|float|double|String|char|boolean)\s+\w+",
        code,
        re.IGNORECASE
    ))

    comments = len(re.findall(
        r"//[^\n]*|#[^\n]*|/\*|\*/",
        code
    ))

    comment_ratio = comments / loc if loc else 0

    # Approximate nesting depth for brace-based languages
    nesting = 0
    current_nesting = 0

    for line in lines:

        stripped = line.strip()

        if stripped.startswith("}"):
            current_nesting = max(0, current_nesting - 1)

        current_nesting += line.count("{")

        nesting = max(nesting, current_nesting)

    return {
        "loc": loc,
        "functions": functions,
        "loops": loops,
        "conditions": conditions,
        "variables": variables,
        "comment_ratio": comment_ratio,
        "nesting": nesting
    }


# ==========================================
# STATIC BUG PATTERN ANALYSIS
# ==========================================

def detect_bug_patterns(code):

    warnings = []
    risk_score = 0

    # 1. Possible array index out of bounds
    has_array_access = bool(re.search(
        r"\b\w+\s*\[\s*\w+\s*\]",
        code
    ))

    has_unsafe_length_condition = bool(re.search(
        r"<=\s*\w+\.length\b",
        code
    ))

    if has_array_access and has_unsafe_length_condition:

        warnings.append(
            "🚨 HIGH RISK: Possible array index out of bounds. "
            "Check loops using <= array.length. For a Java array, "
            "valid indexes normally range from 0 to length - 1."
        )

        risk_score += 3

    # 2. Possible division by zero using an arithmetic expression
    if re.search(
        r"/\s*\(\s*\w+\s*[-+*/]\s*\d+\s*\)",
        code
    ):

        warnings.append(
            "🚨 HIGH RISK: Possible division by zero. "
            "The arithmetic expression used as a divisor "
            "may evaluate to zero."
        )

        risk_score += 3

    # 3. Direct division by literal zero
    if re.search(
        r"/\s*0+(?:\.0*)?\b",
        code
    ):

        warnings.append(
            "🚨 HIGH RISK: Division by a literal zero detected."
        )

        risk_score += 3

    # 4. Multiple loops
    loop_count = len(re.findall(
        r"\b(for|while|do)\b",
        code,
        re.IGNORECASE
    ))

    if loop_count >= 3:

        warnings.append(
            "⚠️ Multiple loops detected. Review code complexity "
            "and possible performance issues."
        )

        risk_score += 1

    # 5. Multiple conditions
    condition_count = len(re.findall(
        r"\b(if|else\s+if|switch|case)\b",
        code,
        re.IGNORECASE
    ))

    if condition_count >= 5:

        warnings.append(
            "⚠️ Multiple conditional branches detected. "
            "Review the logic and test different cases."
        )

        risk_score += 1

    # 6. Approximate structural complexity
    if code.count("{") >= 8:

        warnings.append(
            "⚠️ High structural complexity: many code blocks "
            "were detected."
        )

        risk_score += 1

    return warnings, risk_score


# ==========================================
# HEADER
# ==========================================

st.title("🐞 Bug-Prone Code Prediction System")

st.markdown(
    """
    **Machine Learning Based Bug-Prone Code Prediction**

    Analyze source code using software complexity metrics,
    a trained machine-learning model, and selected static
    bug-pattern checks.
    """
)

st.divider()


# ==========================================
# ABOUT PROJECT
# ==========================================

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This system uses two complementary approaches.

        **1. Machine Learning**
        Predicts the class using seven software complexity metrics.

        **2. Static Bug Pattern Analysis**
        Checks for selected patterns such as possible array-index
        errors, possible division by zero, multiple loops, and
        deep structural nesting.

        The ML output is kept separate from the static warnings.
        The final risk assessment considers both approaches.
        """
    )


# ==========================================
# CODE INPUT
# ==========================================

st.subheader("💻 Source Code Analysis")

code = st.text_area(
    "Paste your source code below:",
    height=350,
    placeholder="Paste Java, C/C++, or Python code here..."
)


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Predict Bug Risk",
    type="primary",
    use_container_width=True
):

    if not code.strip():

        st.warning("⚠️ Please enter source code first.")

    else:

        # --------------------------------------
        # EXTRACT METRICS
        # --------------------------------------

        features = extract_features(code)

        feature_columns = [
            "loc",
            "functions",
            "loops",
            "conditions",
            "variables",
            "comment_ratio",
            "nesting"
        ]

        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        # --------------------------------------
        # ACTUAL ML PREDICTION
        # --------------------------------------

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        if 1 not in classes:
            st.error(
                "The loaded model does not contain class 1 "
                "(Bug-Prone). Check the training labels."
            )
            st.stop()

        bug_class_index = classes.index(1)

        # Actual probability estimated by the trained model
        probability = float(probabilities[bug_class_index])

        # --------------------------------------
        # STATIC ANALYSIS
        # --------------------------------------

        warnings, risk_score = detect_bug_patterns(code)

        st.divider()

        # ======================================
        # ML RESULTS
        # ======================================

        st.subheader("🤖 Machine Learning Prediction")

        if prediction == 1:

            st.error("🔴 ML Prediction: Bug-Prone")

        else:

            st.success("🟢 ML Prediction: Not Bug-Prone")

        st.metric(
            "ML Bug-Prone Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        st.caption(
            "This is the probability estimated by the trained model. "
            "Because the model was trained on synthetic labels, "
            "this is not a validated probability of a real defect."
        )

        # ======================================
        # STATIC ANALYSIS RESULTS
        # ======================================

        st.subheader("🔎 Static Bug Pattern Analysis")

        if warnings:

            st.error(
                f"{len(warnings)} potential risk pattern(s) detected."
            )

            for warning in warnings:
                st.write(warning)

        else:

            st.success(
                "No patterns covered by the current static checks "
                "were detected. This does not prove the code is safe."
            )

        # ======================================
        # FINAL RISK ASSESSMENT
        # ======================================

        st.divider()

        st.subheader("🚦 Final Risk Assessment")

        # Serious static patterns or a positive ML classification
        # lead to a high-risk screening result.
        if risk_score >= 3 or prediction == 1:

            st.error(
                "🔴 HIGH RISK — Potential Bug-Prone Code"
            )

            st.write(
                "Review the detected warnings and test the code "
                "before using it."
            )

        elif risk_score >= 1:

            st.warning(
                "🟠 MEDIUM RISK — Review Recommended"
            )

            st.write(
                "Some complexity-related warning patterns were "
                "detected. Review and test the code."
            )

        else:

            st.success(
                "🟢 LOW RISK — No Major Risk Pattern Detected"
            )

            st.write(
                "The ML model and current static checks did not "
                "identify a major risk. This does not guarantee "
                "that the code is bug-free."
            )

        st.caption(
            "The final risk level is a screening decision based on "
            "the ML classification and selected static rules. "
            "It is not a calibrated probability."
        )

        # ======================================
        # CODE METRICS
        # ======================================

        st.divider()

        st.subheader("📊 Extracted Code Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Lines of Code", features["loc"])
        col2.metric("Functions", features["functions"])
        col3.metric("Loops", features["loops"])
        col4.metric("Conditions", features["conditions"])

        col5, col6, col7 = st.columns(3)

        col5.metric("Variables", features["variables"])

        col6.metric(
            "Comment Ratio",
            f'{features["comment_ratio"]:.2f}'
        )

        col7.metric("Nesting Depth", features["nesting"])


# ==========================================
# MODEL COMPARISON
# ==========================================

st.divider()

st.subheader("🤖 Machine Learning Model Comparison")

display_comparison = comparison.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for column in metric_columns:

    if column in display_comparison.columns:

        display_comparison[column] = (
            display_comparison[column] * 100
        ).round(2)

st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# ACCURACY CHART
# ==========================================

st.subheader("📈 Model Accuracy Comparison")

if (
    "Model" in comparison.columns
    and "Accuracy" in comparison.columns
):

    chart_data = comparison.set_index("Model")[["Accuracy"]]

    st.bar_chart(chart_data)


# ==========================================
# LIMITATIONS
# ==========================================

st.divider()

st.warning(
    """
    **Project Limitation**

    The ML model was trained using a synthetic educational dataset
    whose labels were generated from a complexity score. It has
    not been trained on a representative collection of real bugs.

    The static analyzer uses simple pattern matching. It may miss
    bugs or flag code that is actually safe. The final assessment
    is a preliminary screening result, not proof of correctness.
    """
)


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "Machine Learning Based Bug-Prone Code Prediction "
    "| College Micro Project"
)
