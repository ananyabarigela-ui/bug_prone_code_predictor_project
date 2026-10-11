from pathlib import Path
import pickle
import re

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bug-Prone Code Predictor",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

FEATURES = [
    "loc",
    "functions",
    "loops",
    "conditions",
    "variables",
    "comment_ratio",
    "nesting",
]
FEATURE_LABELS = {
    "loc": "Lines of code",
    "functions": "Functions / methods",
    "loops": "Loops",
    "conditions": "Conditions",
    "variables": "Variable declarations",
    "comment_ratio": "Comment ratio",
    "nesting": "Maximum nesting depth",
}

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1250px;}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(120deg, #102a43 0%, #174a6e 60%, #287d8e 100%);
        color: white; margin-bottom: 1.1rem;
      }
      .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {color: #e6f4f7; margin: 0; font-size: 1rem;}
      .section-card {
        border: 1px solid rgba(128, 128, 128, .25); border-radius: 13px;
        padding: 1rem 1.1rem; margin-bottom: .75rem;
      }
      .result-card {border-radius: 14px; padding: 1.1rem 1.25rem; margin: .4rem 0 1rem 0;}
      .result-card h2 {margin: 0 0 .35rem 0; font-size: 1.55rem;}
      .result-card p {margin: 0;}
      .small-note {font-size: .88rem; opacity: .82;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem; border-radius: 12px;}
      .stButton button {border-radius: 9px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🐞 Bug-Prone Code Predictor</h1>
      <p>Analyze code complexity, view an ML prediction, and review possible warning patterns.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("College project prototype • Educational synthetic dataset • Not a replacement for testing or code review")

# ------------------------------------------------------------
# Feature extraction and simple rule-based checks
# ------------------------------------------------------------
def extract_features(code: str) -> dict:
    lines = code.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    loc = len(non_empty_lines)

    # Approximate source-code metrics; intended for demonstration, not a full parser.
    function_pattern = r"\b(?:def\s+\w+|function\s+\w+|(?:public|private|protected|static|\s)+\w[\w<>\[\]]*\s+\w+\s*\([^;]*\)\s*\{)"
    functions = len(re.findall(function_pattern, code))
    loops = len(re.findall(r"\b(?:for|while|do)\b", code))
    conditions = len(re.findall(r"\b(?:if|else\s+if|elif|switch|case)\b", code))

    variable_pattern = (
        r"\b(?:int|long|short|float|double|char|String|boolean|var|let|const)"
        r"\s+([A-Za-z_$][\w$]*)"
    )
    variables = len(set(re.findall(variable_pattern, code)))

    comment_lines = sum(
        1 for line in lines
        if line.strip().startswith(("//", "#", "*", "/*"))
    )
    comment_ratio = round(comment_lines / max(loc, 1), 3)

    # Estimate nesting from braces; fall back to indentation for Python-like code.
    depth = 0
    max_depth = 0
    for char in code:
        if char == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == "}":
            depth = max(0, depth - 1)

    if max_depth == 0:
        indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
        max_depth = (max(indents) // 4) if indents else 0

    return {
        "loc": loc,
        "functions": functions,
        "loops": loops,
        "conditions": conditions,
        "variables": variables,
        "comment_ratio": comment_ratio,
        "nesting": max_depth,
    }


def detect_bug_patterns(code: str) -> list[tuple[str, str]]:
    findings = []

    # Simple array/list boundary pattern, such as i <= values.length.
    if re.search(
        r"\b\w+\s*<=\s*\w+\s*\.\s*(?:length|size)\b|"
        r"\b\w+\s*<=\s*\w+\s*\.\s*length\s*\(\s*\)",
        code,
        re.IGNORECASE,
    ):
        findings.append(("Possible boundary issue", "A loop uses <= with a length/size value. Check whether the final index should use < instead."))

    # Direct division by zero.
    if re.search(r"/\s*0(?:\.0+)?\b", code):
        findings.append(("Division by zero", "The code appears to divide directly by zero."))

    # Basic case: variable assigned zero and then used as a divisor.
    zero_vars = set(re.findall(
        r"\b(?:int|long|short|float|double|var|let|const)?\s*([A-Za-z_$][\w$]*)\s*=\s*0(?:\.0+)?\s*;?",
        code,
    ))
    for var in sorted(zero_vars):
        if re.search(r"/\s*" + re.escape(var) + r"\b", code):
            findings.append(("Possible division by zero", f"'{var}' is assigned zero and later used as a divisor. Check the value before dividing."))
            break

    loop_count = len(re.findall(r"\b(?:for|while|do)\b", code))
    if loop_count >= 3:
        findings.append(("Many loops", f"The code contains {loop_count} loop statements. Review performance, especially if loops are nested."))

    condition_count = len(re.findall(r"\b(?:if|else\s+if|elif|switch|case)\b", code))
    if condition_count >= 5:
        findings.append(("Many conditions", f"The code contains {condition_count} conditional statements. Consider whether the logic can be simplified."))

    return findings


# ------------------------------------------------------------
# Load saved model and model comparison
# ------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as file:
        saved = pickle.load(file)
    # Current training script saves a dictionary bundle.
    if isinstance(saved, dict) and "model" in saved:
        return saved
    # Graceful support for an older pickle containing only a model.
    return {"model": saved, "feature_names": FEATURES, "best_model_name": "Saved model", "dataset_type": "synthetic"}


bundle = load_model_bundle()

with st.sidebar:
    st.header("Project guide")
    st.markdown("**1. Paste code**\n\n**2. Select Analyze Code**\n\n**3. Review the prediction and metrics**")
    st.divider()
    st.markdown("**Dataset**")
    st.write("Educational synthetic data (500 generated examples).")
    st.markdown("**Important**")
    st.caption("The ML prediction is based on patterns learned from generated labels. It does not prove that code contains or does not contain a real bug.")

if bundle is None:
    st.error("The trained model was not found.")
    st.info("Open PowerShell in your project folder and run: `python train_model.py`")
    st.stop()

model = bundle["model"]
model_features = bundle.get("feature_names", FEATURES)
model_name = bundle.get("best_model_name", "Selected model")

# ------------------------------------------------------------
# Main tabs
# ------------------------------------------------------------
tab_analyze, tab_performance, tab_about = st.tabs(["🔎 Analyze code", "📊 Model performance", "ℹ️ About project"])

with tab_analyze:
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.subheader("Code input")
        st.write("Paste a code snippet below. Java and Python-like snippets work best with these simple metrics.")
        sample_choice = st.selectbox(
            "Load an example",
            ["Choose an example…", "Simple example", "Example with possible issue"],
            index=0,
        )
        safe_sample = '''public class SumExample {
    public static void main(String[] args) {
        int[] numbers = {10, 20, 30, 40, 50};
        int total = 0;

        for (int i = 0; i < numbers.length; i++) {
            total += numbers[i];
        }

        System.out.println("Total: " + total);
    }
}'''
        risky_sample = '''public class ExampleWithIssue {
    public static void main(String[] args) {
        int[] numbers = {10, 20, 30};
        int divisor = 0;

        for (int i = 0; i <= numbers.length; i++) {
            System.out.println(numbers[i]);
        }

        int answer = 10 / divisor;
        System.out.println(answer);
    }
}'''
        default_code = safe_sample if sample_choice == "Simple example" else risky_sample if sample_choice == "Example with possible issue" else ""
        code = st.text_area(
            "Source code",
            value=default_code,
            height=330,
            placeholder="Paste your Java or Python code here…",
            help="This tool estimates a few complexity metrics; it does not fully parse or compile the code.",
            key=f"source_code_{sample_choice}",
        )
        analyze_clicked = st.button("Analyze Code", type="primary", use_container_width=True)

    with right:
        st.subheader("How to read the result")
        st.markdown(
            """
            <div class="section-card">
              <b>ML prediction</b><br/>The selected machine-learning model classifies the code using its extracted metrics.
            </div>
            <div class="section-card">
              <b>Code warnings</b><br/>Simple rule-based checks look for a few recognizable patterns, such as a possible loop boundary issue.
            </div>
            <div class="section-card">
              <b>Complexity metrics</b><br/>Counts such as loops and conditions describe the snippet; higher counts do not automatically mean a bug exists.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        if not code.strip():
            st.warning("Paste code or choose an example before analyzing.")
        else:
            metrics = extract_features(code)
            input_df = pd.DataFrame([{feature: metrics.get(feature, 0) for feature in model_features}], columns=model_features)
            prediction = int(model.predict(input_df)[0])

            bug_score = None
            if hasattr(model, "predict_proba"):
                try:
                    classes = list(model.classes_)
                    if 1 in classes:
                        bug_score = float(model.predict_proba(input_df)[0][classes.index(1)])
                except Exception:
                    bug_score = None

            st.divider()
            st.subheader("Analysis result")
            if prediction == 1:
                st.markdown(
                    "<div class='result-card' style='background:#fff0ed;border:1px solid #e7a79c;color:#7c2419;'>"
                    "<h2>⚠️ Higher predicted risk</h2><p>The ML model classified this snippet as <b>bug-prone</b> based on its extracted metrics.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-card' style='background:#eaf7ef;border:1px solid #9bc9aa;color:#205b35;'>"
                    "<h2>✓ Lower predicted risk</h2><p>The ML model classified this snippet as <b>not bug-prone</b> based on its extracted metrics. This does not guarantee the code is bug-free.</p></div>",
                    unsafe_allow_html=True,
                )

            result_cols = st.columns(3)
            result_cols[0].metric("Prediction", "Bug-prone" if prediction == 1 else "Not bug-prone")
            result_cols[1].metric("Model used", str(model_name))
            result_cols[2].metric("Model-estimated score", f"{bug_score:.0%}" if bug_score is not None else "Not available")
            st.caption("The score is the model's estimate for the bug-prone class, not a verified real-world probability.")

            st.subheader("Code complexity at a glance")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Lines of code", metrics["loc"])
            metric_cols[1].metric("Loops", metrics["loops"])
            metric_cols[2].metric("Conditions", metrics["conditions"])
            metric_cols[3].metric("Nesting depth", metrics["nesting"])
            extra_cols = st.columns(3)
            extra_cols[0].metric("Functions / methods", metrics["functions"])
            extra_cols[1].metric("Variable declarations", metrics["variables"])
            extra_cols[2].metric("Comment ratio", f"{metrics['comment_ratio']:.0%}")

            with st.expander("View all extracted metrics"):
                metric_table = pd.DataFrame({
                    "Metric": [FEATURE_LABELS.get(name, name) for name in FEATURES],
                    "Extracted value": [metrics[name] for name in FEATURES],
                })
                st.dataframe(metric_table, hide_index=True, use_container_width=True)

            st.subheader("Rule-based code warnings")
            findings = detect_bug_patterns(code)
            if findings:
                for title, detail in findings:
                    st.warning(f"**{title}:** {detail}")
            else:
                st.success("No supported suspicious patterns were detected by the simple rule checks.")
            st.caption("Rule-based warnings and the ML prediction are separate checks, so their results may differ.")

with tab_performance:
    st.subheader("Model comparison")
    st.write("These scores measure performance on a held-out test split from the generated synthetic dataset.")
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        # Clean up column labels for presentation without changing the source file.
        st.dataframe(comparison.round(3), hide_index=True, use_container_width=True)
        numeric_columns = [col for col in comparison.columns if pd.api.types.is_numeric_dtype(comparison[col])]
        preferred_metrics = [col for col in ["Accuracy", "Precision", "Recall", "F1 Score"] if col in numeric_columns]
        if preferred_metrics:
            chart_df = comparison.set_index(comparison.columns[0])[preferred_metrics]
            st.markdown("#### Score comparison")
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.info("Model comparison file not found. Run `python train_model.py` to generate it.")

    st.info("Because the dataset is synthetic, these results demonstrate the ML workflow; they should not be presented as real-world bug-detection accuracy.")

with tab_about:
    st.subheader("Project overview")
    st.write("This project demonstrates how machine learning can classify code snippets using simple software-complexity metrics.")
    st.markdown("**Workflow**")
    st.markdown("1. Generate an educational synthetic dataset.\n2. Train and compare classification models.\n3. Extract metrics from a pasted code snippet.\n4. Use the selected model to predict a class.\n5. Run separate rule-based checks for a few recognizable warning patterns.")

    st.markdown("**What this prototype can do**")
    st.markdown("- Display an ML classification based on the extracted metrics.\n- Show basic complexity indicators.\n- Flag a small number of suspicious code patterns.\n- Compare trained models on a synthetic test split.")

    st.markdown("**Limitations to mention during presentation**")
    st.markdown("- The training dataset is synthetic and generated for this college project.\n- Metrics are estimated with simple text patterns, not a full language parser.\n- The model may miss bugs or flag code that is actually correct.\n- The output is educational and should not replace testing, debugging, or code review.")
