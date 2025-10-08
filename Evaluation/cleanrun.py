# Evaluation/matrixesEvaluation.py
import pytest
import nltk

from Evaluation.prefilters import (
    compute_keyword_overlap,
    compute_cosine_similarity,
)
from Evaluation.deepeval_runner import run_deepeval_allmetrics
from Evaluation.utils import load_excel_context_data
from Evaluation.shared_results import results

# ---------- Config ----------
COSINE_W = 0.6
KEYWORD_W = 0.5
PREFILTER_THRESHOLD = 0.55

DEEPEVAL_THRESHOLDS = {
    "AnswerRelevancy":     0.60,
    "ContextualPrecision": 0.60,
    "ContextualRecall":    0.60,
    "Faithfulness":        0.60,
}

DATA_PATH = "/Users/sunilbarik/PycharmProjects/PythonProject/LLMEvaluation/Data/test_eval_data_updated.xlsx"

# ---------- Data ----------
test_data = load_excel_context_data(DATA_PATH)

@pytest.mark.parametrize(
    "query, expected_answer, mock_response, expected_keywords, context",
    test_data
)
def test_llm_response(query, expected_answer, mock_response, expected_keywords, context):
    # Robust fallback if context is blank/NaN-like
    if not context or (isinstance(context, float) and str(context) == "nan"):
        context = expected_answer

    # 1) Prefilter (cosine + keywords)
    cos = compute_cosine_similarity(expected_answer, mock_response)
    kw  = compute_keyword_overlap(expected_keywords, mock_response)
    pre = round(COSINE_W * cos + KEYWORD_W * kw, 2)

    row = {
        "query": query,
        "expected_answer": expected_answer,
        "mock_response": mock_response,
        "context": context,
        "cosine_similarity": round(cos, 2),
        "keyword_overlap": round(kw, 2),
        "prefilter_score": pre,
        "prefilter_threshold": PREFILTER_THRESHOLD,
        "deepeval_run": False,
        "deepeval_scores": {},
        "failed_metrics": "",
        "final_result": "",
        "FailureReason": "",
    }

    if pre < PREFILTER_THRESHOLD:
        row["final_result"] = f"FAIL (Prefilter {pre} < {PREFILTER_THRESHOLD})"
        row["FailureReason"] = "Prefilter"
        results.append(row)
        pytest.fail(row["final_result"], pytrace=False)

    # 2) DeepEval (only after prefilter passes)
    if context is not None and not isinstance(context, list):
        context = [context]

    scores = run_deepeval_allmetrics(
        query=query,
        actual_output=mock_response,
        expected_output=expected_answer,
        context=context,
    )
    row["deepeval_run"] = True
    row["deepeval_scores"] = scores

    # If DeepEval errored (runner returns _error), mark clearly
    if isinstance(scores, dict) and scores.get("_error"):
        row["failed_metrics"] = f"DeepEvalError: {scores['_error']}"
        row["final_result"] = "FAIL (DeepEvalError)"
        row["FailureReason"] = "DeepEvalError"
        results.append(row)
        pytest.fail(row["failed_metrics"], pytrace=False)

    # Threshold checks
    failed = []
    for metric_name, threshold in DEEPEVAL_THRESHOLDS.items():
        sc = scores.get(metric_name, None)
        if sc is None or sc < threshold:
            failed.append(metric_name)

    if failed:
        row["failed_metrics"] = ", ".join(
            f"{m}={scores.get(m) if scores.get(m) is not None else 'None'}(<{DEEPEVAL_THRESHOLDS[m]})"
            for m in failed
        )
        row["final_result"] = "FAIL (DeepEval)"
        row["FailureReason"] = ", ".join(failed)
        results.append(row)
        pytest.fail(f"DeepEval failed → {row['failed_metrics']}", pytrace=False)
    else:
        row["final_result"] = "PASS"
        row["FailureReason"] = ""
        results.append(row)

# ---------- Ensure NLTK (harmless) ----------
@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
