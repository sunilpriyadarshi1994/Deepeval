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
KEYWORD_W = 0.4
PREFILTER_THRESHOLD = 0.50

DEEPEVAL_THRESHOLDS = {
    "AnswerRelevancy":     0.60,
    "ContextualPrecision": 0.60,
    "ContextualRecall":    0.60,
    "Faithfulness":        0.60,
}

DATA_PATH = "/Users/sunilbarik/PycharmProjects/PythonProject/LLMEvaluation/Data/test_eval_data.xlsx"

# ---------- Data ----------
test_data = load_excel_context_data(DATA_PATH)

@pytest.mark.parametrize(
    "query, expected_answer, mock_response, expected_keywords, context",
    test_data
)
def test_llm_response(query, expected_answer, mock_response, expected_keywords, context):
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
    }

    if pre < PREFILTER_THRESHOLD:
        row["final_result"] = f"FAIL (Prefilter {pre} < {PREFILTER_THRESHOLD})"
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

    # Threshold checks
    failed = []
    for metric_name, threshold in DEEPEVAL_THRESHOLDS.items():
        sc = scores.get(metric_name, None)
        if sc is None or sc < threshold:
            failed.append(f"{metric_name}={'None' if sc is None else f'{sc:.2f}'}(<{threshold})")

    if failed:
        row["failed_metrics"] = ", ".join(failed)
        row["final_result"] = "FAIL (DeepEval)"
        results.append(row)
        pytest.fail(f"DeepEval failed → {row['failed_metrics']}", pytrace=False)
    else:
        row["final_result"] = "PASS"
        results.append(row)

# ---------- Ensure NLTK (harmless) ----------
@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
