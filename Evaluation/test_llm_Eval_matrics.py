import pytest
import nltk
import pandas as pd

from Evaluation.prefilters import (
    compute_keyword_overlap,
    compute_confidence_score,
    compute_cosine_similarity,
)
from Evaluation.deepeval_runner import run_deepeval_metrics, run_deepeval_allmetrics
from Evaluation.utils import load_excel_data

# =========================
# Config (tune as needed)
# =========================
COSINE_W = 0.5
KEYWORD_W = 0.3
CONF_W = 0.2

PREFILTER_THRESHOLD = 0.50  # fail early if below this
DEEPEVAL_THRESHOLDS = {
    "AnswerRelevancy": 0.60,
    "ContextualPrecision": 0.60,
    "Faithfulness": 0.60,
    "ContextualRecall": 0.60,
}

# Collect results for reporting
from Evaluation.shared_results import results

# =========================
# Load test data
# =========================
test_data = load_excel_data(
    "/Users/sunilbarik/PycharmProjects/PythonProject/LLMEvaluation/Data/llm_test_dataset.xlsx"
)

@pytest.mark.parametrize(
    "query, expected_answer, mock_response, expected_keywords", test_data
)
def test_llm_response(query, expected_answer, mock_response, expected_keywords):
    # --------------------------
    # 1) Prefilter (cheap checks)
    # --------------------------
    cosine_sim = compute_cosine_similarity(expected_answer, mock_response)
    keyword_score = compute_keyword_overlap(expected_keywords, mock_response)
    confscore = compute_confidence_score(expected_answer, mock_response)

    final_score = round(
        (COSINE_W * cosine_sim) + (KEYWORD_W * keyword_score) + (CONF_W * confscore),
        2,
    )

    result = {
        "query": query,
        "expected_answer": expected_answer,
        "mock_response": mock_response,
        "cosine_similarity": cosine_sim,
        "keyword_score": keyword_score,
        "confidence_metric": confscore,
        "prefilter_score": final_score,
        "prefilter_threshold": PREFILTER_THRESHOLD,
        "deepeval_run": False,
        "deepeval_scores": {},
        "final_result": "",
    }

    if final_score < PREFILTER_THRESHOLD:
        # Fail early → DO NOT run DeepEval
        result["final_result"] = "FAIL (Prefilter)"
        results.append(result)
        pytest.fail(f"Prefilter failed (score={final_score})", pytrace=False)

    else:
        # -----------------------------------
        # 2) DeepEval (run ONLY when eligible)
        # -----------------------------------
        scores = run_deepeval_allmetrics(
            query=query,
            actual_output=mock_response,
            expected_output=expected_answer,
            # or pass context if your runner uses it
        )
        result["deepeval_run"] = True
        result["deepeval_scores"] = scores

        # Check each DeepEval metric against its threshold
        failed = []
        for metric_name, threshold in DEEPEVAL_THRESHOLDS.items():
            score = scores.get(metric_name, None)
            # If a metric isn't returned, treat as pass-by-default; change if you prefer strict
            if score is not None and score < threshold:
                failed.append(f"{metric_name}={score:.2f}(<{threshold})")

        if failed:
            result["final_result"] = "FAIL (DeepEval: " + ", ".join(failed) + ")"
            results.append(result)
            pytest.fail(f"DeepEval failed → {', '.join(failed)}", pytrace=False)
        else:
            result["final_result"] = "PASS"
            results.append(result)







@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_resources():
    """Ensure required NLTK data is available before any test starts."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")