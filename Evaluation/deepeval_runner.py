import os

from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, \
    FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def run_deepeval_metrics(query, actual_output, expected_output):
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output
    )

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6)


    answer_relevancy_metric.measure(test_case)
    answer_relevancy_score=answer_relevancy_metric.score


    return {
        "AnswerRelevancy": round(answer_relevancy_score, 2),

    }
# Evaluation/deepeval_runner.py



# Evaluation/deepeval_runner.py

# Evaluation/deepeval_runner.py
# Evaluation/deepeval_runner.py
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

load_dotenv()

def _ensure_list_context(ctx: Optional[object]) -> Optional[List[str]]:
    if ctx is None:
        return None
    if isinstance(ctx, list):
        return [str(x) for x in ctx]
    return [str(ctx)]

def run_deepeval_allmetrics(
    query: str,
    actual_output: str,
    expected_output: Optional[str],
    context: Optional[object] = None,
) -> Dict[str, float]:
    """
    Compatible with deepeval==0.20.91:
    - build LLMTestCase (with retrieval_context)
    - metric.measure(test_case)
    - read metric.score
    """
    retrieval_context = _ensure_list_context(context)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Do NOT pass model objects for 0.20.91 — metrics use OPENAI_API_KEY from env
    ar = AnswerRelevancyMetric(model="gpt-3.5-turbo")
    cp = ContextualPrecisionMetric(model="gpt-3.5-turbo")
    cr = ContextualRecallMetric(model="gpt-3.5-turbo")
    fa = FaithfulnessMetric(model="gpt-3.5-turbo")


    ar.measure(test_case)
    cp.measure(test_case)
    cr.measure(test_case)
    fa.measure(test_case)

    return {
        "AnswerRelevancy": round(float(ar.score), 2),
        "ContextualPrecision": round(float(cp.score), 2),
        "ContextualRecall": round(float(cr.score), 2),
        "Faithfulness": round(float(fa.score), 2),
    }
