import pandas as pd
from Evaluation.shared_results import results

def pytest_sessionfinish(session, exitstatus):
    print("✅ conftest loaded, writing report…")
    if not results:
        print("ℹ️ No results to write.")
        return

    df = pd.DataFrame(results)

    # Expand deepeval_scores dict → columns
    if "deepeval_scores" in df.columns:
        scores_df = df["deepeval_scores"].apply(lambda d: d if isinstance(d, dict) else {}).apply(pd.Series)
        for col in ["AnswerRelevancy", "ContextualPrecision", "ContextualRecall", "Faithfulness"]:
            if col in scores_df.columns:
                df[col] = scores_df[col]

    out_path = "/Users/sunilbarik/PycharmProjects/PythonProject/LLMEvaluation/Reports/eval_llm_test_results.xlsx"
    df.to_excel(out_path, index=False)
    print(f"📊 Report saved to: {out_path}")
    print(f"✅ Passed: {(df['final_result'] == 'PASS').sum()}")
    print(f"❌ Failed: {(df['final_result'] != 'PASS').sum()}")
