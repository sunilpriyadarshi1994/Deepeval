import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
def load_excel_data(filepath: str):
    df = pd.read_excel(filepath)

    # This ensures all keyword entries become lists, if not already
    df["expected_keywords"] = df["expected_keywords"].apply(
        lambda x: [kw.strip().lower() for kw in x.split(",")] if isinstance(x, str) else x
    )

    return list(df[["query", "expected_answer", "mock_response", "expected_keywords"]].itertuples(index=False, name=None))

def get_embeddings(text):
    return model.encode(text)


import pandas as pd

def _normalize_keywords(val):
    if isinstance(val, str):
        return [kw.strip().lower() for kw in val.split(",") if kw.strip()]
    if isinstance(val, list):
        return [str(kw).strip().lower() for kw in val]
    return []

def load_excel_context_data(filepath: str):
    """
    Expects columns:
      query | expected_answer | mock_response | expected_keywords | context
    If 'context' cell is empty, falls back to 'expected_answer'.
    """
    df = pd.read_excel(filepath)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["query", "expected_answer", "mock_response", "expected_keywords", "context"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in Excel: {missing}")

    df["expected_keywords"] = df["expected_keywords"].apply(_normalize_keywords)

    df["context"] = df.apply(
        lambda r: r["context"] if isinstance(r["context"], str) and r["context"].strip()
        else r["expected_answer"],
        axis=1,
    )

    return list(
        df[["query", "expected_answer", "mock_response", "expected_keywords", "context"]]
        .itertuples(index=False, name=None)
    )

# Backward compatibility for earlier imports
load_excel_cotext_data = load_excel_context_data


