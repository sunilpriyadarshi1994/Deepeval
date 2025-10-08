import string
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from Evaluation.utils import get_embeddings

# Download NLTK resources once
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_and_tokenize(text):
    """
    Lowercase, remove punctuation, tokenize, and remove stopwords.
    """
    #print(f"text before processing-------------->>>>>>>>>>>>>{text}")
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)

    #print(f"TOKEN______________________>>>>>>>>>>>>_________________>>>>>>>{tokens}")
    return [word for word in tokens if word not in stop_words]

def compute_keyword_overlap(expected_keywords, response: str) -> float:
    """
    Compare expected keywords (list or string) with response words.
    """
    if isinstance(expected_keywords, str):
        keywords = [kw.strip().lower() for kw in expected_keywords.split(",") if kw.strip()]
    elif isinstance(expected_keywords, list):
        keywords = [kw.strip().lower() for kw in expected_keywords if isinstance(kw, str)]
    else:
        keywords = []

    response_words = response.lower().split()
    matched = [kw for kw in keywords if kw in response_words]
    return round(len(matched) / len(keywords), 2) if keywords else 0.0

def compute_confidence_score(expected_answer: str, actual_response: str) -> float:
    """
    Overlap score after removing stopwords from expected vs actual answer.
    """
    expected_tokens = set(clean_and_tokenize(expected_answer))
    print(f"expected token after processing _____>> {expected_tokens}")
    actual_tokens = set(clean_and_tokenize(actual_response))
    print(f"expected token after processing _____>> {actual_tokens}")

    if not expected_tokens:
        return 0.0

    matched = expected_tokens.intersection(actual_tokens)
    score = len(matched) / len(expected_tokens)
    return round(score, 2)

def compute_cosine_similarity(embedding1, embedding2) -> float:
    """
    Cosine similarity between two embeddings.
    """
    expected=get_embeddings(embedding1)
    actual=get_embeddings(embedding2)
    score = cosine_similarity([expected], [actual])[0][0]
    return round(score, 2)
