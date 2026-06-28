# metrike za evoluaciju rangiranja : precision@K, recall@K, NDCG@K 

from __future__ import annotations

import numpy as np


def precision_at_k(relevances_binary: np.ndarray, k: int) -> float:
    topk = relevances_binary[:k]
    if len(topk) == 0:
        return 0.0
    return float(np.sum(topk) / k)


def recall_at_k(relevances_binary: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return float("nan")  
    topk = relevances_binary[:k]
    return float(np.sum(topk) / total_relevant)


def dcg_at_k(gains: np.ndarray, k: int) -> float:
    gains = np.asarray(gains, dtype=float)[:k]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains * discounts))

# ne gleda samo da li su preporuke relevantne nego da li su one najbolje na vrhu liste
def ndcg_at_k(graded_relevances: np.ndarray, k: int) -> float:
    graded = np.asarray(graded_relevances, dtype=float)
    dcg = dcg_at_k(graded, k)
    ideal = dcg_at_k(np.sort(graded)[::-1], k)
    if ideal == 0:
        return float("nan")  
    return dcg / ideal


def evaluate_at_ks(graded: np.ndarray, binary: np.ndarray,
                   total_relevant: int, ks=(5, 10)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(binary, k)
        out[f"recall@{k}"] = recall_at_k(binary, total_relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(graded, k)
    return out
