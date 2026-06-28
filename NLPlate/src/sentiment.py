# deo za analizu sentimenta korisnickih recenzija

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# ucitavanje
# NLTK VADER odredjuje da li je tekst pozitivan, negativan ili neutralan, pri čemu posebno dobro radi na kratkim tekstovima
_VADER = None
_VADER_AVAILABLE = False
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  

    try:
        _VADER = SentimentIntensityAnalyzer()
        _VADER_AVAILABLE = True
    except LookupError:
        try:
            import nltk  
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
            _VADER_AVAILABLE = True
        except Exception:
            _VADER_AVAILABLE = False
except Exception:
    _VADER_AVAILABLE = False


# za slucaj da NLTK ne radi fallback leksikon 
_POS_WORDS = {
    "good", "great", "delicious", "tasty", "love", "loved", "perfect", "best",
    "excellent", "amazing", "wonderful", "yummy", "favorite", "easy", "quick",
    "fantastic", "awesome", "nice", "moist", "flavorful", "fresh", "healthy",
    "recommend", "recommended", "enjoyed", "enjoy", "hit", "winner", "super",
    "simple", "fluffy", "crispy", "rich", "satisfying", "wow", "incredible",
}
_NEG_WORDS = {
    "bad", "bland", "dry", "tasteless", "awful", "terrible", "horrible",
    "disgusting", "worst", "hate", "hated", "disappointing", "disappointed",
    "soggy", "burnt", "burned", "gross", "inedible", "ruined", "waste",
    "overcooked", "undercooked", "salty", "tough", "mushy", "watery", "boring",
    "mediocre", "lacking", "meh", "nope", "never",
}
_NEGATIONS = {"not", "no", "never", "without", "barely", "hardly"}
_TOKEN_RE = re.compile(r"[a-z']+")

# rucno racunanje sentimenta bez Vader-a
def _fallback_sentiment(text: str) -> float:
    if not text:
        return 0.0
    toks = _TOKEN_RE.findall(str(text).lower())
    if not toks:
        return 0.0
    score = 0
    negate = False
    for tok in toks:
        if tok in _NEGATIONS:
            negate = True
            continue
        val = 0
        if tok in _POS_WORDS:
            val = 1
        elif tok in _NEG_WORDS:
            val = -1
        if val != 0:
            score += -val if negate else val
            negate = False
        else:
            negate = False

    n_colored = sum(1 for t in toks if t in _POS_WORDS or t in _NEG_WORDS)
    if n_colored == 0:
        return 0.0
    return float(np.clip(score / n_colored, -1.0, 1.0))

# racunanje sentimenta za jednu recenziju
def sentiment_compound(text: str) -> float:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0.0
    s = str(text).strip()
    if not s:
        return 0.0
    if _VADER_AVAILABLE and _VADER is not None:
        return float(_VADER.polarity_scores(s)["compound"])
    return _fallback_sentiment(s)


def backend_name() -> str:
    return "nltk_vader" if _VADER_AVAILABLE else "lexicon_fallback"


def compute_recipe_sentiment(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions[["recipe_id", "review"]].copy()
    df = df[df["review"].notna()]
    df["review"] = df["review"].astype(str)
    df = df[df["review"].str.len() > 0]

    print(f"[sentiment] Backend: {backend_name()} | recenzija sa tekstom: {len(df):,}")

    # racunanje sentimenta po recenziji
    df["compound"] = df["review"].map(sentiment_compound)

    # grupisanje po receptu za racunanje prosecnog sentimenta 
    grp = df.groupby("recipe_id")["compound"]
    out = pd.DataFrame({
        "sentiment_compound_mean": grp.mean(),
        "n_reviews_text": grp.count(),
    })
    pos_ratio = (
        df.assign(is_pos=(df["compound"] > 0.05).astype(int))
        .groupby("recipe_id")["is_pos"].mean()
    )
    out["sentiment_pos_ratio"] = pos_ratio
    out = out.reset_index().rename(columns={"recipe_id": "id"})

    out["sentiment"] = ((out["sentiment_compound_mean"] + 1.0) / 2.0).clip(0.0, 1.0)
    return out


if __name__ == "__main__":
    for t in ["This recipe was absolutely delicious and easy to make!",
              "Bland and dry, would not make again.",
              "It was okay, nothing special."]:
        print(f"{sentiment_compound(t):+.3f}  <- {t}")
