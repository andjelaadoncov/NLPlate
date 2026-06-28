# pomocne funkcije za parsiranje i ciscenje teksta
# tags, ingredients, steps, nutrition su zapisane kao stringovi odvojeni zarezom pa je trebalo parsiranje u python listu, a zatim normalizacija teksta

import ast
import re
from typing import Any


_BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "with", "for", "in", "on",
    "i", "want", "need", "would", "like", "some", "me", "my", "is", "are",
    "that", "this", "please", "give", "make", "something", "really",
}

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s\-]")


def safe_parse_list(value: Any) -> list:
   
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float):  # NaN dolazi kao float
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except (ValueError, SyntaxError):
        return [p.strip(" []'\"") for p in s.split(",") if p.strip(" []'\"")]

# fja koja normalizuje tekst: 
# pretvara tekst u mala slova, uklanja nepotrebne karaktere, sredjuje razmake, zadrzava reci sa crticom, i zamenjuje _ razmakom
def normalize_text(text: Any) -> str:
    if text is None or isinstance(text, float):
        return ""
    s = str(text).lower()
    s = s.replace("_", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s

# vraca listu reci iz teksta, bez stop reci osnovnih 
def tokens(text: Any) -> list[str]:
    norm = normalize_text(text)
    return [t for t in norm.split(" ") if t and t not in _BASIC_STOPWORDS]


def clean_tag(tag: Any) -> str:
    return normalize_text(tag).replace(" ", "-")


# fja koja daje tekstualni opis recepta koji se koristi za TF-IDF i sentence bert
def build_recipe_document(
    name: str,
    tags: list[str],
    ingredients: list[str],
    steps: list[str],
    description: str | None = None,
) -> str:
    
    name_n = normalize_text(name)
    ingr_n = " ".join(normalize_text(i) for i in ingredients)
    tags_n = " ".join(normalize_text(t) for t in tags)
    steps_n = " ".join(normalize_text(s) for s in steps)
    desc_n = normalize_text(description) if description else ""
    
    # nazivi i sastojci idu dva puta, da im se da veca tezina kasnije u TF-IDF i sentence bert modelima 
    # najvazniji su u receptima pa zato tako
    parts = [
        name_n, name_n,          
        ingr_n, ingr_n,          
        tags_n,
        desc_n,
        steps_n,
    ]
    doc = " ".join(p for p in parts if p)
    return _WHITESPACE_RE.sub(" ", doc).strip()

# odvajanje dela u datasetu za nutritivne vrednosti, jer su zapisane kao string 
NUTRITION_FIELDS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
]


def parse_nutrition(value: Any) -> dict[str, float]:
    vals = safe_parse_list(value)
    out: dict[str, float] = {}
    for i, field in enumerate(NUTRITION_FIELDS):
        try:
            out[field] = float(vals[i]) if i < len(vals) else float("nan")
        except (ValueError, TypeError):
            out[field] = float("nan")
    return out
