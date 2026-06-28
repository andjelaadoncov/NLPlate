# ovde obradjujem korisnicke zahteve za sastojke, vreme pripreme i tagove 

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text_utils import normalize_text

# mapiranje kljucnih reci iz upita na tagove iz dataset-a, jer dataset nema posebnu kategoriju za tip jela vec se to moze izvuci iz tagova recepata
_KEYWORD_TAGS = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "gluten free": "gluten-free",
    "gluten-free": "gluten-free",
    "healthy": "healthy",
    "low carb": "low-carb",
    "low-carb": "low-carb",
    "dessert": "desserts",
    "desserts": "desserts",
    "breakfast": "breakfast",
    "brunch": "brunch",
    "lunch": "lunch",
    "dinner": "main-dish",
    "main dish": "main-dish",
    "appetizer": "appetizers",
    "appetizers": "appetizers",
    "side dish": "side-dishes",
    "snack": "snacks",
    "soup": "soups-stews",
}

# kako reci kao sto su quick, easy i tako mapiram vremenski
_TIME_KEYWORDS = {
    "quick": 30,
    "fast": 30,
    "easy": 45,
    "30 minute": 30,
    "30-minute": 30,
    "15 minute": 15,
    "weeknight": 45,
}

# pomocni parser za eksplicitne preference
_COMMON_INGREDIENTS = {
    "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "tofu",
    "rice", "pasta", "noodles", "potato", "potatoes", "bread", "egg", "eggs",
    "cheese", "tomato", "tomatoes", "onion", "garlic", "mushroom", "mushrooms",
    "spinach", "broccoli", "carrot", "carrots", "beans", "lentils", "chickpeas",
    "peanuts", "peanut", "shrimp", "bacon", "turkey", "lamb", "avocado",
    "chocolate", "vanilla", "lemon", "lime", "ginger", "basil", "cilantro",
    "vegetables", "veggies", "corn", "peas", "zucchini", "eggplant",
}

# struktura u kojoj se cuvaju zahtevi korisnika 
@dataclass
class Preferences:
    include_ingredients: list[str] = field(default_factory=list)
    exclude_ingredients: list[str] = field(default_factory=list)
    max_minutes: int | None = None
    tags: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.include_ingredients or self.exclude_ingredients
            or self.max_minutes or self.tags
        )


def parse_query_preferences(query: str) -> Preferences:

    q = normalize_text(query)
    prefs = Preferences()

    # vreme pripreme
    for kw, minutes in _TIME_KEYWORDS.items():
        if kw in q:
            prefs.max_minutes = min(prefs.max_minutes or 10_000, minutes)

    # tagovi tj. kategorije
    for kw, tag in _KEYWORD_TAGS.items():
        if re.search(rf"\b{re.escape(kw)}\b", q):
            if tag not in prefs.tags:
                prefs.tags.append(tag)

    # sastojci 
    for ing in _COMMON_INGREDIENTS:
        if re.search(rf"\b{re.escape(ing)}\b", q):
            if ing not in prefs.include_ingredients:
                prefs.include_ingredients.append(ing)

    return prefs


def merge_preferences(a: Preferences, b: Preferences) -> Preferences:
    return Preferences(
        include_ingredients=list(dict.fromkeys(a.include_ingredients + b.include_ingredients)),
        exclude_ingredients=list(dict.fromkeys(a.exclude_ingredients + b.exclude_ingredients)),
        max_minutes=min([x for x in [a.max_minutes, b.max_minutes] if x is not None], default=None),
        tags=list(dict.fromkeys(a.tags + b.tags)),
    )
