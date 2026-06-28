# 🍽️🤖 NLPlate

NLPlate is a hybrid AI-based recipe recommendation system developed for recommending recipes from the Food.com Recipes and Interactions dataset.

The system combines recipe content, semantic similarity, user review sentiment, rating quality and rating agreement in order to generate more relevant and reliable recipe recommendations.

## Project Overview

The main goal of NLPlate is to connect a free-text user query with recipes that are both relevant to the query and supported by positive user feedback.

Example query:

```text
I want a quick and healthy chicken dinner with vegetables
```

The system uses several recommendation components:

* TF-IDF content-based similarity
* Sentence-BERT semantic similarity
* VADER sentiment analysis of user reviews
* Bayesian rating quality
* Rating agreement between users
* Hybrid weighted ranking

The final recommendation score is calculated as a weighted combination of these components.

## Dataset

The project uses the Food.com Recipes and Interactions dataset from Kaggle:

https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

The main raw files used in this project are:

* `RAW_recipes.csv`
* `RAW_interactions.csv`

Raw and preprocessed data files are not included in this repository because of their size.

After downloading the dataset, place the required raw files in:

```text
data/raw/
```

Expected structure:

```text
data/
└── raw/
    ├── RAW_recipes.csv
    └── RAW_interactions.csv
```

Preprocessed files, TF-IDF matrices and Sentence-BERT embeddings can be recreated by running the provided scripts.

## Project Structure

```text
NLPlate/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│
├── src/
│   ├── preprocessing.py
│   ├── content_based.py
│   ├── semantic.py
│   ├── sentiment.py
│   ├── rating_quality.py
│   ├── recommender.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── ...
│
├── scripts/
│   ├── 00_eda.py
│   ├── 01_preprocess.py
│   ├── 02_build_indexes.py
│   ├── 03_evaluate_baselines.py
│   ├── 04_tune_weights.py
│   ├── 05_report_figures.py
│   └── run_all.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the following files in `data/raw/`:

```text
RAW_recipes.csv
RAW_interactions.csv
```

Run preprocessing:

```bash
python scripts/01_preprocess.py
```

Build indexes:

```bash
python scripts/02_build_indexes.py
```

Run evaluation and tune hybrid weights:

```bash
python scripts/03_evaluate_baselines.py
python scripts/04_tune_weights.py --step 0.05 --min-weight 0.05
```

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

## Evaluation

The system was evaluated using a fixed set of representative natural language queries.

Since the dataset does not contain explicit relevance labels for query-recipe pairs, a proxy relevance measure was defined based on:

* matching the query constraints,
* having enough user reviews,
* achieving high Bayesian rating quality.

The main evaluation metrics were:

* Precision@K
* Recall@K
* NDCG@K

The tuned hybrid model improved the ranking quality compared to the initial hybrid configuration.

## Application Features

The Streamlit application allows users to:

* enter a natural language recipe query,
* specify preferred and excluded ingredients,
* set maximum cooking time,
* choose recipe tags/categories,
* select the number of recommendations,
* compare baseline, tuned and manually adjusted weights,
* view explanations for each recommended recipe.

## Notes

Large generated files, such as preprocessed datasets, TF-IDF matrices and Sentence-BERT embeddings, are excluded from the repository due to their size.

They can be recreated from the raw dataset by running the preprocessing and indexing scripts.

