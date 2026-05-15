"""
W4: Lexical Transparency Score computation.

Computes a quantitative "lexical transparency score" for each of the 28 GoEmotions
categories using two complementary methods:
    1. Embedding-based: Cosine similarity between the emotion label embedding and
       the mean text embedding of texts labeled with that emotion.
    2. Lexicon-based: NRC Emotion Lexicon coverage rate for each category.

Then correlates these scores with human-LLM Spearman rho to validate the
lexical-grounding gradient hypothesis.

Usage (on GPU server):
    CUDA_VISIBLE_DEVICES=1 python3 lexical_transparency.py \
        --input_csv /path/to/core_set_with_text.csv \
        --human_parquet /path/to/core_set.parquet \
        --per_category_json /path/to/llm_human_comparison_results.json \
        --output_dir /path/to/output/

Outputs:
    - lexical_transparency_scores.json
    - lexical_gradient_correlation.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Constants
# ============================================================

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# NRC Emotion Lexicon: manually curated representative words for each GoEmotions category
# Based on NRC Emotion Lexicon (Mohammad & Turney, 2013) + domain knowledge
# Categories mapped from NRC's 8 emotions + 2 sentiments to GoEmotions 28
NRC_LEXICON_MAPPING = {
    "admiration": ["admire", "admiration", "respect", "appreciate", "impressive", "wonderful", "excellent", "brilliant", "remarkable", "outstanding"],
    "amusement": ["funny", "hilarious", "laugh", "amuse", "humor", "comedy", "joke", "entertaining", "witty", "comical"],
    "anger": ["angry", "furious", "rage", "mad", "outraged", "infuriated", "livid", "irate", "wrathful", "enraged"],
    "annoyance": ["annoying", "irritating", "frustrating", "bothersome", "aggravating", "obnoxious", "pesky", "vexing", "tiresome", "tedious"],
    "approval": ["approve", "agree", "support", "endorse", "accept", "good", "right", "correct", "valid", "appropriate"],
    "caring": ["care", "caring", "kind", "compassion", "empathy", "concern", "thoughtful", "supportive", "considerate", "gentle"],
    "confusion": ["confused", "puzzled", "perplexed", "bewildered", "baffled", "uncertain", "unclear", "lost", "mixed up", "disoriented"],
    "curiosity": ["curious", "interested", "wondering", "intrigued", "fascinated", "inquisitive", "questioning", "eager", "exploratory", "investigative"],
    "desire": ["want", "desire", "wish", "crave", "yearn", "long", "hope", "aspire", "covet", "need"],
    "disappointment": ["disappointed", "let down", "disheartened", "dismayed", "dissatisfied", "discouraged", "deflated", "crestfallen", "disillusioned", "unfulfilled"],
    "disapproval": ["disapprove", "disagree", "wrong", "bad", "unacceptable", "inappropriate", "condemn", "reject", "oppose", "object"],
    "disgust": ["disgust", "revolting", "repulsive", "nauseating", "sickening", "gross", "vile", "loathsome", "abhorrent", "repugnant"],
    "embarrassment": ["embarrassed", "ashamed", "humiliated", "mortified", "awkward", "self-conscious", "sheepish", "flustered", "abashed", "chagrined"],
    "excitement": ["excited", "thrilled", "enthusiastic", "eager", "pumped", "ecstatic", "exhilarated", "animated", "energized", "electrified"],
    "fear": ["afraid", "scared", "frightened", "terrified", "fearful", "anxious", "panicked", "alarmed", "petrified", "dread"],
    "gratitude": ["grateful", "thankful", "appreciate", "thanks", "thank you", "gratitude", "blessed", "indebted", "obliged", "recognition"],
    "grief": ["grief", "mourning", "loss", "bereaved", "sorrowful", "grieving", "heartbroken", "devastated", "anguish", "lamenting"],
    "joy": ["happy", "joyful", "delighted", "pleased", "cheerful", "glad", "elated", "blissful", "merry", "jubilant"],
    "love": ["love", "adore", "cherish", "affection", "devotion", "beloved", "darling", "sweetheart", "passionate", "romantic"],
    "nervousness": ["nervous", "anxious", "worried", "tense", "uneasy", "apprehensive", "restless", "jittery", "edgy", "fidgety"],
    "optimism": ["optimistic", "hopeful", "positive", "confident", "encouraging", "promising", "bright", "upbeat", "sanguine", "expectant"],
    "pride": ["proud", "pride", "accomplished", "triumphant", "dignified", "self-assured", "honored", "fulfilled", "gratified", "victorious"],
    "realization": ["realize", "understand", "discover", "notice", "recognize", "aware", "epiphany", "insight", "comprehend", "dawn"],
    "relief": ["relief", "relieved", "comforted", "reassured", "calmed", "soothed", "eased", "unburdened", "relaxed", "alleviated"],
    "remorse": ["sorry", "regret", "remorse", "guilty", "apologetic", "contrite", "repentant", "penitent", "rueful", "ashamed"],
    "sadness": ["sad", "unhappy", "depressed", "melancholy", "sorrowful", "gloomy", "heartbroken", "miserable", "tearful", "dejected"],
    "surprise": ["surprised", "astonished", "amazed", "shocked", "startled", "stunned", "unexpected", "wow", "unbelievable", "incredible"],
    "neutral": ["okay", "fine", "normal", "average", "ordinary", "regular", "typical", "standard", "common", "unremarkable"],
}


def compute_embedding_transparency(
    texts: list[str],
    text_ids: list[str],
    human_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
) -> dict:
    """Compute embedding-based lexical transparency scores.

    For each emotion:
      1. Embed the emotion label word
      2. Embed all texts where that emotion has >0 human annotation rate
      3. Score = cosine similarity between label embedding and mean text embedding
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence transformer: {model_name}")
    st_model = SentenceTransformer(model_name, device=device)

    # Embed emotion labels
    print("Embedding emotion labels...")
    label_embeddings = st_model.encode(EMOTION_LABELS, show_progress_bar=False)

    # Build text lookup
    id_to_text = dict(zip(text_ids, texts))

    # Embed all texts
    print("Embedding all texts...")
    all_text_embeddings = st_model.encode(texts, show_progress_bar=True, batch_size=64)
    id_to_embedding = dict(zip(text_ids, all_text_embeddings))

    scores = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        dist_col = f"{emotion}_dist"
        if dist_col not in human_df.columns:
            continue

        # Get texts where this emotion has >0 human rate
        positive_ids = human_df[human_df[dist_col] > 0]["id"].tolist()
        if len(positive_ids) < 5:
            scores[emotion] = {
                "embedding_score": float("nan"),
                "n_positive_texts": len(positive_ids),
            }
            continue

        # Get embeddings for positive texts
        pos_embeddings = np.array([id_to_embedding[tid] for tid in positive_ids if tid in id_to_embedding])
        mean_text_emb = pos_embeddings.mean(axis=0)

        # Cosine similarity
        cos_sim = float(np.dot(label_embeddings[i], mean_text_emb) /
                       (np.linalg.norm(label_embeddings[i]) * np.linalg.norm(mean_text_emb)))

        scores[emotion] = {
            "embedding_score": cos_sim,
            "n_positive_texts": len(positive_ids),
        }

    return scores


def compute_lexicon_coverage(
    texts: list[str],
    text_ids: list[str],
    human_df: pd.DataFrame,
) -> dict:
    """Compute NRC Emotion Lexicon coverage rates per category.

    For each emotion:
      coverage = fraction of positive-labeled texts containing at least one
      lexicon word for that emotion.
    """
    id_to_text = dict(zip(text_ids, texts))

    scores = {}
    for emotion in EMOTION_LABELS:
        dist_col = f"{emotion}_dist"
        if dist_col not in human_df.columns:
            continue

        lexicon_words = NRC_LEXICON_MAPPING.get(emotion, [])
        if not lexicon_words:
            scores[emotion] = {"lexicon_coverage": float("nan"), "n_positive_texts": 0}
            continue

        positive_ids = human_df[human_df[dist_col] > 0]["id"].tolist()
        if len(positive_ids) < 5:
            scores[emotion] = {"lexicon_coverage": float("nan"), "n_positive_texts": len(positive_ids)}
            continue

        n_covered = 0
        for tid in positive_ids:
            text = id_to_text.get(tid, "").lower()
            if any(word.lower() in text for word in lexicon_words):
                n_covered += 1

        coverage = n_covered / len(positive_ids)
        scores[emotion] = {
            "lexicon_coverage": coverage,
            "n_positive_texts": len(positive_ids),
        }

    return scores


def correlate_with_human_llm_rho(
    transparency_scores: dict,
    per_category_results: dict,
    models: list[str],
) -> dict:
    """Correlate transparency scores with human-LLM Spearman rho."""
    results = {}

    for score_type in ["embedding_score", "lexicon_coverage", "combined_score"]:
        scores = []
        rhos = {m: [] for m in models}

        for emotion in EMOTION_LABELS:
            if emotion not in transparency_scores:
                continue
            score = transparency_scores[emotion].get(score_type)
            if score is None or np.isnan(score):
                continue

            scores.append(score)
            for m in models:
                if emotion in per_category_results.get(m, {}):
                    rhos[m].append(per_category_results[m][emotion].get("spearman_rho", 0))
                else:
                    rhos[m].append(0)

        if len(scores) < 5:
            continue

        correlations = {}
        for m in models:
            if len(rhos[m]) == len(scores):
                r, p = spearmanr(scores, rhos[m])
                correlations[m] = {"spearman_r": float(r), "p_value": float(p)}

        # Average across models
        all_rhos = [np.mean([rhos[m][i] for m in models]) for i in range(len(scores))]
        r_avg, p_avg = spearmanr(scores, all_rhos)

        results[score_type] = {
            "per_model": correlations,
            "average_across_models": {"spearman_r": float(r_avg), "p_value": float(p_avg)},
            "n_categories": len(scores),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Core set CSV with text and id columns")
    parser.add_argument("--human_parquet", required=True, help="Human distribution parquet")
    parser.add_argument("--per_category_json", required=True, help="LLM comparison results JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    text_df = pd.read_csv(args.input_csv)
    human_df = pd.read_parquet(args.human_parquet)

    texts = text_df["text"].tolist()
    text_ids = text_df["id"].tolist()

    # Load per-category results
    with open(args.per_category_json, "r") as f:
        comparison_results = json.load(f)

    # Extract per-category rho from results
    # Structure: results[model_key]["per_category_divergence"][...] or similar
    # We need to adapt to actual JSON structure
    per_category_rhos = {}
    model_names = ["gpt-5.4-mini", "claude-haiku-4-5-20251001", "llama3.1-8b", "qwen3-8b"]

    for key, data in comparison_results.items():
        if "tempall" in key:
            model = key.replace("_tempall", "")
            if "per_category" in data:
                per_category_rhos[model] = data["per_category"]

    # If per_category not found at top level, try alternate structure
    if not per_category_rhos:
        # Try reading from per_category_divergence parquet
        per_cat_path = args.per_category_json.replace("llm_human_comparison_results.json", "per_category_divergence.parquet")
        if os.path.exists(per_cat_path):
            print(f"Reading per-category from parquet: {per_cat_path}")
            cat_df = pd.read_parquet(per_cat_path)
            for model in cat_df["model"].unique():
                model_data = cat_df[(cat_df["model"] == model) & (cat_df["temperature"] == "all")]
                per_category_rhos[model] = {}
                for _, row in model_data.iterrows():
                    per_category_rhos[model][row["emotion"]] = {
                        "spearman_rho": row.get("spearman_rho", 0),
                    }

    # 1. Embedding-based transparency
    print("\n=== Computing Embedding-based Transparency ===")
    embedding_scores = compute_embedding_transparency(
        texts, text_ids, human_df, device=args.device
    )

    # 2. Lexicon-based transparency
    print("\n=== Computing Lexicon-based Transparency ===")
    lexicon_scores = compute_lexicon_coverage(texts, text_ids, human_df)

    # 3. Combined score (average of normalized embedding + lexicon)
    transparency_scores = {}
    emb_vals = [embedding_scores[e]["embedding_score"] for e in EMOTION_LABELS
                if not np.isnan(embedding_scores.get(e, {}).get("embedding_score", float("nan")))]
    lex_vals = [lexicon_scores[e]["lexicon_coverage"] for e in EMOTION_LABELS
                if not np.isnan(lexicon_scores.get(e, {}).get("lexicon_coverage", float("nan")))]

    emb_min, emb_max = min(emb_vals), max(emb_vals)
    lex_min, lex_max = min(lex_vals), max(lex_vals)

    for emotion in EMOTION_LABELS:
        emb_score = embedding_scores.get(emotion, {}).get("embedding_score", float("nan"))
        lex_score = lexicon_scores.get(emotion, {}).get("lexicon_coverage", float("nan"))

        # Normalize to [0, 1]
        emb_norm = (emb_score - emb_min) / (emb_max - emb_min + 1e-12) if not np.isnan(emb_score) else float("nan")
        lex_norm = (lex_score - lex_min) / (lex_max - lex_min + 1e-12) if not np.isnan(lex_score) else float("nan")

        valid_scores = [s for s in [emb_norm, lex_norm] if not np.isnan(s)]
        combined = np.mean(valid_scores) if valid_scores else float("nan")

        transparency_scores[emotion] = {
            "embedding_score": emb_score,
            "lexicon_coverage": lex_score,
            "embedding_score_normalized": emb_norm,
            "lexicon_coverage_normalized": lex_norm,
            "combined_score": combined,
            "n_positive_texts": embedding_scores.get(emotion, {}).get("n_positive_texts", 0),
        }

    # Save transparency scores
    scores_path = os.path.join(args.output_dir, "lexical_transparency_scores.json")
    with open(scores_path, "w") as f:
        json.dump(transparency_scores, f, indent=2)
    print(f"Transparency scores saved to: {scores_path}")

    # 4. Correlate with human-LLM rho
    print("\n=== Correlating with Human-LLM Agreement ===")
    models_found = list(per_category_rhos.keys())
    if models_found:
        gradient_results = correlate_with_human_llm_rho(
            transparency_scores, per_category_rhos, models_found
        )

        gradient_path = os.path.join(args.output_dir, "lexical_gradient_correlation.json")
        with open(gradient_path, "w") as f:
            json.dump(gradient_results, f, indent=2)
        print(f"Gradient correlation saved to: {gradient_path}")

        # Print summary
        for score_type, data in gradient_results.items():
            avg = data.get("average_across_models", {})
            print(f"  {score_type}: r={avg.get('spearman_r', 'N/A'):.3f}, p={avg.get('p_value', 'N/A'):.4f}")
    else:
        print("  WARNING: No per-category rho data found. Skipping correlation.")

    # Print top/bottom emotions
    print("\n=== Lexical Transparency Rankings ===")
    sorted_emotions = sorted(
        [(e, s["combined_score"]) for e, s in transparency_scores.items() if not np.isnan(s["combined_score"])],
        key=lambda x: x[1],
        reverse=True,
    )
    print("Top 5 (most transparent):")
    for e, s in sorted_emotions[:5]:
        print(f"  {e}: {s:.3f}")
    print("Bottom 5 (least transparent):")
    for e, s in sorted_emotions[-5:]:
        print(f"  {e}: {s:.3f}")


if __name__ == "__main__":
    main()
