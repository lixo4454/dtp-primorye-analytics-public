# -*- coding: utf-8 -*-
"""
BERTopic-кластеризация ДТП-постов УМВД Приморья.

Что делает:
- Собирает корпус 2122 ДТП-релевантных постов (текст из data/raw/, tg_id из NER-результатов).
- Считает sentence-transformers эмбеддинги (paraphrase-multilingual-MiniLM-L12-v2).
- Обучает BERTopic = UMAP(10D) → HDBSCAN → c-TF-IDF.
- Описывает темы: топ-слова, размер, примеры постов.
- Связывает темы со структурной БД через 482 gold-пары — heatmap topic × em_type.

Зачем нужно:
- 4-я ML-модель портфолио для ВВГУ-магистратуры/Роспатента (после Prophet, CatBoost-baseline, CatBoost-tuned).
- Качественная аналитика: какие сюжеты ДТП доминируют в публичных сводках МВД.
- Нарратив: «темы из текста vs em_type из БД» — две независимые таксономии, можно сравнить.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_TG_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"
NER_RESULTS = PROJECT_ROOT / "data" / "processed" / "telegram_ner_results.jsonl"
GOLD_MATCHES = PROJECT_ROOT / "data" / "processed" / "telegram_db_matches.jsonl"

# Канал УМВД Приморья — каждое ДТП-сообщение содержит эти слова. Без чистки
# они забьют топ c-TF-IDF каждой темы и темы станут неотличимы.
DTP_DOMAIN_STOPWORDS = {
    # ДТП и его синонимы
    "дтп",
    "авария",
    "аварии",
    "аварию",
    "аварией",
    "аварийный",
    "аварийная",
    "происшествие",
    "происшествия",
    "происшествии",
    "происшествий",
    "столкновение",
    "столкновения",
    "столкновении",
    "столкнулись",
    "столкнулась",
    # Водитель/машина — встречаются в каждом посте
    "водитель",
    "водителя",
    "водителю",
    "водителем",
    "водители",
    "водителей",
    "автомобиль",
    "автомобили",
    "автомобиля",
    "автомобилем",
    "автомобилю",
    "автомобилях",
    "авто",
    "машина",
    "машины",
    "машину",
    "машиной",
    "машине",
    "транспорт",
    "транспортного",
    "транспортные",
    "тс",
    # Гос-структуры — клише пресс-службы
    "госавтоинспекция",
    "госавтоинспекции",
    "госавтоинспекцию",
    "гибдд",
    "мвд",
    "увд",
    "умвд",
    "полиция",
    "полицейские",
    "полицейский",
    "инспектор",
    "инспекторы",
    "инспекторов",
    "сотрудник",
    "сотрудники",
    "сотрудниками",
    "наряд",
    "наряды",
    # География (мы все знаем что это Приморье)
    "приморский",
    "приморского",
    "приморском",
    "приморье",
    "приморья",
    "приморью",
    "приморским",
    "приморской",
    "край",
    "края",
    "крае",
    "краю",
    "владивосток",
    "владивостока",
    "владивостоке",
    # Глаголы-связки сводки
    "произошло",
    "произошла",
    "случилось",
    "сообщает",
    "сообщили",
    "пресс",
    "служба",
    "службы",
    "сообщению",
    "напомнил",
    "напомнили",
    "предупреждает",
    "предупреждаем",
    "уважаемые",
    "граждане",
    # Числа-«общие места»
    "год",
    "года",
    "году",
    "годов",
    "час",
    "часа",
    "часов",
    "минут",
    "число",
    "числа",
    "первый",
    "второй",
    "третий",
    # Канцелярит
    "также",
    "ещё",
    "более",
    "менее",
    "около",
    "примерно",
    "после",
    "которые",
    "которая",
    "которое",
    "которого",
    "которому",
    "место",
    "места",
    "месте",
    "местом",
    "местах",
    "случай",
    "случая",
    "случае",
    "случаи",
    "случаев",
}


def russian_stopwords() -> list[str]:
    """NLTK russian (~150 слов) + кастомный ДТП-словарь канала УМВД."""
    import nltk

    try:
        from nltk.corpus import stopwords

        nltk_sw = set(stopwords.words("russian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        nltk_sw = set(stopwords.words("russian"))
    return sorted(nltk_sw | DTP_DOMAIN_STOPWORDS)


# ────────────────────────────────────────────────────────────────────
# Корпус
# ────────────────────────────────────────────────────────────────────


@dataclass
class CorpusRecord:
    tg_id: int
    text: str
    text_clean: str
    date_published: str


def _normalize(text: str) -> str:
    """Минимальный препроцессинг: lowercase, схлопывание пробелов, удаление URL."""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_corpus(
    ner_results_path: Path = NER_RESULTS,
    raw_dir: Path = RAW_TG_DIR,
) -> list[CorpusRecord]:
    """Собирает все ДТП-релевантные посты по tg_id из NER-результатов.

    Возвращает упорядоченный по tg_id список CorpusRecord. Гарантирует 100%
    покрытие — кидает RuntimeError если не нашли текст для какого-то tg_id.
    """
    ner_ids: set[int] = set()
    with ner_results_path.open("r", encoding="utf-8") as f:
        for line in f:
            ner_ids.add(json.loads(line)["tg_id"])

    found: dict[int, dict] = {}
    for fp in sorted(raw_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["tg_id"] in ner_ids and rec["tg_id"] not in found:
                    found[rec["tg_id"]] = rec

    missing = ner_ids - set(found.keys())
    if missing:
        raise RuntimeError(
            f"Не найдены тексты для {len(missing)} tg_id из NER-результатов "
            f"(пример: {sorted(missing)[:5]})"
        )

    records = []
    for tid in sorted(ner_ids):
        rec = found[tid]
        records.append(
            CorpusRecord(
                tg_id=tid,
                text=rec["text"],
                text_clean=_normalize(rec["text"]),
                date_published=rec["date_published"],
            )
        )
    return records


# ────────────────────────────────────────────────────────────────────
# Эмбеддинги
# ────────────────────────────────────────────────────────────────────


def compute_embeddings(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """sentence-transformers emb. Автоматически использует CUDA если доступна."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings


# ────────────────────────────────────────────────────────────────────
# BERTopic
# ────────────────────────────────────────────────────────────────────


def build_bertopic(
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    min_cluster_size: int = 30,
    umap_n_components: int = 10,
    random_state: int = 42,
):
    """Конфигурирует BERTopic с UMAP(10D) → HDBSCAN → c-TF-IDF.

    Не передаёт embedding_model в BERTopic — эмбеддинги считаются заранее
    (быстрее, кэшируются на диск).
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    # ВАЖНО: BERTopic применяет vectorizer к "темовым" документам (1 на тему),
    # а не к 2122 исходным постам. При ~10 темах min_df=2/max_df=0.95 могут
    # обнулить словарь. min_df=1/max_df=1.0 безопасно — c-TF-IDF сам взвесит
    # слова через специфичность к теме.
    vectorizer_model = CountVectorizer(
        stop_words=russian_stopwords(),
        lowercase=True,
        min_df=1,
        max_df=1.0,
        # только кириллические слова длиной 3+ символа (без латиницы и чисел)
        token_pattern=r"(?u)\b[а-яё][а-яё-]{2,}\b",
        ngram_range=(1, 1),
    )

    topic_model = BERTopic(
        # КРИТИЧНО: дефолт language="english" в _preprocess_text применяет
        # `re.sub(r"[^A-Za-z0-9 ]+", "", doc)` и вырезает всю кириллицу,
        # из-за чего CountVectorizer падает с empty vocabulary.
        language="multilingual",
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


def fit_topic_model(topic_model, texts: list[str], embeddings: np.ndarray):
    """Обучает модель и возвращает (topics, probs)."""
    topics, probs = topic_model.fit_transform(texts, embeddings)
    return np.asarray(topics), probs


# ────────────────────────────────────────────────────────────────────
# Описание тем
# ────────────────────────────────────────────────────────────────────


def describe_topics(
    topic_model,
    texts: list[str],
    topics: np.ndarray,
    k_words: int = 10,
    k_examples: int = 5,
) -> list[dict]:
    """Для каждой темы (включая -1 шум): top-k слов c-TF-IDF, размер, примеры."""
    info = topic_model.get_topic_info()  # DataFrame: Topic, Count, Name, ...
    out = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        words_with_scores = topic_model.get_topic(topic_id) or []
        top_words = [w for w, _ in words_with_scores[:k_words]]
        top_scores = [float(s) for _, s in words_with_scores[:k_words]]

        idxs = np.where(topics == topic_id)[0]
        # Примеры — первые k уникальных текстов (в естественном порядке tg_id)
        examples = [texts[i][:300] for i in idxs[:k_examples]]

        out.append(
            {
                "topic_id": topic_id,
                "size": int(row["Count"]),
                "share_pct": round(100.0 * len(idxs) / len(topics), 2),
                "name": str(row["Name"]),
                "top_words": top_words,
                "top_words_scores": top_scores,
                "examples": examples,
            }
        )
    return out


# ────────────────────────────────────────────────────────────────────
# Связь с em_type через gold-пары
# ────────────────────────────────────────────────────────────────────


def load_gold_pairs(
    gold_path: Path = GOLD_MATCHES,
    score_threshold: int = 90,
) -> dict[int, dict]:
    """Читает telegram_db_matches.jsonl и возвращает {tg_id: top_match}
    только для постов с top_score >= threshold (gold)."""
    gold: dict[int, dict] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("top_score", 0) >= score_threshold and rec.get("matches"):
                gold[rec["tg_id"]] = rec["matches"][0]
    return gold


def link_topics_to_em_type(
    tg_ids: list[int],
    topics: np.ndarray,
    gold_pairs: dict[int, dict],
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Строит cross-tab topic × em_type для постов из gold-пар.

    Возвращает:
    - DataFrame (rows=topics, cols=em_type, values=count)
    - dict {topic_id: {em_type: share_pct, dominant: str, n_gold: int}}
    """
    rows = []
    for i, tid in enumerate(tg_ids):
        if tid in gold_pairs:
            rows.append(
                {
                    "tg_id": tid,
                    "topic_id": int(topics[i]),
                    "em_type": gold_pairs[tid].get("em_type") or "Не указан",
                    "severity": gold_pairs[tid].get("severity") or "unknown",
                }
            )
    df_gold = pd.DataFrame(rows)

    if df_gold.empty:
        return pd.DataFrame(), {}

    crosstab = pd.crosstab(df_gold["topic_id"], df_gold["em_type"])

    summary: dict[int, dict] = {}
    for topic_id, row in crosstab.iterrows():
        total = int(row.sum())
        shares = (row / total * 100).round(2).to_dict()
        dominant = row.idxmax()
        summary[int(topic_id)] = {
            "n_gold_posts": total,
            "shares_pct": {k: float(v) for k, v in shares.items()},
            "dominant_em_type": str(dominant),
            "dominant_share_pct": float(shares[dominant]),
        }
    return crosstab, summary


# ────────────────────────────────────────────────────────────────────
# UMAP 2D для визуализации
# ────────────────────────────────────────────────────────────────────


def umap_2d_projection(
    embeddings: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Отдельный 2D UMAP для scatter-plot. Внутри BERTopic другой 10D UMAP
    используется для кластеризации — это ОК, просто считаем второй раз."""
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)
