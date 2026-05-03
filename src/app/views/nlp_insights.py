"""Страница «NLP-инсайты» — BERTopic темы + UMAP + semantic search.

Источники:
- ``GET /nlp/topics`` (FastAPI) — 7 тем + шум, top-10 c-TF-IDF слов,
  5 примеров постов, em_type-связь через 482 gold-пары
- ``GET /nlp/search`` (FastAPI) — semantic search через sentence-
  transformers (paraphrase-multilingual-MiniLM-L12-v2)
- ``data/processed/telegram_umap_2d.npz`` — pre-computed UMAP-2D
  координаты, читаются прямым numpy.load для plotly scatter

Дефект T4: rule-based фильтр пропустил 72 не-ДТП поста
(юбилеи, кадровые перестановки), которые попали в тему -1 (шум).
Главная находка страницы: BERTopic нашёл проблему фильтра,
которую rule-based проверка пропустила».
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.utils.api_client import fetch_topics
from src.app.utils.embeddings import semantic_search
from src.app.utils.styling import page_footer, page_header
from src.app.utils.visualizations import fmt_int

UMAP_PATH = Path("data/processed/telegram_umap_2d.npz")

page_header(
    title="NLP-инсайты по Telegram @prim_police",
    subtitle=(
        "BERTopic (paraphrase-multilingual-MiniLM-L12-v2) на 2 122 постах "
        "канала «Полиция Приморья» (@prim_police, 2018-2025): 7 содержательных "
        "тем + шум. Кластеры связаны с категориями `em_type` БД через 482 "
        "gold-пары (top_score ≥ 75)."
    ),
    icon=":material/psychology:",
)

# ============================================================
# Загрузка данных
# ============================================================
try:
    topics_payload = fetch_topics(include_noise=True)
except httpx.HTTPError as exc:
    st.error(f"FastAPI недоступно: {exc}")
    st.stop()

topics = topics_payload["items"]


# ============================================================
# KPI
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Постов", fmt_int(topics_payload["n_posts"]))
c2.metric("Тем (без шума)", topics_payload["n_topics"])
c3.metric(
    "В шуме (-1)",
    fmt_int(topics_payload["n_noise"]),
    delta=f"{topics_payload['noise_share_pct']:.1f}% корпуса",
    delta_color="off",
)
c4.metric("Эмбеддинг", "MiniLM-L12-v2", help=topics_payload["embedding_model"])

st.divider()

# ============================================================
# Дефект T4 — отдельная плашка с ключевой находкой
# ============================================================
st.warning(
    "**Дефект T4 (post-hoc audit BERTopic):** rule-based фильтр "
    "`is_dtp_related` пропустил **72 не-ДТП поста** "
    "(юбилеи, кадровые перестановки, поздравления), которые тематический "
    "кластеризатор сгруппировал в тему -1 (шум). BERTopic неявно "
    "сделал post-hoc валидацию rule-based фильтра — цена пропуска "
    "≈ 3.4 % корпуса. Решение: усилить keyword-blacklist "
    "по результатам кластеров шума.",
    icon=":material/bug_report:",
)

# ============================================================
# UMAP-2D scatter (статичный, через plotly)
# ============================================================
st.subheader("UMAP-2D проекция всех 2 122 постов")
st.caption(
    "Каждая точка — пост, цвет — назначенная тема. Близкие точки = "
    "семантически похожие посты. Видно компактные сгустки тем 0-6 и "
    "разреженный хвост темы -1 (шум)."
)

# topic_names используется ниже в рендере результатов semantic search,
# поэтому собираем его ДО блока UMAP — иначе при отсутствии UMAP-файла
# запрос в search-боксе падал с NameError.
topic_names = {int(t["topic_id"]): t["name"] for t in topics}

if UMAP_PATH.exists():
    npz = np.load(UMAP_PATH)
    umap_df = pd.DataFrame(
        {
            "x": npz["coords"][:, 0],
            "y": npz["coords"][:, 1],
            "topic": npz["topics"].astype(int),
            "tg_id": npz["tg_ids"].astype(int),
        }
    )
    umap_df["topic_name"] = umap_df["topic"].map(lambda t: topic_names.get(t, f"topic_{t}"))
    # Подсвечиваем шум серым, темы — qualitative palette.
    palette = {
        -1: "#bbbbbb",
        0: "#1f77b4",
        1: "#2ca02c",
        2: "#d62728",
        3: "#9467bd",
        4: "#8c564b",
        5: "#e377c2",
        6: "#17becf",
    }
    # В легенде показываем «N — Краткое имя темы», чтобы зритель сразу
    # видел СОДЕРЖАНИЕ тем, а не голые «0/1/2/...». Это критическая правка
    # audit item 1.25.
    umap_df["topic_legend"] = umap_df.apply(lambda r: f"{r['topic']} — {r['topic_name']}", axis=1)
    legend_color_map = {f"{k} — {topic_names.get(k, f'topic_{k}')}": v for k, v in palette.items()}
    # category_orders: «-1 шум» в самый низ; темы — по возрастанию topic_id.
    legend_order = sorted(
        umap_df["topic_legend"].unique(),
        key=lambda s: (int(s.split(" — ")[0]) if s.split(" — ")[0].lstrip("-").isdigit() else 99),
    )
    fig_umap = px.scatter(
        umap_df,
        x="x",
        y="y",
        color="topic_legend",
        color_discrete_map=legend_color_map,
        category_orders={"topic_legend": legend_order},
        hover_data={
            "tg_id": True,
            "topic_name": True,
            "x": False,
            "y": False,
            "topic_legend": False,
        },
        opacity=0.65,
        height=460,
        labels={"topic_legend": "Тема"},
    )
    fig_umap.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig_umap.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        legend=dict(font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    )
    st.plotly_chart(fig_umap, use_container_width=True)
else:
    st.info(f"UMAP-файл не найден: {UMAP_PATH}")

st.divider()

# ============================================================
# Semantic search
# ============================================================
st.subheader(":material/search: Поиск похожих постов (semantic)")
st.caption(
    "Введи произвольный текст (фразу, описание ДТП). "
    "Sentence-transformer кодирует запрос, FastAPI считает cosine "
    "similarity со всеми 2 122 постами и возвращает топ-K. Первый "
    "запрос ~5 сек (lazy-load модели), последующие — ~80 мс."
)

# Подсказки-чипсы с типичными запросами — клик заполняет input
# через session_state. Помогает первому пользователю понять «как
# формулировать», а не уходить с пустым поиском.
_EXAMPLES = [
    "пьяный водитель перевернулся в кювет",
    "наезд на пешехода ночью",
    "столкновение на перекрёстке",
    "съезд с трассы в гололёд",
]
if "nlp_search_query" not in st.session_state:
    st.session_state["nlp_search_query"] = ""

st.caption(":material/touch_app: Примеры запросов:")
chip_cols = st.columns(len(_EXAMPLES))
for i, ex in enumerate(_EXAMPLES):
    if chip_cols[i].button(ex, key=f"nlp_chip_{i}", use_container_width=True):
        st.session_state["nlp_search_query"] = ex
        st.rerun()

search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    query = st.text_input(
        "Текст-запрос",
        value=st.session_state["nlp_search_query"],
        placeholder="Например: «пьяный водитель перевернулся в кювет»",
        label_visibility="collapsed",
        key="nlp_search_query_input",
    )
with search_col2:
    top_k = st.number_input("top-K", min_value=1, max_value=20, value=5, step=1)

if query:
    try:
        with st.spinner("FastAPI: encode + cosine similarity..."):
            search_resp = semantic_search(query, int(top_k))
    except httpx.HTTPError as exc:
        st.error(f"Ошибка поиска: {exc}")
    else:
        st.caption(
            f"Найдено за {search_resp.get('elapsed_ms', 0):.1f} мс "
            f"(model: `{search_resp.get('embedding_model', '')}`)"
        )
        if not search_resp["items"]:
            st.info("Ничего не найдено")
        else:
            for hit in search_resp["items"]:
                topic_name = topic_names.get(
                    hit.get("topic_id", -999), f"topic_{hit.get('topic_id')}"
                )
                with st.container():
                    cols = st.columns([1, 4])
                    cols[0].metric(
                        "similarity",
                        f"{hit['similarity']:.3f}",
                        delta=f"тема: {hit.get('topic_id')}",
                        delta_color="off",
                    )
                    preview = (hit.get("text_preview") or "*текст недоступен*").strip()
                    cols[1].markdown(f"**tg_id={hit['tg_id']}** · *{topic_name}*  \n" f"{preview}")
                st.divider()

# ============================================================
# Подробный список тем
# ============================================================
st.subheader("Темы BERTopic")

# Сортируем — сначала содержательные темы по убыванию размера, шум (-1)
# в самый конец. Раньше шум шёл первым (первый expander) — пользователь,
# открывавший по умолчанию первый, видел «мусор» вместо самой большой
# содержательной темы.
sorted_topics = sorted(
    topics,
    key=lambda t: (1 if t["topic_id"] == -1 else 0, -t["size"]),
)

for idx, t in enumerate(sorted_topics):
    is_noise = t["topic_id"] == -1
    icon = ":material/blur_on:" if is_noise else ":material/topic:"
    # Самый первый topic-expander (топ-1 по размеру среди содержательных тем)
    # — раскрыт по умолчанию. Показывает «как читать BERTopic-результат»
    # без необходимости тыкать.
    is_first_substantive = idx == 0 and not is_noise
    with st.expander(
        f"{icon} **Тема {t['topic_id']}** · "
        f"{fmt_int(t['size'])} постов "
        f"({t['share_pct']:.1f}%) · "
        f"{', '.join(t['top_words'][:5])}",
        expanded=is_first_substantive,
    ):
        wc1, wc2 = st.columns(2)

        with wc1:
            st.markdown("**Топ-10 c-TF-IDF слов**")
            words_df = pd.DataFrame(
                {"слово": t["top_words"], "rank": range(1, len(t["top_words"]) + 1)}
            )
            st.dataframe(
                words_df,
                hide_index=True,
                use_container_width=True,
                height=380,
            )

            em_link = t.get("em_type_link")
            if em_link and em_link.get("dominant_em_type"):
                st.markdown(f"**Связь с em_type через {em_link['n_gold_posts']} gold-пар:**")
                em_df = pd.DataFrame(
                    [{"em_type": k, "доля, %": v} for k, v in em_link["shares_pct"].items()]
                ).sort_values("доля, %", ascending=False)
                st.dataframe(
                    em_df,
                    hide_index=True,
                    use_container_width=True,
                    height=180,
                    column_config={
                        "доля, %": st.column_config.NumberColumn(format="%.1f %%"),
                    },
                )

        with wc2:
            st.markdown("**5 примеров постов**")
            for i, ex in enumerate(t.get("examples", [])[:5], 1):
                st.markdown(f"_{i}._ {(ex or '').strip()[:300]}")
                st.markdown("")

# Footer — единый для всех 7 страниц.
page_footer()
