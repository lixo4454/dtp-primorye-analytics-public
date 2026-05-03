"""Главная страница: hero + KPI + карточки-разделы + тренд + топ-5 очагов.

Финальный layout:

1. ``page_header`` — единый header (см. ``utils/styling.py``).
2. **Hero** — синий ``st.info`` с описанием проекта и bullet-списком фич.
3. **KPI-strip** — 4 карточки ``st.container(border=True)`` с
   ``st.metric`` (всего ДТП, смертельных, погибших, очагов DBSCAN).
   Карточки получают ``min-height`` из CSS — выглядят как набор плиток
   одного размера, а не «случайные числа в столбик».
4. **Карточки разделов** — 6 кликабельных тайлов через
   ``st.page_link`` в ``st.container(border=True)``, по 3 в ряд. Это
   landing-навигация для впервые открывших дашборд: пользователь сразу
   видит куда идти за картой / прогнозом / стат-аналитикой.
5. **Тренд по годам** + **распределение тяжести** — два графика бок-о-бок.
   Pie-chart предыдущей версии заменён на **horizontal stacked bar**
   (одна полоска 100% с 4 цветными сегментами): меньше «текст наезжает на
   сегменты», читается с расстояния, занимает меньше высоты.
6. **Топ-5 очагов** — превью таблицы.
7. ``page_footer`` — единый footer (timestamp, версии моделей, ссылки).

Источник данных — FastAPI (``/accidents/stats``, ``/clusters/hotspots``).
Это «корректная архитектура» лёгкие агрегаты,
которые уже посчитаны в API, не трогая БД напрямую.
"""

from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils.api_client import fetch_hotspots, fetch_stats
from src.app.utils.styling import page_footer, page_header
from src.app.utils.visualizations import (
    PRIMARY_COLOR,
    SEVERITY_COLORS,
    SEVERITY_LABELS,
    SEVERITY_ORDER_RU,
    fmt_int,
)

# ============================================================
# Header — page_header() helper, единый для всех 7 страниц
# ============================================================
page_header(
    title="Анализ ДТП Приморского края",
    subtitle=(
        "Аналитический дашборд по 29 413 дорожно-транспортных "
        "происшествиям из открытых данных dtp-stat.ru (агрегатор "
        "открытых данных МВД РФ) за 2015 — апрель 2026 года, с "
        "дополнением Telegram-каналом «Полиция Приморья» "
        "(@prim_police, 2 122 поста, 2018 — 2025)."
    ),
    icon=":material/dashboard:",
)

# ============================================================
# Hero — описание проекта + bullet-список фич
# ============================================================
# Назначение блока: за 5 секунд донести до зрителя что вообще такое
# этот дашборд. Иконка info + структурированный markdown — нейтральный,
# не «продающий» тон, уместный для гос-аудитории.
st.info(
    """
    **Что это.** Программа интеллектуального анализа ДТП и поддержки принятия
    решений по безопасности дорожного движения в Приморском крае. Объединяет
    классический ML на табличных данных ГИБДД с обработкой естественного языка
    (NLP) на текстовых сводках МВД, учитывая региональную специфику —
    преобладание праворульных автомобилей (RHD), горный рельеф Владивостока,
    морской климат с туманами и гололёдом.

    **Что внутри.**
    - Кластеризация очагов аварийности (DBSCAN, 30 топ-очагов)
    - Прогноз количества ДТП на 12 месяцев вперёд (Prophet, MAPE 5.4 %)
    - Калиброванная классификация тяжести (CatBoost v2 + isotonic, ECE < 0.05)
    - Тематическое моделирование Telegram-постов (BERTopic, 7 тем)
    - 18 правил рекомендаций по безопасности с evidence base и counterfactual
    - Автообновление через Celery + Redis (еженедельный парсер, ежемесячный retrain)
    """,
    icon=":material/dashboard:",
)

# ============================================================
# Получение данных для KPI и таблиц
# ============================================================
try:
    stats = fetch_stats()
    hotspots = fetch_hotspots(limit=30)
except httpx.HTTPError as exc:
    st.error(f"FastAPI недоступно: {exc}")
    st.info("Запусти: `docker compose up -d api` или `uvicorn src.api.main:app`")
    st.stop()

# ============================================================
# KPI-strip — 4 карточки на одной линии
# ============================================================
total = stats["total"]
in_region = stats["in_region"]
sev = stats["by_severity"]
dead_acc = sev["dead"]
dead_persons = stats["total_dead"]
suffered = stats["total_suffered"]
hotspots_n = len(hotspots["items"])
clustered_pts = hotspots["stats"]["clustered_count"]

c1, c2, c3, c4 = st.columns(4)

with c1.container(border=True):
    st.metric(
        label=":material/local_taxi: Всего ДТП",
        value=fmt_int(total),
        delta=f"в сухопутной границе: {fmt_int(in_region)}",
        delta_color="off",
        help=(
            "Все ДТП в БД (2015 — апрель 2026). «В сухопутной границе» — "
            "фильтр PostGIS-полигоном Приморского края. "
            "Разница ≈ 1 393 точки — Null Island fallback и точки в "
            "акватории из источника."
        ),
    )

with c2.container(border=True):
    pct_dead_acc = dead_acc / total * 100
    st.metric(
        label=":material/heart_broken: Смертельных ДТП",
        value=fmt_int(dead_acc),
        delta=f"{pct_dead_acc:.1f} % от общего",
        delta_color="inverse",
        help=(
            "Категория `severity = 'dead'` из dtp-stat.ru — ДТП, в которых "
            "погиб как минимум один человек (включая водителей и пешеходов)."
        ),
    )

with c3.container(border=True):
    avg_per_dead_acc = dead_persons / dead_acc if dead_acc else 0
    st.metric(
        label=":material/person_off: Погибших всего",
        value=fmt_int(dead_persons),
        delta=f"{avg_per_dead_acc:.2f} на смертельное ДТП",
        delta_color="off",
        help=(
            f"Сумма поля `lost_amount` по всем ДТП. Раненых "
            f"(`suffer_amount`): {fmt_int(suffered)}. Среднее число "
            f"погибших на одно смертельное ДТП — индикатор "
            f"«массовости» аварии."
        ),
    )

with c4.container(border=True):
    st.metric(
        label=":material/location_on: Очагов DBSCAN",
        value=hotspots_n,
        delta=f"кластеризовано {fmt_int(clustered_pts)} ДТП",
        delta_color="off",
        help=(
            "Топ-30 географических кластеров (eps = 300 м, "
            "min_samples = 15) из 120 найденных на этапе кластеризации. Очаг — "
            "плотная группа ДТП на одном перекрёстке/участке улицы."
        ),
    )

st.divider()

# ============================================================
# Карточки-разделы — 6 кликабельных тайлов в 2 ряда по 3
# ============================================================
# Назначение: впервые открывший дашборд видит сразу куда идти. Каждая
# карточка — одна страница. Используем st.page_link (Streamlit ≥ 1.31)
# вместо ручных ссылок: Streamlit сам подсветит активную страницу,
# обработает routing в SPA-режиме.
st.markdown("### :material/explore: Разделы дашборда")
st.markdown(
    "<p class='dtp-section-lead'>Шесть страниц анализа. Каждая отвечает "
    "на один тип вопроса о безопасности дорожного движения в крае.</p>",
    unsafe_allow_html=True,
)

_SECTION_TILES = [
    (
        "views/hotspots_map.py",
        ":material/location_on:",
        "Карта очагов",
        "Тепловая карта 28 020 ДТП + 30 центроидов DBSCAN с интерактивными "
        "рекомендациями по любой точке края.",
    ),
    (
        "views/stats.py",
        ":material/insights:",
        "Статистика",
        "Глубокий разбор по времени, условиям, участникам и местности. "
        "RHD vs LHD: гипотеза МВД о 4.9× опровергнута.",
    ),
    (
        "views/forecast.py",
        ":material/trending_up:",
        "Прогноз на 2026",
        "Prophet с COVID-регрессором, MAPE 5.4 % на hold-out 2025. "
        "Помесячный прогноз на горизонт до 24 месяцев.",
    ),
    (
        "views/severity_predictor.py",
        ":material/sensors:",
        "Предсказатель тяжести",
        "CatBoost v2 + isotonic-калибровка. 4 калиброванные вероятности + "
        "counterfactual «что если бы…» по 7 мерам.",
    ),
    (
        "views/nlp_insights.py",
        ":material/psychology:",
        "NLP-инсайты",
        "BERTopic на 2 122 постах @prim_police: 7 содержательных тем, semantic "
        "search, дефект T4 фильтра.",
    ),
    (
        "views/dtp_types.py",
        ":material/category:",
        "Типы ДТП",
        "Распределение, динамика и severity по типам столкновения. "
        "Наезды на пешеходов и съезды — самые опасные.",
    ),
]

# Раскладываем 6 карточек в 2 ряда по 3 столбца. Для каждой:
# - st.container(border=True) — bordered card, общая стилизация в style.css
# - заголовок с иконкой
# - 1 описательная строка
# - st.page_link "Открыть →"
for row_start in (0, 3):
    cols = st.columns(3)
    for col, tile in zip(cols, _SECTION_TILES[row_start : row_start + 3]):
        page_path, icon, title, desc = tile
        with col.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.caption(desc)
            st.page_link(
                page_path,
                label="Открыть страницу",
                icon=":material/arrow_forward:",
            )

st.divider()

# ============================================================
# Тренд по годам + распределение тяжести (horizontal stacked bar)
# ============================================================
left, right = st.columns([3, 2])

with left:
    st.markdown("### :material/timeline: Тренд ДТП по годам")
    st.markdown(
        "<p class='dtp-section-lead'>Серый столбец — неполный 2026 год "
        "(данные за январь). Снижение 2020–2021 совпадает с пандемическими "
        "ограничениями и временным падением мобильности.</p>",
        unsafe_allow_html=True,
    )

    by_year = pd.DataFrame(sorted(stats["by_year"].items()), columns=["year", "n"])
    by_year["year"] = by_year["year"].astype(int)
    by_year["n"] = by_year["n"].astype(int)

    by_year["is_partial"] = by_year["year"] == 2026
    by_year["color"] = by_year["is_partial"].map({False: PRIMARY_COLOR, True: "#a0a4a8"})

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=by_year["year"],
            y=by_year["n"],
            marker_color=by_year["color"],
            text=by_year["n"].apply(fmt_int),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,} ДТП<extra></extra>",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1, title=None),
        yaxis=dict(title="Кол-во ДТП", gridcolor="#eaeaea"),
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### :material/donut_small: Распределение тяжести")
    st.markdown(
        "<p class='dtp-section-lead'>Доля каждой категории среди всех ДТП. "
        "Тяжёлые и смертельные суммарно ≈ 30 % — заметная часть, требующая "
        "превентивных мер.</p>",
        unsafe_allow_html=True,
    )

    # Horizontal stacked bar (одна полоска на 100% с 4 цветными
    # сегментами) вместо pie с дыркой — читается быстрее, числа
    # внутри сегментов не наезжают на text labels.
    sev_items = [
        (k, v, SEVERITY_LABELS[k], SEVERITY_COLORS[SEVERITY_LABELS[k]]) for k, v in sev.items()
    ]
    # Упорядочиваем как SEVERITY_ORDER_RU (light → severe → severe_multiple → dead).
    sev_items.sort(key=lambda t: SEVERITY_ORDER_RU.index(t[2]) if t[2] in SEVERITY_ORDER_RU else 99)
    fig_sev = go.Figure()
    for _key, count, label_ru, color in sev_items:
        pct = count / total * 100
        fig_sev.add_trace(
            go.Bar(
                x=[count],
                y=[""],
                name=label_ru,
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=1)),
                text=f"{label_ru}<br>{pct:.1f} %",
                textposition="inside",
                textfont=dict(color="white", size=11),
                hovertemplate=(
                    f"<b>{label_ru}</b><br>{count:,} ДТП " f"({pct:.1f} %)<extra></extra>"
                ),
            )
        )
    fig_sev.update_layout(
        barmode="stack",
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_sev, use_container_width=True)
    severe_total_pct = (sev["severe"] + sev["severe_multiple"] + sev["dead"]) / total * 100
    st.caption(
        f":material/warning: Тяжёлых + смертельных в сумме: "
        f"**{severe_total_pct:.1f} %** — на эти категории направлены "
        f"18 правил из СППР (см. страницу «Карта очагов» → режим "
        f"«Интерактив»)."
    )

st.divider()

# ============================================================
# Топ-5 очагов превью
# ============================================================
st.markdown("### :material/emergency_home: Топ-5 очагов аварийности")
st.markdown(
    "<p class='dtp-section-lead'>Ранжировано по количеству ДТП внутри очага "
    "(DBSCAN-кластера). Полная интерактивная карта всех 30 очагов с "
    "popup-статистикой и динамическими рекомендациями — на странице "
    "<b>Карта очагов</b>.</p>",
    unsafe_allow_html=True,
)

items = hotspots["items"][:5]
top5 = pd.DataFrame(
    [
        {
            "Ранг": it["rank"],
            "ДТП в очаге": fmt_int(it["n_points"]),
            "Доля смертельных": f"{it['pct_dead'] * 100:.2f}%",
            "Доля тяж+смерт.": f"{it['pct_severe_or_dead'] * 100:.1f}%",
            "Радиус, м": fmt_int(round(it["radius_meters"])),
            "Топ тип ДТП": (it["top_em_types"][0][0] if it["top_em_types"] else "—"),
            "Топ НП": it["top_np"][0][0] if it["top_np"] else "—",
        }
        for it in items
    ]
)
st.dataframe(top5, hide_index=True, use_container_width=True)

# ============================================================
# Footer — единый helper page_footer()
# ============================================================
page_footer()
