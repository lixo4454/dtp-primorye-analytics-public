"""Анализ типов ДТП: топ типов, динамика, severity-распределение.

Tabs вместо длинного скролла — пользователь переключается между четырьмя
видами одних и тех же данных (top, тренды, тяжесть, raw-таблица).

Источник — direct DB через SQLAlchemy. Кросс-таб ``em_type × year ×
severity`` агрегируется в SQL за ~50 мс (~800 строк) и кешируется на 1 час.
JSON-roundtrip через FastAPI здесь избыточен.

Фильтр ``is_in_region = TRUE`` исключает 1 393 точки вне сухопутного
полигона Приморья (Null Island fallback, дефект #7).
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.app.utils.db import get_em_type_year_severity
from src.app.utils.styling import insight_block, page_footer, page_header
from src.app.utils.visualizations import (
    PRIMARY_COLOR,
    SEVERITY_COLORS,
    SEVERITY_LABELS,
    SEVERITY_ORDER_RU,
    fmt_int,
)

page_header(
    title="Анализ типов ДТП",
    subtitle=(
        "Распределение и динамика по типу столкновения. Все цифры — по "
        "28 020 ДТП в сухопутной границе Приморья (фильтр `is_in_region`). "
        "Тяжесть смотрим первым делом — это главный практический срез."
    ),
    icon=":material/category:",
)

try:
    df = get_em_type_year_severity()
except SQLAlchemyError as exc:
    st.error(f"Ошибка БД: {exc}")
    st.stop()

# ============================================================
# KPI-strip → divider → фильтр диапазона лет (общий для всех вкладок)
# ============================================================
# KPI-плитки идут СНАЧАЛА (полная выборка), затем slider — пользователь
# видит общий контекст до того, как сужает диапазон.
years_avail = sorted(df["year"].unique())

# Initial state slider — полный диапазон БД, чтобы KPI отразили все ДТП.
if "dtp_types_year_range" not in st.session_state:
    st.session_state["dtp_types_year_range"] = (
        int(years_avail[0]),
        int(years_avail[-1]),
    )
_yr0, _yr1 = st.session_state["dtp_types_year_range"]
df_f_initial = df[(df["year"] >= _yr0) & (df["year"] <= _yr1)].copy()

m1, m2, m3, m4 = st.columns(4)
with m1.container(border=True):
    st.metric(
        "ДТП в выборке",
        fmt_int(df_f_initial["n"].sum()),
        help=(
            "Сумма по всем типам столкновений в выбранном диапазоне лет. "
            "Меняется при движении year-slider'а ниже."
        ),
    )
with m2.container(border=True):
    st.metric(
        "Типов ДТП",
        df_f_initial["em_type"].nunique(),
        help=(
            "Различные категории `em_type` из dtp-stat.ru "
            "(столкновение, наезд на пешехода, опрокидывание и т. д.)."
        ),
    )
with m3.container(border=True):
    st.metric(
        "Лет в диапазоне",
        _yr1 - _yr0 + 1,
        help="Количество календарных лет в выбранном диапазоне.",
    )
with m4.container(border=True):
    st.metric(
        "Среднее в год",
        fmt_int(df_f_initial["n"].sum() / max(1, _yr1 - _yr0 + 1)),
        help=(
            "Среднее годовое число ДТП за выбранный период. Полезно "
            "для сравнения с прогнозом Prophet (страница «Прогноз»)."
        ),
    )

st.divider()

with st.container(border=True):
    st.markdown("**:material/calendar_month: Диапазон лет**")
    st.caption(
        "Применяется ко всем 4 вкладкам ниже. 2026 содержит только январь — "
        "учитывай при чтении графика «Динамика по годам»."
    )
    year_range = st.slider(
        "Год ДТП",
        min_value=int(years_avail[0]),
        max_value=int(years_avail[-1]),
        value=st.session_state["dtp_types_year_range"],
        step=1,
        label_visibility="collapsed",
    )
st.session_state["dtp_types_year_range"] = year_range
df_f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()

if df_f.empty:
    st.warning("Нет данных в выбранном диапазоне.")
    st.stop()

st.divider()

totals = df_f.groupby("em_type")["n"].sum().sort_values(ascending=True).reset_index()

# ============================================================
# Tabs — 4 вида данных
# ============================================================
# Порядок tabs: сначала «Топ типов» (общий обзор), затем
# «Распределение тяжести» (главный практический срез), потом
# «Динамика по годам» (long-term тренд) и «Кросс-таб» (raw-данные
# для верификации и экспорта).
tab_top, tab_severity, tab_trend, tab_raw = st.tabs(
    [
        ":material/leaderboard: Топ типов",
        ":material/warning: Распределение тяжести",
        ":material/timeline: Динамика по годам",
        ":material/table_view: Кросс-таб (raw)",
    ]
)

# ---------- Tab 1: топ типов ДТП ----------
with tab_top:
    st.subheader("Распределение ДТП по типам столкновения")
    fig = go.Figure(
        go.Bar(
            x=totals["n"],
            y=totals["em_type"],
            orientation="h",
            marker_color=PRIMARY_COLOR,
            text=totals["n"].apply(fmt_int),
            textposition="outside",
            textfont_size=11,
            hovertemplate="<b>%{y}</b><br>%{x:,} ДТП<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, 38 * len(totals)),
        margin=dict(l=10, r=60, t=10, b=10),
        yaxis=dict(automargin=True),
        xaxis=dict(title="Кол-во ДТП", gridcolor="#eaeaea"),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    # Расширенный insight-блок: топ-3 типа с долями + что они значат
    # практически.
    _top1, _top2, _top3 = totals.iloc[-1], totals.iloc[-2], totals.iloc[-3]
    _total_n = totals["n"].sum()
    insight_block(
        f"**Топ-3 типа ДТП в выбранном периоде:** "
        f"**{_top1['em_type']}** — {fmt_int(_top1['n'])} ДТП "
        f"({_top1['n'] / _total_n * 100:.1f} % всех ДТП); "
        f"**{_top2['em_type']}** — {fmt_int(_top2['n'])} "
        f"({_top2['n'] / _total_n * 100:.1f} %); "
        f"**{_top3['em_type']}** — {fmt_int(_top3['n'])} "
        f"({_top3['n'] / _total_n * 100:.1f} %). "
        f"Эти три категории составляют "
        f"{(_top1['n'] + _top2['n'] + _top3['n']) / _total_n * 100:.1f} % "
        f"всех ДТП. Превентивные меры в первую очередь должны быть "
        f"направлены именно на них."
    )

# ---------- Tab 2: динамика топ-N типов ----------
with tab_trend:
    st.subheader("Динамика по годам")
    n_top = st.slider(
        "Сколько типов показать",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="trend_n_top",
    )
    top_n_types = totals.nlargest(n_top, "n")["em_type"].tolist()
    trend = (
        df_f[df_f["em_type"].isin(top_n_types)]
        .groupby(["year", "em_type"], as_index=False)["n"]
        .sum()
    )
    fig = px.line(
        trend,
        x="year",
        y="n",
        color="em_type",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(line=dict(width=2.4), marker=dict(size=8))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            title=None,
            font=dict(size=11),
        ),
        xaxis=dict(tickmode="linear", dtick=1, title=None),
        yaxis=dict(title="Кол-во ДТП", gridcolor="#eaeaea"),
        plot_bgcolor="white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    insight_block(
        "**Что показано:** число ДТП в год по топ-N типам столкновения. "
        "**Что видно:** общий спад 2020-2021 связан с COVID-локдауном "
        "(падение мобильности и трафика). 2026 содержит только январь — "
        "поэтому последняя точка визуально низкая, но это не «реальное» "
        "снижение, а артефакт неполного года. Стабильно растущих категорий "
        "за период 2015-2025 не наблюдается — общий тренд по большинству "
        "типов либо плоский, либо медленно убывающий."
    )

# ---------- Tab 3: severity-распределение ----------
with tab_severity:
    st.subheader("Распределение тяжести по типам ДТП")
    n_show = st.slider(
        "Сколько типов показать",
        min_value=3,
        max_value=12,
        value=7,
        step=1,
        key="severity_n_show",
    )
    top_show = totals.nlargest(n_show, "n")["em_type"].tolist()
    sev_long = (
        df_f[df_f["em_type"].isin(top_show)]
        .groupby(["em_type", "severity"], as_index=False)["n"]
        .sum()
    )
    total_per_type = sev_long.groupby("em_type")["n"].transform("sum")
    sev_long["pct"] = sev_long["n"] / total_per_type * 100
    sev_long["severity_ru"] = sev_long["severity"].map(SEVERITY_LABELS)

    gravity_rank = (
        sev_long[sev_long["severity"].isin(["dead", "severe", "severe_multiple"])]
        .groupby("em_type")["pct"]
        .sum()
        .sort_values(ascending=True)
    )

    fig = px.bar(
        sev_long,
        x="pct",
        y="em_type",
        color="severity_ru",
        orientation="h",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={
            "em_type": gravity_rank.index.tolist(),
            "severity_ru": SEVERITY_ORDER_RU,
        },
        custom_data=["n", "severity_ru"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{customdata[1]}: <b>%{x:.1f}%</b> (%{customdata[0]:,} ДТП)"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        height=max(380, 60 * n_show),
        barmode="stack",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            title=None,
        ),
        xaxis=dict(title="% от типа ДТП", ticksuffix="%", gridcolor="#eaeaea"),
        yaxis=dict(title=None, automargin=True),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    # Развёрнутый insight: считаем для топ-N severity-долю + конкретные числа.
    _worst = gravity_rank.idxmax()
    _worst_pct = gravity_rank.max()
    _best = gravity_rank.idxmin()
    _best_pct = gravity_rank.min()
    insight_block(
        f"**Что показано:** доля тяжёлых и смертельных ДТП внутри каждого "
        f"типа столкновения (нормировано — каждая полоска = 100 % ДТП "
        f"данного типа в выбранном периоде). Сортировка снизу вверх — "
        f"по возрастанию суммарной доли «опасных» категорий "
        f"(severe + severe_multiple + dead). "
        f"**Что видно:** наиболее опасный тип — **{_worst}** "
        f"({_worst_pct:.1f} % исходов с тяжёлым/смертельным). Наименее "
        f"опасный — **{_best}** ({_best_pct:.1f} %). Это объясняет "
        f"приоритеты СППР (см. страницу «Карта очагов» → правила R01 "
        f"speed-30 для пешеходных зон, R09 cable barrier для трасс)."
    )

# ---------- Tab 4: raw-таблица ----------
with tab_raw:
    st.subheader("Кросс-таб em_type × severity (абсолютные значения)")
    sev_full = df_f.groupby(["em_type", "severity"], as_index=False)["n"].sum().copy()
    sev_full["severity_ru"] = sev_full["severity"].map(SEVERITY_LABELS)
    pivot = sev_full.pivot_table(
        index="em_type",
        columns="severity_ru",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=SEVERITY_ORDER_RU, fill_value=0)
    pivot["Всего"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Всего", ascending=False)
    st.dataframe(
        pivot,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(format="%d") for col in [*SEVERITY_ORDER_RU, "Всего"]
        },
    )
    st.caption(f"Строк: {len(pivot)}, всего ДТП в выборке: **{fmt_int(pivot['Всего'].sum())}**.")

# Footer — единый для всех 7 страниц.
page_footer()
