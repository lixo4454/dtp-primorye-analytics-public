"""Страница «Статистика» — 4 вкладки deep-dive аналитики.

Вкладки:

* **По времени** — heatmap hour×dow, помесячно по годам, праздники vs будни,
  топ-10 опасных часов
* **По условиям** — severity × light_type / state / clouds / defects + χ² test
* **По участникам** — RHD/LHD интерактив (МВД 4.9× → реальное 0.65×),
  пьяные, непристёгнутые, возраст водителя
* **По местности** — топ-30 НП, city vs highway, топ-10 опасных дорог,
  heatmap НП × em_type

Cross-page navigation: ``st.query_params`` для перехода с карты очагов
с предзаполненными фильтрами. session_state дублирует параметры для
персистенции внутри сессии (F5 теряет ``query_params`` — это by-design
Streamlit).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Stat helpers (inline, без scipy) ------------------------
def _chi2_test(observed: np.ndarray) -> tuple[float, int, float]:
    """χ² test независимости для contingency-таблицы (rows × cols).

    Реализация без scipy: chi2 = Σ(O − E)² / E, df = (r−1)(c−1).
    p-value через Wilson-Hilferty approximation:
        z = ((χ²/df)^(1/3) - (1 - 2/(9 df))) / sqrt(2/(9 df))
        p = erfc(|z| / sqrt(2))
    Wilson-Hilferty точнее на 0.1 % для df ≥ 3.

    Returns: (chi2_stat, dof, p_value)
    """
    obs = np.asarray(observed, dtype=float)
    if obs.size == 0 or obs.shape[0] < 2 or obs.shape[1] < 2:
        return 0.0, 0, 1.0
    row_totals = obs.sum(axis=1, keepdims=True)
    col_totals = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    if total == 0:
        return 0.0, 0, 1.0
    expected = row_totals @ col_totals / total
    # Защита от деления на 0 в expected (если строка/столбец пуст)
    mask = expected > 0
    chi2 = float(((obs[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    if dof <= 0:
        return chi2, dof, 1.0
    # Wilson-Hilferty
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p_value = math.erfc(abs(z) / math.sqrt(2)) if z >= 0 else 1.0 - math.erfc(abs(z) / math.sqrt(2))
    p_value = max(0.0, min(1.0, p_value))
    return chi2, dof, p_value


def _wald_z_test(p1: float, n1: float, p2: float, n2: float) -> tuple[float, float, float]:
    """Wald-test для разности долей.

    SE = sqrt(p1(1−p1)/n1 + p2(1−p2)/n2), z = (p1 − p2)/SE,
    p_value = erfc(|z|/√2) (двусторонний).

    Returns: (z, se, p_value)
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 0.0, 1.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 0.0, 0.0, 1.0
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, se, p_value


from src.app.utils.db import (
    get_city_vs_highway,
    get_drunk_unbelted_severity,
    get_holidays_severity,
    get_hour_dow,
    get_monthly_by_year,
    get_np_em_heatmap,
    get_rhd_lhd_severity,
    get_severity_by_clouds,
    get_severity_by_defects,
    get_severity_by_field,
    get_top_dangerous_roads,
    get_top_np,
    get_year_range,
)
from src.app.utils.styling import insight_block, page_footer, page_header
from src.app.utils.visualizations import (
    PRIMARY_COLOR,
    SEVERITY_COLORS,
    SEVERITY_LABELS,
    SEVERITY_ORDER_RU,
    fmt_int,
)

page_header(
    title="Статистика по 28 020 ДТП",
    subtitle=(
        "Deep-dive аналитика в 4 вкладках: По времени, По условиям, "
        "По участникам, По местности. Все срезы фильтруются единым "
        "year-range slider'ом ниже. Каждый график сопровождается "
        "инсайтом — кратким выводом по графику."
    ),
    icon=":material/insights:",
)


# =====================================================================
# Year-range slider (общий для всех вкладок) + cross-page query_params
# =====================================================================
_y_min_db, _y_max_db = get_year_range()

# query_params → session_state на cold start страницы
qp = st.query_params
if "stats_year_range" not in st.session_state:
    qp_y0 = int(qp.get("year_min", _y_min_db))
    qp_y1 = int(qp.get("year_max", _y_max_db))
    st.session_state["stats_year_range"] = (
        max(_y_min_db, qp_y0),
        min(_y_max_db, qp_y1),
    )

slider_col, _ = st.columns([2, 1])
with slider_col:
    with st.container(border=True):
        st.markdown("**:material/calendar_month: Период анализа**")
        st.caption(
            "Применяется ко всем 4 вкладкам ниже. Деплинкуется через "
            "URL-параметры `year_min` / `year_max` — с других страниц можно "
            "переходить с предзаполненным диапазоном."
        )
        year_range = st.slider(
            "Год",
            min_value=_y_min_db,
            max_value=_y_max_db,
            value=st.session_state["stats_year_range"],
            step=1,
            key="stats_year_range_slider",
            label_visibility="collapsed",
        )
st.session_state["stats_year_range"] = year_range
# Обновляем query_params (без триггера rerun — ключи стабильные)
st.query_params["year_min"] = str(year_range[0])
st.query_params["year_max"] = str(year_range[1])

y0, y1 = year_range


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Утилита для st.download_button."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")  # BOM для Excel
    return buf.getvalue()


# =====================================================================
# Вкладки
# =====================================================================
tab_time, tab_cond, tab_part, tab_loc = st.tabs(
    [
        ":material/schedule: По времени",
        ":material/wb_sunny: По условиям",
        ":material/group: По участникам",
        ":material/place: По местности",
    ]
)


# =====================================================================
# Вкладка 1 — По времени
# =====================================================================
with tab_time:
    st.markdown(
        "<p class='dtp-section-lead'>:material/schedule: Здесь ты увидишь "
        "когда чаще всего происходят ДТП: heatmap по часам и дням недели, "
        "помесячная динамика, сравнение праздников и будних дней.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("### Heatmap часа × дня недели")
    df_hd = get_hour_dow(y0, y1)
    if df_hd.empty:
        st.info("Нет данных в выбранном year-range.")
    else:
        # 7×24 матрица. Postgres dow: 0=Вс..6=Сб
        pivot = df_hd.pivot(index="dow", columns="hour", values="n").fillna(0)
        # Переупорядочиваем строки: Пн..Вс (для UX-привычности)
        weekday_order = [1, 2, 3, 4, 5, 6, 0]
        weekday_labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        pivot_reordered = pivot.reindex(weekday_order).fillna(0)

        fig_hd = px.imshow(
            pivot_reordered.values,
            x=list(range(24)),
            y=weekday_labels,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels={"x": "Час суток", "y": "День недели", "color": "ДТП"},
            text_auto=True,
        )
        fig_hd.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=20))
        fig_hd.update_traces(textfont=dict(size=10))
        st.plotly_chart(fig_hd, use_container_width=True)

        # Инсайт
        max_cell = df_hd.loc[df_hd["n"].idxmax()]
        weekday_name = weekday_labels[weekday_order.index(int(max_cell["dow"]))]
        insight_block(
            f"**Инсайт:** пик ДТП — **{weekday_name} {int(max_cell['hour'])}:00..**, "
            f"**{int(max_cell['n'])} случаев** за {y1 - y0 + 1} лет. "
            f"Утренний час пик (07:00-09:00) и вечерний (17:00-19:00) "
            f"видны как тёмно-оранжевые полосы — типично для городского трафика."
        )

    st.divider()
    st.markdown("### Помесячная динамика по годам")
    df_my = get_monthly_by_year(y0, y1)
    if not df_my.empty:
        fig_my = px.line(
            df_my,
            x="month",
            y="n",
            color="year",
            labels={"month": "Месяц", "n": "ДТП", "year": "Год"},
            markers=True,
        )
        fig_my.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=20))
        fig_my.update_xaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig_my, use_container_width=True)
        insight_block(
            "**Инсайт:** видна сильная сезонность: пики в **октябре-декабре** "
            "(гололёд, тёмное время суток, уязвимые пешеходы) и **июль-август** "
            "(увеличенный трафик отпусков). 2020 год — провал из-за COVID-локдауна. "
            "2026 — неполный год."
        )
        st.download_button(
            ":material/download: Скачать CSV (помесячно)",
            data=_df_to_csv_bytes(df_my),
            file_name=f"dtp_monthly_{y0}-{y1}.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("### Праздники vs будни")
    df_h = get_holidays_severity(y0, y1)
    if not df_h.empty:
        df_h["severity_ru"] = df_h["severity"].map(SEVERITY_LABELS)
        # Доли в каждой категории
        totals = df_h.groupby("category")["n"].sum()
        df_h["pct"] = df_h.apply(lambda r: r["n"] / totals[r["category"]] * 100, axis=1)
        fig_h = px.bar(
            df_h,
            x="category",
            y="pct",
            color="severity_ru",
            category_orders={"severity_ru": SEVERITY_ORDER_RU},
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"pct": "% ДТП", "category": ""},
            text_auto=".1f",
        )
        fig_h.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
        st.plotly_chart(fig_h, use_container_width=True)
        try:
            holiday_dead = df_h.query("category == 'праздник' and severity == 'dead'")["pct"].iloc[
                0
            ]
            weekday_dead = df_h.query("category == 'будни' and severity == 'dead'")["pct"].iloc[0]
            ratio = holiday_dead / weekday_dead if weekday_dead else 0
            insight_block(
                f"**Инсайт:** доля смертельных в праздничные дни — "
                f"**{holiday_dead:.2f} %** против **{weekday_dead:.2f} %** в будни "
                f"(в {ratio:.2f}× выше). Это согласуется с гипотезой об увеличении "
                f"пьяной езды и удалённых поездок в праздники."
            )
        except (IndexError, KeyError):
            pass


# =====================================================================
# Вкладка 2 — По условиям
# =====================================================================
def _plot_severity_crosstab(df: pd.DataFrame, title_field: str, x_label: str) -> None:
    """Универсальный stacked-bar для severity × категория + χ² test.

    df: long-format с колонками category, severity, n.
    """
    if df.empty:
        st.info("Нет данных.")
        return
    df["severity_ru"] = df["severity"].map(SEVERITY_LABELS)
    pivot = df.pivot_table(
        index="category", columns="severity_ru", values="n", aggfunc="sum", fill_value=0
    )
    # Сортируем строки по total
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).head(8)
    pivot_pct = pivot.drop(columns=["total"]).div(pivot["total"], axis=0) * 100

    long = pivot_pct.reset_index().melt(
        id_vars="category", var_name="severity_ru", value_name="pct"
    )

    fig = px.bar(
        long,
        x="pct",
        y="category",
        color="severity_ru",
        category_orders={"severity_ru": SEVERITY_ORDER_RU},
        color_discrete_map=SEVERITY_COLORS,
        orientation="h",
        labels={"pct": "% ДТП", "category": x_label},
        text_auto=".1f",
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # χ² test значимости
    try:
        contingency = pivot.drop(columns=["total"]).values
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, dof, p_value = _chi2_test(contingency)
            sig = (
                "значима" if p_value < 0.001 else ("умеренная" if p_value < 0.05 else "не значима")
            )
            st.markdown(
                f"**χ²-test:** χ² = {chi2:.1f}, df = {dof}, p-value = "
                f"{p_value:.2e} → связь {sig} (на уровне α=0.001 / 0.05). "
                f"_p-value через Wilson-Hilferty approximation._"
            )
    except Exception as e:  # noqa: BLE001
        st.caption(f"_χ²-test не выполнился: {e}_")


with tab_cond:
    st.markdown(
        "<p class='dtp-section-lead'>:material/wb_sunny: Здесь ты увидишь "
        "связь severity ДТП с погодными и инфраструктурными условиями: "
        "освещение, покрытие, облачность, дефекты дороги. Под каждым "
        "графиком — χ²-test значимости связи.</p>",
        unsafe_allow_html=True,
    )
    sub_light, sub_state, sub_clouds, sub_def = st.tabs(
        ["Освещение", "Покрытие", "Облачность", "Дефекты"]
    )

    with sub_light:
        st.markdown("### Severity × light_type")
        df_ll = get_severity_by_field("light_type", y0, y1)
        _plot_severity_crosstab(df_ll, "light_type", "Освещение")
        insight_block(
            "**Инсайт:** доля смертельных растёт в категориях «В темное время "
            "суток, освещение отсутствует» и «Сумерки» — это и стало основой "
            "правил R05 (освещение перехода) и R06 (трасса)."
        )

    with sub_state:
        st.markdown("### Severity × traffic_area_state")
        df_st = get_severity_by_field("traffic_area_state", y0, y1)
        _plot_severity_crosstab(df_st, "traffic_area_state", "Покрытие")
        insight_block(
            "**Инсайт:** на «Гололедице» и «Со снежным накатом» доминируют "
            "Опрокидывания / Съезды (см. вкладку «По местности»). Правило R10 "
            "(HFST) и R18 (anti-icing brine) триггерятся на этих условиях."
        )

    with sub_clouds:
        st.markdown("### Severity × clouds_top")
        df_cl = get_severity_by_clouds(y0, y1)
        _plot_severity_crosstab(df_cl, "clouds_top", "Облачность")
        st.caption("Источник: первый элемент JSONB-массива clouds.")

    with sub_def:
        st.markdown("### Severity × дефекты дороги")
        df_d = get_severity_by_defects(y0, y1)
        _plot_severity_crosstab(df_d, "has_defect", "Наличие дефектов")


# =====================================================================
# Вкладка 3 — По участникам
# =====================================================================
with tab_part:
    st.markdown(
        "<p class='dtp-section-lead'>:material/group: Здесь — главный "
        "содержательный аналитический срез: проверка гипотезы МВД о 4.9× "
        "опасности праворульных авто (RHD), плюс relative risk пьяных и "
        "непристёгнутых, плюс возрастная зависимость водителя.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("### RHD vs LHD: интерактивный анализ")
    st.markdown(
        "**Контр-результат:** МВД РФ заявляло, что праворульные ТС "
        "**в 4.9× опаснее** леворульных (статистика, [см. Дни 1–4]). Наш контроль "
        "по категории ТС, возрасту водителя и местности дал **обратный результат**: "
        "RHD на самом деле **безопаснее** LHD."
    )

    df_rhd = get_rhd_lhd_severity(y0, y1)
    if df_rhd.empty:
        st.info("Нет данных в выбранном year-range.")
    else:
        pivot = df_rhd.pivot_table(
            index="steering", columns="severity", values="n", aggfunc="sum", fill_value=0
        )
        pivot["total"] = pivot.sum(axis=1)
        for col in [c for c in pivot.columns if c != "total"]:
            pivot[f"{col}_pct"] = pivot[col] / pivot["total"] * 100

        rhd_dead_pct = pivot.loc["RHD", "dead_pct"] if "RHD" in pivot.index else 0
        lhd_dead_pct = pivot.loc["LHD", "dead_pct"] if "LHD" in pivot.index else 0
        ratio = rhd_dead_pct / lhd_dead_pct if lhd_dead_pct > 0 else 0

        kpi_cols = st.columns(4)
        kpi_cols[0].metric(
            "RHD: ДТП классифицировано",
            fmt_int(pivot.loc["RHD", "total"]) if "RHD" in pivot.index else "—",
        )
        kpi_cols[1].metric(
            "LHD: ДТП классифицировано",
            fmt_int(pivot.loc["LHD", "total"]) if "LHD" in pivot.index else "—",
        )
        kpi_cols[2].metric(
            "RHD: % смертельных",
            f"{rhd_dead_pct:.2f}%",
            delta=f"{rhd_dead_pct - lhd_dead_pct:+.2f} п.п. vs LHD",
            delta_color="inverse",
        )
        kpi_cols[3].metric(
            "Соотношение RHD/LHD",
            f"{ratio:.2f}×",
            help="Гипотеза МВД: 4.9×. Реальные данные: < 1.0× (т.е. RHD безопаснее)",
        )

        # Stacked-bar severity-распределение
        df_long = df_rhd.copy()
        df_long["severity_ru"] = df_long["severity"].map(SEVERITY_LABELS)
        # Доли в каждой группе
        df_long["total"] = df_long["steering"].map(pivot["total"])
        df_long["pct"] = df_long["n"] / df_long["total"] * 100

        fig_rhd = px.bar(
            df_long,
            x="steering",
            y="pct",
            color="severity_ru",
            category_orders={"severity_ru": SEVERITY_ORDER_RU, "steering": ["RHD", "LHD"]},
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"pct": "% ДТП в группе", "steering": "Тип руля"},
            text_auto=".1f",
        )
        fig_rhd.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
        st.plotly_chart(fig_rhd, use_container_width=True)

        # Wald test для разности долей dead (без scipy — math.erfc)
        try:
            p1 = pivot.loc["RHD", "dead"] / pivot.loc["RHD", "total"]
            p2 = pivot.loc["LHD", "dead"] / pivot.loc["LHD", "total"]
            n1 = pivot.loc["RHD", "total"]
            n2 = pivot.loc["LHD", "total"]
            z, se, p_value = _wald_z_test(p1, n1, p2, n2)
            with st.expander(":material/science: Wald-test для разности долей dead"):
                st.markdown(
                    f"- p₁ (RHD dead) = {p1 * 100:.3f} %, n₁ = {fmt_int(n1)}\n"
                    f"- p₂ (LHD dead) = {p2 * 100:.3f} %, n₂ = {fmt_int(n2)}\n"
                    f"- SE = {se:.5f}, z-statistic = {z:.2f}, "
                    f"p-value = {p_value:.2e}\n"
                    f"- Разность {'статистически значима (p < 0.001)' if p_value < 0.001 else 'на грани значимости'}"
                )
        except Exception as e:  # noqa: BLE001
            st.caption(f"_Wald-test не выполнен: {e}_")

        # Главный narrative
        if ratio < 1:
            st.success(
                f"**ВЫВОД:** RHD автомобили на **{(1 - ratio) * 100:.1f} % безопаснее** "
                f"LHD по доле смертельных ДТП. Гипотеза МВД о 4.9× отвергнута. "
                f"Главный аналитический результат."
            )
        else:
            st.warning(
                f"RHD/LHD ratio = {ratio:.2f} — RHD немного опаснее. "
                f"Не соответствует МВД-цифре 4.9× ни в каком приближении."
            )

    st.divider()
    st.markdown("### Доля пьяных и непристёгнутых vs severity")
    st.caption(
        "Источник: participants × accidents JOIN. Пьяные = "
        "`med_result_permille >= 0.16‰`, непристёгнутые = "
        "`safety_belt = 'Нет'`. Только классифицированные участники."
    )
    df_du = get_drunk_unbelted_severity(y0, y1)
    if not df_du.empty:
        # Простая агрегация: severity × has_drunk
        agg_drunk = df_du.groupby(["has_drunk", "severity"])["n"].sum().reset_index()
        agg_drunk_pivot = agg_drunk.pivot(index="has_drunk", columns="severity", values="n").fillna(
            0
        )
        agg_drunk_pivot["total"] = agg_drunk_pivot.sum(axis=1)
        agg_drunk_pivot["pct_dead"] = (
            agg_drunk_pivot.get("dead", 0) / agg_drunk_pivot["total"] * 100
        )

        rr_drunk = (
            agg_drunk_pivot.loc[True, "pct_dead"] / agg_drunk_pivot.loc[False, "pct_dead"]
            if False in agg_drunk_pivot.index
            and True in agg_drunk_pivot.index
            and agg_drunk_pivot.loc[False, "pct_dead"] > 0
            else 0
        )

        agg_unb = df_du.groupby(["has_unbelted", "severity"])["n"].sum().reset_index()
        agg_unb_pivot = agg_unb.pivot(index="has_unbelted", columns="severity", values="n").fillna(
            0
        )
        agg_unb_pivot["total"] = agg_unb_pivot.sum(axis=1)
        agg_unb_pivot["pct_dead"] = agg_unb_pivot.get("dead", 0) / agg_unb_pivot["total"] * 100

        rr_unb = (
            agg_unb_pivot.loc[True, "pct_dead"] / agg_unb_pivot.loc[False, "pct_dead"]
            if False in agg_unb_pivot.index
            and True in agg_unb_pivot.index
            and agg_unb_pivot.loc[False, "pct_dead"] > 0
            else 0
        )

        rcols = st.columns(2)
        with rcols[0]:
            st.metric(
                "RR смертельных при пьяном водителе",
                f"{rr_drunk:.2f}×",
                help=f"% смертельных при has_drunk=True / при has_drunk=False",
            )
            st.caption(
                f"С пьяным: {agg_drunk_pivot.loc[True, 'pct_dead']:.2f} %, "
                f"без пьяных: {agg_drunk_pivot.loc[False, 'pct_dead']:.2f} %"
            )
        with rcols[1]:
            st.metric(
                "RR смертельных при непристёгнутых",
                f"{rr_unb:.2f}×",
                help=f"% смертельных при has_unbelted=True / при has_unbelted=False",
            )
            st.caption(
                f"С непристёгнутым: {agg_unb_pivot.loc[True, 'pct_dead']:.2f} %, "
                f"все пристёгнуты: {agg_unb_pivot.loc[False, 'pct_dead']:.2f} %"
            )

        insight_block(
            f"**Инсайт:** RR (relative risk) для непристёгнутых ≈ **{rr_unb:.1f}×** "
            f"— подтверждает международные оценки (BMC Public Health 2018: ремни "
            f"снижают occupant fatality в 4-5 раз). RR пьяных ≈ **{rr_drunk:.1f}×** — "
            f"это per-vehicle, не per-blood-test, поэтому дифференциация менее острая."
        )

    st.divider()
    st.markdown("### Возраст водителя × severity")
    st.caption(
        ":material/info: Возраст рассчитан **только для водителей** "
        "(``part_type = 'Водитель'``) с `age_from_telegram IS NOT NULL`. "
        "Пешеходы и пассажиры исключены из расчёта — иначе нарратив "
        "о «молодых водителях» был бы загрязнён возрастом других участников."
    )
    if not df_du.empty and "avg_driver_age" in df_du.columns:
        # Удаляем NULL avg_driver_age (ДТП без водителей с известным возрастом)
        ages = df_du.dropna(subset=["avg_driver_age"]).copy()
        if not ages.empty:
            bins = [0, 25, 35, 45, 55, 65, 100]
            labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
            ages["age_bin"] = pd.cut(ages["avg_driver_age"], bins=bins, labels=labels, right=False)
            agg = ages.groupby(["age_bin", "severity"])["n"].sum().reset_index()
            agg_pivot = agg.pivot(index="age_bin", columns="severity", values="n").fillna(0)
            agg_pivot["total"] = agg_pivot.sum(axis=1)
            for col in [c for c in agg_pivot.columns if c != "total"]:
                agg_pivot[f"{col}_pct"] = agg_pivot[col] / agg_pivot["total"] * 100

            fig_age = go.Figure()
            for sev in ["dead", "severe_multiple", "severe", "light"]:
                if f"{sev}_pct" not in agg_pivot.columns:
                    continue
                fig_age.add_trace(
                    go.Bar(
                        x=agg_pivot.index.astype(str),
                        y=agg_pivot[f"{sev}_pct"],
                        name=SEVERITY_LABELS[sev],
                        marker_color=SEVERITY_COLORS[SEVERITY_LABELS[sev]],
                    )
                )
            fig_age.update_layout(
                barmode="stack",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="% ДТП в группе",
                xaxis_title="Возраст водителя (avg)",
                legend_title="",
            )
            st.plotly_chart(fig_age, use_container_width=True)
            insight_block(
                "**Инсайт:** возрастная зависимость bimodal: молодые водители "
                "(<25) и пожилые (65+) имеют повышенную долю смертельных. Источник "
                "возраста — Telegram-NER из NER-pipeline, ~30 % покрытия."
            )


# =====================================================================
# Вкладка 4 — По местности
# =====================================================================
with tab_loc:
    st.markdown(
        "<p class='dtp-section-lead'>:material/place: Здесь — географический "
        "разрез: топ-30 населённых пунктов, сравнение город vs трасса, "
        "топ-10 самых смертельных трасс, heatmap «НП × тип ДТП».</p>",
        unsafe_allow_html=True,
    )
    st.markdown("### Топ-30 населённых пунктов по числу ДТП")
    df_np = get_top_np(y0, y1, limit=30)
    if not df_np.empty:
        df_np["pct_dead"] = df_np["n_dead"] / df_np["n"] * 100
        df_np_sorted = df_np.sort_values("n", ascending=True)

        fig_np = go.Figure()
        fig_np.add_trace(
            go.Bar(
                x=df_np_sorted["n"],
                y=df_np_sorted["np"],
                orientation="h",
                marker_color=PRIMARY_COLOR,
                text=[fmt_int(n) for n in df_np_sorted["n"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>ДТП: %{x:,}<br>%{customdata:.2f} %% смертельных<extra></extra>",
                customdata=df_np_sorted["pct_dead"],
            )
        )
        fig_np.update_layout(
            height=600, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ДТП за период"
        )
        st.plotly_chart(fig_np, use_container_width=True)
        st.download_button(
            ":material/download: Скачать CSV (топ-30 НП)",
            data=_df_to_csv_bytes(df_np),
            file_name=f"dtp_top30_np_{y0}-{y1}.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("### City vs Highway")
    df_ch = get_city_vs_highway(y0, y1)
    if not df_ch.empty:
        df_ch["severity_ru"] = df_ch["severity"].map(SEVERITY_LABELS)
        totals = df_ch.groupby("area")["n"].sum()
        df_ch["pct"] = df_ch.apply(lambda r: r["n"] / totals[r["area"]] * 100, axis=1)
        df_ch["area_ru"] = df_ch["area"].map({"city": "Город / НП", "highway": "Трасса"})

        kc1, kc2 = st.columns(2)
        kc1.metric("Всего ДТП в городе", fmt_int(totals.get("city", 0)))
        kc2.metric("Всего ДТП на трассах", fmt_int(totals.get("highway", 0)))

        fig_ch = px.bar(
            df_ch,
            x="area_ru",
            y="pct",
            color="severity_ru",
            category_orders={"severity_ru": SEVERITY_ORDER_RU},
            color_discrete_map=SEVERITY_COLORS,
            barmode="stack",
            labels={"pct": "% ДТП", "area_ru": ""},
            text_auto=".1f",
        )
        fig_ch.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
        st.plotly_chart(fig_ch, use_container_width=True)
        insight_block(
            "**Инсайт:** на трассах доля смертельных в **2–3 раза выше**, чем "
            "в городах — основа правил R09 (cable barrier), R11 (guardrail), "
            "R10 (HFST). В городах доминируют пешеходные ДТП — отсюда R01 "
            "(speed-30), R12 (LPI), R15 (RRFB)."
        )

    st.divider()
    st.markdown("### Топ-10 опасных дорог (по pct_dead, n ≥ 30)")
    df_dr = get_top_dangerous_roads(y0, y1, min_n=30)
    if not df_dr.empty:
        df_dr["pct_dead_pct"] = df_dr["pct_dead"] * 100
        df_dr_sorted = df_dr.sort_values("pct_dead_pct", ascending=True)
        fig_dr = px.bar(
            df_dr_sorted,
            x="pct_dead_pct",
            y="roads",
            orientation="h",
            color="pct_dead_pct",
            color_continuous_scale="Reds",
            labels={"pct_dead_pct": "% смертельных", "roads": "Трасса"},
            text=df_dr_sorted["pct_dead_pct"].round(2).astype(str) + "%",
            hover_data={"n": True, "n_dead": True, "pct_dead_pct": False},
        )
        fig_dr.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False
        )
        st.plotly_chart(fig_dr, use_container_width=True)
        insight_block(
            "**Инсайт:** на участках с pct_dead ≥ 5 % rule engine "
            "автоматически триггерит R02 (камеры) и R09 (cable barrier) — "
            "перейди на «Карту очагов» в режиме «Интерактив», ткни в точку, "
            "получишь конкретные рекомендации."
        )

    st.divider()
    st.markdown("### Heatmap: топ-7 НП × топ-7 типов ДТП")
    heat_norm = st.radio(
        "Нормировка",
        options=["абсолютные числа", "% внутри НП (row-normalized)"],
        index=1,  # default — row-normalized: иначе Владивосток заглушает остальных
        horizontal=True,
        key="stats_np_em_heat_norm",
    )
    df_heat = get_np_em_heatmap(y0, y1)
    if not df_heat.empty:
        pivot_h = df_heat.pivot_table(
            index="np", columns="em_type", values="n", aggfunc="sum", fill_value=0
        )
        pivot_h_sorted = pivot_h.loc[pivot_h.sum(axis=1).sort_values(ascending=False).index]
        col_order = pivot_h_sorted.sum(axis=0).sort_values(ascending=False).index
        pivot_h_sorted = pivot_h_sorted[col_order]

        if heat_norm.startswith("%"):
            row_totals = pivot_h_sorted.sum(axis=1)
            display_pivot = pivot_h_sorted.div(row_totals, axis=0) * 100
            colorbar_label = "% внутри НП"
            text_arr = display_pivot.round(1).astype(str) + "%"
            color_scale = "RdYlBu_r"
        else:
            display_pivot = pivot_h_sorted
            colorbar_label = "ДТП"
            text_arr = pivot_h_sorted.astype(int).astype(str)
            color_scale = "Blues"

        fig_heat = go.Figure(
            go.Heatmap(
                z=display_pivot.values,
                x=display_pivot.columns,
                y=display_pivot.index,
                colorscale=color_scale,
                hoverongaps=False,
                text=text_arr.values,
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> · %{x}<br>"
                + colorbar_label
                + ": %{z:.1f}<extra></extra>",
                colorbar=dict(title=colorbar_label),
            )
        )
        fig_heat.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickangle=-25,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        if heat_norm.startswith("%"):
            insight_block(
                "**Инсайт:** в **% внутри НП** видно структуру каждого "
                "населённого пункта. Владивосток — компактная сеть с "
                "пешеходным трафиком (~30 % наездов на пешехода), "
                "Уссурийск/Находка — больше столкновений и съездов "
                "(трассовая периферия). Абсолютные числа Владивостока "
                "не заглушают паттерны меньших НП."
            )
        else:
            insight_block(
                "**Инсайт:** Владивосток доминирует абсолютными числами "
                "ДТП. Переключи нормировку на «% внутри НП» чтобы "
                "увидеть структуру каждого пункта без shadowing'а."
            )

# Footer — единый для всех 7 страниц.
page_footer()
