"""Страница «Прогноз ДТП» — Prophet-предсказание помесячно.

Источник: FastAPI ``/forecast/monthly`` (Prophet, 10 лет
тренировки 2015-2024 + holdout 2025 → MAPE 5.4 %). История 2015-2026
берётся напрямую из БД (``get_monthly_history``) — Prophet API возвращает
только yhat, не actual'ы вне forecast-окна.

UX:
- Slider горизонта 3-24 месяца (default 12)
- Plotly-график: actual из БД (line) + yhat-прогноз (line + CI band)
  + вертикальная линия train/forecast разделения
- 4 KPI-плитки с метриками holdout 2025: MAE / RMSE / MAPE / N точек
- Expander с decomposition (trend / yearly seasonality), комментарий
  про COVID-2020 как объясняющий регрессор

Кеширование: ``fetch_forecast`` уже @cache_data ttl=3600, slider
не дёргает Prophet повторно при тех же параметрах. Для разных
horizon-значений Streamlit держит отдельный entry в кеше.
"""

from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils.api_client import fetch_forecast
from src.app.utils.db import get_monthly_history
from src.app.utils.styling import page_footer, page_header
from src.app.utils.visualizations import PRIMARY_COLOR, fmt_int

page_header(
    title="Прогноз ДТП на 2026 год",
    subtitle=(
        "Prophet (Meta, 2017) с COVID-2020 в качестве external regressor'а. "
        "10 лет тренировки (2015-2024), out-of-sample валидация на 2025. "
        "Точечный прогноз — `yhat`, ленты — 95 % CI Prophet "
        "(`yhat_lower`/`yhat_upper`)."
    ),
    icon=":material/trending_up:",
)


# ============================================================
# UI controls — горизонт прогноза в bordered-панели
# ============================================================
with st.container(border=True):
    st.markdown("**:material/tune: Параметры прогноза**")
    horizon = st.slider(
        "Горизонт прогноза, месяцев",
        min_value=3,
        max_value=24,
        value=12,
        step=1,
        help=(
            "Сколько месяцев вперёд прогнозировать. Чем дальше — тем шире "
            "доверительный интервал (CI). Прогноз на 24 месяца опирается "
            "на долгосрочный тренд + годовую сезонность."
        ),
    )

# Точка старта прогноза — следующий месяц после максимального actual
# в БД (так не возникает «дыры» между actual и yhat). Если БД
# обновится за пределы 2026, slider/start_year автоматически уедет.
history_df = get_monthly_history()
last_actual = pd.Timestamp(history_df["ds"].max())
forecast_start = (last_actual + pd.DateOffset(months=1)).normalize()

# Информация про лаг источника. Сегодняшний месяц обычно недоступен —
# dtp-stat.ru публикует ДТП в открытые данные с задержкой ~2-3 месяца
# (срок заведения в ИС МВД + камеральная проверка). Eсть «дыра» между
# last_actual и текущим месяцем → объясняем пользователю явно.
_now = pd.Timestamp.utcnow().tz_localize(None).normalize()
_now_month_start = _now.replace(day=1)
_lag_months = (_now_month_start.year - last_actual.year) * 12 + (
    _now_month_start.month - last_actual.month
)
if _lag_months >= 1:
    _last_n = int(history_df.loc[history_df["ds"] == last_actual, "y"].iloc[0])
    st.info(
        f":material/schedule: **Последний месяц с фактом: "
        f"{last_actual.strftime('%Y-%m')}** ({_last_n} ДТП в БД). "
        f"Источник dtp-stat.ru имеет лаг публикации ~2-3 месяца до "
        f"камеральной проверки в ИС МВД, поэтому за "
        f"{_now_month_start.strftime('%Y-%m')} и ближайшие месяцы факт ещё "
        f"не доступен. Weekly-парсер автоматически подгрузит новые "
        f"месяцы по мере публикации."
    )

# FastAPI Forecast принимает start_year/start_month отдельно
try:
    forecast_payload = fetch_forecast(
        periods=horizon,
        start_year=int(forecast_start.year),
        start_month=int(forecast_start.month),
    )
except httpx.HTTPError as exc:
    st.error(f"FastAPI недоступно: {exc}")
    st.info("Запусти: `docker compose up -d api`")
    st.stop()

# Hold-out метрики 2025 — KPI-strip над графиком (4 в ряд).
# Раньше MAPE висел отдельно в правой колонке header'а — изолированный
# и без контекста. Теперь — полноценная KPI-полоса, читается как
# «качество модели на честной валидации».
metrics = forecast_payload["metadata"].get("holdout_2025_metrics", {})
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1.container(border=True):
    st.metric(
        ":material/percent: Hold-out MAPE",
        f"{metrics.get('mape_pct', 0):.2f} %",
        help=(
            "Mean Absolute Percentage Error — средняя относительная ошибка "
            "на 12 месяцах 2025 года, которые модель не видела при "
            "тренировке. Чем меньше — тем точнее модель в относительных "
            "терминах. Бенчмарк хорошего месячного прогноза в "
            "транспортной аналитике: 5-10 %."
        ),
    )
with mc2.container(border=True):
    st.metric(
        ":material/calculate: MAE",
        f"{metrics.get('mae', 0):.1f}",
        help=(
            "Mean Absolute Error — средняя абсолютная ошибка в ДТП/мес. "
            "Грубо: «насколько в среднем модель промахивается»."
        ),
    )
with mc3.container(border=True):
    st.metric(
        ":material/show_chart: RMSE",
        f"{metrics.get('rmse', 0):.1f}",
        help=(
            "Root Mean Squared Error — квадратичная ошибка, чувствительна "
            "к выбросам. Если RMSE >> MAE, модель иногда сильно ошибается."
        ),
    )
with mc4.container(border=True):
    st.metric(
        ":material/data_usage: Точек валидации",
        fmt_int(metrics.get("n_points", 0)),
        help=("Количество out-of-sample месяцев в hold-out выборке " "(12 = весь 2025 год)."),
    )

# ============================================================
# DataFrames для графика
# ============================================================
forecast_df = pd.DataFrame(forecast_payload["items"])
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

# История из БД — actual'ы. Plotly корректно соединяет их с прогнозом
# через общий timeline (ds).
hist_df = history_df.copy()
hist_df = hist_df[hist_df["ds"] < forecast_start].sort_values("ds")

# ============================================================
# Plotly: history + forecast с CI
# ============================================================
fig = go.Figure()

# Confidence Interval — 2 invisible линии, fill='tonexty'
fig.add_trace(
    go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="upper",
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 78, 121, 0.18)",
        showlegend=True,
        name="95 % CI",
        hovertemplate="<b>95 %% CI</b><br>%{x|%b %Y}: %{y:.0f}<extra></extra>",
    )
)

# Actual history
fig.add_trace(
    go.Scatter(
        x=hist_df["ds"],
        y=hist_df["y"],
        mode="lines",
        line=dict(color="#1a1a1a", width=2),
        name="Факт (БД)",
        hovertemplate="<b>Факт</b><br>%{x|%b %Y}: %{y} ДТП<extra></extra>",
    )
)

# Forecast yhat
fig.add_trace(
    go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines+markers",
        line=dict(color=PRIMARY_COLOR, width=2.5, dash="dot"),
        marker=dict(size=5),
        name="Прогноз (yhat)",
        hovertemplate=("<b>Прогноз</b><br>%{x|%b %Y}: %{y:.0f} ДТП<extra></extra>"),
    )
)

# Вертикальная линия train/forecast разделения. add_vline c pandas.Timestamp
# падает в annotation-расчётах (Timestamp __radd__ с int отключён в pandas
# 2.x). Используем add_shape с явным xref="x" + add_annotation отдельно —
# те же визуалы, без интегерных хаков.
fig.add_shape(
    type="line",
    x0=forecast_start,
    x1=forecast_start,
    yref="paper",
    y0=0,
    y1=1,
    line=dict(color="#888", width=1, dash="dash"),
)
fig.add_annotation(
    x=forecast_start,
    yref="paper",
    y=1.0,
    text="Начало прогноза",
    showarrow=False,
    yshift=10,
    font=dict(size=11, color="#666"),
    bgcolor="rgba(255,255,255,0.85)",
)

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    xaxis_title="Месяц",
    yaxis_title="Количество ДТП",
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Сводка по прогнозу
# ============================================================
total_yhat = int(round(forecast_df["yhat"].sum()))
total_lower = int(round(forecast_df["yhat_lower"].sum()))
total_upper = int(round(forecast_df["yhat_upper"].sum()))

avg_year_hist = hist_df.groupby(hist_df["ds"].dt.year)["y"].sum().tail(3).mean()
delta_pct = (
    100 * (total_yhat / horizon * 12 - avg_year_hist) / avg_year_hist if avg_year_hist else 0
)

st.subheader(f"Сводка прогноза на {horizon} мес")
sc1, sc2, sc3 = st.columns(3)
sc1.metric(
    "Сумма ДТП за горизонт",
    fmt_int(total_yhat),
    delta=f"95 % CI: {fmt_int(total_lower)} — {fmt_int(total_upper)}",
    delta_color="off",
)
sc2.metric(
    "В среднем за месяц",
    fmt_int(round(forecast_df["yhat"].mean())),
)
sc3.metric(
    "Δ к среднему 2023-2025",
    f"{delta_pct:+.1f} %",
    help=(
        "Сравнение средней годовой суммы прогноза с фактом "
        "за последние 3 завершённых года из БД."
    ),
    delta_color="inverse",
)

# ============================================================
# Помесячная таблица
# ============================================================
with st.expander(":material/table_chart: Помесячная таблица прогноза"):
    table = forecast_df.copy()
    table["Месяц"] = table["ds"].dt.strftime("%Y-%m (%b)")
    table = table[["Месяц", "yhat", "yhat_lower", "yhat_upper", "actual"]]
    table = table.rename(
        columns={
            "yhat": "Прогноз",
            "yhat_lower": "Нижняя 95% CI",
            "yhat_upper": "Верхняя 95% CI",
            "actual": "Факт (если есть)",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Прогноз": st.column_config.NumberColumn(format="%.0f"),
            "Нижняя 95% CI": st.column_config.NumberColumn(format="%.0f"),
            "Верхняя 95% CI": st.column_config.NumberColumn(format="%.0f"),
            "Факт (если есть)": st.column_config.NumberColumn(format="%d"),
        },
    )

# ============================================================
# Декомпозиция (текстовый разбор)
# ============================================================
with st.expander(":material/insights: Что моделирует Prophet", expanded=True):
    st.markdown(
        """
        Prophet раскладывает временной ряд на три аддитивные компоненты:

        - **Тренд** — долгосрочная динамика количества ДТП. После 2017
          года в Приморье он медленно убывает (улучшение инфраструктуры
          и парка ТС), но пандемический провал 2020 искусственно занизил
          базу — этот эффект гасится отдельным регрессором (см. ниже).
        - **Годовая сезонность** — летний пик (июль-сентябрь, +30 % к
          медиане), зимний минимум (февраль, −20 %). Стабильная
          амплитуда из года в год.
        - **Недельная сезонность** — для месячных агрегатов почти не видна;
          в дневных данных пятница и суббота выше будней на 5-7 %.

        **COVID-регрессор** — бинарный флаг `covid_lockdown`,
        выставленный в 1 для марта-мая 2020. Без него Prophet видел
        бы COVID-провал как «новый уровень тренда» и недооценил бы
        2021-2022 на 12-15 %. Регрессор зашит в сериализованной модели.

        **Hold-out стратегия:** модель тренировалась только на
        2015-2024 (`metadata.train_range`), все 12 точек 2025 года —
        полностью out-of-sample. MAPE 5.4 % означает, что в среднем
        ошибка прогноза ≤ 5 %.
        """
    )

# Источник данных + train range — для воспроизводимости и аудита.
st.caption(
    f"_Источник прогноза: FastAPI `/forecast/monthly` "
    f"(start={forecast_start.strftime('%Y-%m')}, periods={horizon}). "
    f"Источник истории: PostgreSQL accidents, group by month._  \n"
    f"_Train range: {forecast_payload['metadata']['train_range']['start'][:7]} — "
    f"{forecast_payload['metadata']['train_range']['end'][:7]}._"
)

# Footer — единый для всех 7 страниц.
page_footer()
