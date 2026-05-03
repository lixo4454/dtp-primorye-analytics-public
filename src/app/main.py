"""Streamlit-дашборд — точка входа.

Запуск из корня проекта::

    streamlit run src/app/main.py

Окружение:
- FastAPI: ``http://localhost:8000`` (переопределяется ``DTP_API_URL``)
- Postgres: из ``.env`` через ``src.database.session.build_database_url``

Multipage navigation сделан через ``st.navigation`` + ``st.Page``
(modern API, Streamlit ≥1.36) вместо legacy-папки ``pages/`` —
явный контроль порядка, иконок и URL-путей.

Финальный визуальный язык дашборда — палитра «Государственный»
(`#1f4e79` primary), типографика Inter + JetBrains Mono. Глобальный CSS
лежит в ``.streamlit/style.css`` и подгружается через
``src.app.utils.styling.inject_css()`` ровно один раз на сессию.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path: streamlit run выполняет скрипт напрямую, не как `python -m`,
# поэтому корень проекта надо добавить вручную для import'ов вида
# ``from src.database.session import...``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.app.utils.api_client import fetch_health  # noqa: E402
from src.app.utils.styling import (  # noqa: E402
    PROJECT_REPO_URL,
    PROJECT_VERSION,
    inject_css,
)

st.set_page_config(
    page_title="Анализ ДТП Приморского края",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Аналитический дашборд по 29 413 ДТП Приморского края "
            "(2015–2026). Источник: dtp-stat.ru (открытые данные МВД РФ) + Telegram @prim_police.\n\n"
            "Стек: PostGIS + FastAPI + Streamlit + Folium + Plotly."
        ),
    },
)

# Глобальный CSS (палитра + типографика + spacing) — один раз на сессию.
# Должен идти СТРОГО после set_page_config и до st.navigation, иначе
# Streamlit отрисует первый rerun «голым» и перекрасит на следующий.
inject_css()

# --- Multipage navigation ---
# default=True страница в Streamlit 1.41 ОБЯЗАНА жить на корневом "/",
# без явного url_path. Если задать url_path="home", легаси-маршрутизатор
# при прямом заходе на /hotspots (или ctrl+shift+R) сначала пытается
# найти pages/hotspots.py, не находит → показывает toast «Page not found».
# С пустым url_path у default-page коллизий нет.
home = st.Page(
    "views/home.py",
    title="Главная",
    icon=":material/home:",
    default=True,
)
hotspots = st.Page(
    "views/hotspots_map.py",
    title="Карта очагов",
    icon=":material/location_on:",
    url_path="hotspots",
)
dtp_types = st.Page(
    "views/dtp_types.py",
    title="Типы ДТП",
    icon=":material/category:",
    url_path="types",
)
forecast = st.Page(
    "views/forecast.py",
    title="Прогноз",
    icon=":material/trending_up:",
    url_path="forecast",
)
severity_predictor = st.Page(
    "views/severity_predictor.py",
    title="Предсказатель тяжести",
    icon=":material/sensors:",
    url_path="severity",
)
nlp_insights = st.Page(
    "views/nlp_insights.py",
    title="NLP-инсайты",
    icon=":material/psychology:",
    url_path="nlp",
)
stats = st.Page(
    "views/stats.py",
    title="Статистика",
    icon=":material/insights:",
    url_path="stats",
)

# st.navigation поддерживает sections (dict[label -> list]). Группируем
# страницы по тематикам — Streamlit отрисует sidebar-разделитель и заголовок.
pg = st.navigation(
    {
        "Аналитика": [home, hotspots, dtp_types, stats],
        "ML-модели": [forecast, severity_predictor],
        "NLP": [nlp_insights],
    }
)

# --- Sidebar: бренд + live API-индикатор + полезные ссылки ---
# Брендовый sidebar: вверху — фирменный бренд-блок с версией
# v1.0.0 (badge), под ним группированный live-индикатор API, ниже —
# быстрые ссылки на ключевые ресурсы. Версия моделей показывается
# в footer'е каждой страницы (см. styling.page_footer), здесь не
# дублируется — sidebar остаётся компактным.
with st.sidebar:
    st.markdown(
        f"""
        <div class="dtp-brand">
          <div class="dtp-brand__title">Анализ ДТП Приморья</div>
          <div class="dtp-brand__subtitle">Открытые данные · 2015 — 2026</div>
          <span class="dtp-brand__version">{PROJECT_VERSION}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    health = fetch_health()
    if health and health.get("status") == "ok":
        models_loaded = sum(1 for v in health.get("models_loaded", {}).values() if v)
        models_total = len(health.get("models_loaded", {})) or models_loaded
        accidents_str = f"{health['accidents_count']:,}".replace(",", " ")
        st.success(f":material/check_circle: API live · {accidents_str} ДТП")
        st.caption(f"Загружено моделей: **{models_loaded}/{models_total}**")
    else:
        st.error(":material/error: API недоступно")
        st.caption("Запусти: `docker compose up -d api`")

    st.divider()
    st.caption("**Быстрая навигация**")
    st.caption(
        f"[:material/code: Репозиторий GitHub]({PROJECT_REPO_URL})  \n"
        ":material/api: [FastAPI Swagger UI](http://localhost:8000/docs)  \n"
        ":material/menu_book: [Методология рекомендаций]"
        "(https://github.com/lixo4454/dtp-primorye-analytics/"
        "blob/main/docs/recommendations_methodology.md)"
    )

pg.run()
