"""Helper'ы единого визуального языка дашборда.

Три точки расширения:

- ``inject_css()`` — однократно подгружает ``.streamlit/style.css`` через
  ``st.markdown(unsafe_allow_html=True)``. Вызывается в ``main.py``
  до первого ``pg.run()``; повторные вызовы игнорируются (флаг в
  ``st.session_state``), чтобы не дублировать ``<style>`` теги при rerun'ах.
- ``page_header(title, subtitle, icon)`` — единый header страницы:
  цветной accent-bar слева, крупный title (h1, 1.85 rem / 700), под ним
  серый подзаголовок-контекст (≤ 70ch). Заменяет россыпь
  ``st.title`` + ``st.caption`` по 7 страницам.
- ``page_footer()`` — единый footer: timestamp обновления данных,
  активные версии моделей (Prophet / CatBoost / BERTopic), ссылки на
  репозиторий и API-документацию. Использует ``fetch_active_model_versions``
  и ``fetch_health`` из ``api_client``; падения API подавляются —
  footer всегда рисуется, чтобы не разваливать layout.

Принципы:

- Никаких side-effect'ов на import'е модуля — Streamlit'у не нравится,
  когда CSS подгружается до ``st.set_page_config``.
- Inline-HTML строится через f-string, экранирование подзаголовков —
  ``html.escape`` (зрители могут передать произвольную строку).
- Иконки — Material Symbols через ``:material/<name>:`` (синтаксис
  Streamlit ≥ 1.36).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ``.streamlit/style.css`` лежит рядом с ``config.toml`` в корне проекта,
# а этот файл — в ``src/app/utils/``. Корень = parents[3].
_STYLE_CSS = Path(__file__).resolve().parents[3] / ".streamlit" / "style.css"

_CSS_INJECTED_FLAG = "_dtp_css_injected"

PROJECT_VERSION = "v1.0.0"
PROJECT_REPO_URL = "https://github.com/lixo4454/dtp-primorye-analytics"
PROJECT_API_URL = "/api/docs"  # подменяется на полный URL в footer'е


def inject_css() -> None:
    """Вкатывает глобальный CSS из ``.streamlit/style.css`` один раз на сессию.

    Streamlit при каждом ``st.rerun()`` повторно исполняет файл скрипта;
    если бы мы каждый раз клали ``<style>``-тег в DOM, у пользователя
    к концу сессии было бы 50+ копий стилей. Защищаемся флагом в
    ``st.session_state`` — стиль кладём один раз на жизнь сессии.
    """
    if st.session_state.get(_CSS_INJECTED_FLAG):
        return
    if not _STYLE_CSS.exists():
        # Не разваливаем дашборд если CSS-файл удалили — просто игнорируем.
        st.session_state[_CSS_INJECTED_FLAG] = True
        return
    css = _STYLE_CSS.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_CSS_INJECTED_FLAG] = True


def page_header(
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
) -> None:
    """Единый header страницы.

    Args:
        title: главный заголовок (h1, 1.85rem / 700). Кириллица OK.
        subtitle: серый подзаголовок-контекст. Может быть ``None`` —
            тогда не рисуется.
        icon: параметр сохранён для обратной совместимости с вызовами
            из view-страниц, но в HTML-режиме игнорируется. Streamlit
            парсит синтаксис ``:material/name:`` ТОЛЬКО внутри
            markdown-контейнеров (st.title, st.markdown без HTML),
            а в нашем raw-HTML заголовке — нет. Визуальный indicator
            страницы — синий accent-bar слева, его достаточно. Если
            нужна иконка-эмодзи в title — передавай её как часть
            ``title`` напрямую.
    """
    del icon  # игнорируем — см. docstring
    safe_title = html.escape(title)
    sub_html = ""
    if subtitle:
        sub_html = f'<p class="dtp-page-header__subtitle">{html.escape(subtitle)}</p>'
    st.markdown(
        f"""
        <div class="dtp-page-header">
          <div class="dtp-page-header__bar"></div>
          <div class="dtp-page-header__body">
            <h1 class="dtp-page-header__title">{safe_title}</h1>
            {sub_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_footer() -> None:
    """Единый footer страницы — три колонки через ``st.columns``.

    Раньше footer строился через единый ``st.markdown(unsafe_allow_html)``
    с CSS flex-вёрсткой на классах ``.dtp-footer / .dtp-footer__col``.
    Streamlit при рендере оборачивает наш HTML в свои блок-элементы
    (``element-container``, ``stMarkdown``), которые ломают flex parent
    и в итоге три колонки складываются вертикально стопкой. Решение —
    использовать нативный ``st.columns(3)``: Streamlit гарантирует
    горизонтальную раскладку через свой grid.

    Содержимое колонок:

    * **Данные** — описание источников + число ДТП в БД (live через
      ``fetch_health``) + timestamp рендера.
    * **Модели** — активные версии (Prophet / CatBoost v2 / v1 /
      BERTopic) с датой обучения через ``fetch_active_model_versions``.
    * **Ресурсы** — ссылки на репозиторий, Swagger UI и версия v1.0.0.

    Все сетевые вызовы обёрнуты в ``try/except`` и подавляются —
    footer ВСЕГДА рисуется, даже если API лежит.
    """
    from src.app.utils.api_client import (
        fetch_active_model_versions,
        fetch_health,
    )

    accidents_count: int | None = None
    try:
        health = fetch_health()
        if health and health.get("status") == "ok":
            accidents_count = health.get("accidents_count")
    except Exception:
        accidents_count = None
    accidents_str = f"{accidents_count:,}".replace(",", " ") if accidents_count else "—"

    pretty_names = {
        "prophet_dtp": "Prophet",
        "catboost_severity_v2": "CatBoost v2",
        "catboost_severity_v1_calibrated": "CatBoost v1",
        "bertopic_dtp": "BERTopic",
    }
    try:
        versions = fetch_active_model_versions() or []
    except Exception:
        versions = []

    now = datetime.now(timezone.utc).astimezone()
    rendered_at = now.strftime("%Y-%m-%d %H:%M %Z")

    # Тонкая разделительная линия — визуально отделяет footer от контента.
    st.markdown(
        "<hr style='margin-top: 2.5rem; margin-bottom: 1rem; "
        "border:0; border-top:1px solid #e1e5eb;'>",
        unsafe_allow_html=True,
    )

    col_data, col_models, col_links = st.columns([1.4, 1.2, 1.0])

    with col_data:
        st.markdown(
            "<div style='font-weight:600;color:#1a1a1a;font-size:0.9rem;"
            "margin-bottom:0.4rem;'>:material/database: Данные</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Источник: [dtp-stat.ru](https://dtp-stat.ru/) "
            f"(открытые данные МВД РФ) + Telegram @prim_police  \n"
            f"В БД: **{accidents_str}** ДТП  \n"
            f"Отрендерено: {rendered_at}"
        )

    with col_models:
        st.markdown(
            "<div style='font-weight:600;color:#1a1a1a;font-size:0.9rem;"
            "margin-bottom:0.4rem;'>:material/precision_manufacturing: Активные модели</div>",
            unsafe_allow_html=True,
        )
        if versions:
            lines = []
            for v in versions[:4]:
                name = pretty_names.get(v["model_name"], v["model_name"])
                ts = (v.get("trained_at") or "")[:10] or "без даты"
                lines.append(f"**{name}** · _{ts}_")
            st.caption("  \n".join(lines))
        else:
            st.caption("_версии моделей недоступны_")

    with col_links:
        st.markdown(
            "<div style='font-weight:600;color:#1a1a1a;font-size:0.9rem;"
            "margin-bottom:0.4rem;'>:material/link: Ресурсы</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"[GitHub-репозиторий]({PROJECT_REPO_URL})  \n"
            f"[FastAPI Swagger UI](http://localhost:8000/docs)  \n"
            f"{PROJECT_VERSION} · Приморский край · 2026"
        )


def insight_block(text: str, *, icon: str = ":material/lightbulb:") -> None:
    """Единый стиль блока «Инсайт».

    Раньше на разных страницах инсайты оформлялись по-разному:
    ``st.markdown("**Инсайт:**...")``, ``st.success(...)`` без иконки,
    голым ``st.caption``. Теперь — один helper, везде одинаково:
    нежно-зелёный фон, иконка-лампочка, чёткое визуальное выделение.

    Реализовано через ``st.success`` (в ``style.css`` мы переопределили
    его фон/border на нашу палитру). Альтернатива — кастомный div через
    markdown — но тогда теряем семантический ARIA для скринридеров.
    """
    st.success(text, icon=icon)
