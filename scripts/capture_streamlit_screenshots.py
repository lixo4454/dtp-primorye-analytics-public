"""Снимки 3 страниц Streamlit-дашборда через Playwright (headless Chromium).

Использование:
    # 1) Streamlit должен быть запущен
    streamlit run src/app/main.py

    # 2) В отдельном терминале (PYTHONIOENCODING=utf-8 на Windows)
    python scripts/capture_streamlit_screenshots.py

Зачем: артефакт презентации/собеседования + воспроизводимый smoke
визуальный тест дашборда.

Установка:
    pip install playwright
    python -m playwright install chromium
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "streamlit_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8501"

# Streamlit контент скроллится внутри <section class="main">, тело
# страницы фиксировано высотой viewport. Чтобы full_page-скриншот
# захватил весь контент, делаем высокий viewport — содержимое
# умещается без внутреннего скролла.
VIEWPORT_BASE = {"width": 1600, "height": 1100}  # для home (~900px)


# Каждая страница: URL + last-element-text (маркер, что весь скрипт
# отработал) + filename + viewport_height. Default-страница (home)
# доступна на "/", не на "/home" — там Streamlit показывает
# toast "Page not found".
PAGES = [
    {
        "url": "/",
        "wait_for_text": "Топ-5 очагов аварийности",
        "filename": "home_screenshot.png",
        "viewport_height": 1100,
        "extra_wait_ms": 2000,
    },
    {
        "url": "/hotspots",
        "wait_for_text": "Все 30 очагов",
        "filename": "hotspots_map_screenshot.png",
        "viewport_height": 1700,
        "extra_wait_ms": 8000,
    },
    {
        "url": "/types",
        "wait_for_text": "Анализ типов ДТП",
        "filename": "dtp_types_screenshot.png",
        "viewport_height": 1300,
        "extra_wait_ms": 3000,
    },
    {
        "url": "/forecast",
        "wait_for_text": "Прогноз ДТП",
        "filename": "forecast_screenshot.png",
        "viewport_height": 1500,
        "extra_wait_ms": 3500,
    },
    {
        "url": "/severity",
        "wait_for_text": "Предсказатель тяжести ДТП",
        "filename": "severity_predictor_screenshot.png",
        "viewport_height": 1500,
        "extra_wait_ms": 2500,
    },
    {
        "url": "/nlp",
        "wait_for_text": "NLP-инсайты",
        "filename": "nlp_insights_screenshot.png",
        "viewport_height": 1800,
        "extra_wait_ms": 4000,
    },
    # --- скриншоты СППР + Stats ---
    {
        "url": "/stats",
        "wait_for_text": "Heatmap часа × дня недели",
        "filename": "stats_time_screenshot.png",
        "viewport_height": 1800,
        "extra_wait_ms": 4000,
    },
    {
        "url": "/stats",
        "wait_for_text": "Severity × light_type",
        "filename": "stats_conditions_screenshot.png",
        "viewport_height": 1500,
        "extra_wait_ms": 4500,
        "click_tab_text": "По условиям",
    },
    {
        "url": "/stats",
        "wait_for_text": "RHD vs LHD: интерактивный анализ",
        "filename": "stats_participants_screenshot.png",
        "viewport_height": 1900,
        "extra_wait_ms": 4500,
        "click_tab_text": "По участникам",
    },
    {
        "url": "/stats",
        "wait_for_text": "Топ-30 населённых пунктов",
        "filename": "stats_location_screenshot.png",
        "viewport_height": 2000,
        "extra_wait_ms": 4500,
        "click_tab_text": "По местности",
    },
    {
        "url": "/severity",
        "wait_for_text": "Симуляция мер (counterfactual)",
        "filename": "severity_counterfactual_screenshot.png",
        "viewport_height": 1700,
        "extra_wait_ms": 3000,
    },
    {
        "url": "/hotspots",
        "wait_for_text": "Рекомендации",  # side-panel заголовок
        "filename": "hotspots_recommendations_screenshot.png",
        "viewport_height": 1300,
        "extra_wait_ms": 6000,
        "click_toggle_label": "Интерактив",
    },
    # --- автообновление + snap-to-road ---
    {
        "url": "/",
        "wait_for_text": "Данные обновляются автоматически",
        "filename": "home_autoupdate_banner_screenshot.png",
        "viewport_height": 600,
        "extra_wait_ms": 2000,
    },
    {
        "url": "/",
        "wait_for_text": "Prophet (прогноз)",
        "filename": "home_model_versions_footer_screenshot.png",
        "viewport_height": 1300,
        "extra_wait_ms": 2500,
    },
    {
        "url": "/hotspots",
        "wait_for_text": "Snap-to-road",
        "filename": "hotspots_snap_toggle_screenshot.png",
        "viewport_height": 1100,
        "extra_wait_ms": 7000,
        "click_toggle_label": "Snap-to-road",
    },
]


def wait_for_page_ready(page: Page, marker_text: str, extra_wait_ms: int) -> None:
    """Ждём пока Streamlit отработает скрипт + последний маркер появится."""
    try:
        page.wait_for_load_state("networkidle", timeout=60000)
    except Exception:
        # Streamlit держит WebSocket — networkidle может не сработать
        # на тяжёлых страницах. Падаем на маркер ниже.
        pass

    # Ждём пока пропадёт running-индикатор
    try:
        page.wait_for_function(
            """
            () => {
                const w = document.querySelector('[data-testid="stStatusWidget"]');
                return !w || !w.textContent.toLowerCase().includes('running');
            }
            """,
            timeout=20000,
        )
    except Exception:
        pass

    # Маркер — гарантия что page.run() дошёл до конца. Для drill-down с
    # 28k FastMarkerCluster — может занять до 30 сек на первой загрузке.
    page.get_by_text(marker_text).first.wait_for(state="attached", timeout=60000)

    # Скроллим до низа и обратно — заставляет Plotly/lazy-элементы
    # отрисоваться (Streamlit рендерит виджеты по мере вьюпорта)
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(800)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(extra_wait_ms)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for spec in PAGES:
            # Каждой странице — свой viewport (контент Streamlit
            # скроллится внутри section.main, body фиксирован)
            context = browser.new_context(
                viewport={"width": 1600, "height": spec["viewport_height"]},
                device_scale_factor=2,
                locale="ru-RU",
            )
            page = context.new_page()

            url = f"{BASE_URL}{spec['url']}"
            print(f"-> {url} (viewport h={spec['viewport_height']})", flush=True)
            page.goto(url, timeout=30000)

            wait_for_page_ready(
                page,
                marker_text=spec["wait_for_text"],
                extra_wait_ms=spec["extra_wait_ms"],
            )

            # Опционально кликнуть toggle перед скриншотом (для drill-down)
            if "click_toggle_label" in spec:
                page.locator(f'label:has-text("{spec["click_toggle_label"]}")').first.click()
                wait_for_page_ready(
                    page,
                    marker_text=spec["wait_for_text"],
                    extra_wait_ms=spec["extra_wait_ms"],
                )

            # Опционально кликнуть на st.tabs-вкладку перед скриншотом
            if "click_tab_text" in spec:
                # Streamlit рисует tabs как button[role="tab"] с текстом вкладки
                tab_btn = page.get_by_role("tab", name=spec["click_tab_text"]).first
                tab_btn.click()
                page.wait_for_timeout(800)
                wait_for_page_ready(
                    page,
                    marker_text=spec["wait_for_text"],
                    extra_wait_ms=spec["extra_wait_ms"],
                )

            # Опционально нажать button (для severity-result)
            if "click_button_label" in spec:
                page.get_by_role("button", name=spec["click_button_label"]).first.click()
                wait_for_page_ready(
                    page,
                    marker_text=spec["wait_for_text"],
                    extra_wait_ms=spec["extra_wait_ms"],
                )

            out_path = OUTPUT_DIR / spec["filename"]
            page.screenshot(path=str(out_path), full_page=False)
            size_kb = out_path.stat().st_size // 1024
            rel = out_path.relative_to(PROJECT_ROOT)
            print(f"  saved -> {rel} ({size_kb} KB)", flush=True)

            context.close()

        browser.close()
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"Done in {time.time() - t0:.1f} s")
    sys.exit(rc)
