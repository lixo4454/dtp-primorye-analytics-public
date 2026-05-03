"""Карта очагов аварийности — 3 слоя визуализации одних и тех же 28 020 ДТП.

UX-решения (после ревью):

- **HeatMap (default ON)** — тепловая карта плотности всех 28 020 ДТП
  в регионе. Показывает РЕАЛЬНОЕ распределение по улицам: видно что
  «горячие точки» внутри кластера №6 (центр Владивостока) находятся
  на Светланской и Алеутской, а не на Сопке Тигровой куда попадает
  математический центроид.
- **Очаги-центроиды (default ON)** — 30 кликабельных маркеров с popup
  (статистика по каждому DBSCAN-кластеру: топ-3 типа ДТП, % смерт.,
  топ НП). Нужны для агрегатной статистики, НЕ для геолокации.
- **Drill-down (опциональный, через st.toggle)** — все 28 020 ДТП как
  ``FastMarkerCluster``: при zoom in ≥ 16 распадается на индивидуальные
  маркеры. Color-coded по severity. Грузится только по запросу
  (~700 КБ HTML), default OFF.

Все три слоя переключаются через `LayerControl` (top-right).

- ``scrollWheelZoom=True`` (default Leaflet) — зум колесом мыши, как
  привычно по Google Maps / Mapbox. Это удобнее чем кнопки +/−, особенно
  для трекпадов с pinch.

Источник aggregates — FastAPI ``/clusters/hotspots``; источник 28 020
координат — direct DB через ``get_accident_coords`` (``ST_X``/``ST_Y``
из PostGIS-колонки ``point``).

Для встраивания — ``streamlit.components.v1.html``, не ``st_folium``
(он авторесайзит iframe под содержимое).
"""

from __future__ import annotations

import math

import folium
import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import FastMarkerCluster, HeatMap
from streamlit_folium import st_folium

from src.app.utils.api_client import (
    fetch_hotspots,
    fetch_recommendations_hotspot,
    fetch_recommendations_point,
)
from src.app.utils.db import get_accident_coords, get_heatmap_grid, get_year_range
from src.app.utils.styling import page_footer, page_header
from src.app.utils.visualizations import color_by_pct_dead, fmt_int

# CSS + JS для скрытия бренд-флага Leaflet attribution (1.9.x вшивает
# украинский флаг в attribution-control как inline-SVG c классом
# `leaflet-attribution-flag`). CSS — основной способ; JS — страховка
# на случай если стиль не успел применить к моменту первой отрисовки
# или кастомные templates folium'а перебивают селектор.
_FLAG_HIDE_HTML = """
<style>
.leaflet-attribution-flag,
.leaflet-control-attribution svg,
.leaflet-control-attribution svg.leaflet-attribution-flag,
.leaflet-control-attribution > svg,
.leaflet-control-attribution a > svg,
.leaflet-control-attribution a[href*="leafletjs"] svg,
a.leaflet-attribution-flag {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
}
</style>
<script>
(function() {
  function nukeFlag() {
    var sels = [
      '.leaflet-attribution-flag',
      '.leaflet-control-attribution svg',
      '.leaflet-control-attribution a[href*="leafletjs"] svg'
    ];
    sels.forEach(function(s) {
      document.querySelectorAll(s).forEach(function(el) { el.remove(); });
    });
  }
  // Несколько попыток — Leaflet может смонтировать attribution
  // асинхронно (после первой отрисовки тайлов).
  setTimeout(nukeFlag, 50);
  setTimeout(nukeFlag, 300);
  setTimeout(nukeFlag, 1200);
})();
</script>
"""

page_header(
    title="Карта очагов аварийности",
    subtitle=(
        "Тепловая карта 28 020 ДТП Приморского края + 30 центроидов "
        "DBSCAN-кластеров со статистикой. Слои переключаются через "
        "LayerControl справа сверху, drill-down и интерактивный "
        "режим — тогглами ниже."
    ),
    icon=":material/location_on:",
)

# Длинный hint про точность координат и snap-to-road вынесен в свёрнутый
# expander. На странице с картой 540+ px высоты длинный caption под title
# теряется при scroll'е; expander виден всегда, а при необходимости
# раскрывается одним кликом.
with st.expander(
    ":material/info: О точности координат и snap-to-road",
    expanded=False,
):
    st.markdown(
        """
        **Точность координат — ±20 м по источнику dtp-stat.ru.** Часть
        точек в исходных GeoJSON-выгрузках снапается на полигон ближайшего
        здания вместо точки на дороге (особенность парсера ГИБДД при
        автоматическом геокодинге адреса). На тепловую карту это почти
        не влияет — blur 18 px компенсирует, плотность улицы видна
        правильно. На сильном zoom in (≥ 16) индивидуальные маркеры
        могут быть смещены на 10-30 м от реального места ДТП.

        **Snap-to-road через OSM road graph.** 192 934 рёбер
        дорожной сети Приморья были загружены из OpenStreetMap и
        собраны в граф с пространственным индексом. Для каждого ДТП
        ищем ближайшее ребро в радиусе 50 м — 87.5 % точек успешно
        снапаются (median смещение 2.1 м). Оставшиеся 12.5 % —
        парковки, площади, дворы — остаются на raw-координате и
        отображаются крестом × вместо круглого маркера.

        **Тоггл «Snap-to-road» (default ON)** переключает источник
        координат для drill-down: snap-to-road путь даёт визуально
        более чистую картину ДТП «на улицах», а off-режим показывает
        исходные координаты ГИБДД с возможным смещением.
        """
    )

try:
    payload = fetch_hotspots(limit=30)
except httpx.HTTPError as exc:
    st.error(f"FastAPI недоступно: {exc}")
    st.stop()

items = payload["items"]
stats = payload["stats"]


# ============================================================
# UI controls — режим интерактива + drill-down + snap-to-road + year-range
# ============================================================
# Все 4 контрола карты собраны в одну bordered-панель «Параметры карты»,
# чтобы визуально читались как единый функциональный блок, а не как
# случайная россыпь тогглов сверху страницы.
_y_min_full, _y_max_full = get_year_range()
with st.container(border=True):
    st.markdown("**:material/tune: Параметры карты**")
    ctrl_cols = st.columns([1.3, 1.3, 1.3])
    with ctrl_cols[0]:
        interactive_mode = st.toggle(
            ":material/touch_app: Интерактив: клик на карте → рекомендации",
            value=False,
            help=(
                "В этом режиме:\n"
                "- клик на маркер очага → топ-3 рекомендации для DBSCAN-кластера\n"
                "- клик в произвольную точку → анализ радиуса (slider 30-1000 м) "
                "с динамическими рекомендациями\n\n"
                "Минусы: drill-down 28 020 ДТП ВЫКЛЮЧЕН (st_folium тяжелее "
                "components.html в 2-3×). Для drill-down переключи toggle обратно."
            ),
        )
    with ctrl_cols[1]:
        show_drill = st.toggle(
            ":material/zoom_in: Drill-down: показать все 28 020 ДТП",
            value=True,
            help=(
                "Загрузить отдельные ДТП. На низком зуме группируются Leaflet-"
                "кластеризатором с цифрами, при приближении распадаются на "
                "индивидуальные маркеры (color-coded по тяжести) с popup'ом "
                "(дата / тип / погибшие / адрес) при клике. Работает в обоих режимах."
            ),
        )
    with ctrl_cols[2]:
        snap_to_road = st.toggle(
            ":material/route: Snap-to-road (точки на дорогах)",
            value=True,
            help=(
                "Перенести точки ДТП на ближайшее ребро OSM road graph (cutoff 50 м). "
                "87.5% точек снапаются (median 2.1 м), 12.5% off-road (парковки, "
                "площади) остаются на raw-координате."
            ),
        )
    # Year-range slider — фильтр для drill-down 28 020 ДТП. На центроиды
    # (DBSCAN-агрегат) НЕ влияет — они зафиксированы на полной выборке.
    year_range = st.slider(
        ":material/calendar_month: Период (год ДТП)",
        min_value=_y_min_full,
        max_value=_y_max_full,
        value=(_y_min_full, _y_max_full),
        step=1,
        help=(
            f"Полный диапазон БД: {_y_min_full}–{_y_max_full}. "
            "Фильтр применяется к drill-down точкам, "
            "но НЕ к DBSCAN-кластерам (они зафиксированы на полной выборке)."
        ),
    )
year_min, year_max = year_range


# ============================================================
# Popup-рендер для очага
# ============================================================
def _format_popup(it: dict) -> str:
    top_types = "<br>".join(
        f"&nbsp;{i + 1}. {t[0]} — <b>{fmt_int(t[1])}</b>"
        for i, t in enumerate(it["top_em_types"][:3])
    )
    top_np = "<br>".join(
        f"&nbsp;{i + 1}. {n[0]} — <b>{fmt_int(n[1])}</b>" for i, n in enumerate(it["top_np"][:2])
    )
    return f"""
    <div style="font-family: sans-serif; font-size: 13px; line-height: 1.5;">
      <div style="font-size: 14px; font-weight: 700; margin-bottom: 6px;
                  color: #1f4e79;">
        Очаг #{it["rank"]}
      </div>
      <table style="border-collapse: collapse;">
        <tr><td>ДТП в очаге:</td><td><b>{fmt_int(it["n_points"])}</b></td></tr>
        <tr><td>Смертельных:</td><td><b>{it["pct_dead"] * 100:.2f}%</b></td></tr>
        <tr><td>Тяжёлых+смерт.:</td><td><b>{it["pct_severe_or_dead"] * 100:.1f}%</b></td></tr>
        <tr><td>Радиус:</td><td>{fmt_int(round(it["radius_meters"]))} м</td></tr>
      </table>
      <div style="margin-top: 8px;"><i>Топ типов ДТП:</i><br>{top_types}</div>
      <div style="margin-top: 8px;"><i>Топ НП:</i><br>{top_np}</div>
    </div>
    """


# ============================================================
# JS-callback'и для drill-down: с popup'ом (default) и без (interactive)
# ============================================================
# В default-режиме клик по маркеру открывает popup с деталями ДТП.
# В interactive-режиме popup'ы убраны, чтобы клик по drill-маркеру
# проходил «насквозь» на карту и срабатывал триггер «точка в радиусе»
# (выбор центра рекомендаций), без перехвата popup'ом.
_DRILL_CALLBACK_WITH_POPUP_JS = """
function (row) {
    var sev2color = {light: '#3498db', severe: '#f39c12',
                     severe_multiple: '#e67e22', dead: '#c0392b'};
    var sev2label = {light: 'Лёгкое', severe: 'Тяжёлое',
                     severe_multiple: 'Тяжёлое мн.',
                     dead: 'Смертельное'};
    var c = sev2color[row[2]] || '#999';
    var datetime = row[4] || '—';
    var em_type  = row[5] || '—';
    var lost     = row[6] || 0;
    var suffer   = row[7] || 0;
    var address  = row[8] || '—';
    var sev_label = sev2label[row[2]] || row[2];
    var off_road  = (row[3] === 'unchanged' || row[3] === 'failed');

    var popup_html =
        '<div style="font-family:sans-serif;font-size:12.5px;line-height:1.5;min-width:240px;">'
      + '<div style="font-weight:700;font-size:13.5px;color:' + c + ';'
      + 'border-bottom:1px solid #ddd;padding-bottom:4px;margin-bottom:6px;">'
      + '● ' + sev_label
      + (off_road ? ' <span style="font-size:11px;color:#666;">(off-road)</span>' : '')
      + '</div>'
      + '<div><b>Когда:</b> ' + datetime + '</div>'
      + '<div><b>Тип:</b> ' + em_type + '</div>'
      + '<div><b>Погибло:</b> ' + lost + ' &nbsp; <b>Ранено:</b> ' + suffer + '</div>'
      + '<div style="margin-top:4px;color:#555;"><b>Где:</b> ' + address + '</div>'
      + '</div>';

    var marker;
    if (off_road) {
        marker = L.marker(new L.LatLng(row[0], row[1]), {
            icon: L.divIcon({
                className: 'off-road-cross',
                iconSize: [10, 10],
                html: '<div style="width:10px;height:10px;color:' + c +
                      ';font-weight:900;font-size:14px;line-height:10px;' +
                      'text-align:center;">×</div>'
            })
        });
    } else {
        marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {radius: 4, color: c, weight: 1, fillOpacity: 0.75});
    }
    marker.bindPopup(popup_html, {maxWidth: 320});
    return marker;
}
"""

# Вариант для interactive: маркер тот же (цвет / off-road крест), но
# БЕЗ bindPopup и БЕЗ event-handler'ов — клик по маркеру всплывает в
# map.click → st_folium возвращает last_clicked → используется как
# точка радиуса для рекомендаций.
_DRILL_CALLBACK_NO_POPUP_JS = """
function (row) {
    var sev2color = {light: '#3498db', severe: '#f39c12',
                     severe_multiple: '#e67e22', dead: '#c0392b'};
    var c = sev2color[row[2]] || '#999';
    var off_road = (row[3] === 'unchanged' || row[3] === 'failed');

    var marker;
    if (off_road) {
        marker = L.marker(new L.LatLng(row[0], row[1]), {
            icon: L.divIcon({
                className: 'off-road-cross',
                iconSize: [10, 10],
                html: '<div style="width:10px;height:10px;color:' + c +
                      ';font-weight:900;font-size:14px;line-height:10px;' +
                      'text-align:center;">×</div>'
            }),
            interactive: false
        });
    } else {
        marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {radius: 4, color: c, weight: 1, fillOpacity: 0.75,
             interactive: false});
    }
    return marker;
}
"""


@st.cache_data(ttl=3600, show_spinner=False)
def _build_drill_payload(
    snap_to_road: bool,
    year_min: int | None,
    year_max: int | None,
) -> tuple[list, int]:
    """Готовит payload для FastMarkerCluster: list of
    [lat, lon, severity, snap_method, datetime_str, em_type, lost, suffer, address].

    Кешируется через cache_data по (snap, year_min, year_max).
    Сбор payload'а на 28k записей занимает ~150-300 мс — кеш критичен
    для частых rerun'ов в interactive-режиме.
    """
    coords_df = get_accident_coords(
        snap_to_road=snap_to_road,
        year_min=year_min,
        year_max=year_max,
    )
    if coords_df.empty:
        return [], 0

    df = coords_df
    datetime_str = (
        df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        if "datetime" in df
        else pd.Series([""] * len(df))
    )
    address = (
        (
            df["np"].fillna("").astype(str)
            + ", "
            + df["street"].fillna("").astype(str)
            + " "
            + df["house"].fillna("").astype(str)
        )
        .str.replace(r"^,\s*", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(", ")
    )
    address = address.replace("", "адрес не указан")
    data = list(
        zip(
            df["lat"].tolist(),
            df["lon"].tolist(),
            df["severity"].tolist(),
            df["snap_method"].fillna("none").tolist(),
            datetime_str.tolist(),
            df["em_type"].fillna("—").tolist(),
            df["lost_amount"].fillna(0).astype(int).tolist(),
            df["suffer_amount"].fillna(0).astype(int).tolist(),
            address.tolist(),
        )
    )
    return data, len(df)


def _add_drill_layer(
    m: folium.Map,
    snap_to_road: bool,
    year_min: int | None,
    year_max: int | None,
    *,
    with_popup: bool = True,
) -> None:
    """Добавляет FastMarkerCluster со всеми 28k ДТП непосредственно в Map.

    НЕ через ``feature_group_to_add`` (тот провоцирует JS-конфликты в
    streamlit-folium iframe). Слой добавляется как часть исходного Map
    при сборке — folium-сгенерированные имена переменных уникальны для
    каждой Map-инстанции.

    ``with_popup=False`` — для interactive-режима: маркеры рисуются
    как ``interactive: false``, popup'ы не привязываются. Клик по
    drill-маркеру проходит «насквозь» на карту и попадает в map.click,
    что нужно для механики «точка радиуса для рекомендаций».
    """
    data, n = _build_drill_payload(snap_to_road, year_min, year_max)
    if not data:
        return
    callback = _DRILL_CALLBACK_WITH_POPUP_JS if with_popup else _DRILL_CALLBACK_NO_POPUP_JS
    FastMarkerCluster(
        data=data,
        name=f"🎯 Все ДТП ({fmt_int(n)}) — drill-down",
        callback=callback,
        overlay=True,
        show=True,
        options={
            "spiderfyOnMaxZoom": False,
            "showCoverageOnHover": False,
            "maxClusterRadius": 60,
            "disableClusteringAtZoom": 16,
            "removeOutsideVisibleBounds": True,
            "chunkedLoading": True,
            "zoomToBoundsOnClick": True,
        },
    ).add_to(m)


# ============================================================
# Сборка карты (cache_resource — строка immutable, без deepcopy)
# ============================================================
@st.cache_resource(show_spinner="Рендеринг карты очагов...")
def build_map_html(
    _items_sig: tuple,
    show_drill: bool,
    snap_to_road: bool,
    year_min: int,
    year_max: int,
) -> str:
    """HTML self-contained Folium-карты для встраивания в iframe.

    Кеш ключ — все аргументы (кроме `_items_sig` который sig items'ов).
    Toggle любого параметра инвалидирует кеш и запускает rebuild.

    ``snap_to_road``/``year_min``/``year_max`` — фильтры drill-down
    payload'а. Heatmap (precision=3 grid) и центроиды НЕ зависят
    от этих фильтров.
    """
    centroids = [(it["centroid"]["lat"], it["centroid"]["lon"]) for it in items]
    lats = [c[0] for c in centroids]
    lons = [c[1] for c in centroids]
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        tiles=None,
        control_scale=True,
        zoom_control=True,
        # prefer_canvas=True — все vector-объекты (CircleMarker/Polygon)
        # рендерятся на одном HTML5 canvas вместо отдельных SVG-нод.
        # Для 30 центроидов разница незаметна, но если drill-down ON и
        # на viewport попадает 200-500 индивидуальных маркеров — это
        # на 1-2 порядка быстрее SVG. SVG хранит каждый маркер как DOM
        # node с listeners; canvas — pixel buffer. Pan/zoom плавный.
        prefer_canvas=True,
        # Ограничиваем максимальный zoom — на zoom 18 каждое движение
        # требует ~25 новых OSM-тайлов (256x256), холодный cache даёт
        # 3-5 сек подвисаний. Zoom 16 = детализация улицы, для
        # аналитического дашборда достаточно. Если нужно «к окну дома»,
        # пользователь идёт в Яндекс/Google Maps по координатам.
        max_zoom=16,
        # scrollWheelZoom=True (default) — зум колесом мыши, привычно
        # по Google Maps / Mapbox; удобнее кнопок +/−
    )

    # max_native_zoom=15: Leaflet прекращает запрашивать новые tile'ы выше
    # zoom 15, последний загруженный tile растягивается ("over-zoom").
    # Качество немного «мыльное» при zoom 16, но zero network roundtrip
    # на zoom-in — нет 5-секундных подвисаний при перемещении в новый
    # район. Совпадает с прагматикой "дашборд показывает агрегаты, не
    # уличные дома".
    # Только OpenStreetMap. CartoDB Positron был раньше как «светлая»
    # альтернатива, но Carto с 2022 года добавляет в attribution
    # политический украинский флаг — для гос-проекта (ГИБДД Приморья,
    # для гос-проекта это неуместно. OSM нейтрален.
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        max_zoom=16,
        max_native_zoom=15,
    ).add_to(m)

    # ---------- Слой 1: HeatMap плотности 28k ДТП (default ON) ----------
    # Главный визуал — показывает РЕАЛЬНОЕ распределение по улицам.
    # Точки внутри кластера №6 центра Владивостока ясно ложатся на
    # Светланскую/Алеутскую/Океанский, а не на Сопку Тигровую куда
    # попадает мат. центроид. Это решает основную UX-претензию.
    # Heatmap использует precision=3 grid (≈ 110 м bins), не raw 28k —
    # snap-to-road / year-фильтр на уровень heatmap НЕ применяется
    # (плотность в одном районе не сильно меняется от года к году).
    # По умолчанию OFF — пользователь явно сказал «без теплоты при
    # открытии» (UX-ревью). Включается через LayerControl справа
    # сверху для обзора плотности по улицам. После spatial-binning
    # лагов на zoom-in нет, так что включение быстрое.
    heat_fg = folium.FeatureGroup(name="🔥 Тепловая карта плотности ДТП", show=False).add_to(m)
    # Spatial-binning: 28 020 точек → ~5-7k weighted-bins (precision=3
    # ≈ 110 м). heatmap.js на сильном zoom пересчитывает projection
    # КАЖДОЙ точки при каждом move/zoom, отсюда лаги. Weighted-bins
    # дают тот же визуальный результат (blur=18 всё равно сглаживает
    # ячейки), но в 4-5 раз быстрее. Кеш на час: первый рендер ~80 мс,
    # последующие из cache_data моментально.
    heatmap_data = get_heatmap_grid(precision=3)
    HeatMap(
        heatmap_data,
        radius=14,
        blur=20,
        min_opacity=0.35,
        # max_zoom=15: heatmap.js не пересчитывает gradient выше zoom 15.
        # При zoom 16 (наш maxZoom) тепловая карта замораживается на
        # последнем canvas — никакого CPU-cost'а, плавный pan/zoom.
        max_zoom=15,
        gradient={
            "0.2": "#3498db",
            "0.4": "#f1c40f",
            "0.6": "#e67e22",
            "0.85": "#c0392b",
        },
    ).add_to(heat_fg)

    # ---------- Слой 2: очаги-центроиды (default ON) ----------
    # Кликабельные маркеры со статистикой каждого DBSCAN-кластера.
    # НЕ показывают геолокацию ДТП (для этого heatmap), показывают
    # только агрегаты (топ типов, % смерт., радиус в метрах).
    # show=False — центроиды визуально перекрывают drill-down кластеры
    # и при default-входе на страницу не нужны (drill-down показывает
    # больше). Включается через LayerControl когда нужны popup'ы со
    # статистикой по DBSCAN-кластерам.
    hotspots_fg = folium.FeatureGroup(
        name="📍 Центроиды 30 очагов (статистика)", show=False
    ).add_to(m)

    for it in items:
        radius_px = max(6.0, math.log10(max(it["n_points"], 10)) * 9.0)
        color = color_by_pct_dead(it["pct_dead"])
        center = [it["centroid"]["lat"], it["centroid"]["lon"]]

        folium.CircleMarker(
            location=center,
            radius=radius_px,
            color="#ffffff",  # белая обводка для контраста на heatmap
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            popup=folium.Popup(_format_popup(it), max_width=340),
            tooltip=(
                f"#{it['rank']} · {fmt_int(it['n_points'])} ДТП · "
                f"{it['pct_dead'] * 100:.1f}% смерт."
            ),
        ).add_to(hotspots_fg)

    # ---------- Слой 3: drill-down на отдельные ДТП (опционально) ----------
    if show_drill:
        _add_drill_layer(m, snap_to_road, year_min, year_max)

    # Легенда
    legend_html = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,0.95); padding: 10px 14px;
                border: 1px solid #d0d0d0; border-radius: 6px;
                font-family: sans-serif; font-size: 12px; line-height: 1.7;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 4px;">
        Тепловая карта — плотность ДТП
      </div>
      <span style="color:#3498db; font-size: 16px;">●</span>&nbsp;разреженно<br>
      <span style="color:#f1c40f; font-size: 16px;">●</span>&nbsp;средне<br>
      <span style="color:#e67e22; font-size: 16px;">●</span>&nbsp;плотно<br>
      <span style="color:#c0392b; font-size: 16px;">●</span>&nbsp;очень плотно<br>
      <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ddd;">
      <div style="font-weight: 700; margin-bottom: 4px;">
        Маркеры-центроиды — % смертельных
      </div>
      <span style="color:#3498db; font-size: 16px;">●</span>&nbsp;&lt; 2%<br>
      <span style="color:#f39c12; font-size: 16px;">●</span>&nbsp;2 — 5%<br>
      <span style="color:#c0392b; font-size: 16px;">●</span>&nbsp;≥ 5%<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Скрыть бренд-флаг Leaflet attribution.
    # CSS: несколько селекторов на случай разных версий Leaflet
    # (1.9.x — `<svg class="leaflet-attribution-flag">` внутри
    # `<a href="https://leafletjs.com">`).
    # JS: страховка — удаляем элементы из DOM, если CSS почему-то
    # не сработал (бывает при кастомных templates folium'а).
    m.get_root().html.add_child(folium.Element(_FLAG_HIDE_HTML))

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    m.fit_bounds(
        [[min(lats), min(lons)], [max(lats), max(lons)]],
        padding=(40, 40),
    )
    return m.get_root().render()


cache_key = tuple((it["cluster_id"], it["n_points"]) for it in items)


# ============================================================
# Side-panel рендер рекомендаций (общий для обоих режимов клика)
# ============================================================
def _render_recommendation_card(rec: dict) -> None:
    """Карточка одной рекомендации в side-panel'и."""
    priority_color = {1: "#c0392b", 2: "#e67e22", 3: "#3498db"}.get(rec["priority"], "#7f8c8d")
    priority_label = {1: "🔴 Критично", 2: "🟠 Важно", 3: "🔵 Превентивно"}.get(rec["priority"], "")
    eff = rec["expected_effect"]
    point_pct = eff["point_estimate"] * 100
    ci_low_pct = eff["ci_low"] * 100
    ci_high_pct = eff["ci_high"] * 100

    with st.container(border=True):
        st.markdown(
            f"<div style='font-size: 14px; color:{priority_color}; "
            f"font-weight: 700;'>{priority_label} · {rec['rule_id']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{rec['title']}**")
        st.caption(rec["trigger_human"])

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"**Ожидаемый эффект:** {point_pct:+.1f} % "
                f"(95 % CI: {ci_low_pct:+.1f} %.. {ci_high_pct:+.1f} %)"
            )
            st.caption(f"_{eff['note']}_")
        with c2:
            stars = "★" * round(rec["confidence"] * 5) + "☆" * (5 - round(rec["confidence"] * 5))
            st.markdown(f"**Confidence:** {stars}")
            st.caption(f"Стоимость: {rec['implementation_cost']}")

        with st.expander(":material/menu_book: Источники"):
            for cit in rec["evidence_basis"]:
                st.markdown(f"- [{cit['label']}]({cit['url']}) — {cit['quoted_effect']}")


def _render_profile_summary(profile: dict) -> None:
    """Сводка профиля очага под рекомендациями."""
    with st.expander(":material/analytics: Профиль выборки", expanded=False):
        sub_cols = st.columns(2)
        sub_cols[0].metric("ДТП", fmt_int(profile["n_points"]))
        sub_cols[1].metric("% смертельных", f"{profile['pct_dead'] * 100:.2f}%")
        if profile.get("top_em_type"):
            st.caption(f"**Тип:** {profile['top_em_type']}")
        if profile.get("top_np"):
            st.caption(f"**НП:** {profile['top_np']}")
        if profile.get("dominant_light_type"):
            st.caption(f"**Освещение:** {profile['dominant_light_type']}")
        if profile.get("dominant_state"):
            st.caption(f"**Покрытие:** {profile['dominant_state']}")
        st.caption(
            f"is_highway={profile['is_highway']}, "
            f"has_night_dominant={profile.get('has_night_dominant', False)}, "
            f"has_winter_spike={profile.get('has_winter_spike', False)}"
        )


# ============================================================
# Рендер карты + side-panel в интерактив-режиме
# ============================================================
# Стратегия (после ревью UX):
# 1. Базовая карта (тайлы + 30 маркеров очагов) строится ОДИН раз
#    через @st.cache_resource — Folium не пересобирается при кликах,
#    Leaflet keeps zoom/pan
# 2. Динамический Circle (радиус анализа) добавляется через
#    st_folium(feature_group_to_add=...) — slot для overlay-слоя,
#    меняется без reinit карты
# 3. Side-panel + radius-slider обёрнуты в @st.fragment — клик в
#    карту перезапускает только этот фрагмент (не всю страницу),
#    позиция/zoom карты сохраняются автоматически
# 4. Diff-логика клика: session_state хранит «прошлый клик»;
#    только новый, отличающийся клик триггерит новый API-запрос —
#    избавляет от двойного fetch'а на rerun-эхо
def _build_interactive_base_map(
    _cache_key: tuple,
    show_drill: bool,
    snap_to_road: bool,
    year_min: int | None,
    year_max: int | None,
) -> folium.Map:
    """Базовая карта interactive-режима: тайлы + 30 центроидов очагов
    + опционально drill-down + флаг-CSS.

    Drill-down добавляется СРАЗУ в Map (не через
    ``feature_group_to_add`` — это провоцировало JS-конфликты в
    streamlit-folium iframe в одной из ранних версий). За счёт unique folium-IDs
    каждой Map-инстанции коллизий не будет.
    """
    centroids = [(it["centroid"]["lat"], it["centroid"]["lon"]) for it in items]
    avg_lat = sum(c[0] for c in centroids) / len(centroids)
    avg_lon = sum(c[1] for c in centroids) / len(centroids)
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=8,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        max_zoom=16,
    )
    folium.TileLayer(
        "OpenStreetMap",
        attr="© OpenStreetMap contributors",
        max_zoom=16,
        max_native_zoom=15,
    ).add_to(m)
    for it in items:
        radius_px = max(6.0, math.log10(max(it["n_points"], 10)) * 9.0)
        color = color_by_pct_dead(it["pct_dead"])
        center = [it["centroid"]["lat"], it["centroid"]["lon"]]
        folium.CircleMarker(
            location=center,
            radius=radius_px,
            color="#ffffff",
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            # Tooltip встроенно содержит cluster_id — парсится в side-panel'е
            # для запроса /recommendations/hotspot/{cluster_id}.
            tooltip=(
                f"#{it['rank']} · {fmt_int(it['n_points'])} ДТП · "
                f"{it['pct_dead'] * 100:.1f}% смерт · CL{it['cluster_id']}"
            ),
        ).add_to(m)
    # Drill-down 28k ДТП — добавляется ВНУТРЬ Map'ы при сборке.
    # with_popup=False: в interactive popup'ы не привязываются, чтобы
    # клик по drill-маркеру не перехватывался — он должен «провалиться»
    # на карту, чтобы сработала механика «точка радиуса».
    if show_drill:
        _add_drill_layer(m, snap_to_road, year_min, year_max, with_popup=False)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    lats = [c[0] for c in centroids]
    lons = [c[1] for c in centroids]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(40, 40))
    # Скрыть бренд-флаг Leaflet attribution.
    m.get_root().html.add_child(folium.Element(_FLAG_HIDE_HTML))
    return m


def _build_selection_overlay(
    clicked: tuple[float, float] | None, radius_m: int
) -> folium.FeatureGroup | None:
    """Динамический FeatureGroup: маркер кликнутой точки + Circle радиуса.

    Возвращает None если точка не выбрана — st_folium корректно обрабатывает.
    """
    if clicked is None:
        return None
    fg = folium.FeatureGroup(name="selection")
    folium.Circle(
        location=list(clicked),
        radius=radius_m,
        color="#1f4e79",
        weight=2,
        fill=True,
        fillColor="#1f4e79",
        fillOpacity=0.10,
        popup=f"Радиус анализа: {radius_m} м",
    ).add_to(fg)
    folium.Marker(
        location=list(clicked),
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        tooltip=f"Точка анализа · {clicked[0]:.5f}, {clicked[1]:.5f}",
    ).add_to(fg)
    return fg


def _parse_cluster_id(tooltip: str | None) -> int | None:
    """Извлекает cluster_id из tooltip-формата '... · CL{id}'."""
    if not tooltip:
        return None
    # Ищем 'CLxxx' в конце строки
    parts = tooltip.split("CL")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1].strip().split()[0].rstrip("."))
    except (ValueError, IndexError):
        return None


@st.fragment
def _render_interactive_map_panel() -> None:
    """Фрагмент с картой + side-panel.

    Изоляция rerun'ов: клик в карту, движение radius-slider'а или
    кнопки сброса перезапускают ТОЛЬКО эту функцию, остальная
    страница (заголовок, метрики, таблица очагов) не моргает.
    """
    # 1) Top-of-fragment баннер с инструкцией (всегда виден)
    st.info(
        ":material/help_outline: **1) Выбери радиус справа** → "
        "**2) Кликни в карту** (любая точка ИЛИ маркер очага) → "
        "**3) Получи топ-3 рекомендации**. Карта сохраняет zoom/центр между кликами."
    )

    map_col, panel_col = st.columns([2, 1.1])

    # 2) Side-panel: radius-slider + статус + кнопки управления + рекомендации
    with panel_col:
        # --- Header строка ---
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown("### :material/lightbulb: Рекомендации")

        # --- Радиус-slider ВВЕРХУ side-panel'а (UX-фикс) ---
        radius = st.slider(
            "Радиус динамического очага, м",
            min_value=30,
            max_value=2000,
            value=int(st.session_state.get("recs_radius", 100)),
            step=1,
            key="recs_radius_slider",
            help=(
                "Любое целое число от 30 до 2000 м с шагом 1. "
                "По умолчанию 100 м — компромисс между точностью и "
                "достаточной выборкой ДТП в радиусе."
            ),
        )
        st.session_state["recs_radius"] = radius

        # --- Кнопка «Сбросить выбор» (если есть selection) ---
        has_selection = (
            st.session_state.get("recs_clicked_point") is not None
            or st.session_state.get("recs_cluster_id") is not None
        )
        with head_r:
            if has_selection:
                if st.button(
                    ":material/clear:",
                    help="Сбросить выбранную точку / очаг",
                    key="recs_clear_btn",
                    use_container_width=True,
                ):
                    st.session_state["recs_clicked_point"] = None
                    st.session_state["recs_cluster_id"] = None
                    st.session_state["recs_last_click_sig"] = None
                    st.rerun(scope="fragment")

    # 3) Map column: рендер карты + чтение клика
    with map_col:
        clicked_point = st.session_state.get("recs_clicked_point")
        # Базовая карта строится свежей при каждом изменении drill/snap/
        # year (тогда же меняется st_folium key → fresh iframe → нет
        # streamlit-folium const collision'а). Внутри одного состояния
        # toggle'ов (только клик/radius меняется) iframe переиспользуется
        # — Leaflet keep zoom/pan, click state сохраняется.
        base_map = _build_interactive_base_map(
            cache_key, show_drill, snap_to_road, year_min, year_max
        )
        # Selection-circle (1 элемент, лёгкий) — через feature_group_to_add,
        # без перерисовки base map.
        selection_fg = _build_selection_overlay(clicked_point, radius)

        # Key зависит от тех параметров, которые меняют HEAVY-содержимое
        # карты (drill 28k, snap snapshot, year-фильтр). Меняется key →
        # Streamlit считает widget новым → fresh iframe для streamlit-
        # folium → нет JS const-конфликтов. clicked_point НЕ в key, чтобы
        # фрагмент-rerun на клик не сбрасывал zoom/pan карты.
        st_folium_key = (
            f"hotspots_interactive_{int(show_drill)}_{int(snap_to_road)}" f"_{year_min}_{year_max}"
        )

        click_data = st_folium(
            base_map,
            feature_group_to_add=selection_fg,
            # Высота 660 px — компромисс: достаточно вертикального
            # пространства для карты Приморья (горизонтально-вытянутый
            # регион), но под картой ещё помещаются сводные метрики
            # DBSCAN и таблица 30 очагов без слишком долгого скролла.
            height=660,
            width=None,
            returned_objects=["last_clicked", "last_object_clicked_tooltip"],
            key=st_folium_key,
        )

    # 4) Diff клика: триггерим новый API-запрос только когда
    # пользователь реально кликнул в новое место (а не просто rerun)
    new_clicked_pt = (click_data or {}).get("last_clicked")
    new_tooltip = (click_data or {}).get("last_object_clicked_tooltip")

    # Сигнатура текущего клика — чтобы детектить «новый» клик vs rerun-эхо
    current_sig: tuple | None = None
    if new_tooltip and "CL" in new_tooltip:
        current_sig = ("cluster", new_tooltip)
    elif new_clicked_pt and new_clicked_pt.get("lat") is not None:
        current_sig = (
            "point",
            round(new_clicked_pt["lat"], 5),
            round(new_clicked_pt["lng"], 5),
        )

    last_sig = st.session_state.get("recs_last_click_sig")
    if current_sig is not None and current_sig != last_sig:
        # Новый клик — обновляем state
        st.session_state["recs_last_click_sig"] = current_sig
        if current_sig[0] == "cluster":
            cid = _parse_cluster_id(new_tooltip)
            if cid is not None:
                st.session_state["recs_cluster_id"] = cid
                st.session_state["recs_clicked_point"] = None
                st.rerun(scope="fragment")
        elif current_sig[0] == "point":
            st.session_state["recs_clicked_point"] = (current_sig[1], current_sig[2])
            st.session_state["recs_cluster_id"] = None
            st.rerun(scope="fragment")

    # 5) Рендер рекомендаций в side-panel'е
    cluster_id = st.session_state.get("recs_cluster_id")
    point = st.session_state.get("recs_clicked_point")

    with panel_col:
        if cluster_id is not None:
            try:
                with st.spinner("Загружаю рекомендации для очага..."):
                    recs_payload = fetch_recommendations_hotspot(cluster_id, top_k=3)
            except httpx.HTTPError as exc:
                st.error(f"Ошибка API: {exc}")
                return
            profile = recs_payload["profile"]
            st.success(
                f":material/check_circle: **DBSCAN-очаг #{cluster_id}** · "
                f"{fmt_int(profile['n_points'])} ДТП · "
                f"{profile['pct_dead'] * 100:.2f}% смерт"
            )
            _render_profile_summary(profile)
            if recs_payload["items"]:
                st.caption(f"Топ-{len(recs_payload['items'])} мер по score'ингу:")
                for rec in recs_payload["items"]:
                    _render_recommendation_card(rec)
            elif recs_payload.get("note"):
                st.info(recs_payload["note"])
            else:
                st.info("Ни одно из 18 правил не сработало для этого очага.")
        elif point is not None:
            # Year-фильтр прокидываем только если он реально сужен
            # (не на полный диапазон БД). Это позволяет API кешировать
            # общий случай.
            y_min_req = year_min if year_min > _y_min_full else None
            y_max_req = year_max if year_max < _y_max_full else None
            period_str = (
                f" за {year_min}–{year_max}"
                if (y_min_req is not None or y_max_req is not None)
                else ""
            )
            try:
                with st.spinner(f"Анализирую ДТП в радиусе {radius} м{period_str}..."):
                    recs_payload = fetch_recommendations_point(
                        lat=point[0],
                        lon=point[1],
                        radius=radius,
                        top_k=3,
                        year_min=y_min_req,
                        year_max=y_max_req,
                    )
            except httpx.HTTPError as exc:
                st.error(f"Ошибка API: {exc}")
                return
            profile = recs_payload["profile"]
            st.success(
                f":material/check_circle: **Точка ({point[0]:.5f}, {point[1]:.5f})** "
                f"· радиус {radius} м{period_str}"
            )
            metric_cols = st.columns(2)
            metric_cols[0].metric("Найдено ДТП", profile["n_points"])
            metric_cols[1].metric("% смертельных", f"{profile['pct_dead'] * 100:.2f}%")
            _render_profile_summary(profile)
            if recs_payload.get("note"):
                st.warning(recs_payload["note"])
            elif recs_payload["items"]:
                st.caption(f"Топ-{len(recs_payload['items'])} мер по score'ингу:")
                for rec in recs_payload["items"]:
                    _render_recommendation_card(rec)
            else:
                st.info("Ни одно из 18 правил не сработало для этого профиля.")
        else:
            st.info(
                "⬅️ **Карта готова к клику.**\n\n"
                f"Текущий радиус: **{radius} м**. "
                "Меняй радиус выше, потом кликни. Если хочешь "
                "пересчитать с новым радиусом — кликни в ту же точку "
                "ещё раз или нажми :material/clear: рядом с заголовком."
            )


if interactive_mode:
    _render_interactive_map_panel()
else:
    # Default-режим: components.html (быстрый, drill-down 28k)
    map_html = build_map_html(cache_key, show_drill, snap_to_road, year_min, year_max)
    # Высота 660 px (см. комментарий в interactive-блоке).
    components.html(map_html, height=660, scrolling=False)


# ============================================================
# Сводные метрики DBSCAN
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Очагов в выборке", len(items))
c2.metric(
    "Кластеризовано",
    fmt_int(stats["clustered_count"]),
    delta=f"{(1 - stats['noise_pct']) * 100:.1f}% от 28 020",
    delta_color="off",
)
c3.metric("В шуме (вне очагов)", fmt_int(stats["noise_count"]))
c4.metric("Время DBSCAN", f"{stats['elapsed_seconds']:.2f} c")

st.divider()

# ============================================================
# Полная таблица очагов
# ============================================================
st.subheader("Все 30 очагов")

df = pd.DataFrame(
    [
        {
            "Ранг": it["rank"],
            "ДТП": it["n_points"],
            "% смерт.": round(it["pct_dead"] * 100, 2),
            "% тяж+смерт.": round(it["pct_severe_or_dead"] * 100, 1),
            "Радиус, м": round(it["radius_meters"]),
            "lat": round(it["centroid"]["lat"], 5),
            "lon": round(it["centroid"]["lon"], 5),
            "Топ-1 тип ДТП": (it["top_em_types"][0][0] if it["top_em_types"] else ""),
            "Топ-1 НП": it["top_np"][0][0] if it["top_np"] else "",
        }
        for it in items
    ]
)
st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "% смерт.": st.column_config.NumberColumn(format="%.2f%%"),
        "% тяж+смерт.": st.column_config.NumberColumn(format="%.1f%%"),
        "ДТП": st.column_config.NumberColumn(format="%d"),
        "Радиус, м": st.column_config.NumberColumn(format="%d"),
    },
)

# Footer — единый для всех 7 страниц.
page_footer()
