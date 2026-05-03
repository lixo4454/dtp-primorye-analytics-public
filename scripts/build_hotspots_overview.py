"""
4-panel PNG overview of DBSCAN accident hotspots.

Что делает: строит составной 4-панельный график-сводку по результатам
DBSCAN-кластеризации (eps=100м, min_samples=15): (1) гистограмма
размеров кластеров, (2) горизонтальный bar-chart топ-30 очагов с
цветом по доле смертельных ДТП и подписью адресом, (3) scatter-plot
«размер vs %смертельных», (4) геопространственная схема топ-30 на
фоне сухопутной границы Приморья. Сохраняется в
`data/processed/hotspots_overview.png`.

Зачем нужно: один PNG-файл с обзором очагов
(стиль Дней 4 и 5 — `rhd_lhd_severity_analysis.png`,
`telegram_ner_overview.png`); сразу даёт ответы на «какие места
самые опасные», «корреляция размера и летальности», «как очаги
распределены географически».
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap


# Та же цветовая логика, что и в Folium-карте
def color_by_pct_dead(pct: float) -> str:
    if pct < 0.03:
        return "#27ae60"
    if pct < 0.06:
        return "#f1c40f"
    if pct < 0.10:
        return "#e67e22"
    if pct < 0.15:
        return "#e74c3c"
    return "#8e44ad"


def main() -> None:
    summary_path = Path("data/processed/hotspots_eps100_final_summary.json")
    geojson_path = Path("data/processed/hotspots_eps100.geojson")
    boundary_path = Path("data/raw/primorye_land_polygons.geojson")
    out_path = Path("data/processed/hotspots_overview.png")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    geojson_obj = json.loads(geojson_path.read_text(encoding="utf-8"))
    top_clusters = summary["top_clusters"]
    all_clusters = geojson_obj["features"]

    n_max = max(c["n_points"] for c in top_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"DBSCAN-кластеризация очагов аварийности Приморского края "
        f"(eps={summary['params']['eps_meters']:.0f}м, "
        f"min_samples={summary['params']['min_samples']}, "
        f"всего кластеров: {summary['stats']['clusters_count']}, "
        f"шум: {summary['stats']['noise_pct']:.1%})",
        fontsize=14,
        fontweight="bold",
    )

    # === Panel 1: Гистограмма размеров кластеров ===
    ax1 = axes[0, 0]
    sizes = [c["properties"]["n_points"] for c in all_clusters]
    bins = np.logspace(np.log10(min(sizes)), np.log10(max(sizes) + 1), 25)
    ax1.hist(sizes, bins=bins, color="#3498db", edgecolor="#2c3e50", alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Размер кластера (число ДТП), лог-шкала")
    ax1.set_ylabel("Количество кластеров")
    ax1.set_title(f"Распределение размеров кластеров (всего {len(all_clusters)})")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(15, color="red", linestyle="--", alpha=0.5, label="min_samples=15")
    ax1.legend()

    # === Panel 2: Горизонтальный bar-chart топ-30 ===
    ax2 = axes[0, 1]
    labels = []
    n_points = []
    bar_colors = []
    for c in reversed(top_clusters):  # reversed чтобы #1 был сверху на горизонтальном
        street = c["top_streets"][0][0] if c["top_streets"] else f"cluster {c['cluster_id']}"
        # сократим длинный адрес
        street = (
            street.replace("г Владивосток / ", "Влв / ")
            .replace("г Уссурийск / ", "Усс / ")
            .replace("г Находка / ", "Нах / ")
            .replace("г Артем / ", "Арт / ")[:40]
        )
        labels.append(f"#{c['rank']:>2} {street}")
        n_points.append(c["n_points"])
        bar_colors.append(color_by_pct_dead(c["pct_dead"]))
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, n_points, color=bar_colors, edgecolor="#2c3e50", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("ДТП в кластере")
    ax2.set_title("ТОП-30 очагов аварийности (цвет ∝ %смертельных)")
    ax2.grid(True, axis="x", alpha=0.3)

    # === Panel 3: Scatter n_points vs pct_dead ===
    ax3 = axes[1, 0]
    xs = [c["properties"]["n_points"] for c in all_clusters]
    ys = [c["properties"]["pct_dead"] * 100 for c in all_clusters]
    cs = [color_by_pct_dead(c["properties"]["pct_dead"]) for c in all_clusters]
    ax3.scatter(xs, ys, s=40, c=cs, edgecolor="#2c3e50", linewidth=0.5, alpha=0.75)

    # Подписи топ-5
    top5 = sorted(all_clusters, key=lambda c: -c["properties"]["n_points"])[:5]
    for c in top5:
        p = c["properties"]
        street = p["top_streets"][0][0] if p["top_streets"] else f"cl {p['cluster_id']}"
        street = (
            street.replace("г Владивосток / ", "Влв / ")
            .replace("г Уссурийск / ", "Усс / ")
            .replace("г Находка / ", "Нах / ")
            .replace("г Артем / ", "Арт / ")
        )
        ax3.annotate(
            street,
            (p["n_points"], p["pct_dead"] * 100),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 3),
            textcoords="offset points",
        )
    ax3.axhline(8.7, color="red", linestyle="--", alpha=0.5, label="Среднее по Приморью 8.7%")
    ax3.set_xscale("log")
    ax3.set_xlabel("Размер кластера (n_points), лог-шкала")
    ax3.set_ylabel("Доля смертельных ДТП в кластере, %")
    ax3.set_title("Связь размера и летальности (топ-5 подписаны)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # === Panel 4: Гео-схема топ-30 (городские) + топ-30 трассовых ===
    ax4 = axes[1, 1]
    boundary_gdf = gpd.read_file(boundary_path)
    boundary_gdf.plot(ax=ax4, color="#ecf0f1", edgecolor="#7f8c8d", linewidth=0.5)

    # Городские очаги (eps=100м) — кружочки
    for c in top_clusters:
        col = color_by_pct_dead(c["pct_dead"])
        ax4.scatter(
            c["centroid_lon"],
            c["centroid_lat"],
            s=20 + 220 * np.sqrt(c["n_points"] / n_max),
            c=col,
            edgecolor="#2c3e50",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
            marker="o",
            label=None,
        )

    # Трассовые очаги (eps=500м, min_samples=5) — ромбы, отфильтрованные от городских
    highway_path = Path("data/processed/hotspots_highway.geojson")
    if highway_path.exists():
        hw_geo = json.loads(highway_path.read_text(encoding="utf-8"))
        hw_features = sorted(hw_geo["features"], key=lambda f: -f["properties"]["n_points"])
        existing = [(c["centroid_lon"], c["centroid_lat"]) for c in top_clusters]
        hw_unique = []
        for hwf in hw_features[:80]:
            hlon, hlat = hwf["geometry"]["coordinates"]
            too_close = any(
                abs(hlon - elon) < 0.014 and abs(hlat - elat) < 0.01 for (elon, elat) in existing
            )
            if not too_close and hwf["properties"]["n_points"] >= 8:
                hw_unique.append(hwf)
            if len(hw_unique) >= 30:
                break
        hw_n_max = max(f["properties"]["n_points"] for f in hw_unique) if hw_unique else 1
        for f in hw_unique:
            p = f["properties"]
            col = color_by_pct_dead(p["pct_dead"])
            ax4.scatter(
                f["geometry"]["coordinates"][0],
                f["geometry"]["coordinates"][1],
                s=20 + 200 * np.sqrt(p["n_points"] / hw_n_max),
                c=col,
                edgecolor="#34495e",
                linewidth=0.6,
                alpha=0.85,
                zorder=3,
                marker="D",
            )

    # Подписи городов
    for label, lon, lat in [
        ("Владивосток", 131.89, 43.12),
        ("Уссурийск", 131.97, 43.81),
        ("Находка", 132.88, 42.83),
        ("Артём", 132.20, 43.36),
        ("Арсеньев", 133.27, 44.16),
        ("Дальнегорск", 135.55, 44.55),
    ]:
        ax4.annotate(
            label,
            (lon, lat),
            fontsize=8,
            color="#2c3e50",
            xytext=(6, 6),
            textcoords="offset points",
            alpha=0.9,
        )

    ax4.set_xlim(130.0, 140.0)
    ax4.set_ylim(42.0, 48.5)
    ax4.set_xlabel("Долгота")
    ax4.set_ylabel("Широта")
    ax4.set_title("Гео-схема: ● топ-30 городских (eps=100м), " "◆ топ-30 трассовых (eps=500м)")
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal", adjustable="datalim")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: {} ({:.1f} kB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
