"""
CLI orchestrator for DBSCAN accident hotspot clustering.

Что делает: запускает `find_hotspots()` из `src.analysis.dbscan_clustering`
с заданными CLI-параметрами (eps, min_samples), сохраняет два артефакта —
GeoJSON для Folium-карты (`hotspots.geojson` — FeatureCollection
с точками-центроидами и метаданными) и JSON со сводкой по топ-30
очагам (`hotspots_summary.json` — для аналитических отчётов и
интерпретации).

Зачем нужно: фиксирует результат конкретного запуска DBSCAN на диск
для воспроизводимости, для использования в Folium-визуализации
и для дальнейшей выгрузки в FastAPI/Streamlit
(Дни 13-15).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.analysis.dbscan_clustering import Cluster, HotspotsResult, find_hotspots
from src.database import SessionLocal


def cluster_to_geojson_feature(cluster: Cluster) -> dict:
    """Превращает Cluster в Feature GeoJSON (точка-центроид + properties)."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [cluster.centroid_lon, cluster.centroid_lat],
        },
        "properties": {
            "cluster_id": cluster.cluster_id,
            "n_points": cluster.n_points,
            "radius_meters": round(cluster.radius_meters, 1),
            "median_distance_meters": round(cluster.median_distance_meters, 1),
            "pct_dead": round(cluster.pct_dead, 4),
            "pct_severe_or_dead": round(cluster.pct_severe_or_dead, 4),
            "severity_distribution": cluster.severity_distribution,
            "bbox": [
                cluster.bbox_lon_min,
                cluster.bbox_lat_min,
                cluster.bbox_lon_max,
                cluster.bbox_lat_max,
            ],
            "top_em_types": cluster.top_em_types,
            "top_np": cluster.top_np,
            "top_streets": cluster.top_streets,
        },
    }


def result_to_geojson(result: HotspotsResult) -> dict:
    """Превращает HotspotsResult в FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [cluster_to_geojson_feature(c) for c in result.clusters],
        "metadata": {
            "eps_meters": result.eps_meters,
            "min_samples": result.min_samples,
            "region_name": result.region_name,
            "projection": result.projection,
            "total_points": result.total_points,
            "clustered_count": result.clustered_count,
            "noise_count": result.noise_count,
            "clusters_count": len(result.clusters),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def result_to_summary(result: HotspotsResult, top_n: int = 30) -> dict:
    """Подробный JSON-отчёт по топ-N кластерам (без accident_ids — они в geojson не нужны)."""
    top_clusters = []
    for rank, c in enumerate(result.clusters[:top_n], start=1):
        top_clusters.append(
            {
                "rank": rank,
                "cluster_id": c.cluster_id,
                "n_points": c.n_points,
                "centroid_lon": round(c.centroid_lon, 6),
                "centroid_lat": round(c.centroid_lat, 6),
                "radius_meters": round(c.radius_meters, 1),
                "median_distance_meters": round(c.median_distance_meters, 1),
                "pct_dead": round(c.pct_dead, 4),
                "pct_severe_or_dead": round(c.pct_severe_or_dead, 4),
                "severity_distribution": c.severity_distribution,
                "top_em_types": c.top_em_types,
                "top_np": c.top_np,
                "top_streets": c.top_streets,
                "n_accidents_listed": len(c.accident_ids),
                # Сохраняем только первые 10 ID для drill-down — остальные в geojson не пишем,
                # их можно достать из БД по WHERE cluster_id (если нужно сохранять привязку)
                "accident_ids_sample": c.accident_ids[:10],
            }
        )

    # Распределение размеров кластеров (для sensitivity-анализа)
    sizes = [c.n_points for c in result.clusters]
    summary = {
        "params": {
            "eps_meters": result.eps_meters,
            "min_samples": result.min_samples,
            "region_name": result.region_name,
            "projection": result.projection,
        },
        "stats": {
            "total_points": result.total_points,
            "clustered_count": result.clustered_count,
            "noise_count": result.noise_count,
            "noise_pct": (
                round(result.noise_count / result.total_points, 4) if result.total_points else 0.0
            ),
            "clusters_count": len(result.clusters),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "size_max": max(sizes) if sizes else 0,
            "size_median": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
            "size_min": min(sizes) if sizes else 0,
        },
        "top_clusters": top_clusters,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=300.0, help="DBSCAN eps in meters")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--region-name", default="primorye_krai_land")
    parser.add_argument(
        "--top-n", type=int, default=30, help="How many top clusters in summary JSON"
    )
    parser.add_argument(
        "--output-prefix",
        default="hotspots",
        help="Prefix for output files in data/processed/ (default: hotspots → hotspots.geojson + hotspots_summary.json)",
    )
    args = parser.parse_args()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / f"{args.output_prefix}.geojson"
    summary_path = out_dir / f"{args.output_prefix}_summary.json"

    with SessionLocal() as session:
        result = find_hotspots(
            session,
            eps_meters=args.eps,
            min_samples=args.min_samples,
            region_name=args.region_name,
        )

    geojson_obj = result_to_geojson(result)
    summary_obj = result_to_summary(result, top_n=args.top_n)

    geojson_path.write_text(
        json.dumps(geojson_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Saved {} ({:.1f} kB) and {} ({:.1f} kB)",
        geojson_path,
        geojson_path.stat().st_size / 1024,
        summary_path,
        summary_path.stat().st_size / 1024,
    )

    # Краткая сводка top-5 для лога
    logger.info("=== TOP-5 hotspots (by size) ===")
    for c in result.clusters[:5]:
        top_street = c.top_streets[0][0] if c.top_streets else "—"
        logger.info(
            "  cid={:3} n={:>5} pct_dead={:5.2%} | {:<60} | top: {}",
            c.cluster_id,
            c.n_points,
            c.pct_dead,
            f"{c.centroid_lon:.4f}, {c.centroid_lat:.4f}",
            top_street,
        )


if __name__ == "__main__":
    main()
