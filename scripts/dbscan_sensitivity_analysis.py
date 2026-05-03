"""
DBSCAN parameter sensitivity analysis on accident hotspots.

Что делает: запускает DBSCAN-кластеризацию на 28 020 ДТП Приморья
для 4 значений eps (100м, 200м, 300м, 500м) при фиксированном
min_samples=15 и собирает сравнительную таблицу — число кластеров,
доля шумовых точек, размер крупнейшего/30-го/медианного кластера,
эффект «слипания» Владивостока. Полные результаты каждого запуска
сохраняются в `data/processed/hotspots_eps{N}.geojson` и
итоговая сравнительная сводка — в `data/processed/dbscan_sensitivity.json`.

Зачем нужно: показывает «правильный» масштаб eps для содержательной
интерпретации очагов аварийности. С eps=300м весь центр Владивостока
сливается в один кластер на 6 800 точек, что бесполезно для
выявления конкретных опасных участков. С eps=100-150м эта «клякса»
распадается на десятки локальных очагов с реальными адресами
(Золотой мост, объездная Артёма, Гоголя, Снеговая) — то, что
нужно для собеседования.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from scripts.run_dbscan_hotspots import result_to_geojson
from src.analysis.dbscan_clustering import find_hotspots
from src.database import SessionLocal


def main() -> None:
    eps_values = [100.0, 200.0, 300.0, 500.0]
    min_samples = 15

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    with SessionLocal() as session:
        for eps in eps_values:
            logger.info("=" * 70)
            logger.info(">>> Running DBSCAN with eps={}m, min_samples={}", eps, min_samples)
            result = find_hotspots(
                session,
                eps_meters=eps,
                min_samples=min_samples,
            )

            # GeoJSON для каждого варианта
            geojson_path = out_dir / f"hotspots_eps{int(eps)}.geojson"
            geojson_path.write_text(
                json.dumps(result_to_geojson(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Сводка для сравнительной таблицы
            sizes = sorted([c.n_points for c in result.clusters], reverse=True)
            top1 = result.clusters[0] if result.clusters else None
            summary = {
                "eps_meters": eps,
                "min_samples": min_samples,
                "clusters_count": len(result.clusters),
                "noise_count": result.noise_count,
                "noise_pct": round(
                    result.noise_count / result.total_points if result.total_points else 0,
                    4,
                ),
                "clustered_count": result.clustered_count,
                "size_max": sizes[0] if sizes else 0,
                "size_30": sizes[29] if len(sizes) >= 30 else None,
                "size_median": sizes[len(sizes) // 2] if sizes else 0,
                "size_min": sizes[-1] if sizes else 0,
                "top1_centroid": [
                    round(top1.centroid_lon, 4) if top1 else None,
                    round(top1.centroid_lat, 4) if top1 else None,
                ]
                if top1
                else [None, None],
                "top1_n_points": top1.n_points if top1 else 0,
                "top1_top_street": (
                    top1.top_streets[0][0] if (top1 and top1.top_streets) else None
                ),
                "top1_pct_dead": round(top1.pct_dead, 4) if top1 else None,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "geojson_path": str(geojson_path),
            }
            summaries.append(summary)

    # Сохраняем сравнительную сводку
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_samples_fixed": min_samples,
        "eps_swept": eps_values,
        "results": summaries,
    }
    summary_path = out_dir / "dbscan_sensitivity.json"
    summary_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved {}", summary_path)

    # Печатаем сравнительную таблицу
    logger.info("=" * 90)
    logger.info(">>> SENSITIVITY ANALYSIS — DBSCAN eps sweep (min_samples=15) <<<")
    logger.info("=" * 90)
    logger.info(
        "  {:>6} | {:>8} | {:>8} | {:>9} | {:>8} | {:>9} | {}",
        "eps",
        "clusters",
        "noise%",
        "size_max",
        "size_30",
        "top1_dead",
        "top1_top_street",
    )
    logger.info("-" * 90)
    for s in summaries:
        logger.info(
            "  {:>6.0f} | {:>8} | {:>7.1%}  | {:>9} | {:>8} | {:>9} | {}",
            s["eps_meters"],
            s["clusters_count"],
            s["noise_pct"],
            s["size_max"],
            s["size_30"] if s["size_30"] is not None else "—",
            f"{s['top1_pct_dead']:.2%}" if s["top1_pct_dead"] is not None else "—",
            s["top1_top_street"] or "—",
        )


if __name__ == "__main__":
    main()
