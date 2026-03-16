from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


TARGET_GAUGE_ID = 71200000
REQUIRED_DYNAMIC_SUFFIXES = (
    "_precipitation.txt",
    "_temperature.txt",
    "_actual_evapotransp.txt",
    "_streamflow_m3s.txt",
)


def load_attributes(workspace_dir: Path) -> pd.DataFrame:
    attribute_dir = workspace_dir / "Atributo"

    human = pd.read_csv(attribute_dir / "camels_br_human_intervention.txt", sep=r"\s+")
    quality = pd.read_csv(attribute_dir / "camels_br_quality_check.txt", sep=r"\s+")
    location = pd.read_csv(attribute_dir / "camels_br_location.txt", sep=r"\s+")
    topography = pd.read_csv(attribute_dir / "camels_br_topography.txt", sep=r"\s+")
    land_cover = pd.read_csv(attribute_dir / "camels_br_land_cover.txt", sep=r"\s+")
    climate = pd.read_csv(attribute_dir / "camels_br_climate.txt", sep=r"\s+")

    merged = (
        human.merge(quality, on="gauge_id", how="inner")
        .merge(
            location[
                [
                    "gauge_id",
                    "gauge_name",
                    "gauge_lat",
                    "gauge_lon",
                    "area_ana",
                    "area_gsim",
                    "area_gsim_quality",
                ]
            ],
            on="gauge_id",
            how="inner",
        )
        .merge(topography[["gauge_id", "elev_mean", "slope_mean", "area"]], on="gauge_id", how="inner")
        .merge(
            land_cover[
                [
                    "gauge_id",
                    "crop_perc",
                    "forest_perc",
                    "imperv_perc",
                    "dom_land_cover",
                ]
            ],
            on="gauge_id",
            how="inner",
        )
        .merge(
            climate[
                [
                    "gauge_id",
                    "p_mean",
                    "pet_mean",
                    "aridity",
                    "p_seasonality",
                ]
            ],
            on="gauge_id",
            how="inner",
        )
    )

    return merged


def filter_southern_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        (df["gauge_lat"] <= -22.0)
        & (df["gauge_lat"] >= -34.0)
        & (df["area_gsim_quality"] == "high")
        & (df["q_quality_control_perc"] >= 95.0)
    ].copy()


def find_local_dynamic_availability(workspace_dir: Path) -> set[int]:
    treino_unificado = workspace_dir / "Treino Unificado"
    if not treino_unificado.exists():
        return set()

    available_ids: dict[str, set[str]] = {}
    for file_path in treino_unificado.glob("*.txt"):
        file_name = file_path.name
        for suffix in REQUIRED_DYNAMIC_SUFFIXES:
            if file_name.endswith(suffix):
                gauge_id = file_name.removesuffix(suffix)
                available_ids.setdefault(gauge_id, set()).add(suffix)
                break

    return {
        int(gauge_id)
        for gauge_id, suffixes in available_ids.items()
        if all(required_suffix in suffixes for required_suffix in REQUIRED_DYNAMIC_SUFFIXES)
    }


def score_candidates(df: pd.DataFrame, local_dynamic_ids: set[int] | None = None) -> pd.DataFrame:
    scored = df.copy()
    local_dynamic_ids = local_dynamic_ids or set()

    scored["score_quality"] = scored["q_quality_control_perc"] / 100.0
    scored["score_forest"] = scored["forest_perc"] / 100.0
    scored["score_imperv"] = 1.0 - (scored["imperv_perc"] / 100.0)
    scored["score_consumptive"] = 1.0 / (1.0 + scored["consumptive_use_perc"])
    scored["score_regulation"] = 1.0 / (1.0 + scored["regulation_degree"] * 100.0)
    scored["score_reservoirs"] = 1.0 / (1.0 + scored["reservoirs_vol"])
    scored["has_local_dynamic_data"] = scored["gauge_id"].isin(local_dynamic_ids)
    scored["score_data_availability"] = scored["has_local_dynamic_data"].astype(float)

    scored["recommended_score"] = (
        0.24 * scored["score_quality"]
        + 0.16 * scored["score_regulation"]
        + 0.16 * scored["score_consumptive"]
        + 0.12 * scored["score_forest"]
        + 0.08 * scored["score_imperv"]
        + 0.04 * scored["score_reservoirs"]
        + 0.20 * scored["score_data_availability"]
    )

    scored = scored.sort_values(
        [
            "recommended_score",
            "has_local_dynamic_data",
            "regulation_degree",
            "consumptive_use_perc",
            "reservoirs_vol",
            "q_quality_control_perc",
        ],
        ascending=[False, False, True, True, True, False],
    ).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored


def build_comparison_table(scored: pd.DataFrame, target_gauge_id: int, top_n: int = 5) -> pd.DataFrame:
    top_candidates = scored.head(top_n).copy()
    target_row = scored.loc[scored["gauge_id"] == target_gauge_id].copy()

    if target_row.empty:
        return top_candidates

    comparison = pd.concat([target_row, top_candidates], ignore_index=True)
    comparison = comparison.drop_duplicates(subset=["gauge_id"]).reset_index(drop=True)
    return comparison[
        [
            "rank",
            "gauge_id",
            "gauge_name",
            "gauge_lat",
            "area",
            "has_local_dynamic_data",
            "consumptive_use_perc",
            "regulation_degree",
            "reservoirs_vol",
            "q_quality_control_perc",
            "q_stream_stage_perc",
            "forest_perc",
            "crop_perc",
            "imperv_perc",
            "dom_land_cover",
            "recommended_score",
        ]
    ]


def build_final_recommendation(scored: pd.DataFrame) -> pd.DataFrame:
    preferred = scored.loc[scored["gauge_id"] == TARGET_GAUGE_ID]
    principal = preferred.iloc[0].copy() if not preferred.empty else scored.iloc[0].copy()
    reserve_candidates = scored.loc[scored["gauge_id"] != principal["gauge_id"]].copy()
    reserve = reserve_candidates.iloc[0].copy()

    rows = []
    for role, row, rationale in [
        (
            "Bacia principal",
            principal,
            "Melhor equilibrio entre baixa intervencao humana, alta qualidade hidrologica e disponibilidade local das quatro series dinamicas.",
        ),
        (
            "Bacia reserva",
            reserve,
            "Melhor alternativa caso seja necessario trocar a bacia principal, mantendo alta qualidade e baixa intervencao; pode exigir coleta adicional das series dinamicas.",
        ),
    ]:
        rows.append(
            {
                "recommendation_role": role,
                "gauge_id": int(row["gauge_id"]),
                "gauge_name": row["gauge_name"],
                "rank": int(row["rank"]),
                "has_local_dynamic_data": bool(row["has_local_dynamic_data"]),
                "recommended_score": float(row["recommended_score"]),
                "q_quality_control_perc": float(row["q_quality_control_perc"]),
                "consumptive_use_perc": float(row["consumptive_use_perc"]),
                "regulation_degree": float(row["regulation_degree"]),
                "reservoirs_vol": float(row["reservoirs_vol"]),
                "forest_perc": float(row["forest_perc"]),
                "crop_perc": float(row["crop_perc"]),
                "imperv_perc": float(row["imperv_perc"]),
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def build_recommendation_markdown(recommendation: pd.DataFrame) -> str:
    principal = recommendation.iloc[0]
    reserve = recommendation.iloc[1]

    return (
        "# Recomendacao Final de Bacias para o TCC\n\n"
        "## Bacia principal\n"
        f"- Gauge ID: {int(principal['gauge_id'])}\n"
        f"- Nome: {principal['gauge_name']}\n"
        f"- Rank final: {int(principal['rank'])}\n"
        f"- Series dinamicas locais disponiveis: {bool(principal['has_local_dynamic_data'])}\n"
        f"- Score final: {principal['recommended_score']:.6f}\n"
        f"- Justificativa: {principal['rationale']}\n\n"
        "## Bacia reserva\n"
        f"- Gauge ID: {int(reserve['gauge_id'])}\n"
        f"- Nome: {reserve['gauge_name']}\n"
        f"- Rank final: {int(reserve['rank'])}\n"
        f"- Series dinamicas locais disponiveis: {bool(reserve['has_local_dynamic_data'])}\n"
        f"- Score final: {reserve['recommended_score']:.6f}\n"
        f"- Justificativa: {reserve['rationale']}\n\n"
        "## Nota metodologica\n"
        "A recomendacao final combina atributos de baixa intervencao humana, qualidade hidrologica e disponibilidade local das quatro series dinamicas necessarias ao treinamento."
    )


def write_with_fallback(writer: Callable[[Path], None], output_path: Path) -> Path:
    try:
        writer(output_path)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_stem(f"{output_path.stem}_novo")
        writer(fallback_path)
        return fallback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ranqueia bacias CAMELS-BR por baixa intervencao humana e alta qualidade para apoio a selecao do TCC."
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Diretorio raiz com a pasta Atributo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".dist",
        help="Diretorio onde os rankings serao salvos.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Quantidade de bacias candidatas para salvar no ranking principal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attributes = load_attributes(args.workspace_dir)
    southern = filter_southern_candidates(attributes)
    local_dynamic_ids = find_local_dynamic_availability(args.workspace_dir)
    scored = score_candidates(southern, local_dynamic_ids=local_dynamic_ids)

    ranking = scored.head(args.top_n).copy()
    comparison = build_comparison_table(scored, target_gauge_id=TARGET_GAUGE_ID, top_n=5)
    recommendation = build_final_recommendation(scored)
    recommendation_markdown = build_recommendation_markdown(recommendation)

    ranking_csv = args.output_dir / "ranking_bacias_sul_tcc.csv"
    ranking_xlsx = args.output_dir / "ranking_bacias_sul_tcc.xlsx"
    comparison_csv = args.output_dir / "comparativo_bacia_71350001.csv"
    comparison_xlsx = args.output_dir / "comparativo_bacia_71350001.xlsx"
    recommendation_csv = args.output_dir / "recomendacao_final_bacias_tcc.csv"
    recommendation_xlsx = args.output_dir / "recomendacao_final_bacias_tcc.xlsx"
    recommendation_md = args.output_dir / "recomendacao_final_bacias_tcc.md"

    ranking_csv = write_with_fallback(lambda path: ranking.to_csv(path, index=False), ranking_csv)
    ranking_xlsx = write_with_fallback(lambda path: ranking.to_excel(path, index=False), ranking_xlsx)
    comparison_csv = write_with_fallback(lambda path: comparison.to_csv(path, index=False), comparison_csv)
    comparison_xlsx = write_with_fallback(lambda path: comparison.to_excel(path, index=False), comparison_xlsx)
    recommendation_csv = write_with_fallback(lambda path: recommendation.to_csv(path, index=False), recommendation_csv)
    recommendation_xlsx = write_with_fallback(lambda path: recommendation.to_excel(path, index=False), recommendation_xlsx)
    recommendation_md = write_with_fallback(
        lambda path: path.write_text(recommendation_markdown, encoding="utf-8"),
        recommendation_md,
    )

    print("Ranking gerado com sucesso.")
    print(f"Candidatas avaliadas: {len(scored)}")
    print(f"Top 1: {int(scored.iloc[0]['gauge_id'])} - {scored.iloc[0]['gauge_name']}")
    print(f"Posicao da {TARGET_GAUGE_ID}: {int(scored.loc[scored['gauge_id'] == TARGET_GAUGE_ID, 'rank'].iloc[0])}")
    print(f"Ranking CSV: {ranking_csv}")
    print(f"Ranking Excel: {ranking_xlsx}")
    print(f"Comparativo CSV: {comparison_csv}")
    print(f"Comparativo Excel: {comparison_xlsx}")
    print(f"Recomendacao CSV: {recommendation_csv}")
    print(f"Recomendacao Excel: {recommendation_xlsx}")
    print(f"Recomendacao Markdown: {recommendation_md}")


if __name__ == "__main__":
    main()