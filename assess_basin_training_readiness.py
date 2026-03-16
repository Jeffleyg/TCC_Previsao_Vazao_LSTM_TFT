from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReadinessThresholds:
    required_start_date: pd.Timestamp
    required_end_date: pd.Timestamp
    max_missing_pct_streamflow: float
    max_missing_pct_precip: float
    max_missing_pct_temperature: float
    max_missing_pct_evapotransp: float
    min_q_quality_control_perc: float
    max_consumptive_use_perc: float
    max_regulation_degree: float
    max_reservoirs_vol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia se uma bacia CAMELS-BR esta pronta para treino de modelos hidrologicos (LSTM/TFT)."
    )
    parser.add_argument("--gauge-id", type=str, required=True, help="Codigo da bacia/estacao (ex.: 71200000).")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Diretorio raiz com as pastas Atributo e Treino Unificado.",
    )
    parser.add_argument(
        "--required-start-date",
        type=str,
        default="1980-01-01",
        help="Data inicial minima exigida para cobertura temporal.",
    )
    parser.add_argument(
        "--required-end-date",
        type=str,
        default="2018-12-31",
        help="Data final minima exigida para cobertura temporal (padrao: 2018-12-31).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / ".dist")
    return parser.parse_args()


def read_dynamic_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", na_values=["nan", "NaN"]) 
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df


def resolve_dynamic_files(workspace_dir: Path, gauge_id: str) -> dict[str, Path]:
    base = workspace_dir / "Treino Unificado"
    return {
        "precipitation": base / f"{gauge_id}_precipitation.txt",
        "temperature": base / f"{gauge_id}_temperature.txt",
        "evapotransp": base / f"{gauge_id}_actual_evapotransp.txt",
        "streamflow": base / f"{gauge_id}_streamflow_m3s.txt",
    }


def extract_primary_missing_pct(variable_name: str, df: pd.DataFrame) -> float:
    if variable_name == "streamflow":
        return float(df["streamflow_m3s"].isna().mean() * 100.0)

    if variable_name == "precipitation":
        cols = [c for c in df.columns if c.startswith("p_")]
    elif variable_name == "temperature":
        cols = [c for c in df.columns if c.startswith("t")]
    else:
        cols = [c for c in df.columns if c.startswith("aet_")]

    if not cols:
        return 100.0

    col_missing = {col: float(df[col].isna().mean() * 100.0) for col in cols}
    return min(col_missing.values())


def evaluate_dynamic_data(
    dynamic_files: dict[str, Path],
    thresholds: ReadinessThresholds,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, object]] = []
    all_ok = True

    for variable_name, file_path in dynamic_files.items():
        exists = file_path.exists()
        if not exists:
            rows.append(
                {
                    "variable": variable_name,
                    "file_exists": False,
                    "rows": 0,
                    "date_min": pd.NaT,
                    "date_max": pd.NaT,
                    "coverage_start_ok": False,
                    "coverage_end_ok": False,
                    "missing_pct_best_source": 100.0,
                    "variable_ready": False,
                }
            )
            all_ok = False
            continue

        df = read_dynamic_file(file_path)
        date_min = df["date"].min()
        date_max = df["date"].max()
        coverage_start_ok = bool(pd.notna(date_min) and date_min <= thresholds.required_start_date)
        coverage_end_ok = bool(pd.notna(date_max) and date_max >= thresholds.required_end_date)
        missing_pct = extract_primary_missing_pct(variable_name, df)

        if variable_name == "streamflow":
            missing_ok = missing_pct <= thresholds.max_missing_pct_streamflow
        elif variable_name == "precipitation":
            missing_ok = missing_pct <= thresholds.max_missing_pct_precip
        elif variable_name == "temperature":
            missing_ok = missing_pct <= thresholds.max_missing_pct_temperature
        else:
            missing_ok = missing_pct <= thresholds.max_missing_pct_evapotransp

        variable_ready = bool(coverage_start_ok and coverage_end_ok and missing_ok)
        all_ok = all_ok and variable_ready

        rows.append(
            {
                "variable": variable_name,
                "file_exists": True,
                "rows": int(len(df)),
                "date_min": date_min,
                "date_max": date_max,
                "coverage_start_ok": coverage_start_ok,
                "coverage_end_ok": coverage_end_ok,
                "missing_pct_best_source": round(missing_pct, 4),
                "variable_ready": variable_ready,
            }
        )

    return pd.DataFrame(rows), all_ok


def evaluate_human_intervention(workspace_dir: Path, gauge_id: str, thresholds: ReadinessThresholds) -> tuple[pd.DataFrame, bool]:
    attr_dir = workspace_dir / "Atributo"
    human = pd.read_csv(attr_dir / "camels_br_human_intervention.txt", sep=r"\s+")
    quality = pd.read_csv(attr_dir / "camels_br_quality_check.txt", sep=r"\s+")
    location = pd.read_csv(attr_dir / "camels_br_location.txt", sep=r"\s+")
    land = pd.read_csv(attr_dir / "camels_br_land_cover.txt", sep=r"\s+")

    merged = (
        human.merge(quality, on="gauge_id", how="inner")
        .merge(location[["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area_gsim_quality"]], on="gauge_id", how="left")
        .merge(land[["gauge_id", "forest_perc", "crop_perc", "imperv_perc", "dom_land_cover"]], on="gauge_id", how="left")
    )
    row = merged.loc[merged["gauge_id"] == int(gauge_id)]

    if row.empty:
        return pd.DataFrame([{"gauge_id": gauge_id, "attribute_found": False}]), False

    r = row.iloc[0]
    q_ok = bool(float(r["q_quality_control_perc"]) >= thresholds.min_q_quality_control_perc)
    c_ok = bool(float(r["consumptive_use_perc"]) <= thresholds.max_consumptive_use_perc)
    reg_ok = bool(float(r["regulation_degree"]) <= thresholds.max_regulation_degree)
    res_ok = bool(float(r["reservoirs_vol"]) <= thresholds.max_reservoirs_vol)

    summary = pd.DataFrame(
        [
            {
                "gauge_id": int(gauge_id),
                "attribute_found": True,
                "gauge_name": r.get("gauge_name", ""),
                "q_quality_control_perc": float(r["q_quality_control_perc"]),
                "consumptive_use_perc": float(r["consumptive_use_perc"]),
                "regulation_degree": float(r["regulation_degree"]),
                "reservoirs_vol": float(r["reservoirs_vol"]),
                "forest_perc": float(r.get("forest_perc", 0.0)),
                "crop_perc": float(r.get("crop_perc", 0.0)),
                "imperv_perc": float(r.get("imperv_perc", 0.0)),
                "dom_land_cover": r.get("dom_land_cover", ""),
                "q_quality_ok": q_ok,
                "consumptive_ok": c_ok,
                "regulation_ok": reg_ok,
                "reservoir_ok": res_ok,
                "human_intervention_ready": bool(q_ok and c_ok and reg_ok and res_ok),
            }
        ]
    )

    return summary, bool(q_ok and c_ok and reg_ok and res_ok)


def build_decision_table(
    gauge_id: str,
    dynamic_ok: bool,
    intervention_ok: bool,
    dynamic_summary: pd.DataFrame,
    intervention_summary: pd.DataFrame,
) -> pd.DataFrame:
    if dynamic_ok and intervention_ok:
        verdict = "APROVADA"
        rationale = "A bacia atende cobertura temporal, faltantes maximos e criterios de intervencao humana."
    elif (not dynamic_ok) and intervention_ok:
        verdict = "REPROVADA_DADOS_DINAMICOS"
        rationale = "Atributos de intervencao estao bons, mas as series dinamicas nao atendem cobertura/faltantes."
    elif dynamic_ok and (not intervention_ok):
        verdict = "REPROVADA_INTERVENCAO"
        rationale = "Series dinamicas estao boas, mas a intervencao humana/qualidade nao atende os criterios definidos."
    else:
        verdict = "REPROVADA"
        rationale = "A bacia nao atende aos criterios minimos de dados dinamicos e/ou intervencao humana."

    return pd.DataFrame(
        [
            {
                "gauge_id": gauge_id,
                "verdict": verdict,
                "dynamic_data_ready": dynamic_ok,
                "human_intervention_ready": intervention_ok,
                "missing_dynamic_files": int((~dynamic_summary["file_exists"]).sum()) if "file_exists" in dynamic_summary else 4,
                "rationale": rationale,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = ReadinessThresholds(
        required_start_date=pd.Timestamp(args.required_start_date),
        required_end_date=pd.Timestamp(args.required_end_date),
        max_missing_pct_streamflow=5.0,
        max_missing_pct_precip=5.0,
        max_missing_pct_temperature=5.0,
        max_missing_pct_evapotransp=5.0,
        min_q_quality_control_perc=95.0,
        max_consumptive_use_perc=0.20,
        max_regulation_degree=0.01,
        max_reservoirs_vol=0.0,
    )

    dynamic_files = resolve_dynamic_files(args.workspace_dir, args.gauge_id)
    dynamic_summary, dynamic_ok = evaluate_dynamic_data(dynamic_files, thresholds)
    intervention_summary, intervention_ok = evaluate_human_intervention(args.workspace_dir, args.gauge_id, thresholds)
    decision = build_decision_table(args.gauge_id, dynamic_ok, intervention_ok, dynamic_summary, intervention_summary)

    base_name = f"readiness_{args.gauge_id}"
    dynamic_csv = args.output_dir / f"{base_name}_dynamic_summary.csv"
    intervention_csv = args.output_dir / f"{base_name}_intervention_summary.csv"
    decision_csv = args.output_dir / f"{base_name}_decision.csv"
    report_xlsx = args.output_dir / f"{base_name}_report.xlsx"

    dynamic_summary.to_csv(dynamic_csv, index=False)
    intervention_summary.to_csv(intervention_csv, index=False)
    decision.to_csv(decision_csv, index=False)

    with pd.ExcelWriter(report_xlsx) as writer:
        dynamic_summary.to_excel(writer, sheet_name="dynamic_summary", index=False)
        intervention_summary.to_excel(writer, sheet_name="intervention_summary", index=False)
        decision.to_excel(writer, sheet_name="decision", index=False)

    print("Avaliacao concluida.")
    print(f"Gauge ID: {args.gauge_id}")
    print(f"Veredito: {decision.iloc[0]['verdict']}")
    print(f"Decision CSV: {decision_csv}")
    print(f"Report Excel: {report_xlsx}")


if __name__ == "__main__":
    main()
