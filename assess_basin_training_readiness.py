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
    streamflow_event_quantile: float
    streamflow_event_min_duration_days: int
    lstm_lookback_days: int
    lstm_lookback_auto_from_events: bool


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
    parser.add_argument(
        "--streamflow-event-quantile",
        type=float,
        default=0.90,
        help="Quantil usado como limiar para detectar eventos de vazao (padrao: 0.90).",
    )
    parser.add_argument(
        "--streamflow-event-min-duration-days",
        type=int,
        default=1,
        help="Duracao minima, em dias, para considerar um evento de vazao (padrao: 1).",
    )
    parser.add_argument(
        "--lstm-lookback-days",
        type=int,
        default=30,
        help="Quantidade de dias anteriores ao inicio do evento usada como entrada da LSTM (padrao: 30).",
    )
    parser.add_argument(
        "--lstm-lookback-auto-from-events",
        action="store_true",
        help=(
            "Se ativo, define automaticamente a janela de entrada da LSTM pela recorrencia media dos eventos "
            "(dias entre inicios de eventos)."
        ),
    )
    return parser.parse_args()


def read_dynamic_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", na_values=["nan", "NaN"]) 
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df


def detect_streamflow_events(
    streamflow_file: Path,
    event_quantile: float,
    min_duration_days: int,
    analysis_start_date: pd.Timestamp,
    analysis_end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_columns = [
        "event_id",
        "event_start_date",
        "event_end_date",
        "peak_date",
        "peak_streamflow_m3s",
        "event_duration_days",
        "event_mean_streamflow_m3s",
    ]
    summary_columns = [
        "analysis_start_date",
        "analysis_end_date",
        "streamflow_start_date",
        "streamflow_peak_date",
        "streamflow_peak_m3s",
        "days_from_streamflow_start_to_peak",
        "event_count",
        "average_event_duration_days",
        "average_inter_event_gap_days",
        "median_inter_event_gap_days",
        "average_days_between_event_starts",
        "median_days_between_event_starts",
        "event_recurrence_every_n_days",
        "event_frequency_per_year",
        "average_peak_streamflow_m3s",
        "peak_flow_date",
        "peak_flow_streamflow_m3s",
        "first_event_start_date",
        "last_event_end_date",
        "event_threshold_streamflow_m3s",
        "event_quantile",
        "min_duration_days",
        "suggested_lstm_lookback_days",
    ]

    if not streamflow_file.exists():
        return pd.DataFrame(columns=event_columns), pd.DataFrame([{col: pd.NA for col in summary_columns}])

    df = read_dynamic_file(streamflow_file)
    if "streamflow_m3s" not in df.columns:
        raise ValueError(f"Arquivo de vazao sem coluna streamflow_m3s: {streamflow_file}")

    series = df.loc[df["date"].notna(), ["date", "streamflow_m3s"]].copy()
    series = series.sort_values("date").reset_index(drop=True)
    series = series.loc[
        (series["date"] >= analysis_start_date) & (series["date"] <= analysis_end_date)
    ].copy()

    if series.empty:
        empty_summary = {col: pd.NA for col in summary_columns}
        empty_summary["analysis_start_date"] = analysis_start_date
        empty_summary["analysis_end_date"] = analysis_end_date
        empty_summary["event_count"] = 0
        return pd.DataFrame(columns=event_columns), pd.DataFrame([empty_summary])

    valid_values = series["streamflow_m3s"].dropna()

    if valid_values.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame([{col: pd.NA for col in summary_columns}])

    series_with_flow = series.loc[series["streamflow_m3s"].notna()].copy()
    streamflow_start_date = series_with_flow["date"].min()
    streamflow_peak_idx = series_with_flow["streamflow_m3s"].idxmax()
    streamflow_peak_row = series_with_flow.loc[streamflow_peak_idx]
    streamflow_peak_date = streamflow_peak_row["date"]
    streamflow_peak_m3s = float(streamflow_peak_row["streamflow_m3s"])
    days_start_to_peak = int((streamflow_peak_date - streamflow_start_date).days)

    threshold = float(valid_values.quantile(event_quantile))
    if pd.isna(threshold):
        threshold = float(valid_values.max())

    is_event_day = series["streamflow_m3s"].fillna(-float("inf")) >= threshold
    event_group = (is_event_day != is_event_day.shift(fill_value=False)).cumsum()

    event_rows: list[dict[str, object]] = []
    event_id = 0
    for _, group in series.loc[is_event_day].groupby(event_group[is_event_day]):
        event_start = group["date"].min()
        event_end = group["date"].max()
        duration_days = int((event_end - event_start).days) + 1

        if duration_days < min_duration_days:
            continue

        peak_idx = group["streamflow_m3s"].idxmax()
        peak_row = group.loc[peak_idx]
        event_id += 1
        event_rows.append(
            {
                "event_id": event_id,
                "event_start_date": event_start,
                "event_end_date": event_end,
                "peak_date": peak_row["date"],
                "peak_streamflow_m3s": float(peak_row["streamflow_m3s"]),
                "event_duration_days": duration_days,
                "event_mean_streamflow_m3s": float(group["streamflow_m3s"].mean()),
            }
        )

    events = pd.DataFrame(event_rows, columns=event_columns)
    if events.empty:
        empty_summary = {col: pd.NA for col in summary_columns}
        empty_summary["event_count"] = 0
        return events, pd.DataFrame([empty_summary])

    events = events.sort_values("event_start_date").reset_index(drop=True)
    events["previous_event_end_date"] = events["event_end_date"].shift(1)
    events["next_event_start_date"] = events["event_start_date"].shift(-1)
    events["days_between_event_starts"] = (
        events["event_start_date"] - events["event_start_date"].shift(1)
    ).dt.days
    events["inter_event_gap_days"] = (
        events["event_start_date"] - events["previous_event_end_date"]
    ).dt.days - 1
    events["days_until_next_event_start"] = (
        events["next_event_start_date"] - events["event_start_date"]
    ).dt.days
    events["days_until_next_event_from_end"] = (
        events["next_event_start_date"] - events["event_end_date"]
    ).dt.days - 1

    max_peak_idx = events["peak_streamflow_m3s"].idxmax()
    max_peak_row = events.loc[max_peak_idx]
    inter_event_gaps = events["inter_event_gap_days"].dropna()
    between_starts = events["days_between_event_starts"].dropna()

    if between_starts.empty:
        recurrence_days = pd.NA
    else:
        recurrence_days = round(float(between_starts.mean()), 4)

    span_days = int((events["event_end_date"].max() - events["event_start_date"].min()).days) + 1
    if span_days <= 0:
        event_frequency_per_year = pd.NA
    else:
        event_frequency_per_year = round(float(len(events) / (span_days / 365.25)), 4)

    suggested_lookback_days = pd.NA
    if pd.notna(recurrence_days):
        suggested_lookback_days = max(1, int(round(float(recurrence_days))))

    summary = pd.DataFrame(
        [
            {
                "analysis_start_date": analysis_start_date,
                "analysis_end_date": analysis_end_date,
                "streamflow_start_date": streamflow_start_date,
                "streamflow_peak_date": streamflow_peak_date,
                "streamflow_peak_m3s": round(streamflow_peak_m3s, 4),
                "days_from_streamflow_start_to_peak": days_start_to_peak,
                "event_count": int(len(events)),
                "average_event_duration_days": round(float(events["event_duration_days"].mean()), 4),
                "average_inter_event_gap_days": round(float(inter_event_gaps.mean()), 4) if not inter_event_gaps.empty else pd.NA,
                "median_inter_event_gap_days": round(float(inter_event_gaps.median()), 4) if not inter_event_gaps.empty else pd.NA,
                "average_days_between_event_starts": recurrence_days,
                "median_days_between_event_starts": round(float(between_starts.median()), 4) if not between_starts.empty else pd.NA,
                "event_recurrence_every_n_days": recurrence_days,
                "event_frequency_per_year": event_frequency_per_year,
                "average_peak_streamflow_m3s": round(float(events["peak_streamflow_m3s"].mean()), 4),
                "peak_flow_date": max_peak_row["peak_date"],
                "peak_flow_streamflow_m3s": round(float(max_peak_row["peak_streamflow_m3s"]), 4),
                "first_event_start_date": events["event_start_date"].min(),
                "last_event_end_date": events["event_end_date"].max(),
                "event_threshold_streamflow_m3s": round(threshold, 4),
                "event_quantile": event_quantile,
                "min_duration_days": min_duration_days,
                "suggested_lstm_lookback_days": suggested_lookback_days,
            }
        ]
    )
    return events, summary


def select_lstm_lookback_days(
    event_summary: pd.DataFrame,
    default_lookback_days: int,
    auto_from_events: bool,
) -> tuple[int, str]:
    if (not auto_from_events) or event_summary.empty:
        return default_lookback_days, "fixed_default"

    suggested = event_summary.iloc[0].get("suggested_lstm_lookback_days", pd.NA)
    if pd.isna(suggested):
        return default_lookback_days, "fixed_default_no_event_recurrence"

    return int(suggested), "auto_event_recurrence"


def build_lstm_training_windows(events: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    window_columns = [
        "event_id",
        "input_start_date",
        "input_end_date",
        "target_start_date",
        "target_end_date",
        "input_window_days",
        "target_window_days",
        "event_start_date",
        "event_end_date",
        "peak_date",
        "peak_streamflow_m3s",
    ]

    if events.empty:
        return pd.DataFrame(columns=window_columns)

    windows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        target_start = pd.Timestamp(event["event_start_date"])
        target_end = pd.Timestamp(event["event_end_date"])
        input_end = target_start - pd.Timedelta(days=1)
        input_start = target_start - pd.Timedelta(days=lookback_days)

        windows.append(
            {
                "event_id": int(event["event_id"]),
                "input_start_date": input_start,
                "input_end_date": input_end,
                "target_start_date": target_start,
                "target_end_date": target_end,
                "input_window_days": int((input_end - input_start).days) + 1,
                "target_window_days": int((target_end - target_start).days) + 1,
                "event_start_date": target_start,
                "event_end_date": target_end,
                "peak_date": event["peak_date"],
                "peak_streamflow_m3s": float(event["peak_streamflow_m3s"]),
            }
        )

    return pd.DataFrame(windows, columns=window_columns)


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


def write_with_fallback(writer, output_path: Path) -> Path:
    try:
        writer(output_path)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_stem(f"{output_path.stem}_novo")
        writer(fallback_path)
        return fallback_path


def can_write_to_path(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("w", encoding="utf-8"):
                pass
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    try:
        with path.open("a", encoding="utf-8"):
            pass
        return True
    except OSError:
        return False


def apply_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_stem(f"{path.stem}{suffix}")


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
        streamflow_event_quantile=args.streamflow_event_quantile,
        streamflow_event_min_duration_days=args.streamflow_event_min_duration_days,
        lstm_lookback_days=args.lstm_lookback_days,
        lstm_lookback_auto_from_events=args.lstm_lookback_auto_from_events,
    )

    dynamic_files = resolve_dynamic_files(args.workspace_dir, args.gauge_id)
    dynamic_summary, dynamic_ok = evaluate_dynamic_data(dynamic_files, thresholds)
    intervention_summary, intervention_ok = evaluate_human_intervention(args.workspace_dir, args.gauge_id, thresholds)
    decision = build_decision_table(args.gauge_id, dynamic_ok, intervention_ok, dynamic_summary, intervention_summary)
    streamflow_events, streamflow_event_summary = detect_streamflow_events(
        dynamic_files["streamflow"],
        thresholds.streamflow_event_quantile,
        thresholds.streamflow_event_min_duration_days,
        thresholds.required_start_date,
        thresholds.required_end_date,
    )
    selected_lookback_days, lookback_source = select_lstm_lookback_days(
        streamflow_event_summary,
        thresholds.lstm_lookback_days,
        thresholds.lstm_lookback_auto_from_events,
    )
    streamflow_event_summary["selected_lstm_lookback_days"] = selected_lookback_days
    streamflow_event_summary["selected_lstm_lookback_source"] = lookback_source
    lstm_windows = build_lstm_training_windows(streamflow_events, selected_lookback_days)
    lookback_summary = pd.DataFrame(
        [
            {
                "gauge_id": args.gauge_id,
                "selected_lstm_lookback_days": selected_lookback_days,
                "selected_lstm_lookback_source": lookback_source,
                "default_lstm_lookback_days": thresholds.lstm_lookback_days,
                "auto_from_events_enabled": thresholds.lstm_lookback_auto_from_events,
            }
        ]
    )

    base_name = f"readiness_{args.gauge_id}"
    dynamic_csv_base = args.output_dir / f"{base_name}_dynamic_summary.csv"
    intervention_csv_base = args.output_dir / f"{base_name}_intervention_summary.csv"
    decision_csv_base = args.output_dir / f"{base_name}_decision.csv"
    events_csv_base = args.output_dir / f"{base_name}_streamflow_events.csv"
    event_summary_csv_base = args.output_dir / f"{base_name}_streamflow_event_summary.csv"
    lstm_windows_csv_base = args.output_dir / f"{base_name}_lstm_windows.csv"
    lookback_summary_csv_base = args.output_dir / f"{base_name}_lstm_lookback_summary.csv"
    report_xlsx_base = args.output_dir / f"{base_name}_report.xlsx"

    planned_paths = [
        dynamic_csv_base,
        intervention_csv_base,
        decision_csv_base,
        events_csv_base,
        event_summary_csv_base,
        lstm_windows_csv_base,
        lookback_summary_csv_base,
        report_xlsx_base,
    ]
    output_suffix = "" if all(can_write_to_path(path) for path in planned_paths) else "_novo"

    dynamic_csv = apply_suffix(dynamic_csv_base, output_suffix)
    intervention_csv = apply_suffix(intervention_csv_base, output_suffix)
    decision_csv = apply_suffix(decision_csv_base, output_suffix)
    events_csv = apply_suffix(events_csv_base, output_suffix)
    event_summary_csv = apply_suffix(event_summary_csv_base, output_suffix)
    lstm_windows_csv = apply_suffix(lstm_windows_csv_base, output_suffix)
    lookback_summary_csv = apply_suffix(lookback_summary_csv_base, output_suffix)
    report_xlsx = apply_suffix(report_xlsx_base, output_suffix)

    dynamic_csv = write_with_fallback(lambda path: dynamic_summary.to_csv(path, index=False), dynamic_csv)
    intervention_csv = write_with_fallback(lambda path: intervention_summary.to_csv(path, index=False), intervention_csv)
    decision_csv = write_with_fallback(lambda path: decision.to_csv(path, index=False), decision_csv)
    events_csv = write_with_fallback(lambda path: streamflow_events.to_csv(path, index=False), events_csv)
    event_summary_csv = write_with_fallback(
        lambda path: streamflow_event_summary.to_csv(path, index=False),
        event_summary_csv,
    )
    lstm_windows_csv = write_with_fallback(lambda path: lstm_windows.to_csv(path, index=False), lstm_windows_csv)
    lookback_summary_csv = write_with_fallback(
        lambda path: lookback_summary.to_csv(path, index=False),
        lookback_summary_csv,
    )

    def write_report(path: Path) -> None:
        with pd.ExcelWriter(path) as writer:
            dynamic_summary.to_excel(writer, sheet_name="dynamic_summary", index=False)
            intervention_summary.to_excel(writer, sheet_name="intervention_summary", index=False)
            decision.to_excel(writer, sheet_name="decision", index=False)
            streamflow_events.to_excel(writer, sheet_name="streamflow_events", index=False)
            streamflow_event_summary.to_excel(writer, sheet_name="streamflow_event_summary", index=False)
            lstm_windows.to_excel(writer, sheet_name="lstm_windows", index=False)
            lookback_summary.to_excel(writer, sheet_name="lstm_lookback_summary", index=False)

    report_xlsx = write_with_fallback(write_report, report_xlsx)

    print("Avaliacao concluida.")
    print(f"Gauge ID: {args.gauge_id}")
    print(f"Veredito: {decision.iloc[0]['verdict']}")
    print(f"Decision CSV: {decision_csv}")
    print(f"Streamflow events CSV: {events_csv}")
    print(f"Streamflow event summary CSV: {event_summary_csv}")
    print(f"LSTM windows CSV: {lstm_windows_csv}")
    print(f"LSTM lookback summary CSV: {lookback_summary_csv}")
    print(f"Report Excel: {report_xlsx}")


if __name__ == "__main__":
    main()
