from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATION_ID = "71200000"
EXPECTED_PERIODS = ("Passado", "Recente", "Teste")
RECOMMENDED_START_DATE = "1980-01-01"
RECOMMENDED_DYNAMIC_FEATURES = [
    "p_mswep",
    "tmin_era5land",
    "tmean_era5land",
    "tmax_era5land",
    "aet_era5land",
    "oni",
    "month_sin",
    "month_cos",
]


@dataclass(frozen=True)
class DatasetArtifacts:
    dataframe: pd.DataFrame
    scaling_metadata: dict[str, dict[str, float]]
    static_categoricals: list[str]
    static_reals: list[str]
    time_varying_known_reals: list[str]
    time_varying_unknown_reals: list[str]


@dataclass(frozen=True)
class PeriodDataFrames:
    treino_passado: pd.DataFrame
    treino_recente: pd.DataFrame
    treino_teste: pd.DataFrame


def read_space_separated_table(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=r"\s+", na_values=["nan", "NaN"])


def build_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()
    daily["date"] = pd.to_datetime(daily[["year", "month", "day"]])
    return daily.drop(columns=["year", "month", "day"])


def load_daily_series(data_dir: Path, station_id: str) -> pd.DataFrame:
    file_map = {
        "precipitation": data_dir / f"{station_id}_precipitation.txt",
        "streamflow_m3s": data_dir / f"{station_id}_streamflow_m3s.txt",
        "temperature": data_dir / f"{station_id}_temperature.txt",
        "actual_evapotransp": data_dir / f"{station_id}_actual_evapotransp.txt",
    }

    frames: list[pd.DataFrame] = []
    for file_path in file_map.values():
        frame = build_daily_frame(read_space_separated_table(file_path))
        frame = frame.sort_values("date")
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged.insert(0, "station_id", station_id)
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["day"] = merged["date"].dt.day
    return merged


def load_oni_monthly(oni_file: Path) -> pd.DataFrame:
    month_columns = [f"oni_{month:02d}" for month in range(1, 13)]
    oni = pd.read_csv(
        oni_file,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["year", *month_columns],
        na_values=["nan", "NaN"],
    )
    oni["year"] = pd.to_numeric(oni["year"], errors="coerce")
    oni = oni.loc[oni["year"].between(1900, 2100, inclusive="both")].copy()
    oni["year"] = oni["year"].astype(int)
    oni[month_columns] = oni[month_columns].apply(pd.to_numeric, errors="coerce")
    oni[month_columns] = oni[month_columns].mask(oni[month_columns] <= -99.0)

    monthly = oni.melt(id_vars="year", var_name="month_name", value_name="oni")
    monthly["month"] = monthly["month_name"].str[-2:].astype(int)
    return monthly[["year", "month", "oni"]].sort_values(["year", "month"])


def add_oni_to_daily_frame(df: pd.DataFrame, oni_monthly: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(oni_monthly, on=["year", "month"], how="left")
    return merged.sort_values("date").reset_index(drop=True)


def load_static_attributes(attributes_dir: Path, station_id: str) -> tuple[dict[str, object], list[str], list[str]]:
    static_values: dict[str, object] = {"station_id": station_id}
    static_categoricals: list[str] = []
    static_reals: list[str] = []

    for file_path in sorted(attributes_dir.glob("camels_br_*.txt")):
        attr_name = file_path.stem.replace("camels_br_", "")
        frame = read_space_separated_table(file_path)
        frame["gauge_id"] = frame["gauge_id"].astype(str)
        station_row = frame.loc[frame["gauge_id"] == station_id]
        if station_row.empty:
            continue

        record = station_row.iloc[0].to_dict()
        for column_name, value in record.items():
            if column_name == "gauge_id":
                continue

            prefixed_name = f"static_{attr_name}_{column_name}"
            static_values[prefixed_name] = value
            if pd.api.types.is_number(value):
                static_reals.append(prefixed_name)
            else:
                static_categoricals.append(prefixed_name)

    return static_values, sorted(static_categoricals), sorted(static_reals)


def attach_static_attributes(df: pd.DataFrame, static_values: dict[str, object]) -> pd.DataFrame:
    enriched = df.copy()
    for column_name, value in static_values.items():
        if column_name == "station_id":
            continue
        enriched[column_name] = value
    return enriched


def add_cyclical_month_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    month_angle = 2 * np.pi * (enriched["month"] - 1) / 12.0
    enriched["month_sin"] = np.sin(month_angle)
    enriched["month_cos"] = np.cos(month_angle)
    return enriched


def interpolate_short_gaps(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    interpolated = df.copy()
    numeric_columns = interpolated.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = {"year", "month", "day", "time_idx"}
    columns_to_interpolate = [column for column in numeric_columns if column not in excluded_columns]

    interpolated[columns_to_interpolate] = interpolated[columns_to_interpolate].interpolate(
        method="linear",
        limit=limit,
        limit_direction="both",
        limit_area="inside",
    )
    return interpolated


def assign_period(date_series: pd.Series) -> pd.Series:
    period = pd.Series(index=date_series.index, dtype="object")
    period.loc[(date_series >= "1980-01-01") & (date_series <= "1990-12-31")] = "Passado"
    period.loc[(date_series >= "1991-01-01") & (date_series <= "2010-12-31")] = "Recente"
    period.loc[(date_series >= "2011-01-01") & (date_series <= "2018-12-31")] = "Teste"
    return period


def min_max_scale(
    df: pd.DataFrame,
    fit_periods: Iterable[str] = ("Passado", "Recente"),
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    scaled = df.copy()
    numeric_columns = scaled.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = {"year", "month", "day", "time_idx"}
    columns_to_scale = [column for column in numeric_columns if column not in excluded_columns]
    fit_mask = scaled["period"].isin(list(fit_periods))
    scaling_metadata: dict[str, dict[str, float]] = {}

    for column in columns_to_scale:
        fit_values = scaled.loc[fit_mask, column].dropna()
        if fit_values.empty:
            continue

        col_min = float(fit_values.min())
        col_max = float(fit_values.max())
        scaling_metadata[column] = {"min": col_min, "max": col_max}

        if np.isclose(col_min, col_max):
            scaled[column] = np.where(scaled[column].notna(), 0.0, np.nan)
            continue

        scaled[column] = (scaled[column] - col_min) / (col_max - col_min)

    return scaled, scaling_metadata


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    finalized = df.copy()
    finalized["period"] = assign_period(finalized["date"])
    finalized = finalized.loc[finalized["period"].isin(EXPECTED_PERIODS)].copy()
    finalized["time_idx"] = (finalized["date"] - finalized["date"].min()).dt.days.astype(int)

    object_columns = [
        column for column in finalized.columns if finalized[column].dtype == "object"
    ]
    for column in object_columns:
        finalized[column] = finalized[column].astype("string")

    return finalized.sort_values("date").reset_index(drop=True)


def infer_feature_roles(
    df: pd.DataFrame,
    static_categoricals: list[str],
    static_reals: list[str],
) -> tuple[list[str], list[str]]:
    known_reals = [
        "time_idx",
        "oni",
        "month_sin",
        "month_cos",
        *static_reals,
    ]
    known_reals = [column for column in known_reals if column in df.columns]

    excluded = {
        "time_idx",
        "year",
        "month",
        "day",
        "oni",
        "month_sin",
        "month_cos",
        "qual_control_by_ana",
        "qual_flag",
        *static_reals,
    }
    unknown_reals = [
        column
        for column in df.select_dtypes(include=[np.number]).columns.tolist()
        if column not in excluded
    ]
    return known_reals, unknown_reals


def prepare_camels_br_dataset(
    workspace_dir: Path,
    station_id: str = STATION_ID,
) -> DatasetArtifacts:
    data_dir = workspace_dir / "Treino Unificado"
    oni_file = workspace_dir / "ONI" / "oni.data.txt"
    attributes_dir = workspace_dir / "Atributo"

    daily_df = load_daily_series(data_dir=data_dir, station_id=station_id)
    oni_monthly = load_oni_monthly(oni_file)
    static_values, static_categoricals, static_reals = load_static_attributes(attributes_dir, station_id)

    prepared = add_oni_to_daily_frame(daily_df, oni_monthly)
    prepared = attach_static_attributes(prepared, static_values)
    prepared = add_cyclical_month_features(prepared)
    prepared = interpolate_short_gaps(prepared, limit=3)
    prepared = finalize_dataframe(prepared)
    prepared, scaling_metadata = min_max_scale(prepared, fit_periods=("Passado", "Recente"))
    time_varying_known_reals, time_varying_unknown_reals = infer_feature_roles(
        prepared,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
    )

    return DatasetArtifacts(
        dataframe=prepared,
        scaling_metadata=scaling_metadata,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
    )


def split_dataframes_by_period(df: pd.DataFrame) -> PeriodDataFrames:
    treino_passado = df.loc[df["period"] == "Passado"].copy().reset_index(drop=True)
    treino_recente = df.loc[df["period"] == "Recente"].copy().reset_index(drop=True)
    treino_teste = df.loc[df["period"] == "Teste"].copy().reset_index(drop=True)

    return PeriodDataFrames(
        treino_passado=treino_passado,
        treino_recente=treino_recente,
        treino_teste=treino_teste,
    )


def build_training_ready_dataset(
    df: pd.DataFrame,
    start_date: str | None = None,
    target_column: str = "streamflow_m3s",
    dynamic_features: list[str] | None = None,
) -> pd.DataFrame:
    if dynamic_features is None:
        dynamic_features = RECOMMENDED_DYNAMIC_FEATURES.copy()

    if start_date is None:
        target_non_null = df.loc[df[target_column].notna(), "date"]
        if target_non_null.empty:
            start_date = RECOMMENDED_START_DATE
        else:
            inferred_start = target_non_null.min()
            start_date = max(pd.Timestamp(RECOMMENDED_START_DATE), inferred_start).strftime("%Y-%m-%d")

    static_columns = [column for column in df.columns if column.startswith("static_")]
    base_columns = [
        "station_id",
        "date",
        "period",
        "time_idx",
        "year",
        "month",
        "day",
        target_column,
        *dynamic_features,
        *static_columns,
    ]
    selected_columns = [column for column in base_columns if column in df.columns]

    curated = df.loc[df["date"] >= pd.Timestamp(start_date), selected_columns].copy()
    required_columns = [target_column, *[column for column in dynamic_features if column in curated.columns]]
    curated = curated.dropna(subset=required_columns)

    return curated.sort_values("date").reset_index(drop=True)


def build_training_ready_splits(
    df: pd.DataFrame,
    start_date: str | None = None,
    target_column: str = "streamflow_m3s",
    dynamic_features: list[str] | None = None,
) -> PeriodDataFrames:
    curated = build_training_ready_dataset(
        df=df,
        start_date=start_date,
        target_column=target_column,
        dynamic_features=dynamic_features,
    )
    return split_dataframes_by_period(curated)


def build_adaptive_monthly_window_grid(
    df: pd.DataFrame,
    window_min_months: int = 1,
    window_max_months: int = 24,
    target_periods: tuple[str, ...] = ("Teste",),
    required_columns: list[str] | None = None,
    require_complete_data: bool = True,
) -> pd.DataFrame:
    if window_min_months < 1:
        raise ValueError("window_min_months deve ser >= 1")
    if window_max_months < window_min_months:
        raise ValueError("window_max_months deve ser >= window_min_months")

    if required_columns is None:
        required_columns = ["streamflow_m3s", *RECOMMENDED_DYNAMIC_FEATURES]

    data = df.copy()
    data["month_start"] = data["date"].dt.to_period("M").dt.to_timestamp()

    month_period = (
        data.groupby("month_start", as_index=False)["period"]
        .agg(lambda values: pd.NA if values.dropna().empty else values.dropna().iloc[0])
    )
    month_period = month_period.loc[month_period["period"].isin(target_periods)].copy()
    target_months = month_period["month_start"].sort_values().tolist()

    rows: list[dict[str, object]] = []
    for target_start in target_months:
        target_end = target_start + pd.offsets.MonthEnd(0)
        target_label = target_start.strftime("%Y-%m")

        for window_months in range(window_min_months, window_max_months + 1):
            input_start = target_start - pd.DateOffset(months=window_months)
            input_end = target_start - pd.Timedelta(days=1)

            input_mask = (data["date"] >= input_start) & (data["date"] <= input_end)
            target_mask = (data["date"] >= target_start) & (data["date"] <= target_end)

            input_rows = data.loc[input_mask].copy()
            target_rows = data.loc[target_mask].copy()

            input_month_count = int(input_rows["date"].dt.to_period("M").nunique())
            has_full_input_months = input_month_count == window_months
            has_target_month = not target_rows.empty

            cols_to_check = [column for column in required_columns if column in data.columns]
            input_missing = int(input_rows[cols_to_check].isna().sum().sum()) if cols_to_check else 0
            target_missing = int(target_rows[cols_to_check].isna().sum().sum()) if cols_to_check else 0
            has_complete_data = input_missing == 0 and target_missing == 0

            is_candidate = has_full_input_months and has_target_month
            if require_complete_data:
                is_candidate = is_candidate and has_complete_data

            rows.append(
                {
                    "target_month": target_label,
                    "window_months": window_months,
                    "input_start_date": input_start,
                    "input_end_date": input_end,
                    "target_start_date": target_start,
                    "target_end_date": target_end,
                    "input_rows": int(len(input_rows)),
                    "target_rows": int(len(target_rows)),
                    "has_full_input_months": has_full_input_months,
                    "has_target_month": has_target_month,
                    "input_missing_values": input_missing,
                    "target_missing_values": target_missing,
                    "has_complete_data": has_complete_data,
                    "is_candidate": is_candidate,
                }
            )

    return pd.DataFrame(rows).sort_values(["target_month", "window_months"]).reset_index(drop=True)


def build_timeseries_dataset(
    df: pd.DataFrame,
    target: str = "streamflow_m3s",
    max_encoder_length: int = 365,
    max_prediction_length: int = 30,
):
    try:
        TimeSeriesDataSet = importlib.import_module("pytorch_forecasting").TimeSeriesDataSet
    except ImportError as exc:
        raise ImportError(
            "PyTorch Forecasting nao esta instalado. Instale 'pytorch-forecasting' para usar este helper."
        ) from exc

    static_categoricals = [column for column in df.columns if column.startswith("static_") and df[column].dtype == "string"]
    static_reals = [
        column
        for column in df.columns
        if column.startswith("static_") and pd.api.types.is_numeric_dtype(df[column])
    ]
    time_varying_known_reals = [
        column
        for column in ["time_idx", "oni", "month_sin", "month_cos", *static_reals]
        if column in df.columns
    ]
    time_varying_unknown_reals = [
        column
        for column in df.select_dtypes(include=[np.number]).columns.tolist()
        if column not in {"time_idx", "year", "month", "day", "oni", "month_sin", "month_cos", *static_reals}
    ]
    if target in time_varying_unknown_reals:
        time_varying_unknown_reals = [
            column for column in time_varying_unknown_reals if column != target
        ]

    dataset_frame = df.copy()
    for column in ["station_id", "period", *static_categoricals]:
        if column in dataset_frame.columns:
            dataset_frame[column] = dataset_frame[column].astype(str)

    return TimeSeriesDataSet(
        dataset_frame,
        time_idx="time_idx",
        target=target,
        group_ids=["station_id"],
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target, *time_varying_unknown_reals],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        allow_missing_timesteps=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara dados CAMELS-BR para LSTM/TFT seguindo CRISP-DM (padrao: estacao 71200000)."
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Diretorio raiz com as pastas Atributo, ONI e Treino Unificado.",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default=STATION_ID,
        help="Codigo da estacao CAMELS-BR.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Caminho opcional para salvar o dataset final em CSV.",
    )
    parser.add_argument(
        "--output-scaling-json",
        type=Path,
        default=None,
        help="Caminho opcional para salvar os parametros de escala em JSON.",
    )
    parser.add_argument(
        "--split-periods",
        action="store_true",
        help="Separa o dataset final em tres DataFrames por periodo e mostra um resumo no terminal.",
    )
    parser.add_argument(
        "--export-period-csvs",
        action="store_true",
        help="Exporta treino_passado.csv, treino_recente.csv e treino_teste.csv para o diretorio de saida.",
    )
    parser.add_argument(
        "--export-period-excels",
        action="store_true",
        help="Exporta treino_passado.xlsx, treino_recente.xlsx e treino_teste.xlsx para o diretorio de saida.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".dist",
        help="Diretorio de saida para os CSVs por periodo quando --export-period-csvs for usado.",
    )
    parser.add_argument(
        "--export-model-ready-csvs",
        action="store_true",
        help="Exporta CSVs curados para treino, removendo o inicio sem vazao e variaveis com cobertura fraca.",
    )
    parser.add_argument(
        "--export-adaptive-window-grid",
        action="store_true",
        help="Exporta grade de janelas mensais para LSTM (ex.: 1 a 24 meses anteriores por mes-alvo).",
    )
    parser.add_argument(
        "--window-min-months",
        type=int,
        default=1,
        help="Menor tamanho da janela mensal para o grid adaptativo.",
    )
    parser.add_argument(
        "--window-max-months",
        type=int,
        default=24,
        help="Maior tamanho da janela mensal para o grid adaptativo.",
    )
    parser.add_argument(
        "--window-target-periods",
        type=str,
        default="Teste",
        help="Periodos alvo separados por virgula para gerar o grid (ex.: Recente,Teste).",
    )
    parser.add_argument(
        "--window-allow-missing",
        action="store_true",
        help="Permite janelas candidatas com valores faltantes (por padrao exige dados completos).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = prepare_camels_br_dataset(
        workspace_dir=args.workspace_dir,
        station_id=args.station_id,
    )
    splits = split_dataframes_by_period(artifacts.dataframe)

    print("Dataset preparado com sucesso.")
    print(f"Linhas: {len(artifacts.dataframe)}")
    print(f"Colunas: {len(artifacts.dataframe.columns)}")
    print(f"Periodos: {artifacts.dataframe['period'].value_counts(dropna=False).to_dict()}")

    if args.split_periods:
        print(f"Treino Passado: {splits.treino_passado.shape}")
        print(f"Treino Recente: {splits.treino_recente.shape}")
        print(f"Treino Teste: {splits.treino_teste.shape}")

    if args.output_csv is not None:
        artifacts.dataframe.to_csv(args.output_csv, index=False)
        print(f"CSV salvo em: {args.output_csv}")

    if args.output_scaling_json is not None:
        with args.output_scaling_json.open("w", encoding="utf-8") as file_obj:
            json.dump(artifacts.scaling_metadata, file_obj, indent=2, ensure_ascii=True)
        print(f"Escalonamento salvo em: {args.output_scaling_json}")

    if args.export_period_csvs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        passado_path = args.output_dir / "treino_passado.csv"
        recente_path = args.output_dir / "treino_recente.csv"
        teste_path = args.output_dir / "treino_teste.csv"

        splits.treino_passado.to_csv(passado_path, index=False)
        splits.treino_recente.to_csv(recente_path, index=False)
        splits.treino_teste.to_csv(teste_path, index=False)

        print(f"Treino Passado salvo em: {passado_path}")
        print(f"Treino Recente salvo em: {recente_path}")
        print(f"Treino Teste salvo em: {teste_path}")

    if args.export_period_excels:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        passado_xlsx = args.output_dir / "treino_passado.xlsx"
        recente_xlsx = args.output_dir / "treino_recente.xlsx"
        teste_xlsx = args.output_dir / "treino_teste.xlsx"

        try:
            splits.treino_passado.to_excel(passado_xlsx, index=False)
            splits.treino_recente.to_excel(recente_xlsx, index=False)
            splits.treino_teste.to_excel(teste_xlsx, index=False)
        except ImportError as exc:
            raise ImportError(
                "Para exportar Excel, instale o pacote 'openpyxl' (ex.: py -m pip install openpyxl)."
            ) from exc

        print(f"Treino Passado salvo em: {passado_xlsx}")
        print(f"Treino Recente salvo em: {recente_xlsx}")
        print(f"Treino Teste salvo em: {teste_xlsx}")

    if args.export_model_ready_csvs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_ready_splits = build_training_ready_splits(artifacts.dataframe)
        passado_model_path = args.output_dir / "treino_passado_model_ready.csv"
        recente_model_path = args.output_dir / "treino_recente_model_ready.csv"
        teste_model_path = args.output_dir / "treino_teste_model_ready.csv"

        model_ready_splits.treino_passado.to_csv(passado_model_path, index=False)
        model_ready_splits.treino_recente.to_csv(recente_model_path, index=False)
        model_ready_splits.treino_teste.to_csv(teste_model_path, index=False)

        print(f"Treino Passado model-ready salvo em: {passado_model_path}")
        print(f"Treino Recente model-ready salvo em: {recente_model_path}")
        print(f"Treino Teste model-ready salvo em: {teste_model_path}")

    if args.export_adaptive_window_grid:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_ready = build_training_ready_dataset(artifacts.dataframe)
        target_periods = tuple(period.strip() for period in args.window_target_periods.split(",") if period.strip())
        window_grid = build_adaptive_monthly_window_grid(
            model_ready,
            window_min_months=args.window_min_months,
            window_max_months=args.window_max_months,
            target_periods=target_periods,
            require_complete_data=not args.window_allow_missing,
        )

        grid_path = args.output_dir / "lstm_adaptive_window_grid.csv"
        summary_path = args.output_dir / "lstm_adaptive_window_summary.csv"
        window_grid.to_csv(grid_path, index=False)

        summary = (
            window_grid.loc[window_grid["is_candidate"]]
            .groupby("target_month", as_index=False)
            .agg(
                candidate_windows=("window_months", "count"),
                min_window_months=("window_months", "min"),
                max_window_months=("window_months", "max"),
            )
        )
        summary.to_csv(summary_path, index=False)

        print(f"Grid adaptativo LSTM salvo em: {grid_path}")
        print(f"Resumo do grid adaptativo salvo em: {summary_path}")


if __name__ == "__main__":
    main()