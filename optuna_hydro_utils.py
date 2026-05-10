from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

OUTPUT_DIR = Path(__file__).parent / ".dist"
GAUGE_ID = "71200000"
TARGET_VAR = "streamflow_m3s"

# Static features selected for drought-oriented streamflow modeling.
SELECTED_STATIC_FEATURES = {
    "static_climate_aridity",
    "static_climate_p_seasonality",
    "static_topography_area",
    "static_topography_slope_mean",
    "static_topography_elev_mean",
    "static_geology_geol_permeability",
    "static_soil_bedrock_depth",
    "static_human_intervention_regulation_degree",
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def read_space_table(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=r"\s+", na_values=["nan", "NaN"])


def load_static_attributes(attributes_dir: Path, station_id: str) -> tuple[dict[str, object], list[str], list[str]]:
    static_values: dict[str, object] = {"station_id": station_id}
    static_categoricals: list[str] = []
    static_reals: list[str] = []

    for file_path in sorted(attributes_dir.glob("camels_br_*.txt")):
        attr_name = file_path.stem.replace("camels_br_", "")
        frame = read_space_table(file_path)
        if "gauge_id" not in frame.columns:
            continue

        frame["gauge_id"] = frame["gauge_id"].astype(str)
        station_row = frame.loc[frame["gauge_id"] == station_id]
        if station_row.empty:
            continue

        record = station_row.iloc[0].to_dict()
        for column_name, value in record.items():
            if column_name == "gauge_id":
                continue

            prefixed_name = f"static_{attr_name}_{column_name}"
            if prefixed_name not in SELECTED_STATIC_FEATURES:
                continue
            static_values[prefixed_name] = value
            if pd.api.types.is_number(value):
                static_reals.append(prefixed_name)
            else:
                static_categoricals.append(prefixed_name)

    return static_values, sorted(static_categoricals), sorted(static_reals)


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


def assign_period(date_series: pd.Series) -> pd.Series:
    period = pd.Series(index=date_series.index, dtype="string")
    period.loc[(date_series >= "1980-01-01") & (date_series <= "1990-12-31")] = "Passado"
    period.loc[(date_series >= "1991-01-01") & (date_series <= "2010-12-31")] = "Recente"
    period.loc[(date_series >= "2011-01-01") & (date_series <= "2018-12-31")] = "Teste"
    return period


def load_daily_data(
    station_id: str = GAUGE_ID,
    data_dir: str = "Treino Unificado",
    oni_relative_path: str = "ONI/oni.data.txt",
    attributes_relative_dir: str = "Atributo",
    master_csv_relative_path: str | None = None,
    rebuild_master: bool = False,
    export_period_splits: bool = True,
) -> pd.DataFrame:
    workspace_dir = Path(__file__).parent
    if master_csv_relative_path is None:
        master_csv_relative_path = f".dist/{station_id}_master_dataset.csv"

    master_csv_path = workspace_dir / master_csv_relative_path
    required_columns = {
        "date",
        "precipitation",
        "temperature",
        "actual_evapotransp",
        "streamflow_m3s",
        "oni",
        "period",
    }

    if master_csv_path.exists() and not rebuild_master:
        cached = pd.read_csv(master_csv_path, parse_dates=["date"])
        if required_columns.issubset(set(cached.columns)):
            cached["station_id"] = cached["station_id"].astype(str)
            return cached.sort_values("date").reset_index(drop=True)

    base = Path(__file__).parent / data_dir
    file_map = {
        "precipitation": base / f"{station_id}_precipitation.txt",
        "temperature": base / f"{station_id}_temperature.txt",
        "actual_evapotransp": base / f"{station_id}_actual_evapotransp.txt",
        "streamflow_m3s": base / f"{station_id}_streamflow_m3s.txt",
    }
    preferred_columns = {
        "precipitation": "p_mswep",
        "temperature": "tmean_era5land",
        "actual_evapotransp": "aet_era5land",
        "streamflow_m3s": "streamflow_m3s",
    }

    frames: list[pd.DataFrame] = []
    for alias, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
        df = read_space_table(path)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        preferred_col = preferred_columns[alias]
        if preferred_col not in df.columns:
            raise ValueError(f"Coluna esperada '{preferred_col}' nao encontrada em {path.name}")
        frames.append(df[["date", preferred_col]].rename(columns={preferred_col: alias}))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    oni_file = workspace_dir / oni_relative_path
    if not oni_file.exists():
        raise FileNotFoundError(f"Arquivo ONI nao encontrado: {oni_file}")

    attributes_dir = workspace_dir / attributes_relative_dir
    if not attributes_dir.exists():
        raise FileNotFoundError(f"Diretorio de atributos nao encontrado: {attributes_dir}")

    static_values, static_categoricals, _ = load_static_attributes(attributes_dir, station_id)

    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged = merged.merge(load_oni_monthly(oni_file), on=["year", "month"], how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["sin_day"] = np.sin(2 * math.pi * merged["day_of_year"] / 366.0)
    merged["cos_day"] = np.cos(2 * math.pi * merged["day_of_year"] / 366.0)
    merged["sin_month"] = np.sin(2 * math.pi * merged["month"] / 12.0)
    merged["cos_month"] = np.cos(2 * math.pi * merged["month"] / 12.0)
    merged["period"] = assign_period(merged["date"])
    merged["station_id"] = station_id

    for column_name, value in static_values.items():
        if column_name == "station_id":
            continue
        merged[column_name] = value

    for column_name in static_categoricals:
        if column_name in merged.columns:
            merged[column_name] = merged[column_name].astype("string")

    merged["time_idx"] = np.arange(len(merged), dtype=int)

    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear", limit=5)
    merged = merged.dropna().reset_index(drop=True)

    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(master_csv_path, index=False)

    if export_period_splits and "period" in merged.columns:
        stem = master_csv_path.stem
        period_to_suffix = {
            "Passado": "passado",
            "Recente": "recente",
            "Teste": "teste",
        }
        for period_name, suffix in period_to_suffix.items():
            period_df = merged.loc[merged["period"] == period_name].copy()
            if not period_df.empty:
                split_path = master_csv_path.parent / f"{stem}_{suffix}.csv"
                period_df.to_csv(split_path, index=False)

    return merged


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMv2(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return y_hat, loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def training_step(self, batch, batch_idx):
        _, loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, loss = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def invert_target(y_scaled: np.ndarray, scaler_target: StandardScaler) -> np.ndarray:
    y_log = scaler_target.inverse_transform(y_scaled)
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


def compute_hydro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    alpha = float(np.std(y_pred) / (np.std(y_true) + 1e-8))
    beta = float(np.mean(y_pred) / (np.mean(y_true) + 1e-8))
    kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100.0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NSE": r2,
        "KGE": kge,
        "sMAPE": smape,
    }


def compute_drought_metrics(y_true: np.ndarray, y_pred: np.ndarray, q90_ref: float) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    eps = 1e-6
    y_true_log = np.log(np.clip(y_true, eps, None))
    y_pred_log = np.log(np.clip(y_pred, eps, None))
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    log_nse = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    low_flow_mask = y_true <= q90_ref
    if np.any(low_flow_mask):
        rmse_dry = float(np.sqrt(np.mean((y_true[low_flow_mask] - y_pred[low_flow_mask]) ** 2)))
        n_dry = int(np.sum(low_flow_mask))
    else:
        rmse_dry = float("nan")
        n_dry = 0

    return {
        "Q90_reference": q90_ref,
        "LogNSE": log_nse,
        "RMSE_dry": rmse_dry,
        "n_dry_samples": n_dry,
    }


def collect_targets_from_loader(dataloader) -> np.ndarray:
    parts = []
    for _, y in dataloader:
        target = y[0] if isinstance(y, (tuple, list)) else y
        parts.append(target.detach().cpu().numpy())
    return np.concatenate(parts, axis=0).reshape(-1)


def save_plots(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    output_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(y_true_val, label="Real", alpha=0.8)
    axes[0, 0].plot(y_pred_val, label="Predito", alpha=0.8)
    axes[0, 0].set_title("Validacao - Serie")
    axes[0, 0].legend()

    axes[0, 1].plot(y_true_test, label="Real", alpha=0.8)
    axes[0, 1].plot(y_pred_test, label="Predito", alpha=0.8)
    axes[0, 1].set_title("Teste - Serie")
    axes[0, 1].legend()

    axes[1, 0].scatter(y_true_val, y_pred_val, alpha=0.4)
    lo_v = float(min(y_true_val.min(), y_pred_val.min()))
    hi_v = float(max(y_true_val.max(), y_pred_val.max()))
    axes[1, 0].plot([lo_v, hi_v], [lo_v, hi_v], "r--")
    axes[1, 0].set_title("Validacao - Scatter")

    axes[1, 1].scatter(y_true_test, y_pred_test, alpha=0.4)
    lo_t = float(min(y_true_test.min(), y_pred_test.min()))
    hi_t = float(max(y_true_test.max(), y_pred_test.max()))
    axes[1, 1].plot([lo_t, hi_t], [lo_t, hi_t], "r--")
    axes[1, 1].set_title("Teste - Scatter")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_sequences(
    df: pd.DataFrame,
    lookback_days: int,
    forecast_horizon_days: int,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    val_fraction: float = 0.15,
    feature_names: list[str] | None = None,
    include_static: bool = True,
    group_normalize: bool = False,
    group_column: str = "station_id",
) -> dict:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    target_raw = df[TARGET_VAR].clip(lower=0.0).to_numpy().reshape(-1, 1)
    target_log = np.log1p(target_raw)

    # Default feature set (matches TFT temporal inputs)
    default_features = [
        "precipitation",
        "temperature",
        "actual_evapotransp",
        "sin_day",
        "cos_day",
        "sin_month",
        "cos_month",
    ]

    # Build feature list explicitly to avoid accidental omission
    if feature_names is None:
        excluded_cols = {"date", TARGET_VAR, "station_id", "day_of_year", "month", "period"}
        feature_cols = [
            column
            for column in df.columns
            if column not in excluded_cols and pd.api.types.is_numeric_dtype(df[column])
        ]
    else:
        feature_cols = [f for f in feature_names if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    # Optionally ensure default temporal features are present
    for f in default_features:
        if f in df.columns and f not in feature_cols:
            feature_cols.append(f)

    # Optionally include selected static attributes
    if include_static:
        for col in sorted(SELECTED_STATIC_FEATURES):
            if col in df.columns and col not in feature_cols and pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

    features = df[feature_cols].to_numpy() if feature_cols else np.empty((len(df), 0))

    scaler_target = StandardScaler()
    # Support both global and per-group normalisation for fairness with TFT
    scaler_features = StandardScaler()
    scaler_features_group: dict[str, StandardScaler] | None = None

    target_scaled = scaler_target.fit_transform(target_log)

    if group_normalize and group_column in df.columns:
        scaler_features_group = {}
        features_scaled = np.zeros_like(features, dtype=float)
        groups = df[group_column].astype(str).to_numpy()
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            scaler = StandardScaler()
            try:
                scaler.fit(features[mask])
                features_scaled[mask] = scaler.transform(features[mask])
            except Exception:
                # Fallback: if group has too few samples, use zeros
                features_scaled[mask] = 0.0
            scaler_features_group[g] = scaler
        # also keep a global scaler for compatibility with callers
        try:
            scaler_features.fit(features)
        except Exception:
            pass
    else:
        features_scaled = scaler_features.fit_transform(features)

    all_scaled = np.hstack([target_scaled, features_scaled])
    target_dates = df["date"].to_numpy()
    target_raw_series = target_raw.reshape(-1)

    X, y, y_raw, y_dates = [], [], [], []
    for index in range(len(all_scaled) - lookback_days - forecast_horizon_days + 1):
        X.append(all_scaled[index : index + lookback_days])
        y_idx = index + lookback_days + forecast_horizon_days - 1
        y.append(all_scaled[y_idx, 0])
        y_raw.append(target_raw_series[y_idx])
        y_dates.append(target_dates[y_idx])

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    y_raw = np.asarray(y_raw).reshape(-1, 1)
    y_dates = pd.to_datetime(np.asarray(y_dates))

    use_date_windows = all(v is not None for v in [train_start, train_end, test_start])

    if use_date_windows:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end) if test_end is not None else y_dates.max()

        train_pool_mask = (y_dates >= train_start_ts) & (y_dates <= train_end_ts)
        test_mask = (y_dates >= test_start_ts) & (y_dates <= test_end_ts)

        train_pool_idx = np.where(train_pool_mask)[0]
        if len(train_pool_idx) < 20:
            raise ValueError("Janela de treino temporal gerou poucas amostras.")

        n_val = max(1, int(round(val_fraction * len(train_pool_idx))))
        n_val = min(n_val, len(train_pool_idx) - 1)
        val_idx = train_pool_idx[-n_val:]
        train_idx = train_pool_idx[:-n_val]
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            raise ValueError("Janela de teste temporal gerou 0 amostras.")
    else:
        n_train = int(0.7 * len(X))
        n_val = int(0.85 * len(X))
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_val)
        test_idx = np.arange(n_val, len(X))

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "y_train_raw": y_raw[train_idx],
        "y_val_raw": y_raw[val_idx],
        "y_test_raw": y_raw[test_idx],
        "y_train_dates": y_dates[train_idx],
        "y_val_dates": y_dates[val_idx],
        "y_test_dates": y_dates[test_idx],
        "scaler_target": scaler_target,
        "scaler_features": scaler_features,
        "scaler_features_group": scaler_features_group,
        "feature_cols": feature_cols,
        "target_transform": "log1p",
    }