from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer  # pyright: ignore[reportMissingImports]
from pytorch_forecasting.metrics import MAE  # pyright: ignore[reportMissingImports]

from tune_tft_v2_optuna import build_time_series_datasets, load_daily_data, set_seed


DEFAULT_STUDIES = [
    "tft_passado_lb96_h7",
    "tft_passado_lb192_h15",
    "tft_passado_lb192_h30",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera a Figura 4 com interpretabilidade nativa do TFT.")
    parser.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES)
    parser.add_argument("--plots-dir", type=str, default=".dist/plots")
    parser.add_argument("--optuna-dir", type=str, default=".dist/optuna")
    parser.add_argument("--output", type=str, default="docs/figuras/figura4_tft_interpretability.png")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="Treino Unificado")
    parser.add_argument("--master-csv", type=str, default=".dist/71200000_master_dataset.csv")
    return parser.parse_args()


def parse_study_name(study_name: str) -> tuple[str, int, int]:
    match = re.match(r"^tft_(passado|recente)_lb(\d+)_h(\d+)$", study_name)
    if not match:
        raise ValueError(f"Nome de estudo invalido: {study_name}")
    period = match.group(1)
    lookback = int(match.group(2))
    horizon = int(match.group(3))
    return period, lookback, horizon


def load_best_payload(optuna_dir: Path, study_name: str) -> dict:
    best_json_path = optuna_dir / f"{study_name}_best.json"
    if not best_json_path.exists():
        raise FileNotFoundError(f"Arquivo best.json nao encontrado: {best_json_path}")
    with open(best_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_args_from_search_config(search_config: dict, data_dir: str, master_csv: str) -> SimpleNamespace:
    return SimpleNamespace(
        train_start=search_config.get("train_start", "1980-01-01"),
        train_end=search_config.get("train_end", "2010-12-31"),
        test_start=search_config.get("test_start", "2011-01-01"),
        test_end=search_config.get("test_end", "2018-12-31"),
        val_fraction=search_config.get("val_fraction", 0.15),
        data_dir=data_dir,
        master_csv=master_csv,
    )


def tensor_to_percent(values: torch.Tensor, labels: list[str]) -> dict[str, float]:
    array = values.detach().cpu().numpy().astype(float)
    total = float(np.nansum(array))
    if total <= 0.0:
        return {label: 0.0 for label in labels}
    normalized = (array / total) * 100.0
    return {label: float(score) for label, score in zip(labels, normalized)}


def train_and_interpret(
    study_name: str,
    payload: dict,
    data_args: SimpleNamespace,
    max_epochs: int,
    seed: int,
) -> dict:
    period, lookback_days, horizon_days = parse_study_name(study_name)
    best_params = payload["best_params"]

    df = load_daily_data(
        station_id="71200000",
        data_dir=data_args.data_dir,
        master_csv_relative_path=data_args.master_csv,
        rebuild_master=False,
    )
    args = SimpleNamespace(
        train_start=data_args.train_start,
        train_end=data_args.train_end,
        test_start=data_args.test_start,
        test_end=data_args.test_end,
        val_fraction=data_args.val_fraction,
    )

    set_seed(seed)
    training, validation, test, _, _ = build_time_series_datasets(
        df=df,
        args=args,
        lookback_days=lookback_days,
        forecast_horizon_days=horizon_days,
    )

    train_loader = training.to_dataloader(train=True, batch_size=int(best_params["batch_size"]), num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=int(best_params["batch_size"]), num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=int(best_params["batch_size"]), num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(best_params["learning_rate"]),
        hidden_size=int(best_params["hidden_size"]),
        attention_head_size=int(best_params["attention_head_size"]),
        hidden_continuous_size=int(best_params["hidden_size"]),
        dropout=float(best_params["dropout"]),
        output_size=1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=2,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    batch = next(iter(test_loader))
    batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
    raw_output = model(batch_x)

    attention_rows: list[np.ndarray] = []
    oni_curve: list[float] = []
    encoder_importance_rows: list[dict[str, float]] = []
    decoder_importance_rows: list[dict[str, float]] = []

    for prediction_step in range(horizon_days):
        interpretation = model.interpret_output(
            raw_output,
            reduction="mean",
            attention_prediction_horizon=prediction_step,
        )

        attention = interpretation["attention"].detach().cpu().numpy().astype(float)
        attention_rows.append(attention)

        encoder_importance = tensor_to_percent(interpretation["encoder_variables"], list(model.encoder_variables))
        decoder_importance = tensor_to_percent(interpretation["decoder_variables"], list(model.decoder_variables))
        encoder_importance_rows.append(encoder_importance)
        decoder_importance_rows.append(decoder_importance)
        oni_curve.append(decoder_importance.get("oni", 0.0))

    max_attention_len = max(len(row) for row in attention_rows)
    attention_matrix = np.full((len(attention_rows), max_attention_len), np.nan, dtype=float)
    for index, row in enumerate(attention_rows):
        attention_matrix[index, : len(row)] = row

    payload_out = {
        "study_name": study_name,
        "period": period,
        "lookback": lookback_days,
        "forecast_horizon": horizon_days,
        "attention": attention_matrix.tolist(),
        "oni_curve": oni_curve,
        "encoder_variables": encoder_importance_rows,
        "decoder_variables": decoder_importance_rows,
        "best_params": best_params,
    }

    return payload_out


def main() -> int:
    args = parse_args()
    optuna_dir = Path(args.optuna_dir)
    plots_dir = Path(args.plots_dir)
    output_path = Path(args.output)

    studies_data = []
    for study_name in args.studies:
        payload = load_best_payload(optuna_dir, study_name)
        data_args = build_args_from_search_config(payload.get("search_config", {}), args.data_dir, args.master_csv)
        studies_data.append(train_and_interpret(study_name, payload, data_args, args.max_epochs, args.seed))

    fig = plt.figure(figsize=(18, 15))
    grid = fig.add_gridspec(len(studies_data), 2, hspace=0.30, wspace=0.18)

    for row_index, data in enumerate(studies_data):
        heat_ax = fig.add_subplot(grid[row_index, 0])
        matrix = np.asarray(data["attention"], dtype=float)
        heat = heat_ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        heat_ax.set_title(f"Attention heatmap - {data['study_name']}")
        heat_ax.set_xlabel("Time index")
        heat_ax.set_ylabel("Prediction step")
        fig.colorbar(heat, ax=heat_ax, fraction=0.046, pad=0.04)

        oni_ax = fig.add_subplot(grid[row_index, 1])
        horizon_axis = np.arange(1, data["forecast_horizon"] + 1)
        oni_ax.plot(horizon_axis, data["oni_curve"], marker="o", linewidth=2.5, color="#C0392B")
        oni_ax.set_title(f"ONI temporal importance - {data['study_name']}")
        oni_ax.set_xlabel("Forecast step (dias)")
        oni_ax.set_ylabel("ONI (%)")
        oni_ax.set_xticks(horizon_axis)
        oni_ax.grid(alpha=0.3)

    fig.suptitle("Figura 4 - Interpretabilidade do TFT: atenção, ONI e pesos por horizonte", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for data in studies_data:
        json_path = plots_dir / f"{data['study_name']}_figure4_interpretability.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Figura 4 salva em: {output_path}")
    print("Estudos usados:")
    for data in studies_data:
        print(f"  - {data['study_name']} (h={data['forecast_horizon']}d, lb={data['lookback']}d)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())