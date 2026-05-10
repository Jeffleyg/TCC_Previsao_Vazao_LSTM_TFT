from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightning.pytorch as pl  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from lightning.pytorch.callbacks import EarlyStopping  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]

try:
    import optuna  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna nao encontrado. Instale com: py -m pip install optuna"
    ) from exc

from optuna_hydro_utils import (
    GAUGE_ID,
    OUTPUT_DIR,
    LSTMv2,
    SequenceDataset,
    compute_drought_metrics,
    compute_hydro_metrics,
    invert_target,
    load_daily_data,
    prepare_sequences,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tuning de hiperparametros LSTM v2 com Optuna")
    parser.add_argument("--data-dir", type=str, default="Treino Unificado")
    parser.add_argument("--master-csv", type=str, default=f".dist/{GAUGE_ID}_master_dataset.csv")
    parser.add_argument("--rebuild-master-dataset", action="store_true")
    parser.add_argument("--study-name", type=str, default="lstm_v2_optuna")
    parser.add_argument("--metric", type=str, default="val_loss", choices=["val_loss", "RMSE", "MAE", "NSE", "KGE", "sMAPE"])
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=None, help="Tempo maximo em segundos")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-lookback", type=int, default=None, help="Fixa o lookback (se None, Optuna varia entre 30/60/90/120/180)")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Horizonte de previsão em dias (padrão: 1)")

    # Reuso das mesmas janelas temporais do treino principal.
    parser.add_argument("--train-start", type=str, default="1980-01-01")
    parser.add_argument("--train-end", type=str, default="2010-12-31")
    parser.add_argument("--test-start", type=str, default="2011-01-01")
    parser.add_argument("--test-end", type=str, default="2018-12-31")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def suggest_params(trial: optuna.Trial, fixed_lookback: int | None = None) -> dict:
    if fixed_lookback is not None:
        lookback = fixed_lookback
    else:
        lookback = trial.suggest_categorical("lookback_days", [48, 96, 144, 192])
    
    return {
        "lookback_days": lookback,
        "hidden_size": trial.suggest_int("hidden_size", 64, 320, step=32),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def objective_factory(df: pd.DataFrame, args: argparse.Namespace):
    metric_name = args.metric

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, fixed_lookback=args.fixed_lookback)
        set_seed(args.seed + trial.number)

        data = prepare_sequences(
            df=df,
            lookback_days=params["lookback_days"],
            forecast_horizon_days=args.forecast_horizon,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            val_fraction=args.val_fraction,
        )

        train_loader = DataLoader(
            SequenceDataset(data["X_train"], data["y_train"]),
            batch_size=params["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            SequenceDataset(data["X_val"], data["y_val"]),
            batch_size=params["batch_size"],
            shuffle=False,
        )

        model = LSTMv2(
            input_size=data["X_train"].shape[2],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=10,
        )

        trainer.fit(model, train_loader, val_loader)

        val_loss_tensor = trainer.callback_metrics.get("val_loss")
        if val_loss_tensor is None:
            raise RuntimeError("val_loss nao foi registrado no treino")
        val_loss = float(val_loss_tensor.detach().cpu().item())

        model.eval()
        with torch.no_grad():
            y_val_pred_scaled = model(torch.as_tensor(data["X_val"], dtype=torch.float32)).cpu().numpy()

        y_val_true = invert_target(data["y_val"], data["scaler_target"])
        y_val_pred = invert_target(y_val_pred_scaled, data["scaler_target"])
        val_metrics = compute_hydro_metrics(y_val_true, y_val_pred)

        y_train_true = invert_target(data["y_train"], data["scaler_target"])
        q90_ref = float(np.quantile(y_train_true.reshape(-1), 0.10))
        val_metrics.update(compute_drought_metrics(y_val_true, y_val_pred, q90_ref))
        val_metrics["val_loss"] = val_loss

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                trial.set_user_attr(key, float(value))

        value = float(val_metrics[metric_name])
        if not np.isfinite(value):
            return float("inf") if metric_name in {"val_loss", "RMSE", "MAE", "sMAPE"} else -1e9
        return value

    return objective


def direction_for_metric(metric: str) -> str:
    if metric in {"val_loss", "RMSE", "MAE", "sMAPE"}:
        return "minimize"
    return "maximize"


def plot_optuna_param_importance(
    study: optuna.Study,
    metric_name: str,
    output_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib nao encontrado. Instale com: py -m pip install matplotlib") from exc

    importances = optuna.importance.get_param_importances(study)
    if not importances:
        raise RuntimeError("Nao foi possivel calcular importancia de parametros para este estudo.")

    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color="#2E86AB")
    ax.set_title(f"Optuna - Importancia dos Hiperparametros ({metric_name})")
    ax.set_xlabel("Importancia")
    ax.invert_yaxis()

    best_params_txt = "\n".join([f"{k}: {v}" for k, v in study.best_params.items()])
    fig.text(
        0.99,
        0.01,
        f"Melhor trial: {study.best_trial.number}\n{best_params_txt}",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_optuna_optimization_history(
    study: optuna.Study,
    metric_name: str,
    direction: str,
    output_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib nao encontrado. Instale com: py -m pip install matplotlib") from exc

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed_trials:
        raise RuntimeError("Nenhum trial completo para montar optimization history.")

    x = [t.number for t in completed_trials]
    y = [float(t.value) for t in completed_trials]

    best_so_far = []
    running_best = y[0]
    for value in y:
        if direction == "minimize":
            running_best = min(running_best, value)
        else:
            running_best = max(running_best, value)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", linestyle="-", alpha=0.6, label="Objective por trial")
    ax.plot(x, best_so_far, linestyle="--", linewidth=2, label="Melhor ate agora")
    ax.set_title(f"Optuna Optimization History ({metric_name})")
    ax.set_xlabel("Trial")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_best_and_plot_hydrograph(
    df: pd.DataFrame,
    args: argparse.Namespace,
    best_params: dict,
    output_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib nao encontrado. Instale com: py -m pip install matplotlib") from exc

    lookback_days = int(best_params.get("lookback_days", args.fixed_lookback or 30))

    data = prepare_sequences(
        df=df,
        lookback_days=lookback_days,
        forecast_horizon_days=1,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        val_fraction=args.val_fraction,
    )

    train_loader = DataLoader(
        SequenceDataset(data["X_train"], data["y_train"]),
        batch_size=int(best_params["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(data["X_val"], data["y_val"]),
        batch_size=int(best_params["batch_size"]),
        shuffle=False,
    )

    model = LSTMv2(
        input_size=data["X_train"].shape[2],
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        dropout=float(best_params["dropout"]),
        learning_rate=float(best_params["learning_rate"]),
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        y_test_pred_scaled = model(torch.as_tensor(data["X_test"], dtype=torch.float32)).cpu().numpy()

    y_test_true = invert_target(data["y_test"], data["scaler_target"]).reshape(-1)
    y_test_pred = invert_target(y_test_pred_scaled, data["scaler_target"]).reshape(-1)
    y_test_dates = pd.to_datetime(data["y_test_dates"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_test_dates, y_test_true, label="Vazao Real", linewidth=1.5)
    ax.plot(y_test_dates, y_test_pred, label="Vazao Prevista", linewidth=1.5, alpha=0.85)
    ax.set_title(f"Hidrograma de Teste - {args.study_name}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Vazao (m3/s)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Carregando dados...")
    df = load_daily_data(
        station_id=GAUGE_ID,
        data_dir=args.data_dir,
        master_csv_relative_path=args.master_csv,
        rebuild_master=args.rebuild_master_dataset,
    )
    print(f"Dias carregados: {len(df)}")

    direction = direction_for_metric(args.metric)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(study_name=args.study_name, direction=direction, sampler=sampler)

    print(f"Iniciando Optuna: study={args.study_name} metric={args.metric} direction={direction}")
    study.optimize(objective_factory(df, args), n_trials=args.n_trials, timeout=args.timeout)

    results_dir = OUTPUT_DIR / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_json = results_dir / f"{args.study_name}_best.json"
    trials_csv = results_dir / f"{args.study_name}_trials.csv"

    best_payload = {
        "study_name": args.study_name,
        "metric": args.metric,
        "direction": direction,
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs,
        "search_config": {
            "n_trials": args.n_trials,
            "timeout": args.timeout,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "data_dir": args.data_dir,
            "master_csv": args.master_csv,
            "rebuild_master_dataset": args.rebuild_master_dataset,
            "seed": args.seed,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "val_fraction": args.val_fraction,
        },
    }

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state),
            "objective": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(trials_csv, index=False)

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_plot = plots_dir / f"{args.study_name}_optuna_optimization_history.png"
    optuna_plot = plots_dir / f"{args.study_name}_optuna_importance.png"
    hydro_plot = plots_dir / f"{args.study_name}_hydrogram_test.png"

    try:
        plot_optuna_optimization_history(study, args.metric, direction, history_plot)
        print(f"Historico Optuna salvo em: {history_plot}")
    except Exception as exc:
        print(f"[Aviso] Nao foi possivel gerar historico de otimizacao: {exc}")

    try:
        plot_optuna_param_importance(study, args.metric, optuna_plot)
        print(f"Grafico Optuna salvo em: {optuna_plot}")
    except Exception as exc:
        print(f"[Aviso] Nao foi possivel gerar grafico do Optuna: {exc}")

    try:
        train_best_and_plot_hydrograph(df, args, study.best_params, hydro_plot)
        print(f"Hidrograma salvo em: {hydro_plot}")
    except Exception as exc:
        print(f"[Aviso] Nao foi possivel gerar hidrograma: {exc}")

    print("\n=== OPTUNA CONCLUIDO ===")
    print(f"Melhor trial: {study.best_trial.number}")
    print(f"Melhor valor ({args.metric}): {study.best_value:.6f}")
    print(f"Melhores hiperparametros: {study.best_params}")
    print(f"Resumo: {result_json}")
    print(f"Trials: {trials_csv}")


if __name__ == "__main__":
    main()
