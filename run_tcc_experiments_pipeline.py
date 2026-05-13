from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).parent
OPTUNA_DIR = PROJECT_DIR / ".dist" / "optuna"
PLOTS_DIR = PROJECT_DIR / ".dist" / "plots"
LOGS_DIR = PROJECT_DIR / ".dist" / "logs"
REPORTS_DIR = PROJECT_DIR / ".dist" / "reports"

LOOKBACKS = [30, 60, 90, 120, 150, 180]
PERIODS = {
    "passado": {"train_start": "1980-01-01", "train_end": "1990-12-31"},
    "recente": {"train_start": "1991-01-01", "train_end": "2010-12-31"},
}
MODELS = {
    "lstm": "tune_lstm_v2_optuna.py",
    "tft": "tune_tft_v2_optuna.py",
}


@dataclass
class ExperimentResult:
    study_name: str
    model: str
    period: str
    lookback: int
    status: str
    nse: float | None
    kge: float | None
    mae: float | None
    val_loss: float | None
    best_trial: int | None
    best_json: str | None
    log_file: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline TCC: 24 experimentos (LSTM/TFT x Passado/Recente x 6 lookbacks)"
    )
    parser.add_argument("--n-trials", type=int, default=40, choices=[40], help="Numero de trials (fixo em 40)")
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="val_loss", choices=["val_loss"])
    parser.add_argument("--master-csv", type=str, default=".dist/71200000_master_dataset.csv")
    parser.add_argument("--test-start", type=str, default="2011-01-01")
    parser.add_argument("--test-end", type=str, default="2018-12-31")
    parser.add_argument("--rebuild-master-dataset", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(
    python_executable: str,
    script_name: str,
    study_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lookback: int,
    args: argparse.Namespace,
) -> tuple[int, Path]:
    log_path = LOGS_DIR / f"{study_name}.log"
    cmd = [
        python_executable,
        script_name,
        "--study-name",
        study_name,
        "--metric",
        args.metric,
        "--train-start",
        train_start,
        "--train-end",
        train_end,
        "--test-start",
        test_start,
        "--test-end",
        test_end,
        "--n-trials",
        str(args.n_trials),
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(args.seed),
        "--master-csv",
        args.master_csv,
        "--fixed-lookback",
        str(lookback),
    ]
    if args.rebuild_master_dataset:
        cmd.append("--rebuild-master-dataset")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Comando: {' '.join(cmd)}\n\n")
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return result.returncode, log_path


def load_best_metrics(study_name: str) -> tuple[dict, Path]:
    best_json_path = OPTUNA_DIR / f"{study_name}_best.json"
    with open(best_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload, best_json_path


def expected_plot_paths(study_name: str) -> list[Path]:
    return [
        PLOTS_DIR / f"{study_name}_optuna_optimization_history.png",
        PLOTS_DIR / f"{study_name}_optuna_importance.png",
        PLOTS_DIR / f"{study_name}_hydrogram_test.png",
    ]


def make_master_comparison_plot(results_df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10, 6))

    series_order = [
        ("lstm", "passado", "LSTM-Passado", "#1f77b4"),
        ("lstm", "recente", "LSTM-Recente", "#ff7f0e"),
        ("tft", "passado", "TFT-Passado", "#2ca02c"),
        ("tft", "recente", "TFT-Recente", "#d62728"),
    ]

    for model, period, label, color in series_order:
        subset = results_df[
            (results_df["status"] == "success")
            & (results_df["model"] == model)
            & (results_df["period"] == period)
        ].copy()
        if subset.empty:
            continue

        subset = subset.sort_values("lookback")
        plt.plot(
            subset["lookback"],
            subset["nse"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    plt.title("Comparacao Geral - NSE por Lookback")
    plt.xlabel("Lookback (dias)")
    plt.ylabel("NSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[ExperimentResult] = []

    total = len(MODELS) * len(PERIODS) * len(LOOKBACKS)
    step = 0

    print("=" * 80)
    print("Pipeline TCC iniciado")
    print(f"Total de experimentos: {total}")
    print(f"Trials por estudo: {args.n_trials}")
    print(f"Lookbacks: {LOOKBACKS}")
    print("=" * 80)

    for model_name, script_name in MODELS.items():
        for period_name, period_cfg in PERIODS.items():
            for lookback in LOOKBACKS:
                step += 1
                study_name = f"{model_name}_{period_name}_lb{lookback}"
                print(f"\n[{step}/{total}] Executando {study_name}")

                exit_code, log_path = run_experiment(
                    python_executable=sys.executable,
                    script_name=script_name,
                    study_name=study_name,
                    train_start=period_cfg["train_start"],
                    train_end=period_cfg["train_end"],
                    test_start=args.test_start,
                    test_end=args.test_end,
                    lookback=lookback,
                    args=args,
                )

                if exit_code != 0:
                    print(f"  -> FALHA (exit_code={exit_code}). Pulando para o proximo.")
                    results.append(
                        ExperimentResult(
                            study_name=study_name,
                            model=model_name,
                            period=period_name,
                            lookback=lookback,
                            status="failed",
                            nse=None,
                            kge=None,
                            mae=None,
                            val_loss=None,
                            best_trial=None,
                            best_json=None,
                            log_file=str(log_path),
                            error=f"exit_code={exit_code}",
                        )
                    )
                    continue

                try:
                    payload, best_json = load_best_metrics(study_name)
                    missing_plots = [str(path) for path in expected_plot_paths(study_name) if not path.exists()]
                    if missing_plots:
                        raise RuntimeError(f"graficos ausentes: {missing_plots}")

                    best_metrics = payload.get("best_metrics", {})
                    results.append(
                        ExperimentResult(
                            study_name=study_name,
                            model=model_name,
                            period=period_name,
                            lookback=lookback,
                            status="success",
                            nse=best_metrics.get("NSE"),
                            kge=best_metrics.get("KGE"),
                            mae=best_metrics.get("MAE"),
                            val_loss=best_metrics.get("val_loss"),
                            best_trial=payload.get("best_trial"),
                            best_json=str(best_json),
                            log_file=str(log_path),
                            error=None,
                        )
                    )
                    print("  -> OK")
                except Exception as exc:
                    print(f"  -> FALHA ao carregar resultados: {exc}")
                    results.append(
                        ExperimentResult(
                            study_name=study_name,
                            model=model_name,
                            period=period_name,
                            lookback=lookback,
                            status="failed",
                            nse=None,
                            kge=None,
                            mae=None,
                            val_loss=None,
                            best_trial=None,
                            best_json=None,
                            log_file=str(log_path),
                            error=f"postprocess_error={exc}",
                        )
                    )

    df = pd.DataFrame([vars(item) for item in results])

    summary_csv = REPORTS_DIR / f"tcc_pipeline_summary_{run_id}.csv"
    summary_json = REPORTS_DIR / f"tcc_pipeline_summary_{run_id}.json"
    master_plot = PLOTS_DIR / f"tcc_nse_comparison_by_lookback_{run_id}.png"

    df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(df.to_dict(orient="records"), file, ensure_ascii=False, indent=2)

    try:
        make_master_comparison_plot(df, master_plot)
        print(f"\nGrafico mestre salvo em: {master_plot}")
    except Exception as exc:
        print(f"\n[Aviso] Nao foi possivel gerar grafico mestre: {exc}")

    n_success = int((df["status"] == "success").sum())
    n_failed = int((df["status"] == "failed").sum())

    print("\n" + "=" * 80)
    print("Pipeline TCC finalizado")
    print(f"Sucessos: {n_success}")
    print(f"Falhas: {n_failed}")
    print(f"Resumo CSV: {summary_csv}")
    print(f"Resumo JSON: {summary_json}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
