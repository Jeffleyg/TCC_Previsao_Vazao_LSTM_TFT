#!/usr/bin/env python3
"""
Fase 2: Exploração de Múltiplos Horizontes (7, 15, 30 dias)

Este script expande os melhores modelos da Fase 1 para testar horizontes mais longos.
Usa configuração de lookbacks baseados na física (48, 96, 144, 192 dias).

Cronograma Estimado:
  • ~2-3 horas por horizonte × 2 modelos
  • Horizonte 7d: ~6-9 horas (4 experimentos)
  • Horizonte 15d: ~6-9 horas (4 experimentos)
  • Horizonte 30d: ~6-9 horas (4 experimentos)
  • TOTAL Fase 2: ~18-27 horas

Uso:
  # Teste rápido com horizonte 7d (só TFT para economizar tempo)
  python run_phase2_multi_horizon_experiments.py --horizons 7 --n-trials 10 --skip-lstm

  # Teste de horizonte 7d completo
  python run_phase2_multi_horizon_experiments.py --horizons 7 --n-trials 20 --generate-tft-explanations

  # Todos os horizontes (será longo!)
  python run_phase2_multi_horizon_experiments.py --horizons 7 15 30 --n-trials 20 --generate-tft-explanations

  # Apenas horizonte 30d com lookback reduzido
  python run_phase2_multi_horizon_experiments.py --horizons 30 --lookbacks 48 96 --n-trials 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).parent
OPTUNA_DIR = PROJECT_DIR / ".dist" / "optuna"
PLOTS_DIR = PROJECT_DIR / ".dist" / "plots"
REPORTS_DIR = PROJECT_DIR / ".dist" / "reports"


@dataclass
class ExperimentConfig:
    model: str
    period: str
    lookback_days: int
    forecast_horizon_days: int
    train_start: str
    train_end: str


@dataclass
class ExperimentResult:
    study_name: str
    model: str
    period: str
    lookback: int
    horizon: int
    status: str
    nse: float | None
    kge: float | None
    mae: float | None
    val_loss: float | None
    best_trial: int | None
    best_json: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 2: Exploração de múltiplos horizontes (7, 15, 30 dias)"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[7],
                        help="Horizontes a testar em dias (ex: 7 15 30)")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[48, 96, 144, 192],
                        help="Lookbacks em dias (ex: 48 96 144 192)")
    parser.add_argument("--n-trials", type=int, default=20, help="Trials Optuna por experimento")
    parser.add_argument("--max-epochs", type=int, default=15, help="Máximo de epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate-tft-explanations", action="store_true",
                        help="Ativa explicabilidade do TFT")
    parser.add_argument("--skip-lstm", action="store_true", help="Pula LSTM (só TFT)")
    parser.add_argument("--skip-tft", action="store_true", help="Pula TFT (só LSTM)")
    parser.add_argument("--skip-baseline", action="store_true", help="Pula horizonte 1d (baseline)")
    return parser.parse_args()


def ensure_dirs() -> None:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    """Gera lista de experimentos para todos os horizontes e lookbacks."""
    models = ["lstm", "tft"]
    if args.skip_lstm:
        models.remove("lstm")
    if args.skip_tft:
        models.remove("tft")

    periods = [
        ("passado", "1980-01-01", "1990-12-31"),
        ("recente", "1991-01-01", "2010-12-31"),
    ]

    return [
        ExperimentConfig(
            model=model,
            period=period_name,
            lookback_days=lookback,
            forecast_horizon_days=horizon,
            train_start=train_start,
            train_end=train_end,
        )
        for horizon, model, (period_name, train_start, train_end), lookback in product(
            args.horizons,
            models,
            periods,
            args.lookbacks,
        )
    ]


def run_experiment(config: ExperimentConfig, args: argparse.Namespace) -> int:
    """Executa um único experimento."""
    study_name = f"{config.model}_{config.period}_lb{config.lookback_days}_h{config.forecast_horizon_days}"
    script_name = "tune_lstm_v2_optuna.py" if config.model == "lstm" else "tune_tft_v2_optuna.py"
    
    cmd = [
        sys.executable,
        script_name,
        "--study-name", study_name,
        "--train-start", config.train_start,
        "--train-end", config.train_end,
        "--test-start", "2011-01-01",
        "--test-end", "2018-12-31",
        "--n-trials", str(args.n_trials),
        "--max-epochs", str(args.max_epochs),
        "--seed", str(args.seed),
        "--fixed-lookback", str(config.lookback_days),
        "--forecast-horizon", str(config.forecast_horizon_days),
    ]
    
    if config.model == "tft" and args.generate_tft_explanations:
        cmd.append("--generate-explanations")
    
    emoji_model = "🧠" if config.model == "lstm" else "🎯"
    emoji_period = "📜" if config.period == "passado" else "📊"
    
    print(f"\n{'='*70}")
    print(f"{emoji_model} [{config.model.upper():4s}] {emoji_period} {config.period:8s} | "
          f"lb={config.lookback_days:3d}d | h={config.forecast_horizon_days:2d}d")
    print(f"{'='*70}")
    print(f"Study: {study_name}")
    print(f"Período treino: {config.train_start} → {config.train_end}")
    print(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode


def load_best_metrics(study_name: str) -> tuple[dict, Path]:
    best_json_path = OPTUNA_DIR / f"{study_name}_best.json"
    with open(best_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload, best_json_path


def make_final_summary_plot(results_df: pd.DataFrame, output_png: Path) -> None:
    success_df = results_df[results_df["status"] == "success"].copy()
    if success_df.empty:
        raise RuntimeError("Nao ha resultados validos para montar o grafico final.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=False)
    axes = axes.ravel()

    colors = {"lstm": "#1f77b4", "tft": "#d62728"}
    markers = {"passado": "o", "recente": "s"}
    plots = [
        ("nse", "NSE por Horizonte", "horizon", "Horizonte (dias)", "NSE"),
        ("kge", "KGE por Horizonte", "horizon", "Horizonte (dias)", "KGE"),
        ("mae", "MAE por Lookback", "lookback", "Lookback (dias)", "MAE"),
        ("val_loss", "Val Loss por Lookback", "lookback", "Lookback (dias)", "Val Loss"),
    ]

    for ax, (metric_name, title, group_key, xlabel, ylabel) in zip(axes, plots):
        for model in ["lstm", "tft"]:
            for period in ["passado", "recente"]:
                subset = success_df[(success_df["model"] == model) & (success_df["period"] == period)].copy()
                if subset.empty:
                    continue

                grouped = subset.groupby(group_key, as_index=False)[metric_name].mean().sort_values(group_key)
                ax.plot(
                    grouped[group_key],
                    grouped[metric_name],
                    marker=markers[period],
                    linewidth=2,
                    color=colors[model],
                    label=f"{model.upper()}-{period}",
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Resumo Final - Fase 2 (Horizontes + Lookbacks)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    experiments = generate_experiments(args)
    results: list[ExperimentResult] = []
    
    print("\n" + "="*70)
    print("🔬 FASE 2: EXPLORAÇÃO DE MÚLTIPLOS HORIZONTES")
    print("="*70)
    print(f"\n📊 Configuração:")
    print(f"   • Horizontes: {args.horizons} dias")
    print(f"   • Lookbacks: {args.lookbacks} dias")
    print(f"   • Trials por experimento: {args.n_trials}")
    print(f"   • Total de experimentos: {len(experiments)}")
    print(f"   • Tempo estimado: {len(experiments) * 10 // 60:.1f}h (aprox)")
    print(f"   • Explicabilidade TFT: {'✅ ATIVADA' if args.generate_tft_explanations else '❌ Desativada'}")
    
    print(f"\n📋 Experimentos a rodar:")
    for i, exp in enumerate(experiments, 1):
        emoji_model = "🧠" if exp.model == "lstm" else "🎯"
        emoji_period = "📜" if exp.period == "passado" else "📊"
        print(f"   {i:2d}. {emoji_model} {exp.model.upper():4s} {emoji_period} {exp.period:8s} "
              f"lb={exp.lookback_days:3d}d h={exp.forecast_horizon_days:2d}d")
    
    input("\n⏸️  Pressione ENTER para iniciar a Fase 2...")
    
    failed = []
    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]", end=" ")
        study_name = f"{config.model}_{config.period}_lb{config.lookback_days}_h{config.forecast_horizon_days}"
        exit_code = run_experiment(config, args)
        
        if exit_code != 0:
            failed.append(config)
            print("❌ ERRO")
            results.append(
                ExperimentResult(
                    study_name=study_name,
                    model=config.model,
                    period=config.period,
                    lookback=config.lookback_days,
                    horizon=config.forecast_horizon_days,
                    status="failed",
                    nse=None,
                    kge=None,
                    mae=None,
                    val_loss=None,
                    best_trial=None,
                    best_json=None,
                    error=f"exit_code={exit_code}",
                )
            )
        else:
            try:
                payload, best_json = load_best_metrics(study_name)
                best_metrics = payload.get("best_metrics", {})
                results.append(
                    ExperimentResult(
                        study_name=study_name,
                        model=config.model,
                        period=config.period,
                        lookback=config.lookback_days,
                        horizon=config.forecast_horizon_days,
                        status="success",
                        nse=best_metrics.get("NSE"),
                        kge=best_metrics.get("KGE"),
                        mae=best_metrics.get("MAE"),
                        val_loss=best_metrics.get("val_loss"),
                        best_trial=payload.get("best_trial"),
                        best_json=str(best_json),
                        error=None,
                    )
                )
                print("✅ OK")
            except Exception as exc:
                failed.append(config)
                results.append(
                    ExperimentResult(
                        study_name=study_name,
                        model=config.model,
                        period=config.period,
                        lookback=config.lookback_days,
                        horizon=config.forecast_horizon_days,
                        status="failed",
                        nse=None,
                        kge=None,
                        mae=None,
                        val_loss=None,
                        best_trial=None,
                        best_json=None,
                        error=f"postprocess_error={exc}",
                    )
                )
                print(f"❌ ERRO no pós-processamento: {exc}")
                continue
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FASE 2")
    print(f"{'='*70}")
    print(f"Total: {len(experiments)} experimentos")
    print(f"✅ Sucesso: {len(experiments) - len(failed)}")
    print(f"❌ Falhas: {len(failed)}")

    results_df = pd.DataFrame([vars(item) for item in results])
    summary_csv = REPORTS_DIR / f"phase2_multi_horizon_summary_{args.horizons[0]}_{len(experiments)}.csv"
    summary_png = PLOTS_DIR / f"phase2_final_summary_{args.horizons[0]}_{len(experiments)}.png"
    results_df.to_csv(summary_csv, index=False)

    try:
        make_final_summary_plot(results_df, summary_png)
        print(f"\nImagem final consolidada salva em: {summary_png}")
    except Exception as exc:
        print(f"\n[Aviso] Nao foi possivel gerar a imagem final consolidada: {exc}")

    print(f"Resumo CSV: {summary_csv}")
    
    if failed:
        print(f"\n❌ Experimentos que falharam:")
        for config in failed:
            print(f"   • {config.model.upper()} {config.period} lb={config.lookback_days}d h={config.forecast_horizon_days}d")
        return 1

    print(f"\n🎉 Todos os experimentos completados com sucesso!")
    print(f"\n📁 Arquivos salvos em:")
    print("   • Resultados: .dist/optuna/")
    print("   • Plots: .dist/plots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
