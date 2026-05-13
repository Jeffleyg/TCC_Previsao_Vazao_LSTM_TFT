#!/usr/bin/env python3
"""
Fase 1: Validação da Abordagem Baseada na Física da Bacia

Este script executa 8 experimentos com lookbacks baseados no ciclo físico (~48 dias):
  • 2 modelos: LSTM, TFT
  • 2 períodos: Passado (1980-1990), Recente (1991-2010)
  • 4 lookbacks: [48, 96, 144, 192] dias (múltiplos do ciclo)

Com explicabilidade ativada para o TFT (feature importance + attention).

Cronograma Estimado:
  • ~2-3 horas por modelo (com 20 trials)
  • Total: ~8-12 horas para os 8 experimentos

Uso:
  python run_phase1_physics_based_lookbacks.py --n-trials 20 --generate-tft-explanations
  python run_phase1_physics_based_lookbacks.py --n-trials 10  # Rápido (teste)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / ".dist" / "phase1_results"


@dataclass
class ExperimentConfig:
    model: str  # "lstm" ou "tft"
    period: str  # "passado" ou "recente"
    lookback_days: int
    forecast_horizon_days: int
    train_start: str
    train_end: str


EXPERIMENTS = [
    ExperimentConfig("lstm", "passado", 48, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("lstm", "recente", 48, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("tft", "passado", 48, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("tft", "recente", 48, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("lstm", "passado", 96, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("lstm", "recente", 96, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("tft", "passado", 96, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("tft", "recente", 96, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("lstm", "passado", 144, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("lstm", "recente", 144, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("tft", "passado", 144, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("tft", "recente", 144, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("lstm", "passado", 192, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("lstm", "recente", 192, 1, "1991-01-01", "2010-12-31"),
    ExperimentConfig("tft", "passado", 192, 1, "1980-01-01", "1990-12-31"),
    ExperimentConfig("tft", "recente", 192, 1, "1991-01-01", "2010-12-31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 1: Experimentos com lookbacks baseados na física (48, 96, 144, 192 dias)"
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Número de trials Optuna por experimento")
    parser.add_argument("--max-epochs", type=int, default=15, help="Máximo de epochs por trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate-tft-explanations", action="store_true", help="Ativa explicabilidade (feature importance) do TFT")
    parser.add_argument("--skip-lstm", action="store_true", help="Pula experimentos LSTM (só TFT)")
    parser.add_argument("--skip-tft", action="store_true", help="Pula experimentos TFT (só LSTM)")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Horizonte de previsao em dias (sobreescreve configuracao dos experimentos)")
    return parser.parse_args()


def run_experiment(config: ExperimentConfig, args: argparse.Namespace) -> int:
    """Executa um único experimento."""
    # Permite sobrescrever horizonte via CLI
    horizon = args.forecast_horizon
    study_name = f"{config.model}_{config.period}_lb{config.lookback_days}_h{horizon}"
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
        "--forecast-horizon", str(horizon),
    ]
    
    # Adiciona explicabilidade apenas para TFT se solicitado
    if config.model == "tft" and args.generate_tft_explanations:
        cmd.append("--generate-explanations")
    
    print(f"\n{'='*70}")
    print(f"🚀 [{config.model.upper()}] {config.period.capitalize()} - Lookback {config.lookback_days}d, Horizonte {horizon}d")
    print(f"{'='*70}")
    print(f"Study: {study_name}")
    print(f"Período treino: {config.train_start} → {config.train_end}")
    print(f"Período teste: 2011-01-01 → 2018-12-31")
    print(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode


def main() -> int:
    args = parse_args()
    
    # Filtra experimentos
    experiments = EXPERIMENTS
    if args.skip_lstm:
        experiments = [e for e in experiments if e.model != "lstm"]
    if args.skip_tft:
        experiments = [e for e in experiments if e.model != "tft"]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🧬 FASE 1: VALIDAÇÃO DA ABORDAGEM BASEADA NA FÍSICA DA BACIA")
    print("="*70)
    print(f"\n📊 Configuração:")
    print(f"   • Lookbacks: [48, 96, 144, 192] dias (múltiplos do ciclo ~48d)")
    print(f"   • Experimentos: {len(experiments)}")
    print(f"   • Trials por experimento: {args.n_trials}")
    print(f"   • Max epochs: {args.max_epochs}")
    print(f"   • Explicabilidade TFT: {'✅ ATIVADA' if args.generate_tft_explanations else '❌ Desativada'}")
    print(f"\n📋 Experimentos a rodar:")
    
    for i, exp in enumerate(experiments, 1):
        model_emoji = "🧠" if exp.model == "lstm" else "🎯"
        period_emoji = "📜" if exp.period == "passado" else "📊"
        print(f"   {i:2d}. {model_emoji} {exp.model.upper():4s} {period_emoji} {exp.period:8s} lb={exp.lookback_days:3d}d")
    
    input("\n⏸️  Pressione ENTER para iniciar a Fase 1...")
    
    failed = []
    for i, config in enumerate(experiments, 1):
        status = f"[{i}/{len(experiments)}]"
        exit_code = run_experiment(config, args)
        
        if exit_code != 0:
            failed.append(config)
            print(f"\n❌ ERRO: {status} {config.model} {config.period} lb{config.lookback_days}")
        else:
            print(f"\n✅ SUCESSO: {status} {config.model} {config.period} lb{config.lookback_days}")
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FASE 1")
    print(f"{'='*70}")
    print(f"Total: {len(experiments)} experimentos")
    print(f"✅ Sucesso: {len(experiments) - len(failed)}")
    print(f"❌ Falhas: {len(failed)}")
    
    if failed:
        print(f"\n❌ Experimentos que falharam:")
        for config in failed:
            print(f"   • {config.model.upper()} {config.period} lb{config.lookback_days}d")
        return 1
    else:
        print(f"\n🎉 Todos os experimentos completados com sucesso!")
        print(f"\n📁 Arquivos salvos em:")
        print(f"   • Resultados: .dist/optuna/")
        print(f"   • Plots: .dist/plots/")
        if args.generate_tft_explanations:
            print(f"   • Explicações TFT: .dist/optuna/*_tft_feature_importance.json")
            print(f"   • Atenção TFT: .dist/plots/*_tft_attention_heatmap.png")
        return 0


if __name__ == "__main__":
    sys.exit(main())
