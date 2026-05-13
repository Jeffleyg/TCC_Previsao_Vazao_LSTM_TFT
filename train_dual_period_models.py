"""
Treina 4 modelos com lookback fixo (30 dias):
- LSTM Passado (1980-1990)
- LSTM Recente (1991-2010)
- TFT Passado (1980-1990)
- TFT Recente (1991-2010)

Uso:
  python train_dual_period_models.py --lookback 30 --n-trials 20 --max-epochs 15 --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Executa um comando e retorna o exit code."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Comando: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"\n❌ ERRO em {description} (exit code: {result.returncode})")
        return result.returncode
    else:
        print(f"\n✅ {description} concluído com sucesso")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Treina 4 modelos (LSTM/TFT x Passado/Recente) com lookback fixo"
    )
    parser.add_argument("--lookback", type=int, default=30, help="Lookback em dias")
    parser.add_argument("--n-trials", type=int, default=20, help="Número de trials Optuna")
    parser.add_argument("--max-epochs", type=int, default=15, help="Máximo de epochs por trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--master-csv", type=str, default=".dist/71200000_master_dataset.csv")
    parser.add_argument("--rebuild-master-dataset", action="store_true")
    parser.add_argument("--generate-explanations", action="store_true", help="Gera visualizacoes explicaveis do TFT")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Horizonte de previsao em dias")
    args = parser.parse_args()

    # Configurações dos 4 modelos
    models = [
        {
            "name": "LSTM Passado",
            "script": "tune_lstm_v2_optuna.py",
            "study_name": f"lstm_passado_lb{args.lookback}",
            "train_start": "1980-01-01",
            "train_end": "1990-12-31",
        },
        {
            "name": "LSTM Recente",
            "script": "tune_lstm_v2_optuna.py",
            "study_name": f"lstm_recente_lb{args.lookback}",
            "train_start": "1991-01-01",
            "train_end": "2010-12-31",
        },
        {
            "name": "TFT Passado",
            "script": "tune_tft_v2_optuna.py",
            "study_name": f"tft_passado_lb{args.lookback}",
            "train_start": "1980-01-01",
            "train_end": "1990-12-31",
        },
        {
            "name": "TFT Recente",
            "script": "tune_tft_v2_optuna.py",
            "study_name": f"tft_recente_lb{args.lookback}",
            "train_start": "1991-01-01",
            "train_end": "2010-12-31",
        },
    ]

    print(f"\n🎯 Configuração:")
    print(f"   • Lookback: {args.lookback} dias")
    print(f"   • Trials: {args.n_trials} por modelo")
    print(f"   • Max epochs: {args.max_epochs}")
    print(f"   • Seed: {args.seed}")
    print(f"   • Período teste: 2011-01-01 a 2018-12-31")
    print(f"\n📋 Modelos a treinar:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model['name']} ({model['train_start']} → {model['train_end']})")

    # Executa os 4 treinamentos
    failed_models = []
    for i, model in enumerate(models, 1):
        cmd = [
            sys.executable,
            model["script"],
            "--study-name", model["study_name"],
            "--train-start", model["train_start"],
            "--train-end", model["train_end"],
            "--test-start", "2011-01-01",
            "--test-end", "2018-12-31",
            "--n-trials", str(args.n_trials),
            "--max-epochs", str(args.max_epochs),
            "--seed", str(args.seed),
            "--master-csv", args.master_csv,
            "--fixed-lookback", str(args.lookback),
            "--forecast-horizon", str(args.forecast_horizon),
        ]
        
        # Adiciona explicabilidade apenas para TFT
        if args.generate_explanations and "tft" in model["script"]:
            cmd.append("--generate-explanations")
        
        if args.rebuild_master_dataset:
            cmd.append("--rebuild-master-dataset")

        description = f"[{i}/4] {model['name']} (lookback={args.lookback}d)"
        exit_code = run_command(cmd, description)
        
        if exit_code != 0:
            failed_models.append(model["name"])

    # Resumo final
    print(f"\n{'='*70}")
    print("📊 RESUMO FINAL")
    print(f"{'='*70}")
    
    if failed_models:
        print(f"❌ {len(failed_models)} modelo(s) falharam:")
        for name in failed_models:
            print(f"   • {name}")
        return 1
    else:
        print("✅ Todos os 4 modelos treinados com sucesso!")
        print(f"\n📁 Arquivos de saída em: .dist/optuna/")
        print(f"   • lstm_passado_lb{args.lookback}_best.json")
        print(f"   • lstm_recente_lb{args.lookback}_best.json")
        print(f"   • tft_passado_lb{args.lookback}_best.json")
        print(f"   • tft_recente_lb{args.lookback}_best.json")
        return 0


if __name__ == "__main__":
    sys.exit(main())
