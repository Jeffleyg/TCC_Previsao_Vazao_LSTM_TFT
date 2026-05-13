#!/usr/bin/env python3
"""
Script rápido para testar explicabilidade do TFT com um único trial.
Útil para validar a integração antes de rodar experimentos completos.

Uso:
  python test_tft_explanations.py --lookback 30 --n-trials 1
  python test_tft_explanations.py --lookback 48 --generate-explanations
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def main():
    # Argumento: lookback e outras flags
    lookback = "30"
    study_name = f"tft_test_lb{lookback}"
    
    cmd = [
        sys.executable,
        "tune_tft_v2_optuna.py",
        "--study-name", study_name,
        "--train-start", "1980-01-01",
        "--train-end", "2005-12-31",  # Período reduzido para teste rápido
        "--test-start", "2006-01-01",
        "--test-end", "2010-12-31",
        "--n-trials", "1",  # Apenas 1 trial para teste
        "--max-epochs", "3",  # Epochs reduzidos para teste
        "--seed", "42",
        "--fixed-lookback", lookback,
        "--generate-explanations",  # 🔥 EXPLICABILIDADE ATIVADA
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTE RÁPIDO - TFT COM EXPLICABILIDADE")
    print("="*70)
    print(f"Comando: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print("\n📁 Arquivos gerados:")
        print(f"   • JSON (Feature Importance): .dist/optuna/{study_name}_tft_feature_importance.json")
        print(f"   • PNG (Atenção): .dist/plots/{study_name}_tft_attention_heatmap.png")
        print(f"   • Hidrograma: .dist/plots/{study_name}_hydrogram_test.png")
        return 0
    else:
        print(f"\n❌ TESTE FALHOU (exit code: {result.returncode})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
