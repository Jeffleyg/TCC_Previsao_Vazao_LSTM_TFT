from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera grafico consolidado de Variable Importance do TFT."
    )
    parser.add_argument("--input-dir", type=str, default=".dist/plots")
    parser.add_argument("--output", type=str, default="docs/figuras/tft_variable_importance.png")
    parser.add_argument("--top-k", type=int, default=7)
    return parser.parse_args()


def load_feature_importance(input_dir: Path) -> pd.DataFrame:
    pattern = re.compile(r"^tft_(passado|recente)_lb\d+_h\d+_tft_feature_importance\.json$")
    rows: list[dict] = []

    for json_path in sorted(input_dir.glob("*_tft_feature_importance.json")):
        match = pattern.match(json_path.name)
        if not match:
            continue
        period = match.group(1)

        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        feature_map = payload.get("feature_importance", {})
        if not isinstance(feature_map, dict):
            continue

        for feature_name, score in feature_map.items():
            rows.append(
                {
                    "period": period,
                    "feature": str(feature_name),
                    "score": float(score),
                    "source": json_path.name,
                }
            )

    if not rows:
        raise ValueError(
            "Nenhum JSON de feature importance do TFT encontrado em formato esperado."
        )

    return pd.DataFrame(rows)


def build_chart(df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    mean_by_period = (
        df.groupby(["feature", "period"], as_index=False)["score"]
        .mean()
        .pivot(index="feature", columns="period", values="score")
        .fillna(0.0)
    )

    for col in ["passado", "recente"]:
        if col not in mean_by_period.columns:
            mean_by_period[col] = 0.0

    mean_by_period["overall"] = mean_by_period[["passado", "recente"]].mean(axis=1)
    top = mean_by_period.sort_values("overall", ascending=False).head(max(1, top_k))
    top = top.sort_values("overall", ascending=True)

    y = np.arange(len(top))
    h = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.barh(y - h / 2, top["passado"], height=h, color="#2E86DE", label="Passado")
    ax.barh(y + h / 2, top["recente"], height=h, color="#17A589", label="Recente")

    ax.set_yticks(y)
    ax.set_yticklabels(top.index.tolist())
    ax.set_xlabel("Importancia media (score)")
    ax.set_title("TFT - Variable Importance (Consolidado por Regime)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Mostra o valor medio consolidado para facilitar leitura no TCC.
    for i, value in enumerate(top["overall"].to_numpy()):
        ax.text(value + 1.0, i, f"{value:.1f}", va="center", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretorio de entrada nao encontrado: {input_dir}")

    df = load_feature_importance(input_dir)
    build_chart(df, output_path, args.top_k)

    counts = df.groupby("period")["source"].nunique().to_dict()
    print(f"Grafico salvo em: {output_path}")
    print(f"Arquivos usados por regime: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())