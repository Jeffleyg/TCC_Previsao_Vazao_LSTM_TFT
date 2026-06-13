from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera boxplot de LogNSE comparando TFT e LSTM nos regimes passado e recente."
    )
    parser.add_argument("--input", type=str, default="results_summary_tcc.csv")
    parser.add_argument("--output", type=str, default="docs/figuras/boxplot_lognse_tft_lstm_regimes.png")
    return parser.parse_args()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"model", "period", "LogNSE"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    prepared = df.copy()
    prepared["model"] = prepared["model"].astype(str).str.lower().str.strip()
    prepared["period"] = prepared["period"].astype(str).str.lower().str.strip()
    prepared["LogNSE"] = pd.to_numeric(prepared["LogNSE"], errors="coerce")

    prepared = prepared[
        prepared["model"].isin(["lstm", "tft"])
        & prepared["period"].isin(["passado", "recente"])
        & prepared["LogNSE"].notna()
    ].copy()

    if prepared.empty:
        raise ValueError("Nao ha dados validos para model in {lstm, tft} e period in {passado, recente}.")

    return prepared


def plot_lognse_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    order = [
        ("passado", "lstm", "Passado\nLSTM"),
        ("passado", "tft", "Passado\nTFT"),
        ("recente", "lstm", "Recente\nLSTM"),
        ("recente", "tft", "Recente\nTFT"),
    ]
    colors = {
        "lstm": "#1f77b4",
        "tft": "#2ca02c",
    }

    grouped_values: list[np.ndarray] = []
    labels: list[str] = []
    box_colors: list[str] = []

    for period, model, label in order:
        values = df.loc[(df["period"] == period) & (df["model"] == model), "LogNSE"].to_numpy()
        if values.size == 0:
            continue
        grouped_values.append(values)
        labels.append(label)
        box_colors.append(colors[model])

    if not grouped_values:
        raise ValueError("Nao foi possivel montar grupos para o boxplot.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, len(grouped_values) + 1)

    bp = ax.boxplot(
        grouped_values,
        positions=positions,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "#111111", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for pos, values, color in zip(positions, grouped_values, box_colors):
        jitter = rng.normal(0, 0.05, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=28,
            c=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LogNSE")
    ax.set_title("Distribuicao do LogNSE por Modelo e Regime")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    df = pd.read_csv(input_path)
    prepared = _prepare_dataframe(df)
    plot_lognse_boxplot(prepared, output_path)

    print(f"Boxplot salvo em: {output_path}")
    print("Contagens por grupo:")
    print(prepared.groupby(["period", "model"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())