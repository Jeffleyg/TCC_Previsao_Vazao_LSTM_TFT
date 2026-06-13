from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STUDIES = [
    "tft_passado_lb96_h7",
    "tft_passado_lb96_h15",
    "tft_passado_lb96_h30",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera a Figura 4 com interpretabilidade do TFT.")
    parser.add_argument("--plots-dir", type=str, default=".dist/plots")
    parser.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES)
    parser.add_argument("--output", type=str, default="docs/figuras/figura4_tft_interpretability.png")
    return parser.parse_args()


def load_payload(plots_dir: Path, study: str) -> dict:
    payload_path = plots_dir / f"{study}_tft_feature_importance.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Arquivo de interpretabilidade ausente: {payload_path}")
    with open(payload_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    payload["_path"] = str(payload_path)
    return payload


def as_dict(mapping: dict[str, float] | None) -> dict[str, float]:
    return mapping if isinstance(mapping, dict) else {}


def plot_heatmap(ax, attention: list[list[float]], title: str) -> None:
    # convert to numpy and handle NaNs
    matrix = np.asarray(attention, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    # compute mean attention across samples (ignore NaNs)
    mean_att = np.nanmean(matrix, axis=0)

    # downsample to at most 120 points for readability
    max_bins = 120
    n = len(mean_att)
    if n == 0:
        return
    bin_size = max(1, n // max_bins)
    # windowed average
    binned = np.array([np.nanmean(mean_att[i : i + bin_size]) for i in range(0, n, bin_size)])
    # simple gaussian smoothing
    def gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
        x = np.arange(-radius, radius + 1)
        k = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return k / k.sum()

    sigma = max(1.0, bin_size / 2.0)
    kr = int(3 * sigma)
    kernel = gaussian_kernel(sigma, kr)
    smooth = np.convolve(np.nan_to_num(binned, nan=0.0), kernel, mode="same")

    # create a small band to display the mean with some vertical thickness
    nrows = 6
    band = np.tile(smooth, (nrows, 1))
    cmap = "cividis"
    image = ax.imshow(band, aspect="auto", cmap=cmap, interpolation="bicubic")

    # overlay mean attention line for readability
    norm = (smooth - np.nanmin(smooth)) / max(1e-9, (np.nanmax(smooth) - np.nanmin(smooth)))
    line_y = norm * (nrows - 1)
    ax.plot(np.arange(len(smooth)), line_y, color="white", linewidth=1.6)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lags (passado → presente)")
    ax.set_yticks([])
    # annotate recent window (rightmost 10%)
    recent_frac = 0.10
    recent_start = int(len(smooth) * (1 - recent_frac))
    ax.axvspan(recent_start, len(smooth) - 1, color="white", alpha=0.08)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean attention")


def main() -> int:
    args = parse_args()
    plots_dir = Path(args.plots_dir)
    output_path = Path(args.output)

    payloads = []
    for study in args.studies:
        payload = load_payload(plots_dir, study)
        payload["study"] = study
        payloads.append(payload)

    horizons = [int(payload.get("forecast_horizon", 1)) for payload in payloads]

    def get_encoder_feature_series(payload: dict, feature: str) -> np.ndarray:
        enc = payload.get("encoder_variables", [])
        if isinstance(enc, dict):
            return np.array([enc.get(feature, 0.0)])
        if isinstance(enc, list):
            vals = [entry.get(feature, 0.0) for entry in enc if isinstance(entry, dict)]
            return np.array(vals, dtype=float) if vals else np.array([0.0])
        return np.array([0.0])

    # aggregate top variables across payloads (mean across lags)
    top_variables = []
    for payload in payloads:
        enc_list = payload.get("encoder_variables", [])
        if isinstance(enc_list, dict):
            for name, score in enc_list.items():
                top_variables.append((name, float(score)))
        elif isinstance(enc_list, list):
            # compute mean importance per variable across lags
            if len(enc_list) > 0 and isinstance(enc_list[0], dict):
                names = enc_list[0].keys()
                for name in names:
                    vals = [d.get(name, 0.0) for d in enc_list]
                    top_variables.append((name, float(np.nanmean(vals))))
    ranked_names = [name for name, _ in sorted(top_variables, key=lambda item: item[1], reverse=True)[:4]]
    ranked_names = list(dict.fromkeys([name for name in ranked_names if name]))
    if "oni" not in ranked_names:
        ranked_names.insert(0, "oni")

    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    for index, payload in enumerate(payloads):
        ax = fig.add_subplot(grid[0, index])
        plot_heatmap(
            ax,
            payload.get("attention", [[0.0]]),
            f"Attention heatmap - h={horizons[index]}d",
        )

    bar_ax = fig.add_subplot(grid[1, :2])
    x = np.arange(len(horizons))
    bar_width = 0.18
    plotted_names = ranked_names[:4]
    colors = ["#2E86DE", "#17A589", "#D35400", "#7D3C98"]

    for offset, feature_name in enumerate(plotted_names):
        values = []
        for payload in payloads:
            series = get_encoder_feature_series(payload, feature_name)
            values.append(float(np.nanmean(series)))
        # convert to percentage of sum for readability if not already
        vals = np.array(values, dtype=float)
        if np.nanmax(vals) <= 1.0:
            vals = vals * 100.0
        bar_ax.bar(x + (offset - (len(plotted_names) - 1) / 2) * bar_width, vals, width=bar_width, label=feature_name.upper(), color=colors[offset % len(colors)])

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([f"{h}d" for h in horizons])
    bar_ax.set_ylabel("Importância (%)")
    bar_ax.set_title("Pesos por horizonte - variáveis encoder")
    bar_ax.legend(ncol=2, fontsize=9)
    bar_ax.grid(axis="y", alpha=0.3)

    oni_ax = fig.add_subplot(grid[1, 2])
    # plot ONI importance across encoder lags for each study
    for payload, h in zip(payloads, horizons):
        oni_series = get_encoder_feature_series(payload, "oni")
        if oni_series.size == 0:
            continue
        # downsample for plotting
        max_points = 200
        n = len(oni_series)
        if n > max_points:
            k = max(1, n // max_points)
            oni_ds = np.array([np.nanmean(oni_series[i : i + k]) for i in range(0, n, k)])
        else:
            oni_ds = oni_series
        # normalize to max and convert to percent
        oni_norm = (oni_ds - np.nanmin(oni_ds))
        if np.nanmax(oni_norm) > 0:
            oni_norm = oni_norm / np.nanmax(oni_norm) * 100.0
        else:
            oni_norm = oni_norm * 0.0
        xaxis = np.arange(len(oni_norm))
        oni_ax.plot(xaxis, oni_norm, label=f"h={h}d", linewidth=1.8)
    oni_ax.set_title("ONI: importância por lag (normalizada)")
    oni_ax.set_xlabel("Lags (passado → presente)")
    oni_ax.set_ylabel("Rel. importance (%)")
    oni_ax.legend(fontsize=9)
    oni_ax.grid(alpha=0.3)

    fig.suptitle("Figura 4 - Interpretabilidade do TFT: atenção, ONI e pesos por horizonte", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura 4 salva em: {output_path}")
    print(f"Horizontes usados: {horizons}")
    print("ONI: plotado por lag para cada estudo (normalizado)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())