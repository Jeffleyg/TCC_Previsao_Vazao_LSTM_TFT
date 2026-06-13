from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from optuna_hydro_utils import GAUGE_ID, load_daily_data, set_seed
from tune_tft_v2_optuna import train_best_and_plot_hydrograph


DEFAULT_HORIZONS = [7, 15, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera artefatos TFT nativos para a Figura 4.")
    parser.add_argument("--results-csv", type=str, default="results_summary_tcc.csv")
    parser.add_argument("--best-json-dir", type=str, default=".dist/optuna")
    parser.add_argument("--output-dir", type=str, default=".dist/plots")
    parser.add_argument("--period", type=str, default="passado", choices=["passado", "recente"])
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_best_studies(results_csv: Path, period: str, horizons: list[int]) -> list[str]:
    results = pd.read_csv(results_csv)
    results["model"] = results["model"].astype(str).str.lower()
    results["period"] = results["period"].astype(str).str.lower()

    studies: list[str] = []
    for horizon in horizons:
        subset = results[(results["model"] == "tft") & (results["period"] == period) & (results["horizon"] == horizon)].copy()
        if subset.empty:
            raise ValueError(f"Nenhum TFT encontrado para period={period}, horizon={horizon}.")
        best_row = subset.sort_values(["LogNSE", "NSE"], ascending=False).iloc[0]
        studies.append(f"tft_{period}_lb{int(best_row['lookback'])}_h{int(best_row['horizon'])}")
    return studies


def load_best_payload(best_json_dir: Path, study_name: str) -> dict:
    payload_path = best_json_dir / f"{study_name}_best.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Arquivo best.json nao encontrado: {payload_path}")
    with open(payload_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    args = parse_args()
    results_csv = Path(args.results_csv)
    best_json_dir = Path(args.best_json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_names = load_best_studies(results_csv, args.period, args.horizons)
    print(f"Estudos selecionados: {study_names}")

    for study_name in study_names:
        payload = load_best_payload(best_json_dir, study_name)
        best_params = payload.get("best_params", {})
        search_config = payload.get("search_config", {})
        lookback = int(study_name.split("_lb")[-1].split("_h")[0])

        max_epochs = int(args.max_epochs)
        patience = int(args.patience)
        seed = int(args.seed)

        run_args = SimpleNamespace(
            study_name=study_name,
            max_epochs=max_epochs,
            patience=patience,
            generate_explanations=True,
            fixed_lookback=lookback,
            forecast_horizon=int(study_name.split("_h")[-1]),
            train_start=search_config.get("train_start", "1980-01-01"),
            train_end=search_config.get("train_end", "2010-12-31"),
            test_start=search_config.get("test_start", "2011-01-01"),
            test_end=search_config.get("test_end", "2018-12-31"),
            val_fraction=search_config.get("val_fraction", 0.15),
        )
        data_dir = search_config.get("data_dir", "Treino Unificado")
        master_csv = search_config.get("master_csv", f".dist/{GAUGE_ID}_master_dataset.csv")
        rebuild_master = bool(search_config.get("rebuild_master_dataset", False))

        print(f"\nReexecutando {study_name} para gerar interpretabilidade nativa...")
        set_seed(seed)
        df = load_daily_data(
            station_id=GAUGE_ID,
            data_dir=data_dir,
            master_csv_relative_path=master_csv,
            rebuild_master=rebuild_master,
        )
        hydrograph_png = output_dir / f"{study_name}_hydrogram_test.png"
        train_best_and_plot_hydrograph(df=df, args=run_args, best_params=best_params, output_png=str(hydrograph_png))

    print("\nArtefatos de interpretabilidade gerados com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())