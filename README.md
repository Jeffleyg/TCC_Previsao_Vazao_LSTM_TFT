# Optuna Workflow

Este repositório foi enxugado para manter apenas o fluxo usado no treino com Optuna.

## Scripts principais

- [prepare_camels_br_dataset.py](prepare_camels_br_dataset.py) prepara os dados diários e os atributos da bacia.
- [assess_basin_training_readiness.py](assess_basin_training_readiness.py) valida se a bacia está pronta para treino.
- [rank_camels_br_basins.py](rank_camels_br_basins.py) ajuda na seleção da bacia.
- [tune_lstm_v2_optuna.py](tune_lstm_v2_optuna.py) faz tuning e treino do LSTM com Optuna.
- [tune_tft_v2_optuna.py](tune_tft_v2_optuna.py) faz tuning e treino do TFT com Optuna.

## Saídas

Os resultados dos estudos ficam em `.dist/optuna/`, com JSON do melhor trial e CSV com todos os trials.

O pipeline de treino agora tambem usa um dataset mestre com ONI em `.dist/71200000_master_dataset.csv`.
Esse dataset mestre junta `Treino Unificado` + `ONI/oni.data.txt` + atributos estaticos de `Atributo/*.txt` e inclui a coluna `period` com:

- `Passado`: 1980-01-01 a 1990-12-31
- `Recente`: 1991-01-01 a 2010-12-31
- `Teste`: 2011-01-01 a 2018-12-31

No TFT, os campos `static_*` entram como features estaticas (categoricas e numericas), e `oni` entra como covariavel temporal conhecida.

Neste estudo, o dataset mestre mantem apenas 8 estaticos prioritarios:

- `static_climate_aridity`
- `static_climate_p_seasonality`
- `static_topography_area`
- `static_topography_slope_mean`
- `static_topography_elev_mean`
- `static_geology_geol_permeability`
- `static_soil_bedrock_depth`
- `static_human_intervention_regulation_degree`

## Execução

Exemplo para LSTM:

```bash
python tune_lstm_v2_optuna.py --n-trials 25 --max-epochs 25
```

Exemplo para TFT:

```bash
python tune_tft_v2_optuna.py --n-trials 20 --max-epochs 20
```

Para reconstruir o dataset mestre antes do treino:

```bash
python tune_lstm_v2_optuna.py --rebuild-master-dataset
```

Por padrao, os scripts usam treino de 1980-01-01 a 2010-12-31 e teste de 2011-01-01 a 2018-12-31.
