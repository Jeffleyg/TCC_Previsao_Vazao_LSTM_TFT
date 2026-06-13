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

Veja a versão final e enxuta da **Metodologia** para inclusão no TCC em [docs/METODOLOGIA_FINAL.md](docs/METODOLOGIA_FINAL.md).

---

## 3. Metodologia

O desenvolvimento deste estudo segue o framework metodológico **CRISP‑DM** (Cross Industry Standard Process for Data Mining). Abaixo encontra‑se a versão adaptada e alinhada aos scripts e ao dataset mestre utilizado para a bacia **71200000**.

```text
			 +--------------------------------------------+
			 |   3.1. Compreensão do Negócio (Business)    |
			 +---------------------+----------------------+ 
														 |
														 v
			 +---------------------+----------------------+ 
			 |     3.2. Compreensão dos Dados (Data)       |
			 +---------------------+----------------------+
														 |
														 v
			 +---------------------+----------------------+ 
			 |   3.3. Preparação dos Dados (Preparation)   |
			 +---------------------+----------------------+
														 |
														 v
			 +---------------------+----------------------+ 
			 |            3.4. Modelagem (Modeling)       |
			 +---------------------+----------------------+
														 |
														 v
			 +---------------------+----------------------+ 
			 |          3.5. Avaliação (Evaluation)        |
			 +---------------------+----------------------+
														 |
														 v
			 +---------------------+----------------------+ 
			 |         3.6. Implantação (Deployment)      |
			 +---------------------+----------------------+

```

### 3.1 Compreensão do Negócio
O problema foi formulado como previsão multi‑horizonte (7, 15 e 30 dias) da vazão para suporte a decisões de gestão de recursos hídricos, com ênfase na detecção e acompanhamento de episódios de seca extrema. O critério de sucesso operacional é a capacidade de prever o decaimento do fluxo de base em horizontes subsazonais.

### 3.2 Compreensão dos Dados
O inventário deriva do CAMELS‑BR e foi organizado em: (i) variável alvo dinâmica `streamflow_m3s`; (ii) forçantes climáticas diárias (`precipitation`, `temperature`, `actual_evapotransp`) mais o índice ONI (mensal); (iii) covariáveis estáticas da bacia (8 atributos prioritários listados acima). O ONI mensal foi acoplado às séries diárias repetindo o valor mensal para todos os dias do mês (sem interpolação intra‑mês), e são explorados lags até ±3 meses para capturar teleconexões.

### 3.3 Preparação dos Dados
O pipeline de preparação está implementado em `optuna_hydro_utils.py` e produz o dataset mestre `.dist/71200000_master_dataset.csv`.

- Tratamento de anomalias: interpolação linear limitada a lacunas de até 5 dias; lacunas maiores são marcadas como NA e tratadas posteriormente (remoção de janelas incompletas ou imputação específica conforme experimento).
- Transformação de variância: aplicou‑se $\log(y + 1)$ na vazão alvo durante treino para reduzir viés por cheias. As métricas de avaliação podem ser reportadas no espaço original voltando à escala com exponenciação e correção de viés quando necessário (back‑transform com viés corrigido).
- Codificação sazonal: features senoidais/cossenoidais baseadas em dia do ano e mês.
- Normalização: `StandardScaler` ajustado somente no conjunto de treino e aplicado sem refit à validação/teste para evitar vazamento.
- Particionamento temporal: o dataset inclui coluna `period` com as partições cronológicas usadas: `Passado` (1980–1990), `Recente` (1991–2010) e `Teste` (2011–2018). A divisão garante que transformações estatísticas e engenharia de features sejam estimadas exclusivamente no treino.

### 3.4 Modelagem
Foram comparadas duas arquiteturas treinadas e afinadas via Optuna:

- **LSTM (baseline)** — configurada em [tune_lstm_v2_optuna.py](tune_lstm_v2_optuna.py).
- **Temporal Fusion Transformer (TFT, proposto)** — configurado em [tune_tft_v2_optuna.py](tune_tft_v2_optuna.py), integrando atributos estáticos via GRNs e atenção interpretável.

Para testar memória e robustez foram avaliados lookbacks de 48, 96, 144 e 192 dias para cada horizonte (7, 15, 30 dias). A escolha dos lookbacks foi motivada por escalas hidrológicas: dias a meses de memória recente; justificar no texto que os valores foram selecionados por experimento piloto e por limitações computacionais. A calibração de hiperparâmetros usou Optuna com TPE, 30 trials por arranjo e Early Stopping; se desejar, aumentamos `n_trials` para análise de sensibilidade.

Inclui‑se também um baseline persistente (persistência) para referência e, quando aplicável, ensembles simples para quantificação de incerteza. Para estimativas probabilísticas, recomenda‑se usar abordagens quantílicas ou ensembles MC‑dropout.

### 3.5 Avaliação
Os modelos foram selecionados por Optuna e avaliados na janela cega (2011–2018) com métricas hidrológicas: NSE, KGE. Além disso, métricas direcionadas a secas foram calculadas:

- **LogNSE**: sensível aos fluxos de base.
- **RMSE_dry**: RMSE calculada apenas para amostras abaixo do limiar crítico da bacia ($Q_{90}$ histórico).

Esclarecer se as métricas são calculadas no espaço original da variável alvo (recomendado). Relatórios incluem curvas temporais, boxplots por horizonte e análises de falhas em episódios secos.

### 3.6 Implantação
O artefato final é um pipeline reutilizável orquestrado por [run_phase2_multi_horizon_experiments.py](run_phase2_multi_horizon_experiments.py), que gera previsões automáticas até 30 dias e artefatos para integração em sistemas de alerta. Recomenda‑se documentar formato de saída (CSV/Parquet ou API), frequência de execução e política de re‑treino (por exemplo, re‑treino trimestral ou quando novos dados acumularem N dias).

---

### Diagrama do pipeline

```mermaid
flowchart TD
	A[Dados brutos CAMELS-BR + ONI + Atributos] --> B[prepare_camels_br_dataset.py]
	B --> C[optuna_hydro_utils.py (ETL, janelamento, scalers)]
	C --> D{Tuning}
	D -->|LSTM| E[tune_lstm_v2_optuna.py]
	D -->|TFT| F[tune_tft_v2_optuna.py]
	E --> G[run_phase2_multi_horizon_experiments.py]
	F --> G
	G --> H[Resultados: .dist/optuna/, métricas, plots]
	H --> I[Deploy / integração em sistema de alerta]
```

---

Se quiser, faço uma versão final enxuta para inclusão direta no capítulo do TCC (Português acadêmico), ou preparo a tabela com horizontes/lookbacks/metricas para inserir como figura/tabela.
