# Ambiente de Preparação de TCC (Hidrologia + IA)

Este projeto organiza e prepara dados do **CAMELS-BR** para treinamento de modelos de séries temporais (ex.: **LSTM** e **Temporal Fusion Transformer - TFT**), com foco em seleção de bacias de baixa intervenção humana e boa qualidade hidrológica.

## Objetivo

- Integrar dados dinâmicos diários (precipitação, temperatura, evapotranspiração e vazão).
- Integrar índice climático **ONI** (mensal -> diário por broadcast).
- Incorporar atributos estáticos das bacias (solo, topografia, uso do solo, intervenção humana, etc.).
- Gerar bases prontas para treino, validação e teste.
- Avaliar prontidão de bacias para o TCC com critérios formais.
- Gerar ranking final de bacias com recomendação de **bacia principal** e **bacia reserva**.

---

## Estrutura principal de pastas

- `Atributo/`: atributos estáticos CAMELS-BR por bacia.
- `ONI/`: índice ONI mensal (`oni.data.txt`).
- `Treino Unificado/`: séries diárias por estação (`*_precipitation.txt`, `*_temperature.txt`, `*_actual_evapotransp.txt`, `*_streamflow_m3s.txt`).
- `.dist/`: saídas geradas (CSV/XLSX/MD de dados processados e relatórios).

---

## Scripts principais

### 1) `prepare_camels_br_dataset.py`
Pipeline de preparação de dados para modelagem.

**Faz:**
- merge diário por `date`;
- ONI mensal -> diário;
- junção de atributos estáticos;
- features cíclicas de mês (`month_sin`, `month_cos`);
- interpolação linear de lacunas curtas;
- divisão em períodos:
  - `Passado` (1980–1990)
  - `Recente` (1991–2010)
  - `Teste` (2011–2018)
- normalização Min-Max em variáveis numéricas.

**Estação padrão atual:** `71200000`.

Comandos úteis:

```bash
py .\prepare_camels_br_dataset.py --split-periods
py .\prepare_camels_br_dataset.py --export-period-csvs
py .\prepare_camels_br_dataset.py --export-period-excels
py .\prepare_camels_br_dataset.py --export-model-ready-csvs
```

Saídas típicas em `.dist/`:
- `treino_passado.csv/.xlsx`
- `treino_recente.csv/.xlsx`
- `treino_teste.csv/.xlsx`
- `treino_passado_model_ready.csv`
- `treino_recente_model_ready.csv`
- `treino_teste_model_ready.csv`

---

### 2) `assess_basin_training_readiness.py`
Avalia se uma bacia está **aprovada** para treino com critérios de dados dinâmicos + intervenção humana.

**Padrão metodológico atual (TCC):**
- cobertura temporal mínima: `1980-01-01` até `2018-12-31`
- tolerância de faltantes por variável: até `5%`
- qualidade/intervenção:
  - `q_quality_control_perc >= 95`
  - `consumptive_use_perc <= 0.20`
  - `regulation_degree <= 0.01`
  - `reservoirs_vol <= 0`

Exemplo:

```bash
py .\assess_basin_training_readiness.py --gauge-id 71200000
```

Saídas em `.dist/`:
- `readiness_<gauge_id>_dynamic_summary.csv`
- `readiness_<gauge_id>_intervention_summary.csv`
- `readiness_<gauge_id>_decision.csv`
- `readiness_<gauge_id>_report.xlsx`

---

### 3) `rank_camels_br_basins.py`
Ranqueia bacias do Sul com boa qualidade e baixa intervenção humana e gera recomendação formal.

Exemplo:

```bash
py .\rank_camels_br_basins.py
```

Saídas em `.dist/`:
- `ranking_bacias_sul_tcc.csv/.xlsx`
- `comparativo_bacia_71350001.csv/.xlsx`
- `recomendacao_final_bacias_tcc.csv/.xlsx/.md`

**Configuração atual da recomendação formal:**
- bacia principal preferencial: `71200000`
- bacia reserva: melhor alternativa seguinte no ranking.

---

## Dependências

Versão Python usada no projeto: Python 3.14 (via `py`).

Pacotes necessários:

```bash
py -m pip install pandas numpy openpyxl
```

Opcional (para montar `TimeSeriesDataSet`):

```bash
py -m pip install pytorch-forecasting
```

---

## Fluxo recomendado para o TCC

1. Validar prontidão da bacia escolhida:
   - `py .\assess_basin_training_readiness.py --gauge-id 71200000`
2. Preparar datasets:
   - `py .\prepare_camels_br_dataset.py --station-id 71200000 --export-model-ready-csvs`
3. Treinar modelos com:
   - `treino_passado_model_ready.csv`
   - `treino_recente_model_ready.csv`
4. Avaliar no:
   - `treino_teste_model_ready.csv`
5. Documentar escolha da bacia com:
   - `recomendacao_final_bacias_tcc.md`

---

## Observações metodológicas

- O recorte até 2018 é coerente com cobertura observacional de vazão em várias bacias CAMELS-BR.
- Tolerância de 5% de faltantes foi adotada como critério operacional de qualidade para o TCC.
- A escolha da bacia principal considera tanto atributos físicos/humanos quanto viabilidade prática para modelagem.
