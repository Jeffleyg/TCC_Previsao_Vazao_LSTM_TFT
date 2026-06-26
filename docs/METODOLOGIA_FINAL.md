## 3. Metodologia

Esta seção apresenta a versão final e compacta da metodologia adotada, redigida para inclusão no TCC. O estudo segue o framework CRISP‑DM (Cross Industry Standard Process for Data Mining), adaptado ao problema de previsão multi‑horizonte da vazão na bacia 71200000.

### 3.1 Compreensão do Negócio
O objetivo é fornecer previsões operacionais de vazão com horizontes de 7, 15 e 30 dias, com foco na antecipação de episódios de seca e no suporte a decisões de alocação e gestão de recursos hídricos. O critério de sucesso é a capacidade de prever o decaimento do fluxo de base em períodos críticos.

### 3.2 Compreensão dos Dados
Fontes e organização principais:

- Dados hidrométricos diários: `streamflow_m3s` (variável alvo).
- Forçantes climáticas diárias: `precipitation`, `temperature`, `actual_evapotransp`.
- Índice ONI (mensal): acoplado às séries diárias repetindo o valor mensal para cada dia do mês e avaliando lags até ±3 meses para capturar teleconexões.
- Atributos estáticos da bacia: oito covariáveis fisiográficas prioritárias (aridez climática, sazonalidade de precipitação, área, declividade média, elevação média, permeabilidade geológica, profundidade do solo até rocha, grau de regulação antrópica).

O dataset mestre está disponível em `.dist/71200000_master_dataset.csv` e contém a coluna `period` com as partições cronológicas: `Passado` (1980–1990), `Recente` (1991–2010) e `Teste` (2011–2018).

### 3.3 Preparação dos Dados
Resumo do pipeline (implementado em `optuna_hydro_utils.py`):

- Tratamento de lacunas: interpolação linear limitada a 5 dias; lacunas maiores marcadas como NA e tratadas por remoção de janelas ou imputação conforme experimento.
- Transformação da variável alvo: $
\log(y + 1)$ aplicada no treino para atenuar o efeito de cheias; as métricas principais são reportadas preferencialmente no espaço original (back‑transform com correção de viés quando aplicável).
- Codificação sazonal: features senoidais/cossenoidais (dia do ano, mês).
- Normalização: `StandardScaler` ajustado apenas no conjunto de treino e aplicado sem refit a validação/teste.
- Janelamento temporal e partições fixas: treino (1980–2010) e teste cego (2011–2018), com validação interna estratificada cronologicamente (15%).
- Justificativa física dos lookbacks: o lookback-base de 48 dias foi adotado a partir do cálculo da média do ciclo hidrológico da bacia, de modo a representar a escala mínima de memória física do sistema; os lookbacks de 96, 144 e 192 dias correspondem a múltiplos desse ciclo-base e permitem testar como a memória acumulada afeta a resposta da vazão.

Partições temporais (especificação): para garantir reprodutibilidade e clarificar o protocolo experimental, adotou-se explicitamente a seguinte partição cronológica nos experimentos reportados: treino 1980–2010; validação temporal (expanding window) 2011–2014; teste cego 2015–2018. A validação temporal foi configurada para preservar a ordem cronológica (sem embaralhamento) e simular condições de previsão em regime de mudança, refletindo possíveis efeitos de não‑estacionariedade.

Protocolo de seleção da bacia: a escolha da bacia 71200000 foi derivada de um roteiro automatizado de triagem implementado em `rank_camels_br_basins.py`, que classifica unidades hidrológicas segundo qualidade das séries, disponibilidade de covariáveis estáticas e representatividade hidroclimática. O arquivo de saída de referência encontra‑se em `.distAntigo/ranking_bacias_sul_tcc.csv` (linha para `gauge_id=71200000` registra `recommended_score = 0.9634340598601399`).

### 3.4 Modelagem
Modelos avaliados e configuração de experimentos:

- Baseline: LSTM (implementado em `tune_lstm_v2_optuna.py`).
- Proposto: Temporal Fusion Transformer (TFT) (implementado em `tune_tft_v2_optuna.py`) — incorpora atributos estáticos via GRNs e utiliza atenção interpretável.

Variantes A/B e sensibilidade arquitetural: cada família de modelos foi avaliada em variantes A e B para investigar sensibilidade a capacidade e regularização. Exemplos (valores concretos e ranges estão documentados nos artefatos de tuning): `LSTM A` (p.ex. 2 camadas, hidden_size menor, dropout moderado) vs `LSTM B` (maior capacidade e dropout diferenciado); `TFT A` vs `TFT B` com diferenças em hidden_size, número de cabeças de atenção e dropout nos GRNs. Estas variantes permitem avaliar se ganhos de desempenho são atribuíveis a arquitetura ou à capacidade do modelo.

Protocolo de validação temporal: a validação interna foi realizada com expanding window temporal, preservando ordem cronológica e evitando vazamento de informação. Em cada iteração, o modelo é treinado até um corte t0 e validado em uma janela subsequente; o corte t0 é deslocado ao longo do período de validação para capturar variabilidade interanual.

Espaço de busca do Optuna (resumo): hiperparâmetros otimizados incluíram `learning_rate` (1e‑5 → 1e‑2, log‑uniform), `batch_size` (16, 32, 64), `hidden_size` (32 → 512), `num_layers` (1 → 4), `dropout` (0.0 → 0.5), `weight_decay` (0 → 1e‑3), e opção de `lr_scheduler` (None, StepLR, CosineAnnealing). Cada configuração foi avaliada com 30 trials usando TPE; Early Stopping monitorou a métrica de validação com paciência configurável.

Foram testados lookbacks de 48, 96, 144 e 192 dias para cada horizonte (7, 15, 30 dias) para investigar a dependência temporal e a capacidade de memória dos modelos. A busca por hiperparâmetros foi conduzida com Optuna (TPE), 30 trials por configuração e Early Stopping. Inclui‑se baseline de persistência e recomenda‑se, quando necessário, ensembles para estimativa de incerteza (ex.: quantílicas, MC‑dropout).

### 3.5 Avaliação
Os modelos foram avaliados na janela de teste (2011–2018) com métricas hidrológicas e operacionais:

- NSE (Nash–Sutcliffe Efficiency)
- KGE (Kling‑Gupta Efficiency)
- LogNSE (ênfase em fluxos de base)
- RMSE_dry (RMSE calculado apenas para amostras abaixo do percentil histórico $Q_{90}$)

As métricas principais devem ser reportadas no espaço original da vazão, com notas sobre qualquer back‑transform aplicado.

Teste estatístico e intervalos de confiança: para avaliar significância nas diferenças de desempenho entre modelos (por exemplo, TFT vs LSTM), adotou‑se bootstrap com 1000 reamostragens do conjunto de teste para estimar intervalos de confiança (IC) das métricas (NSE, KGE, LogNSE). Diferenças entre modelos foram avaliadas por IC das diferenças e, quando aplicável, por testes não paramétricos pareados (ex.: Wilcoxon).

Interpretabilidade e análise física: a interpretação do TFT utilizou a extração das atenções por passo de tempo e por feature, agregadas por horizonte e por estação do ano. As importâncias foram sumarizadas em tabelas e gráficos para comparar sensibilidade entre lookbacks e regimes (Passado/Recente). As rotinas de extração de atenção e geração de plots são executadas automaticamente pelos scripts quando a flag `--generate-explanations` é ativada.

Reprodutibilidade — comando de exemplo:
```
python run_phase2_multi_horizon_experiments.py --basin 71200000 --seed 42 --model tft --horizon 30 --lookbacks 48 96 144 192 --generate-explanations
```

### 3.6 Implantação
O produto é um pipeline reprodutível orquestrado por `run_phase2_multi_horizon_experiments.py` que gera artefatos em `.dist/optuna/` (melhores trials, CSVs de trials, métricas e plots). Recomenda‑se descrever o formato de saída (CSV/Parquet ou endpoint API), a frequência de execução e a política de re‑treino (por exemplo, re‑treino trimestral ou após acumular N dias de novos dados).

---

### Tabela resumo: horizontes, lookbacks e métricas

| Horizonte | Lookbacks avaliados (dias) | Métricas alvo |
|---:|:---:|:---|
| 7 dias | 48, 96, 144, 192 | NSE, KGE, LogNSE, RMSE_dry (Q90) |
| 15 dias | 48, 96, 144, 192 | NSE, KGE, LogNSE, RMSE_dry (Q90) |
| 30 dias | 48, 96, 144, 192 | NSE, KGE, LogNSE, RMSE_dry (Q90) |

---

### Reprodutibilidade e notas técnicas
- Seeds: indicar semente(s) usadas para treino e tuning nos scripts (parâmetro `random_seed`).
- Ambientes: registrar versões principais de Python e bibliotecas (PyTorch/Lightning, Optuna, pandas, scikit‑learn) no relatório final.
- Recursos: indicar hardware usado (GPU/CPU) e tempo aproximado por trial, para apoiar replicação.

---

