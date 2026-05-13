# 🔬 Nova Abordagem: Física da Bacia + Explicabilidade do TFT

## 🎯 Objetivo

Validar uma **abordagem baseada na física hídrica** para melhorar a resiliência do modelo frente às mudanças climáticas:

1. **Lookbacks Físicos**: Usar ciclos de ~48 dias (em vez de 30/60/90 fixos)
2. **Múltiplos Horizontes**: Testar 7, 15 e 30 dias (em vez de apenas t+1)
3. **Explicabilidade**: Ver quais variáveis o TFT prioriza em cada cenário

---

## 📋 Fase 1: Validação (AGORA)

**Objetivo**: Confirmar que lookbacks [48, 96, 144, 192] dias mantêm boa performance.

### Opção 1️⃣: Teste Rápido (5 min)

```bash
python test_tft_explanations.py
```

Treina 1 trial de TFT com lookback 30d e gera:
- ✅ Feature Importance JSON
- ✅ Visualization PNG
- ✅ Hydrograph PNG

**Saída esperada**:
```
.dist/optuna/tft_test_lb30_tft_feature_importance.json
.dist/plots/tft_test_lb30_tft_attention_heatmap.png
.dist/plots/tft_test_lb30_hydrogram_test.png
```

---

### Opção 2️⃣: Experimentos Completos Fase 1 (8-12 horas)

```bash
# Rápido (teste): 10 trials por experimento, explicabilidade ativada
python run_phase1_physics_based_lookbacks.py --n-trials 10 --generate-tft-explanations

# Robusto: 20 trials por experimento
python run_phase1_physics_based_lookbacks.py --n-trials 20 --generate-tft-explanations

# Alternativas:
python run_phase1_physics_based_lookbacks.py --n-trials 20 --skip-lstm  # Só TFT
python run_phase1_physics_based_lookbacks.py --n-trials 20 --skip-tft  # Só LSTM
```

**O que roda**:
- LSTM Passado × 4 lookbacks = 4 experimentos
- LSTM Recente × 4 lookbacks = 4 experimentos
- TFT Passado × 4 lookbacks = 4 experimentos ✨
- TFT Recente × 4 lookbacks = 4 experimentos ✨

**Total**: 16 experimentos com explicabilidade no TFT

---

### Opção 3️⃣: Abordagem Tradicional (Wrapper)

```bash
# Treina 4 modelos com lookback = 48d
python train_dual_period_models.py --lookback 48 --generate-explanations

# Repete para cada lookback físico
python train_dual_period_models.py --lookback 96 --generate-explanations
python train_dual_period_models.py --lookback 144 --generate-explanations
python train_dual_period_models.py --lookback 192 --generate-explanations
```

---

## 🔍 Interpretando a Explicabilidade do TFT

### Arquivos Gerados

Para cada experimento TFT, você terá:

**1. JSON - Feature Importance**
```json
{
  "feature_importance": {
    "precipitation": 85.5,
    "temperature": 72.3,
    "actual_evapotransp": 65.1,
    ...
  },
  "top_features": ["precipitation", "temperature", "actual_evapotransp"]
}
```

**2. PNG - Visualização**
- Gráfico de barras com as variáveis ordenadas por importância
- Top 3 features destacadas em verde
- Demais variáveis em azul

### Interpretação

```
🔝 Top 3 Features:
• precipitation: 85.5    ← Precipitação é a variável mais importante
• temperature: 72.3      ← Temperatura em 2º
• actual_evapotransp: 65.1  ← Evapotranspiração em 3º
```

**O que significa?**

- **Scores altos** = o modelo presta muita atenção nesta variável
- **Comparar entre lookbacks** = ver se a importância muda com a memória
  - *Lookback 48d*: Pode priorizar variabilidade rápida
  - *Lookback 192d*: Pode priorizar tendências sazonais

---

## 📊 Como Analisar os Resultados Fase 1

Após rodar `run_phase1_physics_based_lookbacks.py`, faça:

### 1. Compare Performance entre Lookbacks

```bash
# Abrir em Excel / Python
.dist/optuna/lstm_passado_lb48_trials.csv
.dist/optuna/lstm_passado_lb96_trials.csv
.dist/optuna/lstm_passado_lb144_trials.csv
.dist/optuna/lstm_passado_lb192_trials.csv
```

Procure por:
- NSE (deve ser positivo)
- KGE (deve estar alto)
- sMAPE (deve ser baixo)

### 2. Analise Feature Importance do TFT

```bash
# Abrir JSONs
.dist/optuna/tft_passado_lb48_tft_feature_importance.json
.dist/optuna/tft_passado_lb96_tft_feature_importance.json
.dist/optuna/tft_passado_lb144_tft_feature_importance.json
.dist/optuna/tft_passado_lb192_tft_feature_importance.json
```

**Pergunta principal**: As top-3 features mudam com o lookback? Se mudarem, significa que o modelo está realmente capturando ciclos diferentes!

### 3. Compare Hidrogramas

```bash
# Visualizar predições
.dist/plots/tft_passado_lb48_hydrogram_test.png
.dist/plots/tft_passado_lb96_hydrogram_test.png
.dist/plots/tft_passado_lb144_hydrogram_test.png
.dist/plots/tft_passado_lb192_hydrogram_test.png
```

Procure por:
- Qual lookback segue melhor a vazão real?
- Qual captura picos de seca melhor?

---

## 📈 Próximas Fases

### ✅ Fase 1 (Agora): Validação da Física
- Lookbacks: [48, 96, 144, 192] dias
- Horizonte: 1 dia
- Objetivo: Confirmar que a "memória" física melhora performance

### ⏳ Fase 2: Exploração de Horizontes
- Usar melhores modelos da Fase 1
- Testar horizontes: 7, 15, 30 dias
- Usar fine-tuning para acelerar

### 🎓 Fase 3: Análise e Defesa
- Comparar com baseline (30/60/90 dias)
- Demonstrar que modelo é mais resiliente
- Apresentar feature importance como prova de raciocínio físico

---

## 💾 Resumo de Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `test_tft_explanations.py` | Teste rápido da explicabilidade |
| `run_phase1_physics_based_lookbacks.py` | Fase 1 completa (16 experimentos) |
| `train_dual_period_models.py` | Treina 4 modelos (wrapper clássico) |
| `.dist/optuna/*_tft_feature_importance.json` | Scores de importância por variável |
| `.dist/plots/*_tft_attention_heatmap.png` | Visualização de feature importance |
| `.dist/plots/*_hydrogram_test.png` | Predições vs vazão real |

---

## 🚀 Comando Recomendado para Começar

```bash
# 1. Teste rápido (valida setup)
python test_tft_explanations.py

# 2. Se OK, rode Fase 1 com explicabilidade
python run_phase1_physics_based_lookbacks.py --n-trials 20 --generate-tft-explanations --skip-lstm
```

Boa sorte com a nova abordagem! 🎯
