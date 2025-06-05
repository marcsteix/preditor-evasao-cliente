# Case de Previsão de Evasão de Clientes (Churn Prediction)

Este projeto tem como objetivo prever a evasão de clientes (churn) de um banco de cartões de crédito, utilizando técnicas de machine learning. Foram aplicados testes estatísticos para seleção de variáveis, engenharia de atributos, balanceamento de classes, otimização de modelos e análise interpretativa. O modelo final (XGBoost) alcançou 97% de acurácia e 90% de recall na identificação de clientes em risco.

## Contexto

Este projeto analisa dados de clientes de cartão de crédito para:
- Identificar padrões comportamentais associados à evasão
- Desenvolver um modelo preditivo de risco de churn
- Sugerir ações estratégicas para retenção de clientes

## Origem dos Dados
**Fonte dos dados**: [Credit Card Churn Dataset - por laotse (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Descrição:** Contém informações demográficas, financeiras e comportamentais de clientes de cartão de crédito, além da variável-alvo que indica se o cliente saiu ou permaneceu ativo.

## Etapas do Projeto

1. **Definição do Problema**
   - Identificar clientes com maior probabilidade de evasão para adoção de estratégias preventivas.
2. **Tratamento e Engenharia de Variáveis**
   - Mapeamento de variáveis categóricas.
   - Criação de atributos derivados como faixa de utilização de limite, variação de uso, etc.
3. **Análise de Distribuições**
   - Visualização por histogramas, boxplots e proporções de evasão.
4. **Seleção de Variáveis Significativas**
   - Testes estatísticos (Qui-Quadrado e Mann-Whitney).
5. **Balanceamento com SMOTE**
   - Aplicado para lidar com o desbalanceamento da variável alvo (evasão).
6. **Modelagem com 3 Algoritmos**
   - Regressão Logística
   - Random Forest Otimizado
   - XGBoost Otimizado
7. **Avaliação de Desempenho**
   - Métricas: Accuracy, Recall, Precision, F1-score, ROC-AUC
   - Interpretação com Feature Importance e SHAP Values
8. **Projeção de Evasão**
   - Aplicação do modelo XGBoost otimizado para prever a probabilidade de evasão por cliente
   - Geração de score contínuo e segmentação em faixas de risco (baixíssimo, baixo, médio, alto, altíssimo)
- Com base no **recall do modelo XGBoost (90%)**, foi realizada uma simulação para estimar a taxa final de churn considerando ações de retenção.
   - Foi assumida uma **eficiência de 50%** nas ações — ou seja, metade dos clientes identificados como em risco e abordados pelo time de retenção permaneceriam ativos.
   - Fórmula aplicada:
     evasão_ajustada = evasão_atual - (evasão_atual × recall × eficiência_da_ação)

   - Com uma taxa atual de churn de **16%**, a evasão projetada cai para **8,79%**.
   - Isso demonstra que, mesmo com ações de retenção de **eficiência moderada (50%)**, o modelo é capaz de **reduzir pela metade a perda de clientes** ao identificar corretamente os casos mais críticos.
9. **Segmentação e Estratégias por Score de Risco**
   - Clientes classificados em 6 categorias com base no score preditivo gerado pelo modelo XGBoost:
     - **Saiu**: clientes que já cancelaram o serviço
     - **Baixíssimo Risco** (probabilidade < 10%): manter ações regulares e observar comportamento
     - **Baixo Risco** (10%–25%): iniciar ações leves de engajamento
     - **Médio Risco** (25%–50%): intensificar contato e oferecer diferenciais
     - **Alto Risco** (50%–75%): aplicar ações de retenção imediata
     - **Altíssimo Risco** (≥ 75%): priorizar abordagem ativa e benefícios de permanência

   - Essa segmentação permite estratégias de retenção personalizadas conforme o nível de risco de cada cliente.
10. **Exportação e Salvamento Final**
  - Salvamento dos modelos treinados (`.pkl`)
  - Geração da base final com scores de risco para futuras análises

## Resultados Detalhados

### Comparação de Modelos

| Modelo               | Acurácia | Recall | Precisão | AUC-ROC |
|----------------------|----------|--------|----------|---------|
| Regressão Logística  | 86%      | 81%    | 54%      | 0.94    |
| Random Forest        | 95%      | 86%    | 85%      | 0.98    |
| XGBoost (Final)      | 97%      | 90%    | 90%      | 0.99    |

### Impacto no Negócio
- **Redução projetada de evasão**: De 16% para 8.79%
- **Clientes em alto risco identificados**: 12.3% da base
- **Variáveis-chave**:
  1. Frequência de transações
  2. Mudança no padrão de uso
  3. Taxa de utilização do limite

## Tecnologias Utilizadas

- **Python 3.12**
- Bibliotecas principais:
  - Pandas, NumPy (manipulação de dados)
  - Matplotlib, Seaborn (visualização)
  - Scikit-learn (modelagem)
  - XGBoost (modelo final)
  - SHAP (explicabilidade do modelo)

## Estrutura dos Arquivos

```plaintext
preditor-evasao-cliente/
├── data/
│   ├── credit_card_churn.csv # Dados brutos
│   └── base_final_risco_score.csv # Dados processados com scores
├── figures/ # Todos os gráficos gerados
├── models/
│   ├── modelo_regressao_logistica.pkl
│   ├── modelo_random_forest_otimizado.pkl
│   └── modelo_xgboost_otimizado.pkl
├── notebooks/
│   └── analise_evasao_clientes.ipynb # Jupyter Notebook principal
├── README.md
└── requirements.txt
```

## Como Executar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/marcsteix/preditor-evasao-cliente.git
   cd preditor-evasao-cliente
   ```
2. **Criação de Ambiente Virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Instale as Depedências:**
   ```bash
   pip install -r requirements.txt
4. **Execute o Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/analise_evasao_clientes.ipynb
   ```

## Como Executar o Modelo Preditivo

Após rodar o notebook principal (`notebooks/analise_evasao_clientes.ipynb`), pode-se reutilizar o modelo treinado:

   ```python
      import joblib
      # Carregar o modelo salvo
      modelo = joblib.load('../models/modelo_xgboost_otimizado.pkl')
      # Fazer a predição
      y_pred = modelo_xgb.predict(X_novos_dados)
      # Retorna risco 0-1
      prob_evasao = modelo.predict_proba(novos_dados)[:, 1] 
   ```

## Autor
 **Marcos Teixeira dos Santos**
 - E-mail: [marcteix@live.com](mailto:marcteix@live.com)
 - LinkedIn: [https://www.linkedin.com/in/marcteix/](https://www.linkedin.com/in/marcteix/)
 - Última atualização: Junho/2025