# Plataforma de Detecção de Fraude para Seguro de Saúde

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Plataforma de inteligência analítica para detecção de fraude, desperdício e abuso em solicitações de reembolso médico. Desenvolvida em Streamlit, permite às seguradoras de saúde identificar padrões suspeitos e priorizar casos para investigação.

---

## Funcionalidades

- **Detecção de Anomalias** — Isolation Forest + Z-score por prestador
- **Risco de Prestadores** — Pontuação agregada com 6 métricas de comportamento
- **Análise de Beneficiários** — Detecção de multi-prestadores e gastos atípicos
- **Custos Atípicos** — Comparação vs. benchmark de pares (procedimento / especialidade / carteira)
- **Pontuação de Risco 0–100** — Composta de 4 módulos com pesos diferenciados
- **Relatório Individual por Beneficiário** — Com exportação PDF e Excel
- **Exportação** — Excel (4 folhas), CSV e PDF
- **Persistência SQLite** — Sessões guardadas localmente, sem necessidade de re-upload
- **Interface em Português** — Tema escuro profissional

---

## Requisitos Mínimos do Ficheiro de Dados

| Coluna | Obrigatório | Descrição |
|--------|-------------|-----------|
| `claim_id` | ✅ | Identificador único da solicitação |
| `member_id` | ✅ | Identificador do beneficiário |
| `provider_id` | ✅ | Identificador do prestador |
| `claim_date` | ✅ | Data de submissão (YYYY-MM-DD) |
| `claim_amount` | ✅ | Valor total facturado |
| `diagnosis_code` | Opcional | Código ICD-10 |
| `procedure_code` | Opcional | Código CPT / HCPCS |
| `service_date` | Opcional | Data de prestação do serviço |
| `paid_amount` | Opcional | Valor aprovado / pago |
| `provider_specialty` | Opcional | Especialidade do prestador |
| `member_age` | Opcional | Idade do beneficiário |
| `member_gender` | Opcional | Género do beneficiário |

A plataforma aceita nomes de colunas alternativos (ex: `npi`, `billed_amount`, `dos`, `icd10`) e mapeia-os automaticamente.

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/Ngaspar20/fraud-detection-health-insurance.git
cd fraud-detection-health-insurance
```

### 2. Criar ambiente Conda (recomendado)

```bash
conda create -n ml_env python=3.11 -y
conda activate ml_env
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar a aplicação

```bash
streamlit run app.py
```

Abrir no browser: [http://localhost:8501](http://localhost:8501)

---

## Gerar Dados de Teste

Para gerar um ficheiro de dados sintéticos com padrões de fraude injectados:

```bash
python generate_mock_data.py
```

Cria o ficheiro `mock_claims.csv` com 1.931 solicitações e 5 padrões de fraude:
- Valores muito acima da média (outliers)
- Solicitações duplicadas
- Facturação em valores redondos suspeitos
- Beneficiários com múltiplos prestadores
- Múltiplas solicitações no mesmo dia

---

## Estrutura do Projecto

```
fraud-detection-health-insurance/
├── app.py                        # Aplicação Streamlit principal
├── generate_mock_data.py         # Gerador de dados sintéticos
├── requirements.txt              # Dependências Python
├── modules/
│   ├── column_detector.py        # Detecção e mapeamento automático de colunas
│   ├── data_loader.py            # Upload de ficheiros + persistência SQLite
│   ├── fraud_detection.py        # Isolation Forest + Z-score + duplicados
│   ├── provider_risk.py          # Pontuação de risco por prestador
│   ├── member_utilization.py     # Análise de utilização por beneficiário
│   ├── cost_outlier.py           # Detecção de custos atípicos
│   ├── risk_scorer.py            # Pontuação de risco composta (0–100)
│   └── exporter.py               # Exportação Excel, CSV e PDF
└── data/
    └── claims.db                 # Base de dados SQLite (criada automaticamente)
```

---

## Fórmula de Pontuação de Risco

```
Pontuação Final =
    Anomalia Estatística  × 35%
  + Risco do Prestador    × 25%
  + Risco do Beneficiário × 20%
  + Custo Atípico         × 20%
```

| Nível | Pontuação | Acção |
|-------|-----------|-------|
| 🔴 Alto | 70 – 100 | Investigação prioritária |
| 🟡 Médio | 40 – 69 | Revisão recomendada |
| 🟢 Baixo | 0 – 39 | Dentro dos padrões normais |

---

## Stack Tecnológico

| Biblioteca | Uso |
|-----------|-----|
| Streamlit | Interface web |
| Pandas / NumPy | Processamento de dados |
| Scikit-learn | Isolation Forest (ML) |
| Plotly | Visualizações interactivas |
| OpenPyXL | Exportação Excel |
| ReportLab | Exportação PDF |
| SQLite | Persistência de sessões |

---

## Licença

MIT License — livre para uso, modificação e distribuição.
