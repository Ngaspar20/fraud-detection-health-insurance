# Project Context — Health Insurance Fraud Detection Platform

> **Last updated:** June 2026  
> **GitHub:** https://github.com/Ngaspar20/fraud-detection-health-insurance  
> **Streamlit Cloud:** https://fraud-health-moz.streamlit.app  
> **Local path:** `C:\Users\Nuno Gaspar\OneDrive - Jhpiego\Desktop\Claude_Workstation\Output\Fraud_Detection_Health_Insurance\`

---

## 1. What This Is

A production-quality Streamlit MVP that helps health insurance companies (specifically targeting mid-tier South African medical schemes) detect **fraud, waste, abuse, billing anomalies, and cost outliers** in claims data.

The platform accepts any CSV or Excel claims export, runs 4 ML/statistical analysis modules in sequence, assigns a **risk score (0–100)** to every claim, and outputs ranked alerts with plain-language explanations — ready for a forensics team to act on.

**Target market:** Mid-tier SA medical schemes (Fedhealth, Bestmed, KeyHealth, Polmed, etc.)  
**Deployment:** Streamlit Community Cloud (free tier)  
**Languages:** Bilingual PT / EN (toggle in sidebar)

---

## 2. File Structure

```
Fraud_Detection_Health_Insurance/
├── app.py                        # Main Streamlit app (~1,400 lines)
├── requirements.txt              # Python dependencies
├── generate_mock_data.py         # Generates mock_claims.csv (1,931 rows)
├── mock_claims.csv               # Synthetic test dataset
├── fix_strings.py                # Utility: batch-replace hardcoded strings with t()
├── data/
│   └── claims.db                 # SQLite session storage (auto-created)
└── modules/
    ├── __init__.py
    ├── lang.py                   # i18n: PT/EN translations + flag translation
    ├── column_detector.py        # Auto-detect + fuzzy-map column names → ColumnProfile
    ├── data_loader.py            # CSV/Excel upload + SQLite session persistence
    ├── fraud_detection.py        # Anomaly detection (Isolation Forest + rules)
    ├── provider_risk.py          # Provider-level risk scoring
    ├── member_utilization.py     # Member utilisation anomaly scoring
    ├── cost_outlier.py           # Cost vs. peer benchmark scoring
    ├── risk_scorer.py            # Composite 0–100 risk score + auto-adjudication
    └── exporter.py               # Excel (openpyxl), CSV, PDF (reportlab) export
```

---

## 3. How to Run

```powershell
# Activate environment
conda activate ml_env

# Navigate to project
cd "C:\Users\Nuno Gaspar\OneDrive - Jhpiego\Desktop\Claude_Workstation\Output\Fraud_Detection_Health_Insurance"

# Launch
streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## 4. Architecture

### Analysis Pipeline (triggered on file upload or demo data load)

```
File Upload / Demo Data
        ↓
column_detector.py     → Fuzzy-maps columns → ColumnProfile (what's available)
        ↓
fraud_detection.py     → Isolation Forest + Z-score + rule-based flags
        ↓
provider_risk.py       → Provider-level risk metrics (volume, avg, dup rate, etc.)
        ↓
member_utilization.py  → Member utilisation patterns (multi-provider, same-day, spend)
        ↓
cost_outlier.py        → Z-score vs. peer group (procedure > specialty > overall)
        ↓
risk_scorer.py         → Composite score + risk_level + adjudication recommendation
        ↓
Streamlit session_state (scored_df, provider_df, member_df, profile)
```

### Composite Risk Score Formula

```
risk_score = (
    anomaly_score       × 0.35  +   # Isolation Forest (always runs)
    provider_risk_score × 0.25  +   # Provider risk (always runs)
    member_risk_score   × 0.20  +   # Member risk (always runs)
    cost_outlier_score  × 0.20      # Cost vs. peers (always runs)
).clip(0, 100)

# Floor: if any flags exist, minimum score = 40 (forces "Medium" minimum)
```

| Score Range | Risk Level | Adjudication |
|-------------|------------|--------------|
| 70 – 100    | High       | Investigar Urgente / Urgent Investigation |
| 40 – 69     | Medium     | Rever Manualmente / Manual Review |
| 0 – 39      | Low        | Aprovar Automaticamente / Auto-Approve |

---

## 5. Key Modules — Technical Details

### `modules/column_detector.py`
- **Required columns** (app refuses without): `claim_id`, `member_id`, `provider_id`, `claim_date`, `claim_amount`
- **Optional columns** (enable/disable features): `diagnosis_code`, `procedure_code`, `service_date`, `paid_amount`, `provider_specialty`, `member_age`, `member_gender`, `drug_name`, `ndc_code`, `claim_type`, `denial_code`
- Fuzzy aliases supported (e.g. `npi` → `provider_id`, `billed_amount` → `claim_amount`, `dos` → `service_date`)
- Returns `ColumnProfile` dataclass consumed by all downstream modules

### `modules/fraud_detection.py`
Flags generated (always ASCII, stored in Portuguese, translated at display via `translate_flag()`):

| Flag | Trigger |
|------|---------|
| `Anomalia detectada por modelo estatístico` | Isolation Forest outlier |
| `Facturado {x}x acima da media do prestador (Z>{z})` | Z-score > 3 vs. provider mean |
| `Valor facturado suspeito (numero redondo)` | Amount % 100 == 0 AND >= 500 |
| `Possivel solicitacao duplicada` | Same member + provider + amount (+ date if available) |
| `Procedimento de alta frequencia vs. prestadores similares` | Upcoding signal (requires `procedure_code`) |
| `Beneficiario com idade elevada (possivel fantasma)` | `member_age` > 85 (requires `member_age`) |
| `Sobreprescricao de medicamentos (>3 por mes)` | >3 pharmacy claims/month (requires `drug_name`) |

### `modules/risk_scorer.py`
- Merges anomaly + provider + member + cost scores
- Minimum score 40 if any flags exist
- `adjudication` column: `"Investigar Urgente"` / `"Rever Manualmente"` / `"Aprovar Automaticamente"`
- `risk_flags` column: semicolon-separated flag strings in Portuguese

### `modules/lang.py`
- `TRANSLATIONS` dict: 100+ keys, each with `"pt"` and `"en"` values
- `t(key)` → returns string in current language (`st.session_state.lang`)
- `translate_flag(flag_text)` → translates Portuguese flag strings to English, including dynamic regex patterns for Z-score and cost flags
- `FLAG_TRANSLATIONS["en"]` dict: exact-match map for static flags

### `modules/data_loader.py`
- SQLite DB at `data/claims.db`
- Table naming: `claims_{session_id}` (ASCII, no special chars)
- `sessions` metadata table: `session_id`, `filename`, `uploaded_at`, `row_count`, `columns`

### `modules/exporter.py`
- **Excel**: 4 sheets — `"Solicitações Sinalizadas"`, `"Todas as Solicitações"`, `"Risco de Prestadores"`, `"Risco de Beneficiários"` — with navy headers, conditional colour formatting on `risk_level`
- **CSV**: filtered scored claims sorted by risk_score descending
- **PDF**: reportlab — summary stats table + top-20 flagged claims table

---

## 6. Streamlit Pages (Sidebar Navigation)

| PT Label | EN Label | Key Content |
|----------|----------|-------------|
| Painel de Controlo | Executive Dashboard | 5 KPI cards, auto-adjudication panel (3 cards), risk distribution bar + cards, Top 10 alerts, monthly trend chart |
| Análise de Solicitações | Claims Analysis | Filter bar (risk level, score, provider, date), card view + table view, export buttons |
| Inteligência de Prestadores | Provider Intelligence | Horizontal bar chart (top N), provider detail drill-down (4 metrics + flags), scatter plot |
| Análise de Beneficiários | Member Analysis | Horizontal bar chart, utilisation scatter plot, high-risk member table |
| Custos Atípicos | Cost Outliers | Scatter (amount vs. peer mean), histogram by risk level, top-100 outlier table |
| Relatório por Beneficiário | Member Risk Report | Individual member profile: header card, 4 KPIs, risk flags, claims table, monthly spend chart, PDF/Excel export |
| Gestão de Dados | Data Management | Demo data button, file upload + contamination slider, previous sessions list, column format reference table |
| Como Funciona | How It Works | 4 tabs: Quick Start, Methodology, Analysis Modules, Data Format |

---

## 7. UI Theme

Dark professional, all custom CSS injected via `st.markdown(THEME_CSS, unsafe_allow_html=True)`.

| Token | Hex | Usage |
|-------|-----|-------|
| BG_DARK | `#080F18` | App background |
| BG_CARD | `#1E2D3D` | Cards, panels |
| BG_SURFACE | `#112233` | KPI cards |
| AMBER | `#F59E0B` | Active nav, medium risk accents |
| RED | `#EF4444` / `#F87171` | High risk, alerts |
| GREEN | `#22C55E` / `#34D399` | Low risk, success |
| BLUE | `#3B82F6` | Neutral highlights |
| TEXT | `#E2E8F0` | Primary text |

Charts: Plotly with `paper_bgcolor="#1E2D3D"`, `plot_bgcolor="#1E2D3D"`, `font_color="#E2E8F0"`.  
Bars use `go.Bar` with explicit `marker=dict(color=[...])` — **not** `px.bar` with `color_continuous_scale` (invisible on dark backgrounds).

---

## 8. Language Toggle

- State: `st.session_state.lang` — `"pt"` (default) or `"en"`
- Sidebar buttons: `PT` (primary when active) / `EN` (primary when active), both `use_container_width=True`
- Label above buttons: `"Idioma / Language"` (always bilingual, not translated)
- All page content uses `t("key")` calls — **must be f-strings or direct calls**, never inside triple-quoted strings (t() won't execute)
- Dynamic flag translation: `translate_flag()` handles both exact-match and regex patterns

### Known Remaining Untranslated Strings (as of last commit)
The `"Como Funciona / How It Works"` page (lines 431–622 of app.py) contains hardcoded Portuguese text inside triple-quoted HTML strings for step descriptions, methodology signals, module descriptions, and the "How It Works" header block. These have NOT yet been converted to `t()` calls.

---

## 9. Mock / Demo Data

- **Button:** "🚀 Carregar Dados de Demonstração" in Data Management page
- **1,931 rows** generated inline in `app.py` (not from file) using `numpy` RNG seed 42
- **40 providers** (PRV0001–PRV0040), **300 members** (MBR00001–MBR00300)
- **5 fraud patterns injected:**
  1. High-amount outliers (FRD prefix, $8k–$25k) — 30 rows, 3 fraud providers
  2. Duplicates (DUP prefix) — first 10 rows duplicated
  3. Round-number billing (RND prefix, $500/$1k/$1.5k/$2k/$5k) — 25 rows
  4. Multi-provider shopping (SHP prefix) — 5 fraud members × 12 providers each
  5. Normal claims with full optional columns — 1,800 rows
- Separate `generate_mock_data.py` also exists → saves `mock_claims.csv`

---

## 10. Git / Deployment

```bash
# Push changes
git add -A
git commit -m "Description"
git push origin main
```

Streamlit Cloud auto-redeploys on every push to `main`.

**Python version:** 3.11 (set in `.devcontainer/devcontainer.json` — required for Streamlit Cloud)  
**Important:** All Python variable names must be ASCII only (no `ç`, `ã`, `ê`) — Linux/Streamlit Cloud requirement.

### Recent Commit History
```
f83b861  Translate member report header, claims subheader, data table and session records
a8d73e2  Translate adjudication panel, risk distribution and card labels to EN
54e18c8  Translate top 10 alert cards and flag texts in EN mode
b803f29  Translate no-data message on overview page
9a26624  Fix platform header title - now translates to EN with language switch
a6a9c1b  Fix demo data section translation + update language label to Idioma/Language
eeb81e2  Fix language buttons + translate all page content to EN/PT
7dac974  Add PT/EN language toggle with full i18n module
9ee4a68  Show adjudication panel, ghost beneficiary and over-prescription in UI
c0ae57a  Add ghost beneficiary, over-prescription, auto-adjudication + premium UI theme
```

---

## 11. Known Issues & Technical Debt

| Issue | Status | Notes |
|-------|--------|-------|
| `"Como Funciona"` page content still in PT only | Open | Triple-quoted HTML strings — need conversion to f-strings + `t()` keys |
| `contamination` slider label hardcoded PT in `app.py` line 358 | Open | `"Sensibilidade a anomalias..."` not yet using `t("data_sensitivity")` |
| Some inline chart labels hardcoded PT (axis labels, titles) | Open | `"Solicitações"`, `"Mês"`, `"Risco do Prestador..."` in chart configs |
| Claims card view (tab1) hardcoded `"Solicitação"`, `"Benef."`, `"Prestador:"` | Open | Lines ~940–960 |
| Provider page: hardcoded `"solicitações deste prestador"` caption | Open | Line 1059 |
| `risk_scorer.py`: adjudication values stored in PT (`"Investigar Urgente"`) | Design choice | Translated at display time via string matching in `app.py` |

---

## 12. Dependencies

```
streamlit>=1.35
pandas>=2.0
numpy>=1.25
scikit-learn>=1.4      # IsolationForest
plotly>=5.20
openpyxl>=3.1          # Excel export
reportlab>=4.0         # PDF export
```

Install: `pip install -r requirements.txt` in `ml_env` conda environment.

---

## 13. Business Context

**Target buyers:** Principal Officers, Fraud/Forensic Heads, CFOs at mid-tier SA medical schemes  
**Pitch:** Enterprise-grade fraud detection without 12-month implementation or six-figure licence  
**Key differentiators:**
- Works with any claims export (no IT integration)
- 90-second analysis for 2,000 claims
- CMS Rule 29 compliance audit trail ready
- Export to Excel/PDF for board reporting
- Auto-adjudication recommendation reduces manual triage time

**Outreach assets created:**
- 30 target contacts across 10 schemes: `SA_Medical_Schemes_CRM.xlsx`
- 6-touchpoint cold outreach sequence (4 emails + 2 LinkedIn messages)
- LinkedIn post (Mozambique market focus)
