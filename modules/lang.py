"""
Internationalisation module — PT / EN
Usage:
    from modules.lang import t, get_lang, FLAG_MAP
    txt = t("page_overview")        # returns string in active language
    en_flag = FLAG_MAP["en"].get(pt_flag, pt_flag)   # translate a flag string
"""

TRANSLATIONS = {
    # ── Navigation ─────────────────────────────────────────────────────────────
    "nav_overview":      {"pt": "Painel de Controlo",          "en": "Executive Dashboard"},
    "nav_claims":        {"pt": "Análise de Solicitações",     "en": "Claims Analysis"},
    "nav_providers":     {"pt": "Inteligência de Prestadores", "en": "Provider Intelligence"},
    "nav_members":       {"pt": "Análise de Beneficiários",    "en": "Member Analysis"},
    "nav_costs":         {"pt": "Custos Atípicos",             "en": "Cost Outliers"},
    "nav_report":        {"pt": "Relatório por Beneficiário",  "en": "Member Risk Report"},
    "nav_data":          {"pt": "Gestão de Dados",             "en": "Data Management"},
    "nav_howto":         {"pt": "Como Funciona",               "en": "How It Works"},
    "nav_eval":          {"pt": "Avaliação da Plataforma",     "en": "Platform Evaluation"},

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    "sidebar_title":     {"pt": "Painel Executivo",            "en": "Executive Panel"},
    "sidebar_session":   {"pt": "Sessão Activa",               "en": "Active Session"},
    "sidebar_claims":    {"pt": "solicitações",                "en": "claims"},
    "sidebar_highrisk":  {"pt": "de alto risco",               "en": "high risk"},
    "sidebar_skipped":   {"pt": "Análises omitidas",           "en": "Skipped analyses"},

    # ── Platform header ─────────────────────────────────────────────────────────
    "platform_title":    {"pt": "Plataforma de Detecção de Fraude para Seguro de Saúde",
                          "en": "Health Insurance Fraud Detection Platform"},
    "platform_sub":      {"pt": "Fraude • Desperdício • Abuso • Risco de Prestadores • Custos Atípicos",
                          "en": "Fraud • Waste • Abuse • Provider Risk • Cost Outliers"},

    # ── Overview Dashboard ──────────────────────────────────────────────────────
    "kpi_total":         {"pt": "Total de Solicitações",       "en": "Total Claims"},
    "kpi_high":          {"pt": "Risco Alto",                  "en": "High Risk"},
    "kpi_medium":        {"pt": "Risco Médio",                 "en": "Medium Risk"},
    "kpi_flagged_rate":  {"pt": "Taxa Sinalizada",             "en": "Flagged Rate"},
    "kpi_high_value":    {"pt": "Valor em Risco Alto",         "en": "High-Risk Value"},
    "adj_panel_label":   {"pt": "Recomendacao de Adjudicacao Automatica",
                          "en": "Auto-Adjudication Recommendation"},
    "adj_investigate":   {"pt": "Investigar Urgente",          "en": "Urgent Investigation"},
    "adj_review":        {"pt": "Rever Manualmente",           "en": "Manual Review"},
    "adj_approve":       {"pt": "Aprovar Automaticamente",     "en": "Auto-Approve"},
    "adj_investigate_desc": {"pt": "Risco >= 70. Encaminhar para equipa de investigacao.",
                             "en": "Risk >= 70. Escalate to investigation team."},
    "adj_review_desc":   {"pt": "Risco 40-69. Requer revisao por gestor de sinistros.",
                          "en": "Risk 40-69. Requires review by claims manager."},
    "adj_approve_desc":  {"pt": "Risco < 40. Dentro dos parametros normais. Auto-adjudicar.",
                          "en": "Risk < 40. Within normal parameters. Auto-adjudicate."},
    "adj_pct":           {"pt": "do total",                    "en": "of total"},
    "risk_dist_label":   {"pt": "Distribuicao de Risco",       "en": "Risk Distribution"},
    "claims_analyzed":   {"pt": "solicitacoes analisadas",     "en": "claims analyzed"},
    "high_risk_label":   {"pt": "Alto Risco",                  "en": "High Risk"},
    "medium_risk_label": {"pt": "Risco Médio",                 "en": "Medium Risk"},
    "low_risk_label":    {"pt": "Baixo Risco",                 "en": "Low Risk"},
    "top10_title":       {"pt": "Top 10 Solicitações de Alto Risco",
                          "en": "Top 10 High-Risk Claims"},
    "claim_label":       {"pt": "Solicitacao",                 "en": "Claim"},
    "member_label":      {"pt": "Benef.",                      "en": "Member"},
    "provider_label":    {"pt": "Prestador",                   "en": "Provider"},
    "value_label":       {"pt": "Valor",                       "en": "Amount"},
    "score_label":       {"pt": "Score",                       "en": "Score"},
    "trend_title":       {"pt": "Evolução Mensal de Solicitações",
                          "en": "Monthly Claims Trend"},
    "trend_month":       {"pt": "Mês",                         "en": "Month"},
    "trend_claims":      {"pt": "Solicitações",                "en": "Claims"},
    "risk_level_label":  {"pt": "Nível de Risco",              "en": "Risk Level"},

    # ── Claims Analysis ─────────────────────────────────────────────────────────
    "claims_title":      {"pt": "Análise de Solicitações",     "en": "Claims Analysis"},
    "filter_risk":       {"pt": "Nível de Risco",              "en": "Risk Level"},
    "filter_min_score":  {"pt": "Pontuação Mínima",            "en": "Minimum Score"},
    "filter_provider":   {"pt": "Prestador",                   "en": "Provider"},
    "filter_all":        {"pt": "Todos",                       "en": "All"},
    "filter_period":     {"pt": "Período",                     "en": "Period"},
    "showing_of":        {"pt": "A mostrar",                   "en": "Showing"},
    "of_label":          {"pt": "de",                          "en": "of"},
    "tab_cards":         {"pt": "🃏  Vista em Cards",           "en": "🃏  Card View"},
    "tab_table":         {"pt": "📋  Vista em Tabela",          "en": "📋  Table View"},
    "no_results":        {"pt": "Nenhum resultado encontrado.", "en": "No results found."},
    "export_title":      {"pt": "Exportar Resultados",         "en": "Export Results"},
    "btn_excel":         {"pt": "📥 Descarregar Excel",        "en": "📥 Download Excel"},
    "btn_csv":           {"pt": "📥 Descarregar CSV",          "en": "📥 Download CSV"},
    "btn_pdf":           {"pt": "📥 Descarregar PDF",          "en": "📥 Download PDF"},

    # ── Provider Intelligence ───────────────────────────────────────────────────
    "prov_title":        {"pt": "Inteligência de Prestadores", "en": "Provider Intelligence"},
    "prov_slider":       {"pt": "Mostrar os N prestadores com maior risco",
                          "en": "Show top N providers by risk"},
    "prov_chart_title":  {"pt": "Prestadores por Pontuação de Risco",
                          "en": "Providers by Risk Score"},
    "prov_detail":       {"pt": "Detalhe do Prestador",        "en": "Provider Detail"},
    "prov_select":       {"pt": "Seleccionar prestador",       "en": "Select provider"},
    "prov_risk_score":   {"pt": "Pontuação de Risco",          "en": "Risk Score"},
    "prov_total_claims": {"pt": "Total de Solicitações",       "en": "Total Claims"},
    "prov_avg_value":    {"pt": "Valor Médio",                 "en": "Avg Amount"},
    "prov_dup_rate":     {"pt": "Taxa de Duplicados",          "en": "Duplicate Rate"},
    "prov_claims_count": {"pt": "solicitações deste prestador","en": "claims for this provider"},
    "prov_scatter_title":{"pt": "Risco do Prestador: Volume vs. Valor Médio",
                          "en": "Provider Risk: Volume vs. Average Amount"},
    "prov_scatter_x":    {"pt": "Número de Solicitações",      "en": "Number of Claims"},
    "prov_scatter_y":    {"pt": "Valor Médio ($)",             "en": "Avg Amount ($)"},

    # ── Member Analysis ─────────────────────────────────────────────────────────
    "mem_title":         {"pt": "Análise de Utilização por Beneficiário",
                          "en": "Member Utilization Analysis"},
    "mem_slider":        {"pt": "Mostrar os N beneficiários com maior risco",
                          "en": "Show top N members by risk"},
    "mem_chart_title":   {"pt": "Beneficiários por Pontuação de Risco",
                          "en": "Members by Risk Score"},
    "mem_scatter_title": {"pt": "Padrão de Utilização: Prestadores vs. Gasto Total",
                          "en": "Utilisation Pattern: Providers vs. Total Spend"},
    "mem_scatter_x":     {"pt": "Prestadores Distintos",       "en": "Distinct Providers"},
    "mem_scatter_y":     {"pt": "Gasto Total ($)",             "en": "Total Spend ($)"},
    "mem_table_title":   {"pt": "Beneficiários de Alto Risco", "en": "High-Risk Members"},

    # ── Cost Outliers ───────────────────────────────────────────────────────────
    "cost_title":        {"pt": "Análise de Custos Atípicos",  "en": "Cost Outlier Analysis"},
    "cost_benchmark":    {"pt": "Grupo de referência (benchmark):",
                          "en": "Peer benchmark group:"},
    "cost_benchmark_proc":  {"pt": "código de procedimento",   "en": "procedure code"},
    "cost_benchmark_spec":  {"pt": "especialidade do prestador","en": "provider specialty"},
    "cost_benchmark_port":  {"pt": "carteira global",          "en": "portfolio overall"},
    "cost_scatter_title": {"pt": "Valor da Solicitação vs. Referência de Pares",
                           "en": "Claim Amount vs. Peer Benchmark"},
    "cost_scatter_x":    {"pt": "Média de Pares ($)",          "en": "Peer Mean ($)"},
    "cost_scatter_y":    {"pt": "Valor Facturado ($)",         "en": "Billed Amount ($)"},
    "cost_hist_title":   {"pt": "Distribuição dos Valores por Nível de Risco",
                          "en": "Amount Distribution by Risk Level"},
    "cost_hist_x":       {"pt": "Valor da Solicitação ($)",    "en": "Claim Amount ($)"},
    "cost_table_title":  {"pt": "Principais Solicitações com Custos Atípicos",
                          "en": "Top Cost Outlier Claims"},

    # ── Member Report ───────────────────────────────────────────────────────────
    "report_title":      {"pt": "Relatório de Risco por Beneficiário",
                          "en": "Member Risk Report"},
    "report_caption":    {"pt": "Seleccione um beneficiário para gerar o seu perfil de risco completo.",
                          "en": "Select a member to generate their full risk profile."},
    "report_select":     {"pt": "Seleccionar Beneficiário",    "en": "Select Member"},
    "report_risk_label": {"pt": "Nível de Risco:",             "en": "Risk Level:"},
    "report_flags":      {"pt": "Sinais de Risco Detectados",  "en": "Detected Risk Signals"},
    "report_no_flags":   {"pt": "Nenhum sinal de risco específico detectado.",
                          "en": "No specific risk signals detected."},
    "report_claims_title": {"pt": "Solicitações do Beneficiário",
                            "en": "Member Claims"},
    "report_trend":      {"pt": "Evolução de Gastos",          "en": "Spending Trend"},
    "report_trend_title":{"pt": "Gasto Mensal do Beneficiário","en": "Monthly Member Spend"},
    "report_export":     {"pt": "Exportar Relatório Individual","en": "Export Individual Report"},
    "report_pdf_btn":    {"pt": "📥 Descarregar PDF do Beneficiário",
                          "en": "📥 Download Member PDF"},
    "report_xlsx_btn":   {"pt": "📥 Descarregar Excel do Beneficiário",
                          "en": "📥 Download Member Excel"},
    "report_total_spend":{"pt": "Gasto Total",                 "en": "Total Spend"},
    "report_providers":  {"pt": "Prestadores Distintos",       "en": "Distinct Providers"},

    # ── Data Management ─────────────────────────────────────────────────────────
    "data_title":        {"pt": "Gestão de Dados",             "en": "Data Management"},
    "data_demo_title":   {"pt": "Experimente com Dados de Demonstração",
                          "en": "Try with Demo Data"},
    "data_demo_desc":    {"pt": "Carregue automaticamente 1.931 solicitações sintéticas com padrões de fraude injectados.",
                          "en": "Automatically load 1,931 synthetic claims with injected fraud patterns."},
    "data_demo_btn":     {"pt": "🚀  Carregar Dados de Demonstração",
                          "en": "🚀  Load Demo Data"},
    "data_upload_title": {"pt": "Carregar Novo Ficheiro de Solicitações",
                          "en": "Upload New Claims File"},
    "data_analyze_btn":  {"pt": "Analisar Ficheiro",           "en": "Analyse File"},
    "data_sessions":     {"pt": "Sessões Anteriores",          "en": "Previous Sessions"},
    "data_no_sessions":  {"pt": "Nenhuma sessão anterior encontrada.",
                          "en": "No previous sessions found."},
    "data_load_btn":     {"pt": "Carregar",                    "en": "Load"},
    "data_records":      {"pt": "registos",                    "en": "records"},
    "data_col_format":   {"pt": "Formato de Colunas Necessário","en": "Required Column Format"},
    "data_col_required": {"pt": "Obrigatório",                 "en": "Required"},
    "data_col_optional": {"pt": "Opcional",                    "en": "Optional"},
    "data_sensitivity":  {"pt": "Sensibilidade a anomalias (% esperada de outliers)",
                          "en": "Anomaly sensitivity (% expected outliers)"},

    # ── Spinners / messages ─────────────────────────────────────────────────────
    "spin_columns":      {"pt": "A detectar colunas...",       "en": "Detecting columns..."},
    "spin_anomaly":      {"pt": "A executar detecção de anomalias...",
                          "en": "Running anomaly detection..."},
    "spin_providers":    {"pt": "A calcular risco por prestador...",
                          "en": "Scoring providers..."},
    "spin_members":      {"pt": "A analisar utilização por beneficiário...",
                          "en": "Analysing member utilisation..."},
    "spin_costs":        {"pt": "A detectar custos atípicos...",
                          "en": "Detecting cost outliers..."},
    "spin_scoring":      {"pt": "A calcular pontuação de risco composta...",
                          "en": "Computing composite risk scores..."},
    "analysis_complete": {"pt": "Análise concluída.",          "en": "Analysis complete."},
    "missing_cols":      {"pt": "Colunas obrigatórias em falta:",
                          "en": "Missing required columns:"},
    "no_data_msg":       {"pt": "Nenhum dado carregado. Aceda a **Gestão de Dados** para carregar um ficheiro.",
                          "en": "No data loaded. Go to **Data Management** to upload a file."},
    "no_data_title":     {"pt": "Nenhum dado carregado",       "en": "No data loaded"},
    "no_data_sub":       {"pt": "Aceda a <strong style='color:#F59E0B'>Gestão de Dados</strong> no menu lateral para carregar um ficheiro CSV ou Excel e iniciar a análise.",
                          "en": "Go to <strong style='color:#F59E0B'>Data Management</strong> in the sidebar to upload a CSV or Excel file and start the analysis."},

    # ── How It Works ────────────────────────────────────────────────────────────
    "howto_title":       {"pt": "Como Funciona a Plataforma",  "en": "How the Platform Works"},
    "howto_sub":         {"pt": "Guia completo de utilização e metodologia de detecção de fraude",
                          "en": "Complete usage guide and fraud detection methodology"},
    "howto_tab1":        {"pt": "🚀  Início Rápido",           "en": "🚀  Quick Start"},
    "howto_tab2":        {"pt": "🧠  Metodologia",             "en": "🧠  Methodology"},
    "howto_tab3":        {"pt": "📊  Módulos de Análise",      "en": "📊  Analysis Modules"},
    "howto_tab4":        {"pt": "📁  Formato dos Dados",       "en": "📁  Data Format"},

    # ── Risk levels ─────────────────────────────────────────────────────────────
    "risk_high":         {"pt": "Alto",                        "en": "High"},
    "risk_medium":       {"pt": "Médio",                       "en": "Medium"},
    "risk_low":          {"pt": "Baixo",                       "en": "Low"},
    "risk_score_col":    {"pt": "Pontuação de Risco",          "en": "Risk Score"},
    "adjudication_col":  {"pt": "Adjudicação",                 "en": "Adjudication"},
    "no_signals":        {"pt": "Sem sinais especificos detectados",
                          "en": "No specific signals detected"},
}

# Flag text translation map (PT -> EN)
FLAG_TRANSLATIONS = {
    "pt": {
        "Anomalia detectada por modelo estatístico": "Anomalia detectada por modelo estatístico",
        "Valor facturado suspeito (numero redondo)": "Valor facturado suspeito (numero redondo)",
        "Possivel solicitacao duplicada":            "Possivel solicitacao duplicada",
        "Procedimento de alta frequencia vs. prestadores similares": "Procedimento de alta frequencia vs. prestadores similares",
        "Beneficiario com idade elevada (possivel fantasma)":        "Beneficiario com idade elevada (possivel fantasma)",
        "Sobreprescricao de medicamentos (>3 por mes)":              "Sobreprescricao de medicamentos (>3 por mes)",
        "Sem sinais especificos detectados":                         "Sem sinais especificos detectados",
    },
    "en": {
        "Anomalia detectada por modelo estatístico": "Statistical model anomaly detected",
        "Valor facturado suspeito (numero redondo)": "Suspicious round-number billing",
        "Possivel solicitacao duplicada":            "Potential duplicate claim",
        "Procedimento de alta frequencia vs. prestadores similares": "High-frequency procedure vs. peer providers",
        "Beneficiario com idade elevada (possivel fantasma)":        "High-age beneficiary (possible ghost member)",
        "Sobreprescricao de medicamentos (>3 por mes)":              "Over-prescription detected (>3/month)",
        "Sem sinais especificos detectados":                         "No specific signals detected",
    }
}


def get_lang() -> str:
    """Return current language from Streamlit session state."""
    try:
        import streamlit as st
        return st.session_state.get("lang", "pt")
    except Exception:
        return "pt"


def t(key: str) -> str:
    """Return translated string for the current language."""
    lang = get_lang()
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("pt", key))


def translate_flag(flag_text: str) -> str:
    """Translate a flag string based on current language."""
    lang = get_lang()
    if lang == "pt":
        return flag_text

    lang_map = FLAG_TRANSLATIONS.get("en", {})

    # Exact match
    if flag_text in lang_map:
        return lang_map[flag_text]

    # Dynamic: "Facturado 3.2x acima da média do prestador (Z>3.5)"
    if "acima da média do prestador" in flag_text:
        import re
        m = re.search(r"Facturado\s+([\d.]+)x\s+acima da média do prestador\s*(\(.*\))?", flag_text)
        if m:
            mult = m.group(1)
            zpart = m.group(2) or ""
            return f"Billed {mult}x above provider average {zpart}".strip()

    # Dynamic: "Custo 738% acima da média de referência (procedimento)"
    if "acima da média de referência" in flag_text:
        import re
        m = re.search(r"Custo\s+(-?[\d.]+)%\s+acima da média de referência\s*\((\w+)\)?", flag_text)
        if m:
            pct   = m.group(1)
            label = m.group(2).replace("procedimento", "procedure") \
                               .replace("especialidade", "specialty") \
                               .replace("carteira", "portfolio")
            return f"Cost {pct}% above peer benchmark ({label})"

    return flag_text
