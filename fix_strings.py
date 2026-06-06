"""Run this script to replace all hardcoded Portuguese strings with t() calls."""

with open('app.py', 'r', encoding='utf-8') as f:
    c = f.read()

replacements = [
    # KPI labels in overview
    ('"Total de Solicitações"',   't("kpi_total")'),
    ('"Risco Alto"',              't("kpi_high")'),
    ('"Risco Médio"',             't("kpi_medium")'),
    ('"Taxa Sinalizada"',         't("kpi_flagged_rate")'),
    ('"Valor em Risco Alto"',     't("kpi_high_value")'),

    # Adjudication panel
    ('"Recomendacao de Adjudicacao Automatica"', 't("adj_panel_label")'),
    ('"das solicitacoes"',        't("adj_pct")'),

    # Risk distribution
    ('"Distribuicao de Risco"',   't("risk_dist_label")'),
    ('"solicitacoes analisadas"', 't("claims_analyzed")'),
    ('"Alto Risco"',              't("high_risk_label")'),
    ('"Baixo Risco"',             't("low_risk_label")'),

    # Top 10
    ('st.subheader("Top 10 Solicitações de Alto Risco")', 'st.subheader(t("top10_title"))'),

    # Trend chart
    ('st.subheader("Evolução Mensal de Solicitações")', 'st.subheader(t("trend_title"))'),

    # Claims analysis filters
    ('st.multiselect("Nível de Risco"', 'st.multiselect(t("filter_risk")'),
    ('st.slider("Pontuação Mínima"',    'st.slider(t("filter_min_score")'),
    ('st.selectbox("Prestador", providers)', 'st.selectbox(t("filter_provider"), providers)'),
    ('providers = ["Todos"]',            'providers = [t("filter_all")]'),
    ('prov_filter != "Todos"',           'prov_filter != t("filter_all")'),
    ('st.date_input("Período"',          'st.date_input(t("filter_period")'),

    # Tab labels
    ('"🃏  Vista em Cards"', 't("tab_cards")'),
    ('"📋  Vista em Tabela"', 't("tab_table")'),

    # Export
    ('st.subheader("Exportar Resultados")', 'st.subheader(t("export_title"))'),
    ('"📥 Descarregar Excel"',  't("btn_excel")'),
    ('"📥 Descarregar CSV"',    't("btn_csv")'),
    ('"📥 Descarregar PDF"',    't("btn_pdf")'),

    # Provider page
    ('st.slider("Mostrar os N prestadores com maior risco"', 'st.slider(t("prov_slider")'),
    ('st.subheader("Detalhe do Prestador")', 'st.subheader(t("prov_detail"))'),
    ('st.selectbox("Seleccionar prestador"', 'st.selectbox(t("prov_select")'),
    ('p1.metric("Pontuação de Risco"', 'p1.metric(t("prov_risk_score")'),
    ('p2.metric("Total de Solicitações"', 'p2.metric(t("prov_total_claims")'),
    ('p3.metric("Valor Médio"', 'p3.metric(t("prov_avg_value")'),
    ('p4.metric("Taxa de Duplicados"', 'p4.metric(t("prov_dup_rate")'),

    # Member page
    ('st.slider("Mostrar os N beneficiários com maior risco"', 'st.slider(t("mem_slider")'),
    ('st.subheader("Beneficiários de Alto Risco")', 'st.subheader(t("mem_table_title"))'),

    # Cost outliers
    ('st.subheader("Principais Solicitações com Custos Atípicos")', 'st.subheader(t("cost_table_title"))'),
    ('"Distribuição dos Valores por Nível de Risco"', 't("cost_hist_title")'),

    # Member report
    ('st.subheader("Sinais de Risco Detectados")', 'st.subheader(t("report_flags"))'),
    ('st.subheader("Evolução de Gastos")', 'st.subheader(t("report_trend"))'),
    ('st.subheader("Exportar Relatório Individual")', 'st.subheader(t("report_export"))'),
    ('"📥 Descarregar PDF do Beneficiário"', 't("report_pdf_btn")'),
    ('"📥 Descarregar Excel do Beneficiário"', 't("report_xlsx_btn")'),
    ('k3.metric("Gasto Total"', 'k3.metric(t("report_total_spend")'),
    ('k4.metric("Prestadores Distintos"', 'k4.metric(t("report_providers")'),

    # Data management
    ('st.subheader("Carregar Novo Ficheiro de Solicitações")', 'st.subheader(t("data_upload_title"))'),
    ('st.subheader("Sessões Anteriores")', 'st.subheader(t("data_sessions"))'),
    ('st.subheader("Formato de Colunas Necessário")', 'st.subheader(t("data_col_format"))'),
    ('"Analisar Ficheiro"',      't("data_analyze_btn")'),
    ('"Carregar"',               't("data_load_btn")'),

    # How it works
    ('"🚀  Início Rápido"',  't("howto_tab1")'),
    ('"🧠  Metodologia"',    't("howto_tab2")'),
    ('"📊  Módulos de Análise"', 't("howto_tab3")'),
    ('"📁  Formato dos Dados"',  't("howto_tab4")'),

    # Platform header HTML
    ('Plataforma de Deteccao de Fraude para Seguro de Saude', '" + t("platform_title") + "'),
    ('Fraude &bull; Desperdicio &bull; Abuso &bull; Risco de Prestadores &bull; Custos Atipicos',
     '" + t("platform_sub") + "'),
]

for old, new in replacements:
    c = c.replace(old, new)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Done - replaced', len(replacements), 'strings')
