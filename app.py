import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Equity Screening Lab", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: #f5f7fa; border: 1px solid #e5e9ef; padding: 14px 16px; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #536174;}
h1, h2, h3 {letter-spacing: -0.02em;}
</style>
""", unsafe_allow_html=True)

st.title("Equity Screening Lab")
st.caption("A three-stage, industry-aware stock screening and portfolio research tool")

DATA_PATH = Path(__file__).with_name("sample_700_companies.csv")

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def process_screen(df, cut1, cut2, cut3, use_industry_demean, portfolio_size, weighting):
    d = df.copy()
    margin_cols = [f'Margin_Tminus{i}' for i in range(7, -1, -1)]
    required = ['Company', 'Ticker', 'Sector', 'Industry', 'STA', 'SNOA', 'Z_Score', 'GPA', 'BTM', 'CFOA']
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    metrics = ['STA','SNOA','Z_Score','GPA','BTM','CFOA']
    for c in metrics:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    # Seven-year window = eight annual margin observations and seven year-over-year growth rates.
    if all(c in d.columns for c in margin_cols):
        for c in margin_cols:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        margin_values = d[margin_cols].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            growth_rates = margin_values[:, 1:] / margin_values[:, :-1] - 1
        growth_rates[~np.isfinite(growth_rates)] = np.nan
        d['MS'] = np.nanmean(growth_rates, axis=1) / np.nanstd(growth_rates, axis=1, ddof=1)
        start_margin = margin_values[:, 0]
        end_margin = margin_values[:, -1]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            d['MG'] = np.where((start_margin > 0) & (end_margin > 0), (end_margin / start_margin) ** (1/7) - 1, np.nan)
        d.loc[~np.isfinite(d['MS']), 'MS'] = np.nan
        d.loc[~np.isfinite(d['MG']), 'MG'] = np.nan
    else:
        # Backward compatibility for uploads containing pre-calculated MS/MG.
        for c in ['MS', 'MG']:
            if c not in d.columns:
                d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['Company','Ticker','Sector','Industry']).drop_duplicates(subset=['Ticker']).reset_index(drop=True)
    d['Stage'] = 'Eligible'
    d['Stage1_Rank'] = np.nan; d['Stage2_Rank'] = np.nan; d['Stage3_Rank'] = np.nan
    d['STA_Adjusted'] = d['STA']; d['SNOA_Adjusted'] = d['SNOA']
    if use_industry_demean:
        d['STA_Adjusted'] = d['STA'] - d.groupby('Industry')['STA'].transform('mean')
        d['SNOA_Adjusted'] = d['SNOA'] - d.groupby('Industry')['SNOA'].transform('mean')
    def rank_metric(frame, col, ascending=True):
        # ascending=True means lower raw value receives rank 1
        return frame[col].rank(method='average', ascending=ascending, na_option='bottom')
    # Stage 1: lower adjusted STA/SNOA good, higher Z good
    d['R_STA'] = rank_metric(d, 'STA_Adjusted', True)
    d['R_SNOA'] = rank_metric(d, 'SNOA_Adjusted', True)
    d['R_Z'] = rank_metric(d, 'Z_Score', False)
    d['Stage1_Total'] = d[['R_STA','R_SNOA','R_Z']].sum(axis=1, min_count=1)
    d['Stage1_Rank'] = d['Stage1_Total'].rank(method='min', ascending=True, na_option='bottom')
    def eliminate(frame, pct, cumulative_col, stage_name):
        frame = frame.copy()
        n_remove = int(np.ceil(len(frame) * pct / 100)) if pct > 0 else 0
        # Highest cumulative rank score is weakest; stable tie break by ticker
        ordered = frame.sort_values([cumulative_col, 'Ticker'], ascending=[False, True], na_position='first')
        removed = ordered.head(n_remove)['Ticker'].tolist()
        frame.loc[frame['Ticker'].isin(removed), 'Stage'] = stage_name + ' — eliminated'
        survivors = frame[~frame['Ticker'].isin(removed)].copy()
        return frame, survivors, removed
    all_rows, survivors1, removed1 = eliminate(d, cut1, 'Stage1_Total', 'Stage 1')
    # Re-rank stage 1 on the new population and carry updated ranks forward
    survivors1['R_STA'] = rank_metric(survivors1, 'STA_Adjusted', True)
    survivors1['R_SNOA'] = rank_metric(survivors1, 'SNOA_Adjusted', True)
    survivors1['R_Z'] = rank_metric(survivors1, 'Z_Score', False)
    survivors1['Stage1_Total'] = survivors1[['R_STA','R_SNOA','R_Z']].sum(axis=1, min_count=1)
    survivors1['Stage1_Rank'] = survivors1['Stage1_Total'].rank(method='min', ascending=True, na_option='bottom')
    # Stage 2: max percentile of MS and MG. Higher percentile is better.
    survivors1['MS_Percentile'] = survivors1['MS'].rank(pct=True, method='average')
    survivors1['MG_Percentile'] = survivors1['MG'].rank(pct=True, method='average')
    survivors1['MM'] = survivors1[['MS_Percentile','MG_Percentile']].max(axis=1, skipna=True)
    survivors1['R_MM'] = rank_metric(survivors1, 'MM', False)
    survivors1['Stage2_Total'] = survivors1['Stage1_Rank'] + survivors1['R_MM']
    survivors1['Stage2_Rank'] = survivors1['Stage2_Total'].rank(method='min', ascending=True, na_option='bottom')
    after2, survivors2, removed2 = eliminate(survivors1, cut2, 'Stage2_Total', 'Stage 2')
    # Refresh Stage 1 and Stage 2 ranks after Stage 2 elimination
    for col, asc in [('STA_Adjusted',True),('SNOA_Adjusted',True),('Z_Score',False)]:
        rcol = {'STA_Adjusted':'R_STA','SNOA_Adjusted':'R_SNOA','Z_Score':'R_Z'}[col]
        survivors2[rcol] = rank_metric(survivors2, col, asc)
    survivors2['Stage1_Total'] = survivors2[['R_STA','R_SNOA','R_Z']].sum(axis=1, min_count=1)
    survivors2['Stage1_Rank'] = survivors2['Stage1_Total'].rank(method='min', ascending=True, na_option='bottom')
    survivors2['MS_Percentile'] = survivors2['MS'].rank(pct=True, method='average')
    survivors2['MG_Percentile'] = survivors2['MG'].rank(pct=True, method='average')
    survivors2['MM'] = survivors2[['MS_Percentile','MG_Percentile']].max(axis=1, skipna=True)
    survivors2['R_MM'] = rank_metric(survivors2, 'MM', False)
    survivors2['Stage2_Total'] = survivors2['Stage1_Rank'] + survivors2['R_MM']
    survivors2['Stage2_Rank'] = survivors2['Stage2_Total'].rank(method='min', ascending=True, na_option='bottom')
    # Stage 3: GPA and CFOA higher is better; BTM higher is treated as better by default, user can adjust outside this version.
    survivors2['R_GPA'] = rank_metric(survivors2, 'GPA', False)
    survivors2['R_BTM'] = rank_metric(survivors2, 'BTM', False)
    survivors2['R_CFOA'] = rank_metric(survivors2, 'CFOA', False)
    survivors2['Stage3_Total'] = survivors2['Stage2_Rank'] + survivors2[['R_GPA','R_BTM','R_CFOA']].sum(axis=1, min_count=1)
    survivors2['Stage3_Rank'] = survivors2['Stage3_Total'].rank(method='min', ascending=True, na_option='bottom')
    after3, survivors3, removed3 = eliminate(survivors2, cut3, 'Stage3_Total', 'Stage 3')
    # Recalculate all ranks for final survivors for transparent final results
    if len(survivors3):
        for col, asc, rcol in [('STA_Adjusted',True,'R_STA'),('SNOA_Adjusted',True,'R_SNOA'),('Z_Score',False,'R_Z'),('MM',False,'R_MM'),('GPA',False,'R_GPA'),('BTM',False,'R_BTM'),('CFOA',False,'R_CFOA')]:
            survivors3[rcol] = rank_metric(survivors3, col, asc)
        survivors3['Stage1_Total'] = survivors3[['R_STA','R_SNOA','R_Z']].sum(axis=1, min_count=1)
        survivors3['Stage1_Rank'] = survivors3['Stage1_Total'].rank(method='min', ascending=True, na_option='bottom')
        survivors3['Stage2_Total'] = survivors3['Stage1_Rank'] + survivors3['R_MM']
        survivors3['Stage2_Rank'] = survivors3['Stage2_Total'].rank(method='min', ascending=True, na_option='bottom')
        survivors3['Stage3_Total'] = survivors3['Stage2_Rank'] + survivors3[['R_GPA','R_BTM','R_CFOA']].sum(axis=1, min_count=1)
        survivors3['Final_Rank'] = survivors3['Stage3_Total'].rank(method='min', ascending=True, na_option='bottom')
        survivors3 = survivors3.sort_values(['Final_Rank','Ticker'])
        if weighting == 'Equal weight':
            survivors3['Portfolio_Weight'] = 1 / len(survivors3)
        else:
            # Rank-based weights favor better ranked survivors; always normalized to 100%.
            raw = 1 / survivors3['Final_Rank'].clip(lower=1)
            survivors3['Portfolio_Weight'] = raw / raw.sum()
    # Create an audit trail with all companies and final disposition
    audit = d[['Company','Ticker','Sector','Industry','STA','SNOA','Z_Score','MS','MG','GPA','BTM','CFOA']].copy()
    audit['Disposition'] = 'Survived all stages'
    audit.loc[audit['Ticker'].isin(removed1), 'Disposition'] = 'Eliminated at Stage 1'
    audit.loc[audit['Ticker'].isin(removed2), 'Disposition'] = 'Eliminated at Stage 2'
    audit.loc[audit['Ticker'].isin(removed3), 'Disposition'] = 'Eliminated at Stage 3'
    return d, survivors1, survivors2, survivors3, audit, (removed1, removed2, removed3)

with st.sidebar:
    st.header('Screen settings')
    st.caption('Change these inputs to rerun the full screening process.')
    cut1 = st.slider('Stage 1: safety metrics removed (%)', 0, 50, 10, 5)
    cut2 = st.slider('Stage 2: combined rank removed (%)', 0, 50, 10, 5)
    cut3 = st.slider('Stage 3: combined rank removed (%)', 0, 50, 10, 5)
    demean = st.checkbox('Demean STA and SNOA by industry', value=True)
    weighting = st.selectbox('Portfolio weighting', ['Equal weight', 'Rank weighted'])
    st.divider()
    st.subheader('Data source')
    uploaded = st.file_uploader('Upload your company CSV', type=['csv'], help='Use the required column names shown below. Uploaded data replaces the demo universe.')
    st.caption('For automatic MS/MG calculations, include Margin_Tminus7 through Margin_T (eight annual margin observations). These produce seven annual growth rates. The included dataset is synthetic demo data, not reported financial data.')

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        data_source = 'Uploaded CSV'
    except Exception as e:
        st.error(f'Could not read the uploaded CSV: {e}')
        st.stop()
else:
    data = load_sample()
    data_source = 'Synthetic demonstration dataset'

try:
    original, s1, s2, final_stage, audit, removed = process_screen(data, cut1, cut2, cut3, demean, 25, weighting)
except Exception as e:
    st.error(f'Could not run the screen: {e}')
    st.info('Required columns: Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, GPA, BTM, CFOA, plus either the eight annual margin columns (recommended) or pre-calculated MS and MG.')
    st.stop()

n0 = len(original); n1 = n0 - len(removed[0]); n2 = len(s1) - len(removed[1]); n3 = len(final_stage)
c1,c2,c3,c4 = st.columns(4)
c1.metric('Starting universe', f'{n0:,}')
c2.metric('After Stage 1', f'{n1:,}', delta=f'-{len(removed[0])} removed')
c3.metric('After Stage 2', f'{n2:,}', delta=f'-{len(removed[1])} removed')
c4.metric('Final survivors', f'{n3:,}', delta=f'-{len(removed[2])} removed')
st.caption(f'Data: {data_source} · Ranking convention: rank 1 is best · Elimination percentages apply sequentially to the current surviving population.')

tabs = st.tabs(['Overview', 'Final portfolio', 'Stage analysis', 'Company database', 'Methodology & CSV template'])
with tabs[0]:
    left, right = st.columns([1.15,1])
    with left:
        st.subheader('Sector representation')
        if len(final_stage):
            sector_counts = final_stage['Sector'].value_counts().rename_axis('Sector').reset_index(name='Companies')
            fig = px.bar(sector_counts, x='Sector', y='Companies', color='Sector', title='Final survivors by sector')
            fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Companies')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('No companies remain. Lower one or more elimination percentages.')
    with right:
        st.subheader('Screen funnel')
        funnel = pd.DataFrame({'Stage':['Starting universe','After Stage 1','After Stage 2','Final survivors'],'Companies':[n0,n1,n2,n3]})
        fig2 = px.funnel(funnel, x='Companies', y='Stage', title='Companies remaining after each stage')
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader('Best-ranked survivors')
    if len(final_stage):
        cols = ['Final_Rank','Company','Ticker','Sector','Industry','Stage1_Rank','Stage2_Rank','Stage3_Total','Portfolio_Weight']
        show = final_stage[cols].copy()
        show['Portfolio_Weight'] = (show['Portfolio_Weight']*100).round(2).astype(str)+'%'
        st.dataframe(show.head(20), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader('Final portfolio')
    st.caption('Portfolio weights are illustrative allocations based on the selected weighting method, not investment advice.')
    if len(final_stage):
        sector_summary = final_stage.groupby('Sector', as_index=False).agg(Companies=('Ticker','count'), Portfolio_Weight=('Portfolio_Weight','sum'))
        sector_summary['Portfolio_Weight'] = (sector_summary['Portfolio_Weight']*100).round(2)
        a,b = st.columns(2)
        a.metric('Portfolio holdings', len(final_stage))
        b.metric('Total portfolio weight', f"{final_stage['Portfolio_Weight'].sum()*100:.1f}%")
        fig = px.pie(sector_summary, names='Sector', values='Portfolio_Weight', title='Portfolio allocation by sector')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(final_stage[['Final_Rank','Company','Ticker','Sector','Industry','STA','SNOA','Z_Score','MS','MG','MM','GPA','BTM','CFOA','Stage1_Rank','Stage2_Rank','Stage3_Total','Portfolio_Weight']], use_container_width=True, hide_index=True)
        st.download_button('Download final portfolio CSV', final_stage.to_csv(index=False).encode('utf-8'), 'final_portfolio.csv', 'text/csv')
    else:
        st.warning('No survivors with the current settings.')

with tabs[2]:
    st.subheader('Stage-by-stage screening')
    st.markdown('**Stage 1 — Financial safety.** STA and SNOA are ranked low-to-high after optional industry demeaning; Z-score is ranked high-to-low. The weakest cumulative ranks are removed.')
    st.markdown('**Stage 2 — Margin and growth.** MS and MG are converted to percentile ranks; MM is the maximum of the two percentiles. Stage 1 rank and MM rank are combined, then the weakest cumulative ranks are removed.')
    st.markdown('**Stage 3 — Profitability and valuation.** GPA, BTM and CFOA are ranked high-to-low and combined with the Stage 2 rank. The weakest cumulative ranks are removed.')
    disp = audit['Disposition'].value_counts().rename_axis('Disposition').reset_index(name='Companies')
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.subheader('Elimination audit')
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.download_button('Download full elimination audit CSV', audit.to_csv(index=False).encode('utf-8'), 'screening_audit.csv', 'text/csv')

with tabs[3]:
    st.subheader('Company database')
    f1,f2 = st.columns(2)
    sectors = sorted(original['Sector'].dropna().unique().tolist())
    selected_sector = f1.selectbox('Filter by sector', ['All sectors'] + sectors)
    search = f2.text_input('Search company or ticker')
    view = original.copy()
    if selected_sector != 'All sectors': view = view[view['Sector'] == selected_sector]
    if search: view = view[view['Company'].str.contains(search, case=False, na=False) | view['Ticker'].str.contains(search, case=False, na=False)]
    st.write(f'{len(view):,} companies shown')
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button('Download current company dataset CSV', original.to_csv(index=False).encode('utf-8'), 'company_universe.csv', 'text/csv')

with tabs[4]:
    st.subheader('Methodology and assumptions')
    st.markdown('''
**Ranking convention:** rank 1 is best. For low-is-good metrics, the smallest value receives rank 1. For high-is-good metrics, the largest value receives rank 1. The company with the highest cumulative rank number is weakest and is removed first.

**Industry demeaning:** `adjusted metric = company metric − industry mean`. This is currently applied to STA and SNOA only, following your stated design.

**Seven-year window:** the app uses eight annual margin observations (T−7 through T) to calculate seven year-over-year growth rates.

**MS (Margin of Safety):** arithmetic mean of the seven annual margin growth rates divided by their sample standard deviation. If the standard deviation is zero or the observations are insufficient, MS is missing.

**MG (Margin of Growth):** annualized geometric growth rate across the seven-year span: `(ending margin / starting margin)^(1/7) − 1`. This requires positive starting and ending margins.

**Input columns:** `Margin_Tminus7`, `Margin_Tminus6`, `Margin_Tminus5`, `Margin_Tminus4`, `Margin_Tminus3`, `Margin_Tminus2`, `Margin_Tminus1`, and `Margin_T`. If these columns are present, the app calculates MS and MG itself. If they are absent, it accepts pre-calculated `MS` and `MG` columns for backward compatibility.

**MM:** `max(percentile(MS), percentile(MG))`. Percentiles are calculated within the current surviving population, with larger percentiles treated as better.

**Stage totals:** Stage 2 adds the refreshed Stage 1 rank to the MM rank. Stage 3 adds the Stage 2 rank to the GPA, BTM and CFOA ranks. Ranks are recalculated after each elimination for the next stage.

**BTM assumption:** this version treats higher book-to-market as better. If your course methodology treats BTM differently, change the ranking direction before using the results.

**Missing values:** missing metric values are placed at the bottom of that metric's ranking. For serious analysis, ensure data coverage is adequate and consistent across companies.
''')
    st.markdown('**Required CSV columns**')
    template = pd.DataFrame(columns=['Company','Ticker','Sector','Industry','STA','SNOA','Z_Score','Margin_Tminus7','Margin_Tminus6','Margin_Tminus5','Margin_Tminus4','Margin_Tminus3','Margin_Tminus2','Margin_Tminus1','Margin_T','GPA','BTM','CFOA'])
    st.dataframe(template, use_container_width=True, hide_index=True)
    st.download_button('Download blank CSV template', template.to_csv(index=False).encode('utf-8'), 'company_data_template.csv', 'text/csv')
    st.info('The included 700-row universe and its eight annual margin observations are synthetic demonstration data. They do not represent actual financial statements, company fundamentals, or investment recommendations. The seven-year formulas are applied automatically when the annual margin columns are present.')

st.divider()
st.caption('Educational research tool. Not investment advice. Verify financial data, formula definitions, fiscal periods and sector classifications before relying on results.')
