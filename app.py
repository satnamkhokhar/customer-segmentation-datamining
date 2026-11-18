"""CSC 4740 Streamlit dashboard: run `streamlit run app.py` from Sources/."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from dashboard_logic import ROOT,RESULTS,read_saved,run_live,weighted_scores
from src.data_loader import load_data
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm

st.set_page_config(page_title='Customer Segmentation | CSC 4740',page_icon='📊',layout='wide')
st.title('Customer Segmentation Research Lab')
st.caption('CSC 4740 · K-Means vs DBSCAN vs Hierarchical · UCI Online Retail · Shabib Hyder Gazi & Satnam Singh Khokhar')

@st.cache_data(show_spinner='Loading and preparing UCI Online Retail transactions…')
def dataset():
    path=ROOT/'data'/'raw'/'Online Retail.xlsx'
    raw=load_data(path)
    clean,ledger=clean_transactions(raw)
    rfm,reference=calculate_rfm(clean)
    return len(raw),len(clean),rfm,ledger,str(reference.date())

@st.cache_data(show_spinner=False)
def saved(name):return read_saved(name)

pages=['Overview & RFM','Live clustering','Algorithm comparison','Stability & QSCS','Preprocessing sensitivity','Code & methodology']
page=st.sidebar.radio('Explore the research',pages)
st.sidebar.caption('Live clustering fits models on real RFM customer records. Other pages show saved full-dataset experiment outputs.')

if page=='Overview & RFM':
    st.header('Dataset and feature engineering')
    try:
        raw_n,clean_n,rfm,ledger,reference=dataset()
    except Exception as exc:
        st.error(f'Unable to load the workbook: {exc}');st.stop()
    a,b,c,d=st.columns(4)
    a.metric('Original transactions',f'{raw_n:,}')
    b.metric('Cleaned purchases',f'{clean_n:,}')
    c.metric('Unique customers',f'{len(rfm):,}')
    d.metric('RFM reference date',reference)
    st.subheader('Sequential cleaning ledger')
    st.dataframe(ledger,use_container_width=True,hide_index=True)
    st.subheader('RFM = Recency · Frequency · Monetary')
    st.write('Recency: days since last purchase; Frequency: distinct invoices; Monetary: sum of positive purchase values. Cancellations are excluded, so monetary is not net revenue.')
    st.dataframe(rfm.head(20),use_container_width=True,hide_index=True)
    metric=st.selectbox('Explore RFM distribution',['Recency','Frequency','Monetary'])
    st.plotly_chart(px.histogram(rfm,x=metric,nbins=55,title=f'{metric} across customers'),use_container_width=True)

elif page=='Live clustering':
    st.header('Run a clustering algorithm on real customers')
    st.write('Choose a model and parameters, then click **Fit model**. This performs a new model fit rather than reading saved cluster labels.')
    try:_,_,rfm,_,_=dataset()
    except Exception as exc:st.error(str(exc));st.stop()
    alg=st.selectbox('Algorithm',['kmeans','dbscan','hierarchical'],format_func=lambda x:{'kmeans':'K-Means','dbscan':'DBSCAN','hierarchical':'Hierarchical clustering'}[x])
    variant=st.selectbox('Feature preprocessing',['baseline','all_log','raw','robust','winsor'],help='Baseline: raw Recency, log1p(Frequency and Monetary), StandardScaler.')
    limit=st.slider('Number of customers for live fit',min_value=100,max_value=len(rfm),value=min(1200,len(rfm)),step=50,help='Use 1,200 for a responsive demonstration; choose all 4,338 to fit the full customer dataset.')
    subset=rfm.sample(n=limit,random_state=42).sort_values('CustomerID').reset_index(drop=True)
    params={}
    if alg!='dbscan':
        params['k']=st.slider('Number of clusters (k)',2,10,3)
    else:
        params['eps']=st.slider('DBSCAN eps',.05,3.0,.60,.05)
        params['min_samples']=st.slider('Minimum samples',2,20,4)
    if alg=='hierarchical':params['linkage']=st.selectbox('Linkage',['ward','complete','average','single'])
    if st.button('Fit model',type='primary'):
        with st.spinner('Fitting clustering model and calculating metrics…'):
            try:st.session_state['live_result']=(*run_live(subset,alg,variant,**params),alg,variant,limit)
            except Exception as exc:st.error(f'Clustering failed: {exc}')
    result=st.session_state.get('live_result')
    if result:
        quality,seconds,profiles,points,old_alg,old_variant,old_n=result
        if (old_alg,old_variant,old_n)!=(alg,variant,limit):st.info('Showing the previous fit. Click Fit model to apply the current settings.')
        a,b,c,d=st.columns(4)
        a.metric('Clusters',int(quality['clusters']))
        b.metric('Coverage',f"{quality['coverage']:.1%}")
        c.metric('Silhouette',f"{quality['silhouette']:.3f}" if pd.notna(quality['silhouette']) else 'Undefined')
        d.metric('Fit time',f'{seconds:.3f} s')
        st.caption('Silhouette is sampled for speed and excludes DBSCAN noise; cluster metrics are undefined for fewer than two usable clusters.')
        st.plotly_chart(px.scatter(points,x='PC1',y='PC2',color='Cluster',hover_data=['CustomerID'],title='PCA projection (visualization only)'),use_container_width=True)
        st.subheader('Segment profiles (original RFM units)')
        st.dataframe(profiles,use_container_width=True,hide_index=True)
        st.download_button('Download live segment profiles',profiles.to_csv(index=False),'live_profiles.csv','text/csv')

elif page=='Algorithm comparison':
    st.header('Full-dataset algorithm comparison')
    df=saved('algorithm_comparison.csv')
    st.dataframe(df[['algorithm','clusters','coverage','silhouette','davies_bouldin','calinski_harabasz','fit_seconds']],use_container_width=True,hide_index=True)
    col=st.selectbox('Metric',['silhouette','davies_bouldin','calinski_harabasz','coverage','fit_seconds'])
    st.plotly_chart(px.bar(df,x='algorithm',y=col,color='algorithm',title=f'Selected models — {col}'),use_container_width=True)
    st.caption('Metrics are not interchangeable: silhouette/CH higher is better, DB lower is better; runtime depends on hardware. DBSCAN quality excludes noise.')
    st.subheader('Model selection search')
    grid=saved('parameter_search.csv'); chosen=st.selectbox('Inspect algorithm grid',sorted(grid.algorithm.unique()))
    st.dataframe(grid[grid.algorithm==chosen],use_container_width=True,hide_index=True)
    st.subheader('Saved segment profiles')
    for alg in ['kmeans','dbscan','hierarchical']:
        with st.expander(alg.title()):st.dataframe(saved(f'{alg}_profiles.csv'),hide_index=True,use_container_width=True)

elif page=='Stability & QSCS':
    st.header('Stability and our project-specific QSCS score')
    st.write('Repeated 80% customer subsampling; ARI compares customer-aligned assignments. DBSCAN noise is excluded from the assigned-only ARI, so inspect coverage and noise agreement as well.')
    stab=saved('stability_summary.csv')
    st.dataframe(stab,use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(stab,x='algorithm',y='ari_mean',color='algorithm',title='Mean ARI from 12 stability trials'),use_container_width=True)
    st.subheader('Quality–Stability–Coverage Score')
    st.latex(r'QSCS = w_Q Q + w_S S + w_C C,\quad w_Q+w_S+w_C=1')
    st.write('Q averages normalized silhouette, inverse Davies–Bouldin, and Calinski–Harabasz; S rescales ARI from [-1,1] to [0,1]; C is assigned-customer coverage. Normalization depends on the candidate model pool.')
    quality=st.slider('Quality weight',0.0,1.0,.45,.05)
    stability=st.slider('Stability weight',0.0,1.0,.35,.05)
    coverage=round(1-quality-stability,4)
    st.caption(f'Coverage weight (remainder): {coverage:.2f}')
    if coverage<0:st.warning('Quality + stability must be ≤ 1. Reduce a weight to see the ranking.')
    else:
        score=weighted_scores(saved('qscs_comparison.csv'),quality,stability,coverage)
        st.dataframe(score,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(score,x='algorithm',y='custom_score',color='algorithm',title='QSCS ranking under chosen weights'),use_container_width=True)
    st.caption('The score is a project-specific proposal, not an externally validated universal measure of customer value.')

elif page=='Preprocessing sensitivity':
    st.header('How preprocessing changes customer segments')
    df=saved('feature_sensitivity.csv')
    st.dataframe(df[['algorithm','variant','clusters','coverage','silhouette','ari_vs_baseline','fit_seconds']],use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(df,x='variant',y='ari_vs_baseline',color='algorithm',barmode='group',title='Agreement with baseline assignments (ARI)'),use_container_width=True)
    alg=st.selectbox('Inspect one algorithm',sorted(df.algorithm.unique()))
    st.plotly_chart(px.line(df[df.algorithm==alg],x='variant',y='silhouette',markers=True,title=f'{alg.title()}: silhouette by preprocessing'),use_container_width=True)
    st.write('The same selected hyperparameters are reused across variants. This isolates preprocessing sensitivity; it does not measure the best achievable score after retuning.')

else:
    st.header('Implementation and reproducibility')
    st.code('python -m pytest -q\npython demo.py\nstreamlit run app.py\npython final_main.py --repeats 12',language='bash')
    st.write('The code lives in `src/`; `final_main.py` runs the full search and 12-trial stability experiment; `part3_sensitivity.py` runs the 15 preprocessing variants. The UI is an additional demonstration layer.')
    st.write('Key limitations: unsupervised labels are not ground truth; DBSCAN can label customers as noise; QSCS weights are subjective; silhouette can favor compact clusters over business interpretability; the reference date is fixed by the dataset, not today.')
    st.info('For the video, run a live clustering fit, show saved full-data experiments, and demonstrate pytest in a terminal. Full parameter search is computationally expensive.')
