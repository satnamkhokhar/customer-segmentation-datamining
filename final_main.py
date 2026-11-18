"""Run final clustering comparison and novel QSCS experiments on UCI Online Retail."""
import argparse, json
from pathlib import Path
import pandas as pd
from src.data_loader import download_dataset,load_data
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm
from src.final_research import run_final

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--download',action='store_true')
    p.add_argument('--dataset',default='data/raw/Online Retail.xlsx')
    p.add_argument('--out',default='results/final')
    p.add_argument('--repeats',type=int,default=12)
    args=p.parse_args()
    dataset=Path(args.dataset)
    if args.download:download_dataset(dataset)
    raw=pd.read_csv('data/raw/online_retail_cached.csv',low_memory=False) if dataset.name=='Online Retail.xlsx' and Path('data/raw/online_retail_cached.csv').exists() else load_data(dataset)
    cleaned,ledger=clean_transactions(raw)
    rfm,_=calculate_rfm(cleaned)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ledger.to_csv(out/'cleaning_summary.csv',index=False)
    rfm[ ['Recency','Frequency','Monetary'] ].describe().to_csv(out/'rfm_statistics.csv')
    results=run_final(rfm,out,repeats=args.repeats)
    print('Selected:',results['selected'])
    print(results['scores'].to_string(index=False))
    print('Results:',out)
if __name__=='__main__':main()
