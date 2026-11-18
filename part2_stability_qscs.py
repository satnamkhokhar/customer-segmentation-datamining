"""Run only Part 2: repeated stability tests and the proposed QSCS score.

Uses Part 1's saved parameter search to avoid refitting the entire tuning grid.
The real UCI dataset is required; no research values are fabricated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm
from src.final_research import features_for, select_models, stability, qscs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='data/raw/Online Retail.xlsx')
    parser.add_argument('--grid',default='results/final/parameter_search.csv')
    parser.add_argument('--out',default='results/final')
    parser.add_argument('--repeats',type=int,default=12)
    parser.add_argument('--fraction',type=float,default=.8)
    args=parser.parse_args()
    if args.repeats<2 or not 0<args.fraction<1:
        parser.error('--repeats must be >=2 and --fraction must be between 0 and 1')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    grid=pd.read_csv(args.grid)
    selected=select_models(grid)
    if not any(selected.values()):
        raise RuntimeError('No valid model selected from the saved parameter grid')
    raw=load_data(Path(args.dataset))
    cleaned,_=clean_transactions(raw)
    rfm,_=calculate_rfm(cleaned)
    x=features_for(rfm)
    trials,summary=stability(x,rfm,selected,grid,repeats=args.repeats,frac=args.fraction)
    scores,bounds=qscs(grid,selected,summary)
    trials.to_csv(out/'stability_trials.csv',index=False)
    summary.to_csv(out/'stability_summary.csv',index=False)
    scores.to_csv(out/'qscs_comparison.csv',index=False)
    manifest={'source_dataset':str(args.dataset),'parameter_grid':str(args.grid),
              'selected_models':selected,'customer_count':len(rfm),
              'repeats':args.repeats,'sample_fraction':args.fraction,
              'seed':42,'weights':{'quality':.45,'stability':.35,'coverage':.20},
              'normalization_bounds':bounds,
              'caveats':['ARI compares the same sampled customer IDs against full-data labels',
                         'Noise-excluded ARI is conditional on shared assigned customers',
                         'QSCS is a proposed project metric, not an externally validated measure',
                         'Model selection is based on the previously saved parameter grid']}
    (out/'part2_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Selected models:',selected)
    print('Stability summary:\n',summary.to_string(index=False))
    print('QSCS:\n',scores.to_string(index=False))
    print('Outputs:',out.resolve())

if __name__=='__main__':main()
