import pandas as pd 
import tqdm
from textrec.automated_analysis import taps_to_type, depunct

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data")
parser.add_argument("out")
opts = parser.parse_args()

df = pd.read_csv(opts.data)

out = []
for row in tqdm.tqdm(df.itertuples(), total=len(df)):
    for action in taps_to_type(depunct(row.text)):
        action['recs_shown'] = ':'.join(action['recs_shown'])
        out.append(dict(row._asdict(), **action))

pd.DataFrame(out).to_csv(opts.out, index=False)
