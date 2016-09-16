from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import pandas as pd

paths = [
    'predictions/predictions_1473898618_0.690694646045.csv',
    'predictions/predictions_1473899898_0.689341091853.csv',
]

def main():
    t_id = []
    probs = []
    for path in paths:
        df = pd.read_csv(path)
        t_id = df['t_id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(paths))
    assert(len(probability) == len(t_id))

    df_pred = pd.DataFrame({
        't_id': t_id,
        'probability': probability,
    })
    csv_path = 'predictions_ensemble_{}.csv'.format(int(time.time()))
    df_pred.to_csv(csv_path, columns=('t_id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()
