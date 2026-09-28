import pandas as pd
import numpy as np

def combine():
    df1 = pd.read_csv('../../baselines/aapl_finmem_only_CFR.csv')
    df2 = pd.read_csv('../../baselines/aapl_finmem_only_ETR.csv')
    df3 = pd.read_csv('../../baselines/aapl_finmem_only_RGR.csv')

    combined = df1.copy()
    sums = df1['direction'] + df2['direction'] + df3['direction']
    combined['direction'] = np.sign(sums).astype(int)

    combined.to_csv('baselines/aapl_finmem_combined.csv', index=False)
    print("Files combined successfully.")

if __name__ == '__main__':
    combine()

