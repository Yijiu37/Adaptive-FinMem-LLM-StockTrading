import pandas as pd
import numpy as np
import os

def combine_actions(output_path='aapl_gpt4.csv'):
    # Load the three improved versions
    df_cfr = pd.read_csv('../../baselines/aapl_finmem_only_CFR.csv')
    df_etr = pd.read_csv('../../baselines/aapl_finmem_only_ETR.csv')
    df_rgr = pd.read_csv('../../baselines/aapl_finmem_only_RGR.csv')

    # Ensure they have the same dates
    assert (df_cfr['date'] == df_etr['date']).all(), "Dates do not match between CFR and ETR"
    assert (df_cfr['date'] == df_rgr['date']).all(), "Dates do not match between CFR and RGR"

    # Create the combined dataframe
    combined = df_cfr.copy()

    # Sum the directions and take the sign for majority vote (since directions are -1, 0, or 1 usually)
    sums = df_cfr['direction'] + df_etr['direction'] + df_rgr['direction']
    combined['direction'] = np.sign(sums).astype(int)

    # Save the combined actions
    combined.to_csv(output_path, index=False)
    print(f'Saved combined action baseline to: {output_path}')

if __name__ == '__main__':
    combine_actions()

