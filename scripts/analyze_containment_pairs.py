from pathlib import Path
import re
import pandas as pd
import numpy as np

DATA_DIR = Path('containment_outputs_sem/csv')
PATTERN = 'containment_pairs_variance_seps_sem_*.csv'

NUMERIC_COLS = [
    'n_nodes','edge_probability','k_roots','seed','pair_index',
    'num_separators_for_xy','num_hasse_edges_for_xy',
    'outer_sep_len','outer_component_len','outer_variance',
    'inner_sep_len','inner_component_len','inner_variance',
    'diff_variance_outer_minus_inner','abs_diff_variance',
    'diff_sep_len_outer_minus_inner','diff_component_len_inner_minus_outer',
    'variance_ratio_outer_div_inner'
]


def load_all(data_dir=DATA_DIR, pattern=PATTERN):
    frames = []
    for path in list(Path(data_dir).glob(pattern)):#sorted(Path(data_dir).glob(pattern)):
        df = pd.read_csv(path)
        # remove repeated header rows that appear inside files
        df = df[df['source_file'].astype(str) != 'source_file'].copy()
        df['input_file'] = path.name
        # extract n,p from filename as a validation/reference
        m = re.search(r'_sem_(\d+)_([0-9.]+)\.csv$', path.name)
        if m:
            df['n_from_filename'] = int(m.group(1))
            df['p_from_filename'] = float(m.group(2))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce')
    data['variance_positive_bool'] = data['diff_variance_outer_minus_inner'] > 0
    data['component_shrinks_bool'] = data['diff_component_len_inner_minus_outer'] < 0
    data['component_equal_bool'] = data['diff_component_len_inner_minus_outer'] == 0
    data['sep_outer_larger_bool'] = data['diff_sep_len_outer_minus_inner'] > 0
    data['run_id'] = (data['n_nodes'].astype(int).astype(str) + '_' +
                      data['edge_probability'].astype(str) + '_' +
                      data['seed'].astype(int).astype(str) + '_' +
                      data['X'].astype(str) + '_' + data['Y'].astype(str))
    return data


def summarize(data):
    rows = len(data)
    print(f'rows={rows:,}')
    print(f'experiments={data.groupby(["n_nodes","edge_probability"]).ngroups}')
    print(f'runs=(n,p,seed,X,Y)={data["run_id"].nunique():,}')
    print('\nGlobal checks:')
    print('  diff variance > 0:', int((data.diff_variance_outer_minus_inner > 0).sum()), '/', rows)
    print('  diff variance = 0:', int((data.diff_variance_outer_minus_inner == 0).sum()))
    print('  diff variance < 0:', int((data.diff_variance_outer_minus_inner < 0).sum()))
    print('  inner component <= outer component:', int((data.diff_component_len_inner_minus_outer <= 0).sum()), '/', rows)
    print('  inner component equal outer component:', int((data.diff_component_len_inner_minus_outer == 0).sum()))
    print('\nBy experiment:')
    g = data.groupby(['n_nodes','edge_probability']).agg(
        rows=('run_id','size'),
        runs=('run_id','nunique'),
        seeds=('seed','nunique'),
        mean_outer_component=('outer_component_len','mean'),
        mean_inner_component=('inner_component_len','mean'),
        mean_component_gap=('diff_component_len_inner_minus_outer','mean'),
        max_outer_component=('outer_component_len','max'),
        max_inner_component=('inner_component_len','max'),
        mean_outer_variance=('outer_variance','mean'),
        mean_inner_variance=('inner_variance','mean'),
        mean_diff_variance=('diff_variance_outer_minus_inner','mean'),
        median_diff_variance=('diff_variance_outer_minus_inner','median'),
        max_diff_variance=('diff_variance_outer_minus_inner','max'),
        mean_ratio=('variance_ratio_outer_div_inner','mean'),
        pct_positive_diff=('variance_positive_bool','mean'),
        pct_component_shrinks=('component_shrinks_bool','mean'),
        mean_outer_sep_len=('outer_sep_len','mean'),
        mean_inner_sep_len=('inner_sep_len','mean'),
        mean_sep_len_diff=('diff_sep_len_outer_minus_inner','mean'),
    ).reset_index()
    with pd.option_context('display.max_rows', 100, 'display.width', 220):
        print(g.to_string(index=False))
    print('\nCorrelations:')
    corr_cols = ['diff_variance_outer_minus_inner','outer_component_len','inner_component_len',
                 'diff_component_len_inner_minus_outer','outer_sep_len','inner_sep_len',
                 'diff_sep_len_outer_minus_inner','edge_probability','n_nodes']
    print(data[corr_cols].corr(numeric_only=True)['diff_variance_outer_minus_inner'].sort_values(ascending=False).to_string())
    print('\nTop 20 largest variance gaps:')
    top = data.sort_values('diff_variance_outer_minus_inner', ascending=False).head(20)
    print(top[['input_file','seed','X','Y','outer_sep_len','outer_component_len','outer_variance','inner_sep_len','inner_component_len','inner_variance','diff_variance_outer_minus_inner','variance_ratio_outer_div_inner','outer_sep','inner_sep']].to_string(index=False))
    return g

if __name__ == '__main__':

    from pathlib import Path

    data_dir = DATA_DIR

    p = Path(data_dir)

    print("exists:", p.exists())
    print("files in dir:")
    for f in p.iterdir():
        print("  ", f.name)

    print("glob result:")
    print(list(p.glob("*.csv")))

    import argparse
    parser = argparse.ArgumentParser(description='Analyze containment-pair variance CSV files.')
    parser.add_argument('data_dir', nargs='?', default='.', help='Directory containing containment_pairs_variance_seps_sem_*.csv')
    args = parser.parse_args()
    data = load_all(DATA_DIR)
    exp_summary = summarize(data)
    data.to_csv(Path(DATA_DIR) / 'containment_pairs_clean_rows.csv', index=False)
    exp_summary.to_csv(Path(DATA_DIR) / 'containment_pairs_experiment_summary.csv', index=False)
