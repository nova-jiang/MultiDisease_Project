import os
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

STEP3_DIR = os.path.join('results', 'step3_models')
OUTPUT_DIR = os.path.join('results', 'final_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_metrics(json_path):
    """Load overall metrics from a nested CV results JSON file."""
    with open(json_path, 'r') as f:
        text = f.read()
    # try direct JSON parsing
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # remove merge conflict markers if present
        clean_lines = [line for line in text.splitlines()
                       if not line.startswith('<<<<<<<')
                       and not line.startswith('=======')
                       and not line.startswith('>>>>>>>')]
        clean_text = '\n'.join(clean_lines)
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            # final fallback: regex extraction
            match = re.search(r'"overall_metrics"\s*:\s*\{([^}]*)\}', text)
            if not match:
                return None
            metrics = json.loads('{' + match.group(1) + '}')
            return metrics
    if 'overall_metrics' in data:
        return data['overall_metrics']
    if 'nested_cv_results' in data and 'overall_metrics' in data['nested_cv_results']:
        return data['nested_cv_results']['overall_metrics']
    return None


def gather_results():
    records = []
    for sub in sorted(os.listdir(STEP3_DIR)):
        subdir = os.path.join(STEP3_DIR, sub)
        if not os.path.isdir(subdir):
            continue
        if not sub.endswith('_nested'):
            continue
        # identify JSON file
        json_files = [f for f in os.listdir(subdir) if f.endswith('_cv_results.json')]
        if not json_files:
            continue
        json_path = os.path.join(subdir, json_files[0])
        metrics = load_metrics(json_path)
        if metrics is None:
            continue
        group = 'mrmr' if sub.startswith('mrmr_') else 'xgb_rfecv' if sub.startswith('xgb_rfecv_') else 'baseline'
        record = {'model': sub, 'group': group}
        record.update(metrics)
        records.append(record)
    return pd.DataFrame(records)


def save_tables(df):
    mrmr_df = df[df['group'] == 'mrmr']
    xgb_df = df[df['group'] == 'xgb_rfecv']
    mrmr_df.to_csv(os.path.join(OUTPUT_DIR, 'mrmr_models_summary.csv'), index=False)
    xgb_df.to_csv(os.path.join(OUTPUT_DIR, 'xgb_rfecv_models_summary.csv'), index=False)


def create_plots(df):
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    if 'auc_macro' in df.columns:
        metrics.append('auc_macro')
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, hue='group', data=df, dodge=False)
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f'{metric}_barplot.png')
        plt.savefig(out_path, dpi=300)
        plt.close()


def main():
    df = gather_results()
    if df.empty:
        print('No results found.')
        return
    save_tables(df)
    create_plots(df)
    print(f'Results saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
