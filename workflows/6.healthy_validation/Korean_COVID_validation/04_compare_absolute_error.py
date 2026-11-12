"""Run in Korean_validation. One row per healthy donor, paired model comparison.
Two-sided within-donor prediction-swap permutation: exchangeability null.
Paired donor percentile-bootstrap CIs are pointwise, not simultaneous.
Holm adjustment jointly covers MAE across 3 cell types (3 tests).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CELL_TYPES = ['NK', 'CD4T', 'CD8T']
OUR_DIR = Path('validation_results')
LI_DIR = Path('Li_et_al_covid_validation_result')
OUTPUT_DIR = Path('figures/model_comparison_absolute_error')
STATE = 'Normal'
EXPECTED_N = 57
N_BOOT = 10000
N_PERM = 20000
SEED = 251219
METRICS = ['MAE']

def metric_values(y, pred):
    """MAE only; last axis is donors."""
    return np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)), axis=-1)[..., None]

def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('All three MAE P values must be finite before Holm correction.')
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result

def load_pairs(ct):
    files = [OUR_DIR / f'{ct}_ElasticNet_validation_predictions_donor_median.csv',
             LI_DIR / f'{ct}_Li_et_al_validation_predictions_donor_median.csv']
    frames = []
    for path in files:
        df = pd.read_csv(path, dtype={'donor': str}, keep_default_na=False)
        required = ['donor', 'state', 'true_age', 'pred_age']
        if not set(required).issubset(df.columns):
            raise ValueError(f'{path}: required columns {required}')
        df = df.loc[df['state'].astype(str).str.strip().eq(STATE), required].copy()
        df['donor'] = df['donor'].str.strip()
        if df.empty or df['donor'].eq('').any() or df['donor'].duplicated().any():
            raise ValueError(f'{path}: empty controls or duplicate/blank donor IDs.')
        for c in ['true_age', 'pred_age']:
            df[c] = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(df[['true_age', 'pred_age']].to_numpy()).all():
            raise ValueError(f'{path}: nonfinite ages/predictions.')
        frames.append(df)
    a, b = frames
    membership = a[['donor']].merge(b[['donor']], on='donor', how='outer', indicator=True)
    membership.to_csv(OUTPUT_DIR / f'{ct}_donor_membership.csv', index=False)
    pairs = a.merge(b, on='donor', suffixes=('_our', '_li'), validate='one_to_one')
    if not np.allclose(pairs.true_age_our, pairs.true_age_li, atol=1e-6, rtol=0):
        raise ValueError(f'{ct}: chronological ages disagree between model files.')
    if len(pairs) < 4:
        raise ValueError(f'{ct}: too few paired donors.')
    print(f'{ct}: Our={len(a)}, Li={len(b)}, paired unique donors={len(pairs)}; expected {EXPECTED_N}')
    if len(pairs) != EXPECTED_N:
        print('  Check donor_membership.csv before reporting the manuscript N.')
    pairs.to_csv(OUTPUT_DIR / f'{ct}_paired_donors.csv', index=False)
    return pairs

def compare_arrays(y, our, li, rng):
    observed_a = metric_values(y, our)
    observed_b = metric_values(y, li)
    observed_diff = observed_a - observed_b
    if not np.isfinite(observed_diff).all():
        raise ValueError('Nonfinite MAE: inspect input values.')
    boot_a, boot_b = [], []
    null_diff = []
    n = len(y)
    for start in range(0, N_BOOT, 500):
        idx = rng.integers(0, n, size=(min(500, N_BOOT-start), n))
        boot_a.append(metric_values(y[idx], our[idx]))
        boot_b.append(metric_values(y[idx], li[idx]))
    for start in range(0, N_PERM, 500):
        swaps = rng.integers(0, 2, size=(min(500, N_PERM-start), n)).astype(bool)
        pa = np.where(swaps, li, our)
        pb = np.where(swaps, our, li)
        null_diff.append(metric_values(y, pa) - metric_values(y, pb))
    boot_a, boot_b = np.concatenate(boot_a), np.concatenate(boot_b)
    null_diff = np.concatenate(null_diff)
    rows = []
    for j, metric in enumerate(METRICS):
        valid = np.isfinite(boot_a[:, j]) & np.isfinite(boot_b[:, j])
        if valid.mean() < .95:
            raise ValueError(f'{metric}: too many undefined bootstrap draws.')
        # Stop rather than silently changing the permutation reference distribution.
        if not np.isfinite(null_diff[:, j]).all():
            raise ValueError(f'{metric}: undefined permutation statistic; inspect input variation.')
        ci_a = np.quantile(boot_a[valid, j], [.025, .975])
        ci_b = np.quantile(boot_b[valid, j], [.025, .975])
        ci_d = np.quantile((boot_a-boot_b)[valid, j], [.025, .975])
        p = (1 + np.count_nonzero(np.abs(null_diff[:, j]) >= abs(observed_diff[j])-1e-12)) / (N_PERM+1)
        rows.append(dict(Metric=metric, N_donors=n, Our=observed_a[j], Li=observed_b[j],
            Our_CI_low=ci_a[0], Our_CI_high=ci_a[1], Li_CI_low=ci_b[0], Li_CI_high=ci_b[1],
            Difference_Our_minus_Li=observed_diff[j], Difference_CI_low=ci_d[0],
            Difference_CI_high=ci_d[1], P_permutation=p, Valid_bootstrap_draws=int(valid.sum())))
    return rows

def main():
    from matplotlib.patches import Patch
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, paired = [], {}
    for ct in CELL_TYPES:
        df = load_pairs(ct)
        paired[ct] = df
        y = df.true_age_our.to_numpy()
        our, li = df.pred_age_our.to_numpy(), df.pred_age_li.to_numpy()
        for row in compare_arrays(y, our, li, rng):
            row.update(Cell_Type=ct, State=STATE)
            rows.append(row)
    result = pd.DataFrame(rows)
    # One MAE comparison per cell type: three tests in this family.
    if len(result) != 3 or set(result.Metric) != {'MAE'}:
        raise ValueError('Expected exactly three MAE comparisons.')
    result['P_Holm_3tests'] = holm_adjust(result.P_permutation)
    mae = result.loc[result.Metric.eq('MAE')].set_index('Cell_Type').loc[CELL_TYPES]
    mae.to_csv(OUTPUT_DIR / 'absolute_error_MAE_statistics.csv')

    colors = {'NK':'#B55475', 'CD4T':'#7D9263', 'CD8T':'#4E75A3'}
    edge, li_color = '#4A4A4A', '#D9D9D9'
    plt.rcParams.update({'font.family':'Arial','font.weight':'normal',
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    tops=[]
    for i,ct in enumerate(CELL_TYPES):
        df=paired[ct]
        errors=[np.abs(df.pred_age_our-df.true_age_our).to_numpy(),
                np.abs(df.pred_age_li-df.true_age_li).to_numpy()]
        top=0
        for values,offset,color in zip(errors,[-.18,.18],[colors[ct],li_color]):
            bp=ax.boxplot(values,positions=[i+offset],widths=.28,patch_artist=True,
                showfliers=False,whis=1.5,
                boxprops=dict(facecolor=color,edgecolor=edge,alpha=.85),
                medianprops=dict(color=edge,linewidth=1.2),
                whiskerprops=dict(color=edge,linewidth=1),
                capprops=dict(color=edge,linewidth=1))
            top=max(top,max(float(np.max(w.get_ydata())) for w in bp['whiskers']))
        tops.append(top)
    scale=max(max(tops),1)
    annotation_tops=[]
    for i,ct in enumerate(CELL_TYPES):
        p=float(mae.loc[ct,'P_Holm_3tests'])
        stars=('****' if p<.0001 else '***' if p<.001 else
               '**' if p<.01 else '*' if p<.05 else 'ns')
        y=tops[i]+scale*.08
        h=scale*.035
        ax.plot([i-.18,i-.18,i+.18,i+.18],[y,y+h,y+h,y],color='black',lw=1)
        ax.text(i,y+h+scale*.02,stars,
                ha='center',va='bottom',fontsize=13)
        annotation_tops.append(y+h+scale*.23)
    ax.set_xticks(range(3),CELL_TYPES,fontsize=13)
    ax.set_ylabel('Absolute error (years)',fontsize=12)
    ax.set_ylim(-1,max(annotation_tops))
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both',labelsize=11)
    ax.grid(False)
    ax.legend(handles=[Patch(facecolor=colors['NK'],edgecolor=edge,label='Our model'),
                       Patch(facecolor=li_color,edgecolor=edge,label='Li_et_al.')],
              title='Model',loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,
              fontsize=8,title_fontsize=9,borderaxespad=0)
    fig.tight_layout()
    for extension in ['pdf','svg','png']:
        fig.savefig(OUTPUT_DIR/f'Absolute_error_Normal_paired.{extension}',
                    bbox_inches='tight',dpi=300)
    plt.show()
    plt.close(fig)
    print(mae[['N_donors','Our','Li','Difference_Our_minus_Li',
               'Difference_CI_low','Difference_CI_high','P_Holm_3tests']].to_string())
    legend = (
        'Donor-level absolute prediction errors in healthy controls with predictions '
        'from both models. Boxes show medians and interquartile ranges; whiskers '
        'extend to the most extreme observations within 1.5 times the interquartile '
        'range. Outlier symbols are omitted for display, but all paired donors are '
        'included in statistical analyses. Brackets report two-sided paired '
        'prediction-swap permutation tests of the difference in mean absolute error '
        '(20,000 permutations), with Holm adjustment across three MAE comparisons '
        '(NK, CD4T, and CD8T). The permutation null assumes '
        'within-donor exchangeability of the model predictions. Pointwise 95% '
        'confidence intervals for MAE differences are estimated using 10,000 paired '
        'donor bootstrap resamples and reported in the statistics table. '
        '*P_adj < 0.05; **P_adj < 0.01; ***P_adj < 0.001; ****P_adj < 0.0001; '
        'ns, not significant. Sample sizes: '
        + ', '.join(f'{ct}, n={len(paired[ct])}' for ct in CELL_TYPES) + '.'
    )
    (OUTPUT_DIR/'figure_legend.txt').write_text(legend,encoding='utf-8')
    return result

if __name__ == '__main__':
    comparison_results = main()
