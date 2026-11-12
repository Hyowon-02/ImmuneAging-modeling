"""Run one curated workflow from its expected directory; never runs all stages implicitly."""
from pathlib import Path
import argparse, os, runpy, subprocess
ROOT = Path(__file__).resolve().parent
STAGES = {
 'prepare-aida': 'workflows/1.preprocessing/prepare_aida.py',
 'pseudobulk': 'workflows/2.DEG_analysis/01_pseudobulk.py',
 'select-aags': 'workflows/2.DEG_analysis/02_select_aags.R',
 'split': 'workflows/4.prediction_model/01_split_donors.py',
 'size-sensitivity': 'workflows/4.prediction_model/02_size_sensitivity.py',
 'pseudocells': 'workflows/4.prediction_model/03_make_pseudocells.py',
 'train': 'workflows/4.prediction_model/04_train_models.py',
 'resampling': 'workflows/4.prediction_model/05_resampling.py',
 'composition-baseline': 'workflows/4.prediction_model/06_composition_baseline.py',
 'external-pseudocells': 'workflows/6.healthy_validation/Korean_COVID_validation/01_make_pseudocells.py',
 'predict': 'workflows/6.healthy_validation/Korean_COVID_validation/02_predict_our_model.py',
 'predict-li': 'workflows/6.healthy_validation/Korean_COVID_validation/03_predict_li_model.py',
 'compare': 'workflows/6.healthy_validation/Korean_COVID_validation/04_compare_absolute_error.py',
}
if __name__ == '__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage', choices=STAGES)
 args=p.parse_args()
 target=ROOT/STAGES[args.stage]
 os.chdir(target.parent)
 if target.suffix=='.R': subprocess.run(['Rscript',target.name],check=True)
 else: runpy.run_path(str(target),run_name='__main__')
