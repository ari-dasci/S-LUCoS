import os
from pathlib import Path

project_folder = Path(__file__).parents[3]
results_folder = project_folder / 'results'

log_folder = results_folder / 'logs'
datasets_folder = project_folder / 'datasets'
ckpt_folder = project_folder / 'checkpoints'
tabpfn_ckpt_folder = ckpt_folder / 'tabpfn_checkpoints'

undersampling_ckpt_folder = ckpt_folder / 'Undersampling'
ssma_results_folder = undersampling_ckpt_folder / 'OriginalSpace' / 'SSMAUnderSampler'


def change_permissions_rw(path):
    try:
        os.chmod(str(path), 0o777)
    except PermissionError:
        pass
