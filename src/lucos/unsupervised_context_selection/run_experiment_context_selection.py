import os
import random
import shutil
import tempfile
import openml
from pathlib import Path
from lucos.utils.log_utils import setup_logging
import torch
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from time import perf_counter
from datetime import datetime
from lucos.utils.dataset import Benchmark
from lucos.utils.paths import datasets_folder, results_folder, change_permissions_rw
from lucos.utils.config_utils import get_obj_from_str, load_yaml_config, parse_folds_arg, save_config

logger = logging.getLogger("lucos")

openml.config.set_root_cache_directory(str(datasets_folder / "openml"))
pd.options.mode.copy_on_write = True


GLOBAL_SEED = 42


def exp_id_with_fold(exp_id: str, fold: int) -> str:
    return f"{str(exp_id)}_f{int(fold)}"


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.reset_max_memory_allocated now calls torch\.cuda\.reset_peak_memory_stats.*",
    category=FutureWarning,
    module=r"torch\.cuda\.memory",
)


class Experiment:

    def __init__(self, config):
        self.config = config
        self.device = config.experiment.device
        self.exp_id = config.experiment.exp_id
        self.folds_to_run = list(config.experiment.get("folds_to_run", [0]))
        self.base_seed = int(config.experiment.get("base_seed", GLOBAL_SEED))
        self.recompute_evaluator = bool(config.experiment.get("recompute_evaluator", False))
        self.save_results_every_appends = max(1, int(config.experiment.get("save_results_every_appends", 1)))
        self._pending_result_appends = 0
        self._results_dirty = False
        self.experiment_results_folder = results_folder / f"results_LUCoS_{self.exp_id}"
        self.results_basename = ""
        self.results_filename = ""

        self.results_columns = [
            "dataset",
            "fold",
            "instance_selection_space",
            "instance_selection_algorithm",
            "n_selected_samples",
            "random_state",
            "evaluator",
            "metric",
            "value",
        ]
        self.result_key_columns = [
            "dataset",
            "fold",
            "instance_selection_space",
            "instance_selection_algorithm",
            "n_selected_samples",
            "random_state",
            "evaluator",
            "metric",
        ]

        self._set_results_file_for_exp_id(self.exp_id)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Folds to run: {self.folds_to_run}")
        logger.info(f"Base seed: {self.base_seed}")
        logger.info(f"Saving results every {self.save_results_every_appends} append(s)")

    def _set_results_file_for_exp_id(self, exp_id: str) -> None:
        if getattr(self, "_results_dirty", False) and self.results_filename:
            self._save_pending_results("before switching results file")

        self.current_exp_id = str(exp_id)
        self.results_basename = f"results_LUCoS_{self.current_exp_id}"
        self.results_filename = str(self.experiment_results_folder / "raw" / f"{self.results_basename}.csv")
        self.config.results_basename = self.results_basename
        self.config.results_filename = self.results_filename
        self.results = self._load_existing_results()
        self._existing_result_keys = self._build_existing_result_keys(self.results)

    def _set_results_file_for_fold(self, fold: int) -> None:
        self._set_results_file_for_exp_id(exp_id_with_fold(self.exp_id, fold))

    def _load_existing_results(self):
        if not os.path.exists(self.results_filename):
            backup_filename = self.results_filename + ".bak"
            if os.path.exists(backup_filename):
                try:
                    logger.warning("Primary results CSV missing. Loading backup results from %s", backup_filename)
                    df = pd.read_csv(backup_filename)
                    return self._normalize_loaded_results(df)
                except Exception as error:
                    logger.warning("Could not read backup results CSV %s (%s). Starting with empty results.", backup_filename, error)
            return pd.DataFrame(columns=self.results_columns)

        try:
            df = pd.read_csv(self.results_filename)
        except Exception as e:
            backup_filename = self.results_filename + ".bak"
            if os.path.exists(backup_filename):
                try:
                    logger.warning(
                        f"Could not read existing results CSV {self.results_filename} ({e}). "
                        f"Trying backup {backup_filename}."
                    )
                    df = pd.read_csv(backup_filename)
                    return self._normalize_loaded_results(df)
                except Exception as backup_error:
                    logger.warning(
                        f"Could not read backup results CSV {backup_filename} ({backup_error}). "
                        "Starting with empty results."
                    )
            else:
                logger.warning(
                    f"Could not read existing results CSV {self.results_filename} ({e}). "
                    "Starting with empty results."
                )
            return pd.DataFrame(columns=self.results_columns)

        return self._normalize_loaded_results(df)

    def _normalize_loaded_results(self, df: pd.DataFrame) -> pd.DataFrame:
        # Backward compatibility with the previous format
        if {"evaluation_method", "AUC"}.issubset(df.columns) and not {"evaluator", "metric", "value"}.issubset(df.columns):
            df = df.rename(columns={"evaluation_method": "evaluator", "AUC": "value"})
            df["metric"] = "AUC"

        optional_missing_fill = {
            "random_state": np.nan,
        }
        missing = [c for c in self.results_columns if c not in df.columns]
        for optional_col, default_value in optional_missing_fill.items():
            if optional_col in missing:
                df[optional_col] = default_value
                missing.remove(optional_col)

        if missing:
            raise ValueError(
                f"Existing results file {self.results_filename} is missing columns {missing}. "
                "Please remove or rename the file to run this experiment."
            )
        return df[self.results_columns]


    def _save_results(self):
        target_path = Path(self.results_filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        total_start = perf_counter()

        temp_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            prefix=f".{target_path.stem}_",
            dir=str(target_path.parent),
            delete=False,
        )
        temp_path = Path(temp_handle.name)
        temp_handle.close()

        try:
            logger.info(
                "Saving results CSV with %s rows to %s",
                len(self.results),
                target_path,
            )
            step_start = perf_counter()
            self.results.to_csv(temp_path, index=False)
            to_csv_time = perf_counter() - step_start

            step_start = perf_counter()
            os.replace(temp_path, target_path)
            replace_time = perf_counter() - step_start

            backup_path = Path(self.results_filename + ".bak")
            backup_time = None
            try:
                # copy2 also tries to preserve metadata. In shared runs across users
                # that can fail even when the backup file is writable.
                step_start = perf_counter()
                shutil.copyfile(target_path, backup_path)
                backup_time = perf_counter() - step_start
            except Exception as error:
                logger.warning("Could not update backup results file %s (%s).", backup_path, error)

            # Set permissions to: Everyone can read and write.
            # If permissions cannot be changed, keep going.
            step_start = perf_counter()
            change_permissions_rw(self.results_filename)
            change_permissions_rw(backup_path)
            permissions_time = perf_counter() - step_start

            total_time = perf_counter() - total_start
            if backup_time is None:
                logger.info(
                    "Saved results CSV to %s in %.3fs (to_csv=%.3fs, replace=%.3fs, backup=failed, chmod=%.3fs)",
                    target_path,
                    total_time,
                    to_csv_time,
                    replace_time,
                    permissions_time,
                )
            else:
                logger.info(
                    "Saved results CSV to %s in %.3fs (to_csv=%.3fs, replace=%.3fs, backup=%.3fs, chmod=%.3fs)",
                    target_path,
                    total_time,
                    to_csv_time,
                    replace_time,
                    backup_time,
                    permissions_time,
                )
            
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _save_pending_results(self, reason: str) -> None:
        if not self._results_dirty:
            return

        logger.info("Saving pending results (%s)", reason)
        self._save_results()
        self._pending_result_appends = 0
        self._results_dirty = False

    def _append_results(self, rows):
        if not rows:
            return

        new_rows = pd.DataFrame(rows).reindex(columns=self.results_columns)
        
        # Check for NaN values in new_rows
        if new_rows.isna().any().any():
            nan_info = new_rows.isna().sum()
            cols_with_nan = nan_info[nan_info > 0].to_dict()
            rows_with_any_nan = new_rows.isna().any(axis=1).sum()
            # WHEN/HOW CAN THIS HAPPEN??
            # I had this error: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation. self.results = pd.concat([self.results, new_rows], ignore_index=True)
            logger.warning(
                f"Found {rows_with_any_nan} row(s) with NaN values in new results. "
                f"Affected columns: {cols_with_nan}"
            )
        
        if new_rows.empty:
            logger.warning("No new results to append after filtering duplicates.")
            return

        duplicate_new_key_mask = new_rows.duplicated(subset=self.result_key_columns, keep=False)
        if duplicate_new_key_mask.any():
            n_duplicate_new_rows = int(duplicate_new_key_mask.sum())
            n_duplicate_new_keys = int(new_rows.loc[duplicate_new_key_mask, self.result_key_columns].drop_duplicates().shape[0])
            if self.recompute_evaluator:
                logger.warning(
                    "Recompute generated %s row(s) for %s duplicate result key(s). "
                    "Keeping the last computed metric value for each key.",
                    n_duplicate_new_rows,
                    n_duplicate_new_keys,
                )
                new_rows = new_rows.drop_duplicates(subset=self.result_key_columns, keep="last").reset_index(drop=True)
            else:
                raise ValueError(
                    f"Generated {n_duplicate_new_rows} row(s) for {n_duplicate_new_keys} duplicate result key(s) "
                    "while recompute_evaluator=False."
                )

        if self.recompute_evaluator and not self.results.empty:
            new_keys = {
                self._make_result_key(
                    dataset_name=row["dataset"],
                    fold=row["fold"],
                    instance_selection_space=row["instance_selection_space"],
                    instance_selection_algorithm=row["instance_selection_algorithm"],
                    n_selected_samples=row["n_selected_samples"],
                    random_state=row["random_state"],
                    evaluator=row["evaluator"],
                    metric=row["metric"],
                )
                for _, row in new_rows.iterrows()
            }
            existing_mask = self.results.apply(
                lambda row: self._make_result_key(
                    dataset_name=row["dataset"],
                    fold=row["fold"],
                    instance_selection_space=row["instance_selection_space"],
                    instance_selection_algorithm=row["instance_selection_algorithm"],
                    n_selected_samples=row["n_selected_samples"],
                    random_state=row["random_state"],
                    evaluator=row["evaluator"],
                    metric=row["metric"],
                ) not in new_keys,
                axis=1,
            )
            removed_rows = int((~existing_mask).sum())
            if removed_rows > 0:
                logger.info("Recomputing evaluator results: replacing %s existing row(s) in %s", removed_rows, self.results_filename)
                self.results = self.results.loc[existing_mask].reset_index(drop=True)

        if self.results.empty:
            self.results = new_rows
        else:
            self.results = pd.concat([self.results, new_rows], ignore_index=True)

        self._pending_result_appends += 1
        self._results_dirty = True

        if self._pending_result_appends >= self.save_results_every_appends:
            self._save_pending_results("save interval reached")
        else:
            logger.info(
                "Buffered results append %s/%s before next CSV save",
                self._pending_result_appends,
                self.save_results_every_appends,
            )

    @staticmethod
    def _make_result_key(
        dataset_name,
        fold,
        instance_selection_space,
        instance_selection_algorithm,
        n_selected_samples,
        random_state,
        evaluator,
        metric,
    ):
        random_state_norm = None if pd.isna(random_state) else int(random_state)
        return (
            str(dataset_name),
            int(fold),
            str(instance_selection_space),
            str(instance_selection_algorithm),
            int(n_selected_samples),
            random_state_norm,
            str(evaluator),
            str(metric),
        )

    def _metric_rows_from_evaluation(
        self,
        *,
        dataset_name,
        fold,
        instance_selection_space,
        instance_selection_algorithm,
        n_selected_samples,
        random_state,
        evaluator_name,
        metrics,
    ):
        rows = []
        for metric_name, metric_value in metrics.items():
            result_key = self._make_result_key(
                dataset_name=dataset_name,
                fold=fold,
                instance_selection_space=instance_selection_space,
                instance_selection_algorithm=instance_selection_algorithm,
                n_selected_samples=n_selected_samples,
                random_state=random_state,
                evaluator=evaluator_name,
                metric=metric_name,
            )
            if result_key in self._existing_result_keys and not self.recompute_evaluator:
                logger.info(
                    f"{dataset_name} Fold {fold}. Skipping already computed combination "
                    f"| evaluator={evaluator_name} | metric={metric_name} | space={instance_selection_space} "
                    f"| method={instance_selection_algorithm} | n_selected={n_selected_samples} | random_state={random_state}"
                )
                continue

            if result_key in self._existing_result_keys and self.recompute_evaluator:
                logger.info(
                    f"{dataset_name} Fold {fold}. Recomputing existing combination "
                    f"| evaluator={evaluator_name} | metric={metric_name} | space={instance_selection_space} "
                    f"| method={instance_selection_algorithm} | n_selected={n_selected_samples} | random_state={random_state}"
                )

            rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "instance_selection_space": instance_selection_space,
                    "instance_selection_algorithm": instance_selection_algorithm,
                    "n_selected_samples": int(n_selected_samples),
                    "random_state": None if random_state is None else int(random_state),
                    "evaluator": evaluator_name,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )
            self._existing_result_keys.add(result_key)

        return rows

    def _build_existing_result_keys(self, df: pd.DataFrame):
        required = {
            "dataset",
            "fold",
            "instance_selection_space",
            "instance_selection_algorithm",
            "n_selected_samples",
            "random_state",
            "evaluator",
            "metric",
        }
        if df is None or df.empty or any(col not in df.columns for col in required):
            return set()

        keys = set()
        for _, row in df.iterrows():
            try:
                key = self._make_result_key(
                    dataset_name=row["dataset"],
                    fold=row["fold"],
                    instance_selection_space=row["instance_selection_space"],
                    instance_selection_algorithm=row["instance_selection_algorithm"],
                    n_selected_samples=row["n_selected_samples"],
                    random_state=row["random_state"],
                    evaluator=row["evaluator"],
                    metric=row["metric"],
                )
            except Exception:
                continue
            keys.add(key)
        return keys

    def _has_all_evaluator_metrics(
        self,
        *,
        dataset_name,
        fold,
        instance_selection_space,
        instance_selection_algorithm,
        n_selected_samples,
        random_state,
        evaluator_name,
        evaluator,
    ):

        if self.recompute_evaluator:
            return False

        return all(
            self._make_result_key(
                dataset_name=dataset_name,
                fold=fold,
                instance_selection_space=instance_selection_space,
                instance_selection_algorithm=instance_selection_algorithm,
                n_selected_samples=n_selected_samples,
                random_state=random_state,
                evaluator=evaluator_name,
                metric=metric_name,
            ) in self._existing_result_keys
            for metric_name in evaluator.metrics
        )

    def _evaluate_all_original_space(self, dataset_name, fold, x_train, y_train, x_test, y_test, n_classes, evaluators):
        rows = []
        n_train = len(x_train)

        for evaluator in evaluators:

            logger.info(
                f"{dataset_name} Fold {fold}. evaluator={evaluator.name} "
                f"| space=OriginalSpace | method=All | n_selected={n_train} | random_state=None"
            )

            if self._has_all_evaluator_metrics(
                dataset_name=dataset_name,
                fold=fold,
                instance_selection_space="OriginalSpace",
                instance_selection_algorithm="All",
                n_selected_samples=n_train,
                random_state=None,
                evaluator_name=evaluator.name,
                evaluator=evaluator,
            ):
                logger.info(
                    f"{dataset_name} Fold {fold}. Skipping already computed evaluator "
                    f"| evaluator={evaluator.name} | space=OriginalSpace | method=All | n_selected={n_train} | random_state=None "
                    f"| metrics={evaluator.metrics}"
                )
                continue

            try:
                metrics = evaluator.eval(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    num_classes=n_classes,
                    device=self.device,
                    dataset_name=dataset_name,
                    fold=fold,
                    instance_selection_space="OriginalSpace",
                    instance_selection_algorithm="All",
                    n_selected_samples=n_train,
                    random_state=None,
                    suitename=self.config.benchmark.suitename,
                )
            except Exception as e:
                logger.exception(
                    f"{dataset_name} Fold {fold}. evaluator={evaluator.name} failed "
                    f"| space=OriginalSpace | method=All | n_selected={n_train} "
                    f"| random_state=None | error={e}"
                )
                continue

            if metrics is None:
                logger.warning(
                    f"{dataset_name} Fold {fold}. evaluator={evaluator.name} returned None "
                    f"| space=OriginalSpace | method=All | n_selected={n_train} | random_state=None"
                )
                continue

            rows.extend(
                self._metric_rows_from_evaluation(
                    dataset_name=dataset_name,
                    fold=fold,
                    instance_selection_space="OriginalSpace",
                    instance_selection_algorithm="All",
                    n_selected_samples=n_train,
                    random_state=None,
                    evaluator_name=evaluator.name,
                    metrics=metrics,
                )
            )

        return rows

    @staticmethod
    def _build_sizes_to_sample(n_train, n_classes, percentages, nclass_multipliers):
        """
        Calculate the list of sample sizes to evaluate 
        based on the total number of training samples (n_train) and the number of classes (n_classes)
        """
        sizes = []

        for percentage in percentages:
            value = int(round((float(percentage) / 100.0) * n_train))
            sizes.append(value)

        for mult in nclass_multipliers:
            sizes.append(int(n_classes * int(mult)))

        normalized_sizes = []
        for size in sizes:
            size = max(1, min(int(size), int(n_train)))
            if size not in normalized_sizes:
                normalized_sizes.append(size)

        return sorted(normalized_sizes, reverse=True)

    def _instantiate_encoder(self, encoder_cfg, dataset=None):
        params = dict(encoder_cfg.get("params", {}))
        params.setdefault("device", self.device)
        params["dataset"] = dataset
        return get_obj_from_str(encoder_cfg.target)(**params)

    @staticmethod
    def _instantiate_sampler(method_cfg, space_name, dataset=None):
        params = dict(method_cfg.get("params", {}))
        params.pop("n_runs", None)
        params["dataset"] = dataset
        params["space_name"] = space_name
        return get_obj_from_str(method_cfg.target)(**params)

    @staticmethod
    def _get_method_runs_and_base_seed(method_cfg):
        n_runs = int(method_cfg.get("n_runs", 1))
        n_runs = max(1, n_runs)
        base_seed = int(method_cfg.get("base_seed", 42))
        return n_runs, base_seed

    def _resolve_evaluators(self):
        evaluators = []
        for ev_cfg in self.config.original_space_evaluation:
            params = dict(ev_cfg.get("params", {}))
            params.setdefault("device", self.device)
            params["dataset"] = None
            evaluator = get_obj_from_str(ev_cfg.target)(**params)
            evaluators.append(evaluator)
        return evaluators

    def _encode_train_space(self, encoder, x_train):

        if hasattr(encoder, "fit_transform"):
            try:
                x_train_encoded = encoder.fit_transform(x_train)
                if x_train_encoded is not None and len(x_train_encoded) == len(x_train):
                    return x_train_encoded
                else:
                    logger.warning(f"ENCODER {encoder.name} fit_transform returned invalid output (None or wrong length).")
                    logger.warning(f"Expected length: {len(x_train)}, got: {len(x_train_encoded) if x_train_encoded is not None else 'None'}")
                    return None
            except Exception:
                logger.warning(f"ENCODER {encoder.name} failed to fit_transform!!!!!!!!!!!!!!!!!!!")
                # trace and print the error
                import traceback
                traceback.print_exc()

                return None
        

    @staticmethod
    def _select_original_train(x_train, y_train, selected_indices):
        """
        The indices returned by samplers are meant for iloc, because samplers transform data to numpy arrays if they are not arrays already
        """
        if isinstance(x_train, pd.DataFrame):
            x_sel = x_train.iloc[selected_indices]
        else:
            x_sel = x_train[selected_indices]

        if isinstance(y_train, pd.Series):
            y_sel = y_train.iloc[selected_indices]
        else:
            y_sel = y_train[selected_indices]

        return x_sel, y_sel

    @staticmethod
    def _has_all_constant_feature(x):
        if isinstance(x, pd.DataFrame):
            if x.shape[1] == 0:
                return True
            return not bool((x.nunique(dropna=False) > 1).any())

        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return True

        try:
            return not bool(np.any(np.nanmax(arr, axis=0) != np.nanmin(arr, axis=0)))
        except Exception:
            for col_idx in range(arr.shape[1]):
                if np.unique(arr[:, col_idx]).shape[0] > 1:
                    return False
            return True

    def eval_dataset_fold(self, dataset, fold):
        logger.debug(f"{dataset.name} Fold {fold}. Starting")

        x_train, y_train, x_test, y_test = dataset.get_data_of_fold(fold)
        n_train = len(x_train)
        n_total_samples = len(x_train) + len(x_test)

        sizes_to_sample = self._build_sizes_to_sample(
            n_train=n_train,
            n_classes=dataset.num_classes,
            percentages=self.config.experiment.subsampler_train_percentages,
            nclass_multipliers=self.config.experiment.subsampler_nclasses_multipliers,
        )

        logger.debug(f"{dataset.name} Fold {fold}. n_train={n_train}, n_classes={dataset.num_classes}, sample_sizes={sizes_to_sample}")

        evaluators = self._resolve_evaluators()
        self._append_results(
            self._evaluate_all_original_space(
                dataset_name=dataset.name,
                fold=fold,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                n_classes=dataset.num_classes,
                evaluators=evaluators,
            )
        )

        # Iterate over the encoders that create instance selection spaces
        for is_idx, is_cfg in enumerate(self.config.instance_selection, start=1):
            encoder = self._instantiate_encoder(is_cfg.instance_selection_space, dataset=dataset)
            encoder_name = encoder.name
            space_name = encoder.space_name
            logger.info(f"{dataset.name} Fold {fold}. Space [{is_idx}/{len(self.config.instance_selection)}]: {space_name}")
            logger.debug(f"{dataset.name} Fold {fold}. Encoding with {encoder_name}")

            x_train_space = self._encode_train_space(encoder, x_train)
            if x_train_space is None:
                logger.warning(f"{dataset.name} Fold {fold}. Encoder {encoder_name} returned None. Skipping.")
                continue
            
            # Iterate over the instance selection methods we want to evaluate in that space
            for method_idx, method_cfg in enumerate(is_cfg.undersampling_methods, start=1):
                sampler = self._instantiate_sampler(method_cfg, space_name=space_name, dataset=dataset)
                sampler_name = getattr(sampler, "name", method_cfg.target.split(".")[-1])
                logger.info(f"{dataset.name} Fold {fold}. Method [{method_idx}/{len(is_cfg.undersampling_methods)}] in {space_name}: {sampler_name}")

                n_runs, base_seed = self._get_method_runs_and_base_seed(method_cfg)
                if n_runs > 1:
                    logger.info(
                        f"{dataset.name} Fold {fold}. Running {sampler_name} for {n_runs} runs "
                        f"with base random_state={base_seed}"
                    )

                # Iterate over each reduction size
                for n_selected_samples_target in sizes_to_sample:
                    size_rows = []

                    # Iterate over the number of times to run each evaluator, with different random states
                    for run_idx in range(n_runs):
                        run_random_state = base_seed + run_idx

                        selected_indices = sampler.select_instances(
                            x_train_space,
                            y_train,
                            n_samples=n_selected_samples_target,
                            return_only_indexes=True,
                            random_state=run_random_state)

                        if len(selected_indices) == 0:
                            logger.warning(f"{dataset.name} Fold {fold}. {space_name}-{sampler_name} selected 0 samples for n={n_selected_samples_target}. Skipping.")
                            continue

                        x_train_sel, y_train_sel = self._select_original_train(x_train, y_train, selected_indices)

                        if self._has_all_constant_feature(x_train_sel):
                            logger.warning(
                                f"{dataset.name} Fold {fold}. {space_name}-{sampler_name} produced all-constant features for n={n_selected_samples_target}. "
                                "Continuing evaluation may fail for some evaluators."
                            )

                        # Iterate over evaluators. Currently we only use TabPFN
                        for evaluator in evaluators:
                            evaluator_name = evaluator.name

                            if self._has_all_evaluator_metrics(
                                dataset_name=dataset.name,
                                fold=fold,
                                instance_selection_space=space_name,
                                instance_selection_algorithm=sampler_name,
                                n_selected_samples=len(selected_indices),
                                random_state=run_random_state,
                                evaluator_name=evaluator_name,
                                evaluator=evaluator,
                            ):
                                logger.info(
                                    f"{dataset.name} Fold {fold}. Skipping already computed evaluator "
                                    f"| evaluator={evaluator_name} | space={space_name} | method={sampler_name} "
                                    f"| n_selected={n_selected_samples_target} | random_state={run_random_state} "
                                    f"| metrics={evaluator.metrics}"
                                )
                                continue

                            logger.info(
                                f"{dataset.name} Fold {fold}. evaluator={evaluator_name} "
                                f"| space={space_name} | method={sampler_name} | n_selected={n_selected_samples_target} "
                                f"| random_state={run_random_state}"
                            )
                            try:
                                metrics = evaluator.eval(
                                    x_train_sel,
                                    y_train_sel,
                                    x_test,
                                    y_test,
                                    num_classes=dataset.num_classes,
                                    device=self.device,
                                    dataset_name=dataset.name,
                                    fold=fold,
                                    instance_selection_space=space_name,
                                    instance_selection_algorithm=sampler_name,
                                    n_selected_samples=n_selected_samples_target,
                                    random_state=run_random_state,
                                    suitename=self.config.benchmark.suitename,
                                )
                            except Exception as e:
                                logger.exception(
                                    f"{dataset.name} Fold {fold}. evaluator={evaluator_name} failed "
                                    f"| space={space_name} | method={sampler_name} | n_selected={n_selected_samples_target} "
                                    f"| random_state={run_random_state} | error={e}"
                                )
                                continue

                            if metrics is None:
                                logger.warning(
                                    f"{dataset.name} Fold {fold}. evaluator={evaluator_name} returned None "
                                    f"| space={space_name} | method={sampler_name} | n_selected={n_selected_samples_target} "
                                    f"| random_state={run_random_state}"
                                )
                                continue

                            size_rows.extend(
                                self._metric_rows_from_evaluation(
                                    dataset_name=dataset.name,
                                    fold=fold,
                                    instance_selection_space=space_name,
                                    instance_selection_algorithm=sampler_name,
                                    n_selected_samples=n_selected_samples_target,
                                    random_state=run_random_state,
                                    evaluator_name=evaluator_name,
                                    metrics=metrics,
                                )
                            )

                    self._append_results(size_rows)

        logger.debug(f"{dataset.name} Fold {fold}. Finished")
        self._save_pending_results(f"finished dataset {dataset.name} fold {fold}")

        return None

    def eval_dataset(self, dataset, i, n):
        available_folds = set(range(dataset.n_folds))
        selected_folds = [fold for fold in self.folds_to_run if fold in available_folds]

        for fold in selected_folds:
            self._set_results_file_for_fold(fold)
            setup_logging(
                log_file=f"experiment_unsupervised_context_selection_{exp_id_with_fold(self.exp_id, fold)}.log",
                log_level=logging.INFO,
            )
            logger.info("\n" + "#" * 100 + "\n" + "#" * 100 + "\n" + "#" * 100)
            logger.info("Dataset [%d/%d]: %s | Fold: %s | results: %s", i + 1, n, dataset.name, fold, self.results_basename)
            logger.info("\n" + "#" * 100 + "\n" + "#" * 100 + "\n" + "#" * 100)
            self.eval_dataset_fold(dataset, fold)

        return None

    def run(self):

        benchmark = Benchmark(**self.config.benchmark)
        
        for i, dataset in enumerate(benchmark.get_datasets()):
            self.eval_dataset(dataset, i, benchmark.num_of_filtered_datasets)


def main():
    logger.info("PYTHON: %s - Starting experiment.py", datetime.now().strftime("%H:%M:%S"))
    seed_everything(GLOBAL_SEED)
    logger.info(f"Global deterministic seed set to {GLOBAL_SEED}")

    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest="config", default="config_exp2_LUCoS.yaml", help="Path to the configuration file")
    parser.add_argument(
        "--folds",
        type=str,
        default="[0]",
        help="List of folds to run. Examples: '[0,1,2,3,4]' or '0,1,2,3,4'. Default: [0]",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Experiment id override. If provided, it takes precedence over the YAML exp_id.",
    )
    args = parser.parse_args()

    config = load_yaml_config(
        args.config,
        "lucos.unsupervised_context_selection",
        "config",
        repo_relative_dir=Path("src") / "lucos" / "unsupervised_context_selection" / "config",
    )

    if args.exp_id is not None and str(args.exp_id).strip() != "":
        # If the experiment name was provided by CLI, use it; otherwise use the YAML one.
        config.experiment.exp_id = str(args.exp_id).strip()
        exp_id_source = "cli --exp_id"
    else:
        config.experiment.exp_id = str(config.experiment.exp_id)
        exp_id_source = "yaml"

    if config.experiment.exp_id == "now":
        config.experiment.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id_source = f"{exp_id_source} (resolved from 'now')"
    if config.experiment.device == "default":
        config.experiment.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config.experiment.folds_to_run = parse_folds_arg(args.folds)

    config.results_folder = str(results_folder / f"results_LUCoS_{config.experiment.exp_id}")
    config.results_basename = f"results_LUCoS_{config.experiment.exp_id}"

    exp_id_message = f"Using exp_id='{config.experiment.exp_id}' (source: {exp_id_source})"

    setup_logging(log_file=f"experiment_unsupervised_context_selection_{config.experiment.exp_id}.log", log_level=logging.INFO)
    logger.info(exp_id_message)
    
    save_config(config)
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
