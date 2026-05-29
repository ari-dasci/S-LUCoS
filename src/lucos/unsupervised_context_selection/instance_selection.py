import logging
import os
import json
import copy
from time import perf_counter
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.utils import resample
from lucos.utils.preprocessing import one_hot_encode
from lucos.utils.evaluation import imbalance_ratio
from lucos.utils.paths import ckpt_folder, change_permissions_rw
from lucos.SSMA.ssma_utils import (
    load_best_chromosomes_curve,
    get_chromosome_in_curve_smaller_or_equal_to_n,
)
logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)



class UnderSampler:

    @staticmethod
    def get_name(**kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __init__(self, space_name=None, dataset=None, use_cache=True, **kwargs):
        self.verbose = False
        self.dataset = dataset
        self.space_name = space_name
        self.use_cache = bool(use_cache)
        self.name = self.get_name(**kwargs)
        
        if space_name is None or dataset is None:
            self.ckpt_folder = None
            self.checkpoint_file = None
            msg = "No checkpoints will be saved & loaded for this instance selection method."
            if space_name is None:
                msg = f"Instantiating {self.name} with no space name." + msg
            if dataset is None:
                self.dataset = SimpleNamespace(name_and_fold="unknown_dataset", suitename="unknown_suitename", cat_cols=[], num_cols=[], bin_cols=[])
                msg = f"Instantiating {self.name} with no dataset information." + msg
            logger.warning(msg)
        else:
            self.ckpt_folder = ckpt_folder / "Undersampling" / self.space_name / self.name / self.dataset.suitename
            self.checkpoint_file = self.ckpt_folder / f"{self.dataset.name_and_fold}.json"
            self.ckpt_folder.mkdir(parents=True, exist_ok=True)

        if self.use_cache:
            logger.info(f"{self.name}: use_cache=True. Selection checkpoints are enabled when path context is available.")
        else:
            logger.warning(
                f"{self.name}: use_cache=False. Selection will always be recomputed and checkpoints will not be loaded/saved."
            )

    def _cache_ready(self):
        return self.space_name is not None and self.checkpoint_file is not None

    def _try_load_cached_indices(self, mode, n_samples, random_state, n_total):
        if not self.use_cache:
            return None

        if not self._cache_ready():
            logger.debug(
                f"{self.name}.select_instances called without complete checkpoint context. "
                "Selection will be computed without loading/saving checkpoints."
            )
            return None

        selected_indxs = self._load_selected_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
        )
        if selected_indxs is not None:
            logger.debug(
                f"Loaded {self.name} selection from checkpoint {self.checkpoint_file} "
                f"(space={self.space_name}, mode={mode}, n_samples={n_samples}, seed={random_state})"
            )
        return selected_indxs

    def _maybe_store_indices(self, mode, n_samples, random_state, n_total, selected_indxs):
        if not self.use_cache or not self._cache_ready():
            return
        self._store_selected_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
            selected_indxs=selected_indxs,
        )

    def _get_or_compute_indices(self, mode, n_samples, random_state, n_total, compute_fn):
        selected_indxs = self._try_load_cached_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
        )
        if selected_indxs is not None:
            return selected_indxs

        logger.info(
            f"{self.name}: computing selection (mode={mode}, n_samples={n_samples}, seed={random_state}, use_cache={self.use_cache})"
        )
        selection_start = perf_counter()
        selected_indxs = compute_fn()
        compute_time = perf_counter() - selection_start
        self._maybe_store_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
            selected_indxs=selected_indxs,
        )
        total_time = perf_counter() - selection_start
        logger.info(
            "%s: computed selection in %.3fs (total with checkpoint save %.3fs; mode=%s, n_samples=%s, seed=%s)",
            self.name,
            compute_time,
            total_time,
            mode,
            n_samples,
            random_state,
        )
        return selected_indxs


    def _load_checkpoint(self):
        if self.checkpoint_file is None or not self.checkpoint_file.exists():
            return {
                "sampler": self.name,
                "space_name": self.space_name,
                "selections": {}
            }
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "selections" not in data or not isinstance(data["selections"], dict):
                data["selections"] = {}
            return data
        except Exception as e:
            logger.warning(f"Could not read checkpoint {self.checkpoint_file}. Recreating it. Error: {e}")
            return {
                "sampler": self.name,
                "space_name": self.space_name,
                "selections": {}
            }

    def _save_checkpoint(self, data):
        if self.checkpoint_file is None:
            return
        tmp_file = self.checkpoint_file.with_suffix(self.checkpoint_file.suffix + ".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp_file.replace(self.checkpoint_file)

        change_permissions_rw(self.checkpoint_file)

    def _load_selected_indices(self, mode, n_samples, random_state, n_total):
        data = self._load_checkpoint()
        mode = str(mode)
        n_key = str(int(n_samples))
        seed_key = str(int(random_state))

        mask = (
            data
            .get("selections", {})
            .get(self.space_name, {})
            .get(mode, {})
            .get(n_key, {})
            .get(seed_key)
        )

        if mask is None:
            return None

        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape[0] != n_total:
            logger.warning(
                f"Checkpoint mask length mismatch for {self.checkpoint_file} "
                f"(expected {n_total}, got {mask.shape[0]}). Recomputing selection."
            )
            return None

        selected_count = int(mask.sum())
        if selected_count != int(n_samples):
            logger.warning(
                f"Checkpoint selection size mismatch for {self.checkpoint_file} "
                f"(expected {n_samples}, got {selected_count}). Recomputing selection."
            )
            return None

        return np.where(mask == 1)[0]

    def _store_selected_indices(self, mode, n_samples, random_state, n_total, selected_indxs):
        data = self._load_checkpoint()
        mode = str(mode)
        n_key = str(int(n_samples))
        seed_key = str(int(random_state))

        selected_indxs = self._normalize_selected_indices(selected_indxs, n_total)

        mask = np.zeros(n_total, dtype=np.uint8)
        mask[selected_indxs] = 1

        selections = data.setdefault("selections", {})
        by_space = selections.setdefault(self.space_name, {})
        by_mode = by_space.setdefault(mode, {})
        by_n = by_mode.setdefault(n_key, {})
        by_n[seed_key] = mask.tolist()

        self._save_checkpoint(data)

    @staticmethod
    def _select_by_indices(X, y, selected_indxs):
        X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
        y_sel = y.iloc[selected_indxs] if isinstance(y, pd.Series) else y[selected_indxs]
        return X_sel, y_sel

    @staticmethod
    def _normalize_selected_indices(selected, n_total):
        # Normalize selector output to always return unique and valid indices.
        # If it receives a binary mask (0/1) of length n_total, convert it to indices.
        selected = np.asarray(selected)
        if selected.ndim == 0:
            selected = selected.reshape(1)

        if selected.shape[0] == n_total:
            unique_values = np.unique(selected)
            if np.all(np.isin(unique_values, [0, 1])):
                return np.where(selected.astype(np.uint8) == 1)[0]

        selected = selected.astype(int)
        selected = selected[(selected >= 0) & (selected < n_total)]
        return np.unique(selected)


    @staticmethod
    def extract_selection_map(checkpoint_data):
        selections = checkpoint_data.get("selections", {}) if isinstance(checkpoint_data, dict) else {}
        if not isinstance(selections, dict):
            return {}

        for space_block in selections.values():
            if not isinstance(space_block, dict):
                continue
            for mode_block in space_block.values():
                if isinstance(mode_block, dict) and len(mode_block) > 0:
                    return mode_block
        return {}



class RandomUnderSampler(UnderSampler):

    @staticmethod
    def get_name(mode, **kwargs):
        return "RandomUnderSampler" + mode
    
    def __init__(self, space_name=None, dataset=None, mode='Unsupervised', use_cache=True, **kwargs):
        super().__init__(space_name=space_name, dataset=dataset, mode=mode, use_cache=use_cache, **kwargs)
        assert mode in ["Min1PerClass", "Balanced", 'Unsupervised']
        self.mode = mode

    @staticmethod
    def _compute_balanced_indices(y_np, n_samples, random_state):
        rng = np.random.RandomState(random_state)
        classes = np.unique(y_np)
        class_to_indices = {c: np.where(y_np == c)[0] for c in classes}
        class_to_target = {c: min(len(class_to_indices[c]), n_samples // len(classes)) for c in classes}

        remaining = n_samples - sum(class_to_target.values())
        while remaining > 0:
            eligible_classes = [c for c in classes if class_to_target[c] < len(class_to_indices[c])]
            if not eligible_classes:
                break

            rng.shuffle(eligible_classes)
            eligible_classes.sort(
                key=lambda c: len(class_to_indices[c]) - class_to_target[c],
                reverse=True,
            )

            progress = False
            for class_label in eligible_classes:
                if remaining == 0:
                    break
                if class_to_target[class_label] >= len(class_to_indices[class_label]):
                    continue
                class_to_target[class_label] += 1
                remaining -= 1
                progress = True

            if not progress:
                break

        selected_indxs = []
        for class_label in classes:
            n_class_samples = class_to_target[class_label]
            if n_class_samples == 0:
                continue
            selected_indxs.append(
                rng.choice(class_to_indices[class_label], size=n_class_samples, replace=False)
            )

        if not selected_indxs:
            return np.array([], dtype=int)

        selected_indxs = np.concatenate(selected_indxs).astype(int)
        rng.shuffle(selected_indxs)
        return selected_indxs

    def _compute_selected_indices(self, n_samples, n_total, random_state, y=None):
        if n_samples >= n_total:
            return np.arange(n_total)

        if self.mode == 'Unsupervised':
            return resample(np.arange(n_total), n_samples=n_samples, replace=False, random_state=random_state)
        
        assert y is not None, "y cannot be None when mode is not 'Unsupervised'"

        y_np = y.values if isinstance(y, pd.Series) else y
        classes = np.unique(y_np)
        if n_samples < len(classes):
            logger.warning(
                f"Requested n_samples={n_samples} is lower than number of classes={len(classes)}. "
                "Falling back to plain random sampling without per-class guarantee."
            )
            return resample(np.arange(n_total), n_samples=n_samples, replace=False, random_state=random_state)

        if self.mode == "Min1PerClass":
            selected_indxs = np.array([], dtype=int)
            for c in classes:
                class_indxs = np.where(y_np == c)[0]
                selected_indxs = np.concatenate([selected_indxs, resample(class_indxs, n_samples=1, random_state=random_state)])

            n_samples_left = n_samples - len(selected_indxs)
            if n_samples_left > 0:
                not_selected_indsx = np.setdiff1d(np.arange(n_total), selected_indxs)
                selected_indxs = np.concatenate([
                    selected_indxs,
                    resample(not_selected_indsx, n_samples=n_samples_left, replace=False, random_state=random_state),
                ])

            return selected_indxs
        if self.mode == "Balanced":
            return self._compute_balanced_indices(
                y_np=y_np,
                n_samples=n_samples,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown mode {self.mode} for RandomUnderSampler")


    def select_instances(
        self,
        X,
        y=None,
        n_samples=None,
        return_only_indexes=False,
        random_state=42):
        """
        Selects random instances from the dataset.
        Supports both pandas DataFrames and numpy ndarrays.
        Returns the same data type as the input.
        """
        n_total = len(X)
        selected_indxs = self._get_or_compute_indices(
            mode=self.mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
            compute_fn=lambda: self._compute_selected_indices(
                y=y,
                n_samples=n_samples,
                n_total=n_total,
                random_state=random_state,
            ),
        )

        if self.verbose:
            logger.debug(f"Reducing dataset of {X.shape[0]} to {n_samples} ({n_samples/len(X)*100:.2f}%) instances using random selection")

        if y is not None:
            x_selected, y_selected = self._select_by_indices(X, y, selected_indxs)
            if self.verbose:
                logger.debug(f'Original imbalance ratio: {imbalance_ratio(y):.2f}, reduced imbalance ratio: {imbalance_ratio(y_selected):.2f}')
            if len(np.unique(y_selected)) < len(np.unique(y)):
                logger.info(f"RandomUnderSampler selected {n_samples} intances, but only {len(np.unique(y_selected))} classes out of {len(np.unique(y))}. This may cause issues for some evaluators!!!!!!!")

        selected_indxs = self._normalize_selected_indices(selected_indxs, n_total)
        if return_only_indexes:
            return selected_indxs

        if y is None:
            X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
            return X_sel, None

        return self._select_by_indices(X, y, selected_indxs)
    

class KMedoidsUnderSampler(UnderSampler):

    # Saving the distance matrix:
    #    - For the smallest datasets: 0.003s to compute
    #    - For har (10299x561): 1.04s to compute, 3.2s to save, 0.05s to load the first time (0.001s the rest of the times)
    #    - For nomao (34465x118): 10s to compute
    #    - For mnist_784 (70000x784): 48s to compute (and loading the distance matrix takes 1.8s the first time and less than 1 second afterwards). In any case, running KMedoids then takes about 5 min
    #      And if you do it for 6 sizes by 5 seeds: 48s * 6 * 5 = 1440s = 24 minutes just to compute the mnist_784 distance matrix for the 30 selections we do in total.

    @staticmethod
    def get_name(init="random", metric="euclidean", **kwargs):
        init_str = str(init)
        init_str = init_str[:1].upper() + init_str[1:]
        metric_str = metric[:1].upper() + metric[1:]
        return "KMedoidsUnderSampler" + init_str + metric_str

    def __init__(
        self,
        space_name=None,
        dataset=None,
        init="random",
        metric="euclidean",
        use_cache=True,
        save_distance_matrices=True, # This saves computing time but may consume a lot of disk space, use with caution. 
        **kwargs,
    ):
        super().__init__(
            space_name=space_name,
            dataset=dataset,
            init=init,
            metric=metric,
            use_cache=use_cache,
            **kwargs,
        )
        self.init = init
        self.metric = metric
        self.save_distance_matrices = bool(save_distance_matrices)

    def _distance_matrix_cache_file(self):
        if self.ckpt_folder is None:
            return None
        dataset_name = getattr(self.dataset, "name_and_fold", "unknown_dataset")
        metric_name = str(self.metric).replace(os.sep, "_")
        return self.ckpt_folder / "distance_matrices" / f"{dataset_name}__{metric_name}.npy"

    def _load_cached_distance_matrix(self, n_total):
        if not self.use_cache or not self._cache_ready():
            return None

        cache_file = self._distance_matrix_cache_file()
        if cache_file is None or not cache_file.exists():
            return None

        start = perf_counter()
        try:
            distance_matrix = np.load(cache_file, mmap_mode="r")
            load_time = perf_counter() - start
        except Exception as exc:
            logger.warning(
                "%s could not load cached distance matrix from %s. It will be recomputed. Error: %s",
                self.name,
                cache_file,
                exc,
            )
            return None

        expected_shape = (n_total, n_total)
        if distance_matrix.shape != expected_shape:
            logger.warning(
                "%s cached distance matrix shape mismatch at %s (expected %s, got %s). It will be recomputed.",
                self.name,
                cache_file,
                expected_shape,
                distance_matrix.shape,
            )
            return None

        logger.info(
            "%s loaded cached %s distance matrix from %s in %.3fs",
            self.name,
            self.metric,
            cache_file,
            load_time,
        )
        return distance_matrix

    def _save_distance_matrix(self, distance_matrix):
        if not self.use_cache or not self._cache_ready():
            return None, None
        if not self.save_distance_matrices:
            return None, None

        cache_file = self._distance_matrix_cache_file()
        if cache_file is None:
            return None, None

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.npy")
        start = perf_counter()
        np.save(tmp_file, distance_matrix)
        tmp_file.replace(cache_file)
        change_permissions_rw(cache_file)
        save_time = perf_counter() - start
        return cache_file, save_time

    def _get_or_compute_distance_matrix(self, X_oh):
        from sklearn.metrics import pairwise_distances

        n_total = len(X_oh)
        distance_matrix = self._load_cached_distance_matrix(n_total=n_total)
        if distance_matrix is not None:
            return distance_matrix

        logger.info(
            "%s computing %s distance matrix for %s instances.",
            self.name,
            self.metric,
            n_total,
        )
        start = perf_counter()
        distance_matrix = pairwise_distances(X_oh, metric=self.metric).astype(np.float32, copy=False)
        compute_time = perf_counter() - start

        cache_file, save_time = self._save_distance_matrix(distance_matrix)
        if cache_file is not None:
            logger.info(
                "%s computed %s distance matrix in %.3fs and saved it to %s in %.3fs",
                self.name,
                self.metric,
                compute_time,
                cache_file,
                save_time,
            )
        else:
            logger.info(
                "%s computed %s distance matrix in %.3fs. Matrix was not saved "
                "(use_cache=%s, save_distance_matrices=%s).",
                self.name,
                self.metric,
                compute_time,
                self.use_cache,
                self.save_distance_matrices,
            )
        return distance_matrix

    def _compute_kmedoids_indices(self, X, n_samples, random_state):
        from sklearn_extra.cluster import KMedoids

        if n_samples >= len(X):
            return np.arange(len(X))

        X_oh = one_hot_encode(X)
        logger.debug(
            f"Fitting KMedoids to select {n_samples} instances from {len(X)}. "
            f"Random state: {random_state}. Metric: {self.metric}. "
            f"save_distance_matrices={self.save_distance_matrices}"
        )
        distance_matrix = self._get_or_compute_distance_matrix(X_oh)
        kmedoids = KMedoids(
            n_clusters=n_samples,
            init=self.init,
            metric="precomputed",
            random_state=random_state,
        )
        fit_start = perf_counter()
        kmedoids.fit(distance_matrix)
        fit_time = perf_counter() - fit_start
        logger.info(
            "%s fit completed in %.3fs (dataset=%s, space=%s, n_samples=%s, seed=%s, init=%s, metric=%s)",
            self.name,
            fit_time,
            getattr(self.dataset, "name_and_fold", "unknown_dataset"),
            self.space_name,
            n_samples,
            random_state,
            self.init,
            self.metric,
        )
        # return kmedoids.medoid_indices_

        # Diagnostic only: this does not modify selected indices, only logs duplicate signatures.
        medoid_indices = np.asarray(kmedoids.medoid_indices_, dtype=int)
        try:
            selected_repr = np.asarray(X_oh)[medoid_indices]
            selected_unique_repr = np.unique(selected_repr, axis=0)
            if selected_unique_repr.shape[0] < selected_repr.shape[0]:
                duplicated_mask = pd.DataFrame(selected_repr).duplicated(keep=False).to_numpy()
                dup_positions = np.where(duplicated_mask)[0]
                dup_original_indices = medoid_indices[dup_positions]

                logger.warning(
                    "%s selected %s medoids but only %s unique feature vectors in selection space "
                    "(dataset=%s, space=%s, n_samples=%s, seed=%s, init=%s).",
                    self.name,
                    selected_repr.shape[0],
                    selected_unique_repr.shape[0],
                    getattr(self.dataset, "name_and_fold", "unknown_dataset"),
                    self.space_name,
                    n_samples,
                    random_state,
                    self.init,
                )
                logger.warning(
                    "%s duplicate medoid positions=%s, original_indices=%s",
                    self.name,
                    dup_positions.tolist(),
                    dup_original_indices.tolist(),
                )

                if dup_positions.size > 0:
                    # print how many examples in the dataset have the same signature as the first duplicated example
                    example_vec = selected_repr[int(dup_positions[0])]
                    example_matches = int(np.all(np.asarray(X_oh) == example_vec, axis=1).sum())
                    logger.warning(
                        "%s example duplicated signature appears %s times in candidate pool.",
                        self.name,
                        example_matches,
                    )
        except Exception as exc:
            logger.debug("%s duplicate-medoid diagnostics failed: %s", self.name, exc)

        return medoid_indices

    def select_instances(
        self,
        X,
        y=None,
        n_samples=None,
        return_only_indexes=False,
        random_state=42):
        """
        Selects instances using KMedoids clustering.
        NOTE: this method does not ensure that at least one instance from each class is selected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        LATER THE EVALUATOR ADDS DUMMY INSTANCES IF ANY CLASS IS MISSING FROM THE TRAINING SET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        n_total = len(X)
        mode = f"init:{self.init}|metric:{self.metric}"
        selected_indxs = self._get_or_compute_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
            compute_fn=lambda: self._compute_kmedoids_indices(
                X=X,
                n_samples=n_samples,
                random_state=random_state,
            ),
        )

        if self.verbose:
            logger.debug(f"Reducing dataset of {X.shape[0]} to {n_samples} ({n_samples/len(X)*100:.2f}%) instances using KMedoids selection")

        if y is not None:
            x_selected, y_selected = self._select_by_indices(X, y, selected_indxs)
            if self.verbose:
                logger.debug(f'Original imbalance ratio: {imbalance_ratio(y):.2f}, reduced imbalance ratio: {imbalance_ratio(y_selected):.2f}')

            # print if not al classes are selected
            if len(np.unique(y_selected)) < len(np.unique(y)):
                logger.info(f"KMedoids selected {n_samples} intances, but only {len(np.unique(y_selected))} classes out of {len(np.unique(y))}. This may cause issues for some evaluators!!!!!!!")

        selected_indxs = self._normalize_selected_indices(selected_indxs, n_total)
        if return_only_indexes:
            return selected_indxs

        if y is None:
            X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
            return X_sel, None

        return self._select_by_indices(X, y, selected_indxs)


class RDSSUnderSampler(UnderSampler):

    @staticmethod
    def get_name(**kwargs):
        return "RDSSUnderSampler"

    def __init__(self, space_name=None, dataset=None, use_cache=True, **kwargs):
        super().__init__(space_name=space_name, dataset=dataset, use_cache=use_cache, **kwargs)

    def _compute_rdss_indices(self, X, n_samples, alpha):
        from RDSS import khsample, gaussiankernel

        if n_samples >= len(X):
            return np.arange(len(X))

        X_oh = one_hot_encode(X)
        X_np = np.asarray(X_oh)
        logger.debug(f"Fitting RDSS to select {n_samples} instances from {len(X)}. Alpha: {alpha:.6f}")
        selected_indxs, _ = khsample(X_np, gaussiankernel, n=n_samples, alpha=alpha)
        return np.asarray(selected_indxs, dtype=int)

    def select_instances(
        self,
        X,
        y=None,
        n_samples=None,
        alpha=None,
        return_only_indexes=False,
        random_state=42):
        """
        Selects instances using RDSS (Representative and Diverse Sample Selection).
        Supports both pandas DataFrames and numpy ndarrays.
        """        
        n_total = len(X)
        if n_samples is None:
            raise ValueError("n_samples must be provided for RDSSUnderSampler")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 for RDSSUnderSampler")

        if alpha is None:
            alpha = 1 - 1 / np.sqrt(max(1, int(n_samples)))
        mode = f"alpha:{float(alpha):.8f}"
        selected_indxs = self._get_or_compute_indices(
            mode=mode,
            n_samples=n_samples,
            random_state=random_state,
            n_total=n_total,
            compute_fn=lambda: self._compute_rdss_indices(
                X=X,
                n_samples=n_samples,
                alpha=float(alpha),
            ),
        )

        if self.verbose:
            logger.debug(f"Reducing dataset of {X.shape[0]} to {n_samples} ({n_samples/len(X)*100:.2f}%) instances using RDSS selection")

        if y is not None:
            _, y_selected = self._select_by_indices(X, y, selected_indxs)
            if self.verbose:
                logger.debug(f'Original imbalance ratio: {imbalance_ratio(y):.2f}, reduced imbalance ratio: {imbalance_ratio(y_selected):.2f}')
            if len(np.unique(y_selected)) < len(np.unique(y)):
                logger.info(f"RDSS selected {n_samples} intances, but only {len(np.unique(y_selected))} classes out of {len(np.unique(y))}. This may cause issues for some evaluators!!!!!!!")

        selected_indxs = self._normalize_selected_indices(selected_indxs, n_total)
        if return_only_indexes:
            return selected_indxs

        if y is None:
            X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
            return X_sel, None

        return self._select_by_indices(X, y, selected_indxs)


class SSMAUnderSampler(UnderSampler):

    @staticmethod
    def get_name(**kwargs):
        return "SSMAUnderSampler"

    def __init__(self, space_name=None, dataset=None, use_cache=True, **kwargs):
        super().__init__(space_name=space_name, dataset=dataset, use_cache=use_cache, **kwargs)

    @staticmethod
    def get_available_runs(dataset):
        from lucos.SSMA import ssma_paths

        base_folder = ssma_paths.ssma_results_folder
        if not os.path.isdir(base_folder):
            return []

        available_runs = []
        for folder_name in os.listdir(base_folder):
            if not folder_name.startswith("ssma_run"):
                continue

            run_str = folder_name.replace("ssma_run", "")
            if not run_str.isdigit():
                continue

            run = int(run_str)
            curve = load_best_chromosomes_curve(dataset, run=run)
            if isinstance(curve, dict) and len(curve) > 0:
                available_runs.append(run)

        return sorted(available_runs)

    def select_instances(
        self,
        X,
        y=None,
        n_samples=None,
        return_only_indexes=False,
        random_state=1,
    ):
        logger.info(
            f"{self.name}: computing selection (n_samples={n_samples}, run={random_state}, use_cache={self.use_cache})"
        )

        if self.dataset is None or self.dataset.name_and_fold == "unknown_dataset":
            logger.warning("SSMAUnderSampler called without dataset context. Returning empty selection.")
            return np.array([], dtype=int)

        best_chromosomes_curve = load_best_chromosomes_curve(self.dataset, run=random_state)
        if not best_chromosomes_curve:
            logger.warning(
                f"SSMA best_chromosomes_curve not found for {self.dataset.name_and_fold}. "
                "Skipping SSMA selection for this dataset/fold."
            )
            return np.array([], dtype=int)

        chromosome = get_chromosome_in_curve_smaller_or_equal_to_n(best_chromosomes_curve, n_samples)

        if chromosome is None:
            logger.warning(
                f"No chromosome found in best_chromosomes_curve for n_samples<={n_samples}. "
                f"dataset_name_and_fold={self.dataset.name_and_fold}. Skipping SSMA selection."
            )
            return np.array([], dtype=int)
        
        selection_mask = chromosome.get_genes_mask_with_validation_set(full_size=len(X))

        if selection_mask.shape[0] != len(X):
            raise ValueError(
                f"Chromosome mask length ({selection_mask.shape[0]}) does not match X length ({len(X)}). "
                f"dataset_name_and_fold={self.dataset.name_and_fold}"
            )

        selected_indxs = self._normalize_selected_indices(selection_mask, len(X))
        if return_only_indexes:
            return selected_indxs

        if y is None:
            X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
            return X_sel, None

        return self._select_by_indices(X, y, selected_indxs)


class ZCoreUnderSampler(UnderSampler):
    """
    Instance selector using ZCore (Zero-Shot Coreset Selection).
    
    Uses core.coreset.zcore_score() to compute importance scores for each instance
    based on coverage and redundancy in an embedding space.
    """

    @staticmethod
    def get_name(redund_nn, redund_exp, **kwargs):
        return f"ZCoreUnderSampler" # _rn{redund_nn}re{redund_exp}"

    def __init__(
        self,
        space_name=None,
        dataset=None,
        use_cache=True,
        sample_dim=2,
        redund_nn=1000,
        redund_exp=4, # A high value penalizes very close neighbors and avoids local duplicates.
        num_workers=16, # Number of threads
        n_sample = 1e6, # how many total random iterations ZCore runs to build the scores. Larger => more stable but slower
        rand_init=True,
        **kwargs
    ):
        from types import SimpleNamespace
        super().__init__(
            space_name=space_name,
            dataset=dataset,
            use_cache=use_cache,
            redund_nn=redund_nn,
            redund_exp=redund_exp,
            **kwargs,
        )
        self.sample_dim = int(sample_dim)
        self.redund_nn = int(redund_nn)
        self.redund_exp = int(redund_exp)
        self.num_workers = int(num_workers)
        self.rand_init = bool(rand_init)
        self.args = SimpleNamespace(
            trial= 0,
            n_sample= n_sample, 
            num_workers= self.num_workers,
            rand_init= self.rand_init,
            sample_dim= 2,
            redund_nn= redund_nn,
            redund_exp= redund_exp,
            dataset = self.dataset.name_and_fold
        )
    
    def _try_load_cached_scores(self):
        """Load pre-computed ZCore scores from checkpoint."""
        if not self.use_cache:
            return None

        if self.space_name is None or self.checkpoint_file is None:
            return None
        
        if not self.checkpoint_file.exists():
            return None
        
        try:
            data = self._load_checkpoint()
            scores_dict = data.get("zcore_scores", {})
            scores = scores_dict.get(str(self.space_name))
            return np.asarray(scores) if scores is not None else None
        except Exception:
            return None
    
    def _compute_zcore_scores(self, embeddings, args):
        """Compute ZCore scores for embeddings using the zcore_score function."""
        import importlib.util
        import sys
        from lucos.utils.paths import project_folder

        logger.debug(f"Computing ZCore scores for dataset {self.dataset.name_and_fold} with selection_space={self.space_name}")

        filtered_embeddings, zcore_args = self._preprocess_embeddings_for_zcore(embeddings, args)
        if filtered_embeddings is None:
            return np.ones(len(embeddings), dtype=np.float32)

        zcore_path = project_folder / "libs" / "zcore" / "core" / "coreset" / "zcore.py"
        spec = importlib.util.spec_from_file_location("_lucos_zcore_score", zcore_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load ZCore scorer from {zcore_path}")
        zcore_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = zcore_module
        spec.loader.exec_module(zcore_module)

        scores = zcore_module.zcore_score(zcore_args, filtered_embeddings)
        return scores

    def _preprocess_embeddings_for_zcore(self, embeddings, args):
        """Drop constant dimensions and adjust zcore args before scoring."""
        min_vals = np.min(embeddings, axis=0)
        max_vals = np.max(embeddings, axis=0)
        valid_dim_mask = max_vals > min_vals
        n_valid_dims = int(np.sum(valid_dim_mask))

        if n_valid_dims == 0:
            logger.warning(
                f"All embedding dimensions are constant for dataset {self.dataset.name_and_fold} with selection_space={self.space_name}. "
                "Returning uniform ZCore scores."
            )
            return None, None

        filtered_embeddings = embeddings[:, valid_dim_mask]
        dropped_dims = embeddings.shape[1] - filtered_embeddings.shape[1]
        if dropped_dims > 0:
            logger.info(
                f"Dropped {dropped_dims} constant embedding dimensions for "
                f"dataset {self.dataset.name_and_fold} with selection_space={self.space_name}."
            )

        zcore_args = copy.copy(args)
        zcore_args.sample_dim = min(int(args.sample_dim), n_valid_dims)
        if zcore_args.sample_dim != int(args.sample_dim):
            logger.info(
                f"Adjusted ZCore sample_dim from {args.sample_dim} to "
                f"{zcore_args.sample_dim} for dataset {self.dataset.name_and_fold} with selection_space={self.space_name}."
            )
        return filtered_embeddings, zcore_args
    
    def _store_scores(self, scores):
        """Store computed ZCore scores in checkpoint."""
        if not self.use_cache:
            return

        if self.checkpoint_file is None or self.space_name is None:
            return
        
        data = self._load_checkpoint()
        
        # Initialize or get zcore_scores dict
        zcore_scores = data.setdefault("zcore_scores", {})
        zcore_scores[self.space_name] = scores.tolist()
        
        self._save_checkpoint(data)
        logger.debug(f"Stored ZCore scores for selection_space={self.space_name}")
    

    def get_scores(self):
        scores = self._try_load_cached_scores()
        if scores is None:
            logger.warning(f"No cached ZCore scores found for {self.dataset.name_and_fold} and selection_space={self.space_name}")
        return scores

    def select_instances(
        self,
        X,
        y=None,
        n_samples=None,
        return_only_indexes=False,
        random_state=42):
        """
        Selects instances using ZCore scoring.
        
        Args:
            X: Embeddings array (n_samples, embedding_dim)
            y: Labels (optional)
            n_samples: Number of instances to select
            return_only_indexes: If True, return only indices
            random_state: Random seed (not used for deterministic selection)
        """
        
        n_total = len(X)
        
        # Try to load cached scores
        scores = self._try_load_cached_scores()
        
        if scores is None:
            logger.info(
                f"{self.name}: computing ZCore scores (n_samples={n_samples}, use_cache={self.use_cache})"
            )
            # Compute scores if not cached
            embeddings = np.asarray(X)
            scores = self._compute_zcore_scores(embeddings, self.args)
            # Store scores for future use
            self._store_scores(scores)
        
        # Select indices with highest scores
        selected_indxs = np.argsort(-scores)[:n_samples]
        
        if self.verbose:
            logger.debug(f"Reducing dataset of {n_total} to {n_samples} ({n_samples/n_total*100:.2f}%) instances using ZCore selection")
        
        if y is not None:
            _, y_selected = self._select_by_indices(X, y, selected_indxs)
            if self.verbose:
                logger.debug(f'Original imbalance ratio: {imbalance_ratio(y):.2f}, reduced imbalance ratio: {imbalance_ratio(y_selected):.2f}')
            
            if len(np.unique(y_selected)) < len(np.unique(y)):
                logger.info(f"ZCore selected {n_samples} instances, but only {len(np.unique(y_selected))} classes out of {len(np.unique(y))}. This may cause issues for some evaluators!!!!!!!")
        
        selected_indxs = self._normalize_selected_indices(selected_indxs, n_total)
        if return_only_indexes:
            return selected_indxs
        
        if y is None:
            X_sel = X.iloc[selected_indxs] if isinstance(X, pd.DataFrame) else X[selected_indxs]
            return X_sel, None
        
        return self._select_by_indices(X, y, selected_indxs)
