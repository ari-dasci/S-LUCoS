import os
import logging
from time import perf_counter
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from lucos.utils.preprocessing import one_hot_encode
from lucos.utils.paths import ckpt_folder, change_permissions_rw


logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


def _to_numpy(values):
    if isinstance(values, pd.Series):
        return values.to_numpy()
    if isinstance(values, pd.DataFrame):
        return values.to_numpy()
    return np.asarray(values)


def _ensure_2d(values):
    if isinstance(values, pd.DataFrame):
        return values.copy()

    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _align_logits(logits, class_labels, all_classes):
    """
    Align logits to the same class labels.
    This function takes the predicted logits and their corresponding class labels, and aligns them to a common set of class labels (all_classes).
    It creates a new array of logits where the columns correspond to the classes in all_classes, and fills in the predicted logits in the correct positions based on class_labels.
    If a class in all_classes is not present in class_labels, its logits will be filled with zeros.
    """
    logits = np.asarray(logits)
    class_labels = np.asarray(class_labels)
    all_classes = np.asarray(all_classes)

    aligned = np.zeros((logits.shape[0], len(all_classes)), dtype=float)
    for source_idx, class_label in enumerate(class_labels):
        matches = np.where(all_classes == class_label)[0]
        if len(matches) == 0:
            continue
        aligned[:, matches[0]] = logits[:, source_idx]
    return aligned


def _safe_auc(y_true, logits, class_labels):
    y_true = _to_numpy(y_true)
    class_labels = np.asarray(class_labels)
    observed_labels = np.unique(y_true)
    observed_positions = []
    for label in observed_labels:
        matches = np.where(class_labels == label)[0]
        if len(matches) == 0:
            continue
        observed_positions.append(matches[0])

    if len(observed_positions) < 2:
        return float("nan")

    observed_labels = np.asarray([class_labels[idx] for idx in observed_positions])
    logits = np.asarray(logits)[:, observed_positions]
    label_to_index = {label: idx for idx, label in enumerate(observed_labels)}
    encoded_y_true = np.asarray([label_to_index[label] for label in y_true])

    try:
        if logits.shape[1] == 2:
            return float(roc_auc_score(encoded_y_true, logits[:, 1]))
        return float(roc_auc_score(encoded_y_true, logits, multi_class="ovr"))
    except Exception:
        logger.exception("AUC could not be computed for the current evaluation.")
        return float("nan")


class Evaluator(ABC):

    def __init__(self, device="cpu", dataset=None, name=None, use_checkpoint=False, **params):
        self.device = device
        self.dataset = dataset
        self.params = params
        self.name = name or self.__class__.__name__
        self.metrics = ["AUC", "ACC", "F1", "BalACC"]
        self.use_checkpoint = bool(use_checkpoint)

    def preprocess(self, x_train, y_train, x_test, y_test):
        return x_train, y_train, x_test, y_test

    @abstractmethod
    def predict_logits(self, x_train, y_train, x_test, y_test):
        raise NotImplementedError

    @staticmethod
    def _normalize_checkpoint_value(value):
        return "none" if value is None else str(value)

    def _checkpoint_file_path(self, *, suitename, instance_selection_space, instance_selection_algorithm, dataset_name, fold):
        return (
            ckpt_folder
            / "evaluators"
            / str(instance_selection_space)
            / str(instance_selection_algorithm)
            / str(self.name)
            / str(suitename)
            / f"{dataset_name}_fold{fold}.npz"
        )

    def _checkpoint_entry_key(
        self,
        *,
        dataset_name,
        fold,
        instance_selection_space,
        instance_selection_algorithm,
        n_selected_samples,
        random_state,
    ):
        return "|".join(
            [
                str(dataset_name),
                str(int(fold)),
                str(instance_selection_space),
                str(instance_selection_algorithm),
                str(int(n_selected_samples)),
                self._normalize_checkpoint_value(random_state),
            ]
        )

    def _checkpoint_context_from_kwargs(self, kwargs):
        required = [
            "dataset_name",
            "fold",
            "instance_selection_space",
            "instance_selection_algorithm",
            "n_selected_samples",
            "suitename",
        ]
        if any(key not in kwargs for key in required):
            return None

        return {
            "dataset_name": kwargs["dataset_name"],
            "fold": kwargs["fold"],
            "instance_selection_space": kwargs["instance_selection_space"],
            "instance_selection_algorithm": kwargs["instance_selection_algorithm"],
            "n_selected_samples": kwargs["n_selected_samples"],
            "random_state": kwargs.get("random_state", None),
            "suitename": kwargs["suitename"],
        }

    def _load_logits_from_checkpoint(self, context):
        checkpoint_path = self._checkpoint_file_path(
            suitename=context["suitename"],
            instance_selection_space=context["instance_selection_space"],
            instance_selection_algorithm=context["instance_selection_algorithm"],
            dataset_name=context["dataset_name"],
            fold=context["fold"],
        )

        if not checkpoint_path.exists():
            return None

        entry_key = self._checkpoint_entry_key(
            dataset_name=context["dataset_name"],
            fold=context["fold"],
            instance_selection_space=context["instance_selection_space"],
            instance_selection_algorithm=context["instance_selection_algorithm"],
            n_selected_samples=context["n_selected_samples"],
            random_state=context["random_state"],
        )

        try:
            entries = self._load_npz_checkpoint_entries(checkpoint_path)
        except Exception as error:
            logger.warning("Could not read checkpoint %s (%s).", checkpoint_path, error)
            return None

        entry = entries.get(entry_key)
        if entry is None:
            return None

        logits, class_labels = entry
        if logits.ndim != 2 or class_labels.ndim != 1 or logits.shape[1] != len(class_labels):
            logger.warning("Invalid checkpoint entry in %s for key %s.", checkpoint_path, entry_key)
            return None

        logger.info("Loaded evaluator logits from checkpoint: %s | key=%s", checkpoint_path, entry_key)
        return logits, class_labels

    @staticmethod
    def _checkpoint_class_labels_array(class_labels):
        labels = np.asarray(class_labels)
        if labels.dtype == object:
            labels = labels.astype(str)
        return labels

    @staticmethod
    def _load_npz_checkpoint_entries(checkpoint_path):
        entries = {}
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            if "__keys__" not in payload:
                return entries

            keys = payload["__keys__"].astype(str).tolist()
            for idx, entry_key in enumerate(keys):
                logits_name = f"logits_{idx}"
                class_labels_name = f"class_labels_{idx}"
                if logits_name not in payload or class_labels_name not in payload:
                    continue
                entries[entry_key] = (
                    np.asarray(payload[logits_name], dtype=float),
                    np.asarray(payload[class_labels_name]),
                )
        return entries

    @staticmethod
    def _write_npz_checkpoint_entries(checkpoint_path, entries):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_name(f".{checkpoint_path.stem}.{os.getpid()}.tmp.npz")

        keys = list(entries.keys())
        payload = {"__keys__": np.asarray(keys, dtype=str)}
        for idx, entry_key in enumerate(keys):
            logits, class_labels = entries[entry_key]
            payload[f"logits_{idx}"] = np.asarray(logits, dtype=np.float32)
            payload[f"class_labels_{idx}"] = Evaluator._checkpoint_class_labels_array(class_labels)

        np.savez(tmp_path, **payload)
        tmp_path.replace(checkpoint_path)
        change_permissions_rw(checkpoint_path)

    def _save_logits_to_checkpoint(self, context, logits, class_labels):
        checkpoint_path = self._checkpoint_file_path(
            suitename=context["suitename"],
            instance_selection_space=context["instance_selection_space"],
            instance_selection_algorithm=context["instance_selection_algorithm"],
            dataset_name=context["dataset_name"],
            fold=context["fold"],
        )

        entry_key = self._checkpoint_entry_key(
            dataset_name=context["dataset_name"],
            fold=context["fold"],
            instance_selection_space=context["instance_selection_space"],
            instance_selection_algorithm=context["instance_selection_algorithm"],
            n_selected_samples=context["n_selected_samples"],
            random_state=context["random_state"],
        )

        entries = {}
        if checkpoint_path.exists():
            try:
                entries = self._load_npz_checkpoint_entries(checkpoint_path)
            except Exception as error:
                logger.warning("Could not read existing checkpoint %s before saving (%s). Recreating it.", checkpoint_path, error)

        entries[entry_key] = (
            np.asarray(logits, dtype=np.float32),
            self._checkpoint_class_labels_array(class_labels),
        )
        self._write_npz_checkpoint_entries(checkpoint_path, entries)

        logger.info("Saved evaluator logits checkpoint: %s | key=%s", checkpoint_path, entry_key)

    def _compute_metrics(self, y_train, y_test, logits, class_labels):
        all_classes = np.unique(np.concatenate([_to_numpy(y_train), _to_numpy(y_test)]))
        logits = _align_logits(logits, class_labels, all_classes)

        y_true = _to_numpy(y_test)
        y_pred = all_classes[np.argmax(logits, axis=1)]

        return {
            "AUC": _safe_auc(y_true, logits, all_classes),
            "ACC": float(accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "BalACC": float(balanced_accuracy_score(y_true, y_pred)),
        }

    def eval(self, x_train, y_train, x_test, y_test, **kwargs):
        total_start = perf_counter()
        train_classes = np.unique(_to_numpy(y_train))
        all_classes = np.unique(np.concatenate([_to_numpy(y_train), _to_numpy(y_test)]))
        if len(train_classes) == 1:
            logits = np.zeros((len(y_test), len(all_classes)), dtype=float)
            class_idx = np.where(all_classes == train_classes[0])[0][0]
            logits[:, class_idx] = 1.0
            metrics_start = perf_counter()
            metrics = self._compute_metrics(y_train, y_test, logits, all_classes)
            metrics_time = perf_counter() - metrics_start
            total_time = perf_counter() - total_start
            logger.info(
                "%s evaluator timing: one_class=%.3fs, metrics=%.3fs, total=%.3fs",
                self.name,
                total_time - metrics_time,
                metrics_time,
                total_time,
            )
            return metrics

        checkpoint_context = None
        cached = None
        checkpoint_load_time = 0.0
        if self.use_checkpoint:
            checkpoint_context = self._checkpoint_context_from_kwargs(kwargs)
            if checkpoint_context is not None:
                checkpoint_start = perf_counter()
                cached = self._load_logits_from_checkpoint(checkpoint_context)
                checkpoint_load_time = perf_counter() - checkpoint_start

        if cached is not None:
            logits, class_labels = cached
            metrics_start = perf_counter()
            metrics = self._compute_metrics(y_train, y_test, logits, class_labels)
            metrics_time = perf_counter() - metrics_start
            total_time = perf_counter() - total_start
            logger.info(
                "%s evaluator timing: checkpoint_load=%.3fs, metrics=%.3fs, total=%.3fs",
                self.name,
                checkpoint_load_time,
                metrics_time,
                total_time,
            )
            return metrics

        preprocess_start = perf_counter()
        x_train, y_train, x_test, y_test = self.preprocess(x_train, y_train, x_test, y_test)
        preprocess_time = perf_counter() - preprocess_start

        inference_start = perf_counter()
        logits, class_labels = self.predict_logits(x_train, y_train, x_test, y_test)
        inference_time = perf_counter() - inference_start

        checkpoint_save_time = 0.0
        if self.use_checkpoint and checkpoint_context is not None:
            checkpoint_start = perf_counter()
            self._save_logits_to_checkpoint(checkpoint_context, logits, class_labels)
            checkpoint_save_time = perf_counter() - checkpoint_start

        metrics_start = perf_counter()
        metrics = self._compute_metrics(y_train, y_test, logits, class_labels)
        metrics_time = perf_counter() - metrics_start
        total_time = perf_counter() - total_start
        logger.info(
            "%s evaluator timing: checkpoint_load=%.3fs, preprocess=%.3fs, inference=%.3fs, "
            "checkpoint_save=%.3fs, metrics=%.3fs, total=%.3fs",
            self.name,
            checkpoint_load_time,
            preprocess_time,
            inference_time,
            checkpoint_save_time,
            metrics_time,
            total_time,
        )
        return metrics


class _SklearnProbabilityEvaluator(Evaluator):
    def preprocess(self, x_train, y_train, x_test, y_test):
        x_train, x_test = _ensure_2d(x_train), _ensure_2d(x_test)
        x_train, x_test = one_hot_encode(x_train, x_test)
        return x_train, y_train, x_test, y_test

    def _build_model(self, x_train, y_train):
        raise NotImplementedError

    def predict_logits(self, x_train, y_train, x_test, y_test):
        model = self._build_model(x_train, y_train)
        model.fit(x_train, y_train)
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"{self.name} must expose predict_proba().")
        logits = model.predict_proba(x_test)
        class_labels = getattr(model, "classes_", np.unique(_to_numpy(y_train)))
        return logits, class_labels


class LR(_SklearnProbabilityEvaluator):
    def _build_model(self, x_train, y_train):
        return LogisticRegression(random_state=42, max_iter=1000)


class KNN(_SklearnProbabilityEvaluator):
    def __init__(self, device="cpu", dataset=None, n_neighbors=5, **params):
        super().__init__(device=device, dataset=dataset, n_neighbors=n_neighbors, **params)
        self.n_neighbors = int(n_neighbors)

    def _build_model(self, x_train, y_train):
        n_neighbors = max(1, min(self.n_neighbors, len(y_train)))
        return KNeighborsClassifier(n_neighbors=n_neighbors)


class XGBoost(_SklearnProbabilityEvaluator):
    def _build_model(self, x_train, y_train):
        device = self.params.get("device", self.device)
        normalized_device = "cuda" if str(device).startswith("cuda") else str(device)

        if normalized_device == "cpu":
            return xgb.XGBClassifier(
                eval_metric="logloss",
                enable_categorical=True,
                random_state=42,
                device="cpu",
            )

        return xgb.XGBClassifier(
            eval_metric="logloss",
            enable_categorical=True,
            random_state=42,
            device="cuda",
            tree_method="hist",
        )

    def predict_logits(self, x_train, y_train, x_test, y_test):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(_to_numpy(y_train))

        model = self._build_model(x_train, y_train_encoded)
        model.fit(x_train, y_train_encoded)

        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"{self.name} must expose predict_proba().")

        logits = model.predict_proba(x_test)
        class_labels = label_encoder.classes_
        return logits, class_labels


class TabPFNv2_5(_SklearnProbabilityEvaluator):
    def __init__(self, device="cpu", dataset=None, use_checkpoint=True, **params):
        super().__init__(device=device, dataset=dataset, use_checkpoint=use_checkpoint, **params)

    def preprocess(self, x_train, y_train, x_test, y_test):
        x_train, x_test = _ensure_2d(x_train), _ensure_2d(x_test)
        return x_train, y_train, x_test, y_test

    def _build_model(self, x_train, y_train):
        num_train_classes = len(np.unique(_to_numpy(y_train)))
        if num_train_classes > 10:
            logger.warning(
                "TabPFNv2.5 should not handle more than 10 classes (%s given). Running it anyway.",
                num_train_classes,
            )
        if x_train.shape[0] > 50000:
            logger.warning(
                "TabPFNv2.5 should not handle more than 50000 samples (%s given). Running it anyway.",
                x_train.shape[0],
            )
        if x_train.shape[1] > 2000:
            logger.warning(
                "TabPFNv2.5 should not handle more than 2000 features (%s given). Running it anyway.",
                x_train.shape[1],
            )
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            device=self.device,
            n_estimators=self.params.get("n_estimators", 8),
            ignore_pretraining_limits=True,
        )


class TabPFNv3(TabPFNv2_5):
    
    def _build_model(self, x_train, y_train):
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            device=self.device,
            n_estimators=self.params.get("n_estimators", 8),
            ignore_pretraining_limits=True,
        )


class TabICLv2(_SklearnProbabilityEvaluator):
    def __init__(self, device="cpu", dataset=None, use_checkpoint=True, **params):
        super().__init__(device=device, dataset=dataset, use_checkpoint=use_checkpoint, **params)

    def preprocess(self, x_train, y_train, x_test, y_test):
        x_train, x_test = _ensure_2d(x_train), _ensure_2d(x_test)
        return x_train, y_train, x_test, y_test

    def _build_model(self, x_train, y_train):
        # Load TabICLv2 with the default parameters
        return TabICLClassifier(
            n_estimators=8,  # number of ensemble members, more = better but slower
            norm_methods=None,  # normalization methods to try
            feat_shuffle_method="latin",  # feature permutation strategy
            class_shuffle_method="shift",  # class permutation strategy
            outlier_threshold=4.0,  # z-score threshold for outlier detection and clipping
            softmax_temperature=0.9,  # temperature to control prediction confidence
            average_logits=True,  # average logits (True) or probabilities (False)
            support_many_classes=True,  # handle >10 classes automatically
            batch_size=8,  # ensemble members processed together, lower to save memory
            kv_cache=False,  # cache training data KV projections for faster repeated inference
            model_path=None,  # path to checkpoint, None downloads from Hugging Face
            allow_auto_download=True,  # auto-download checkpoint if not found locally
            checkpoint_version="tabicl-classifier-v2-20260212.ckpt",  # pretrained checkpoint version. TabICLv2
            device=None,  # inference device, None auto-selects CUDA or CPU; specify "mps" for Apple Silicon
            use_amp="auto",  # automatic mixed precision for faster inference
            use_fa3="auto",  # Flash Attention 3 for Hopper GPUs (e.g. H100)
            offload_mode="auto",  # automatically decide when to use cpu/disk offloading
            disk_offload_dir=None,  # directory for disk offloading
            random_state=42,  # random seed for reproducibility
            n_jobs=None,  # number of PyTorch threads for CPU inference
            verbose=False,  # print detailed information during inference
            inference_config=None,  # fine-grained inference control for advanced users
        )


class UCD(Evaluator):
    def __init__(self, device="cpu", dataset=None, name=None, **params):
        super().__init__(device, dataset, name, **params)
        self.metrics = ["UCD"]

    # Cold PAWS
    def eval(self, x_train, y_train, x_test, y_test, num_classes, **kwargs):
        num_y_train_classes = int(len(np.unique(_to_numpy(y_train))))
        return {"UCD": num_y_train_classes / num_classes}

    def predict_logits(self, x_train, y_train, x_test, y_test):
        raise NotImplementedError("UCD does not predict logits.")
