import os
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from lucos.utils.paths import project_folder, ckpt_folder, change_permissions_rw
from types import SimpleNamespace

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


sup_str = {True: 'Sup', False: 'UnSup'}

class Encoder:
    
    @staticmethod
    def get_name(config):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __init__(self, **kwargs):
        self.name = self.get_name(kwargs)
        self.space_name = self.name.replace("Encoder", "Space")
        self.use_cache = bool(kwargs.get("use_cache", True))
        self.supervised = kwargs.get("supervised", True)
        self.device = kwargs.get("device", 'cuda:0') if torch.cuda.is_available() else 'cpu'
        self.dataset = kwargs.get("dataset", None)

        if self.dataset is None:
            self.dataset = SimpleNamespace(name_and_fold="unknown_dataset", suitename="unknown_suitename", cat_cols=[], num_cols=[], bin_cols=[])
            self.ckpt_folder = None
            self.cache_path = None
            self.log_prefix = f"{self.dataset.name_and_fold}. {self.name}."
            logger.warning(
                "%s No dataset information provided. Caching will be disabled and some encoders may not work properly.",
                self.log_prefix,
            )
        else:
            self.log_prefix = f"{self.dataset.name_and_fold}. {self.name}."
            if self.use_cache:
                self.ckpt_folder = ckpt_folder / "encoders" / self.name / self.dataset.suitename
                os.makedirs(self.ckpt_folder, exist_ok=True)
                self.cache_path = self.ckpt_folder / f"{self.dataset.name_and_fold}.npy"
            else:
                self.ckpt_folder = None
                self.cache_path = None

        if self.use_cache:
            logger.info("%s use_cache=True. Embedding cache load/save is enabled when dataset context is available.", self.log_prefix)
        else:
            logger.warning("%s use_cache=False. Embeddings will always be recomputed and cache I/O is disabled.", self.log_prefix)

    def _can_use_cache(self) -> bool:
        return self.use_cache and self.cache_path is not None

    def fit(self, x_train=None, y_train=None, **kwargs):
        self.x_train = x_train
        self.y_train = y_train

    def _cache_path_for_split(self, split: str) -> Path | None:
        if not self._can_use_cache() or self.ckpt_folder is None:
            return None
        return self.ckpt_folder / f"{self.dataset.name_and_fold}_{split}.npy"

    def _load_cached_embeddings(self) -> np.ndarray | None:
        if not self._can_use_cache():
            if not self.use_cache:
                logger.info("%s Skipping cache load because use_cache=False. Embeddings will be recomputed.", self.log_prefix)
            return None
        if self.cache_path.exists():
            logger.info("%s Loading precomputed encodings from %s", self.log_prefix, self.cache_path)
            self.output = np.load(self.cache_path)
            return self.output
        return None

    def _save_cached_embeddings(self, output: np.ndarray) -> None:
        if not self._can_use_cache():
            return
        np.save(self.cache_path, output)
        logger.info("%s Encodings saved to %s", self.log_prefix, self.cache_path)

    def _load_cached_split_embeddings(self, split: str) -> np.ndarray | None:
        # This is used for PCA and t-SNE
        if not self._can_use_cache():
            if not self.use_cache:
                logger.info(
                    "%s Skipping %s cache load because use_cache=False. Embeddings will be recomputed.",
                    self.log_prefix,
                    split,
                )
            return None
        cache_path = self._cache_path_for_split(split)
        if cache_path is None:
            return None
        if cache_path.exists():
            logger.info("%s Loading %s encodings from %s", self.log_prefix, split, cache_path)
            self.output = np.load(cache_path)
            return self.output
        return None

    def _save_cached_split_embeddings(self, output: np.ndarray, split: str) -> None:
        # This is used for PCA and t-SNE
        if not self._can_use_cache():
            return
        cache_path = self._cache_path_for_split(split)
        if cache_path is None:
            return
        np.save(cache_path, output)
        logger.info("%s %s encodings saved to %s", self.log_prefix, split, cache_path)
        change_permissions_rw(cache_path)


    def fit_transform_with_test(self, x_train, x_test):

        self.dataset.name_and_fold += "_train_test"

        train_idx = np.arange(len(x_train))
        test_idx = np.arange(len(x_train), len(x_train) + len(x_test))

        X_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
        Z_all = self.fit_transform(X_all)

        Z_train = Z_all[train_idx]
        Z_test  = Z_all[test_idx]

        self.dataset.name_and_fold = self.dataset.name_and_fold.replace("_train_test", "")

        return Z_train, Z_test


class DummyEncoder(Encoder):
    
    @staticmethod
    def get_name(config):
        return "DummyEncoder"


class OriginalSpaceEncoder(Encoder):
    @staticmethod
    def get_name(config):
        return f"OriginalEncoder"

    def transform(self, X, **kwargs):
        if isinstance(X, np.ndarray):
            return X.astype(np.float32)
        elif isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        else:
            raise ValueError("Unsupported data type for OriginalSpaceEncoder. Supported types are np.ndarray and pd.DataFrame.")

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X, **kwargs)



class TabClustPFNEncoder(Encoder):

    """
    It encoded all datasets in a reasonable time, except MNIST, which took 13 minutes
    """
    @staticmethod
    def get_name(config):
        return f"TabClustPFNEncoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supervised = False
        self.device = kwargs.get("device", "cuda:0")
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            self.device = "cpu"

        default_checkpoint = project_folder / "libs" / "TabClustPFN" / "checkpoints" / "step-10000.ckpt"
        self.checkpoint_path = Path(kwargs.get("checkpoint_path", str(default_checkpoint)))
        self.num_clusters = kwargs.get("num_clusters", None)
        self.predict_k = kwargs.get("predict_k", self.num_clusters is None)
        self.nan_strategy = kwargs.get("nan_strategy", "zero")
        self.normalization = kwargs.get("normalization", "z-norm")
        self.remove_zero_variance = kwargs.get("remove_zero_variance", True)

        self.model = None
        self.preprocessor = None
        self.max_classes = None

    def _ensure_tabclust_imports(self):
        import importlib
        import sys

        tabclust_root = project_folder / "libs" / "TabClustPFN"
        tabclust_root_str = str(tabclust_root)
        if tabclust_root_str not in sys.path:
            sys.path.insert(0, tabclust_root_str)

        tabcluster_module = importlib.import_module("model.tabcluster")
        preprocess_module = importlib.import_module("utils.preprocess")

        TabClusterIMAB = getattr(tabcluster_module, "TabClusterIMAB")
        DataPreprocessor = getattr(preprocess_module, "DataPreprocessor")

        return TabClusterIMAB, DataPreprocessor

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TabClustPFN checkpoint not found: {self.checkpoint_path}. "
                "Provide encoder.params.checkpoint_path in the config."
            )

        TabClusterIMAB, DataPreprocessor = self._ensure_tabclust_imports()

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(f"Invalid TabClustPFN checkpoint: {self.checkpoint_path}")

        model = TabClusterIMAB(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.max_classes = int(getattr(model, "max_classes", 10))
        self.preprocessor = DataPreprocessor(
            nan_strategy=self.nan_strategy,
            normalization=self.normalization,
            remove_zero_variance=self.remove_zero_variance,
        )

    def fit(self, x_train, y_train, **kwargs):
        super().fit(x_train, y_train, **kwargs)
        if self.model is None or self.preprocessor is None:
            self._load_model()

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return pd.DataFrame(np.asarray(X))

    def _preprocess(self, X):
        X_df = self._to_dataframe(X)
        X_np = self.preprocessor.fit_transform(X_df)
        X_tensor = torch.from_numpy(X_np.astype(np.float64)).float().to(self.device).unsqueeze(0)
        return X_tensor

    def fit_transform(self, X, **kwargs):
        super().fit(X, y_train=None, **kwargs)

        cached = self._load_cached_embeddings()
        if cached is not None:
            return cached

        if self.model is None or self.preprocessor is None:
            self._load_model()

        X_tensor = self._preprocess(X)

        with torch.no_grad():
            col_embeddings = self.model.col_embedder(X_tensor)
            row_representations = self.model.row_interactor(col_embeddings)

        self.output = row_representations.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self._save_cached_embeddings(self.output)
        return self.output
