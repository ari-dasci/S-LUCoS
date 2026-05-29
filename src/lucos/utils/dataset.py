import os
from lucos.utils.paths import datasets_folder
os.environ["OPENML_CACHE_DIR"] = str(datasets_folder / "openml")
import openml
import logging
import numpy as np
import pandas as pd
from lucos.utils.evaluation import imbalance_ratio
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


def _save_datasets_info_excel(path, datasets_info: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets_info.to_excel(path, index=False)


class Benchmark:
    """
    OpenML-CC18 or tabarena-v0.1
    """
    def __init__(self, suitename, only_classification=True, impute_nan=True, min_samples=None, max_samples=None, min_features=None, max_features=None, max_classes=None, lite=False, ascending=True, verbose=True, **kwargs):
        self.suitename = suitename
        # In the future we could also allow regression.
        # We would need to rename the AUC column and add a 'metric' column indicating which metric was used (AUC or RMSE?, R2?)
        # but the number of instances would no longer be chosen based on the number of classes.
        self.only_classification = only_classification
        self.impute_nan = impute_nan
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.lite = lite
        self.verbose = verbose
        self.ascending = ascending
        self.excel_path = datasets_folder / f'{self.suitename}.xlsx'
        self.suite = None
        self.tasks = []

        if suitename == "OpenML-CC18":
            self.n_folds = 10
        elif suitename == "tabarena-v0.1":
            self.n_folds = 3
        else:
            self.n_folds = None

        suite_info_found = self.load_suite_info()
        if suite_info_found:
            self.tasks = self.datasets_info["task_id"].dropna().astype(int).tolist()
            if self.verbose:
                logger.info(f"Loaded suite info from {self.excel_path}. Skipping OpenML suite metadata request.")
        else:
            # only once
            if self.verbose:
                logger.info(f"Suite info not found in {self.excel_path}. Downloading suite and saving info in excel file.")
            self._load_suite_from_openml()
            self.download_suite()
        
        self.filter_datasets_info()

    def _load_suite_from_openml(self):
        try:
            self.suite = openml.study.get_suite(self.suitename)
            self.tasks = self.suite.tasks
        except Exception as exc:
            logger.warning(
                "Could not load OpenML suite metadata for %s. If %s exists, it will be used to run offline; "
                "otherwise the benchmark cannot be initialized. Original error: %s",
                self.suitename,
                self.excel_path,
                exc,
            )
            raise

    def download_suite(self):
        #Download the suite of datasets in the OpenML cache folder and save the datasets info in an excel file.

        columns = [
            'task_id',
            'dataset_id',
            'dataset_name',
            'task_type',
            'n_samples',
            'n_features',
            'cols_num',
            'cols_cat',
            'cols_bin',
            'n_classes',
            'imbalance_ratio',
            'nan_ratio',
            'split_repeats',
            'split_folds',
            'split_samples',
        ]
        rows = []
        if self.verbose:
            logger.info(f"Downloading suite {self.suitename} and saving info in excel file.")
        for task_id in self.tasks:    
            dataset = Dataset(self.suitename, task_id, self.lite, self.impute_nan)

            rows.append([
                task_id,
                dataset.id,
                dataset.name,
                dataset.task_type,
                dataset.X.shape[0],
                dataset.X.shape[1],
                len(dataset.num_cols),
                len(dataset.cat_cols),
                len(dataset.bin_cols),
                dataset.num_classes,
                dataset.imbalance_ratio,
                dataset.nan_ratio,
                dataset.n_repeats,
                dataset.full_folds,
                dataset.split_samples,
            ])

        self.datasets_info = pd.DataFrame(rows, columns=columns)

        _save_datasets_info_excel(self.excel_path, self.datasets_info)
        if self.verbose:
            logger.info(f"Suite {self.suitename} downloaded and info saved in {self.excel_path}.")


    def load_suite_info(self):
        try:
            self.datasets_info = pd.read_excel(self.excel_path)
            return True
        except FileNotFoundError:
            return False


    def filter_datasets_info(self):
        """
        Get datasets from the excel file with the specified conditions.
        """
        self.datasets_info_filtered = self.datasets_info.copy()
        num_total_datasets = len(self.datasets_info_filtered)

        if self.only_classification:
            self.datasets_info_filtered = self.datasets_info_filtered[self.datasets_info_filtered['task_type'] == 'classification']
        if self.min_samples is not None:
            self.datasets_info_filtered = self.datasets_info_filtered[(self.datasets_info_filtered['n_samples'] >= self.min_samples)]
        if self.max_samples is not None:
            self.datasets_info_filtered = self.datasets_info_filtered[(self.datasets_info_filtered['n_samples'] < self.max_samples)]
        if self.min_features is not None:
            self.datasets_info_filtered = self.datasets_info_filtered[(self.datasets_info_filtered['n_features'] >= self.min_features)]
        if self.max_features is not None:
            self.datasets_info_filtered = self.datasets_info_filtered[(self.datasets_info_filtered['n_features'] <= self.max_features)]
        if self.max_classes is not None:
            self.datasets_info_filtered = self.datasets_info_filtered[self.datasets_info_filtered['n_classes'] <= self.max_classes]

        self.datasets_info_filtered = self.datasets_info_filtered.sort_values(by="n_samples", ascending=self.ascending).reset_index(drop=True)
        self.num_of_filtered_datasets = len(self.datasets_info_filtered)

        if self.verbose:
            logger.info(f"Number of datasets: {len(self.datasets_info_filtered)}/{num_total_datasets}")
            logger.info("DATASETS: %s", self.datasets_info_filtered["dataset_name"].values)


    def get_datasets(self):
        for task_id in self.datasets_info_filtered['task_id'].values:
            dataset = Dataset(self.suitename, int(task_id), self.lite, self.impute_nan)
            yield dataset


class Dataset:

    def __init__(self, suitename, task_id, lite=False, impute_nan=False):
        self.suitename = suitename
        self.task = openml.tasks.get_task(task_id)
        self.task_type = Dataset._resolve_task_type(self.task)
        self.dataset = openml.datasets.get_dataset(self.task.dataset_id)
        self.id = self.dataset.dataset_id
        self.name = self.dataset.name
        self.fold = None
        self.lite = lite
        self.impute_nan = impute_nan
        self.n_repeats, self.full_folds, self.split_samples  = self.task.get_split_dimensions()
        self.n_folds = self.full_folds
        if lite:
            logger.info(f"Using lite version of dataset {self.name} with only 1 fold and 1 repeat. Original was {self.n_folds} folds, {self.n_repeats} repeats and {self.split_samples} samples.")
            self.n_folds = 1
            self.n_repeats = 1
            self.split_samples = 1

        self.name_and_fold = f"{self.name}_foldX"

        self.X, self.y, categorical_indicator, attribute_names = self.dataset.get_data(dataset_format='dataframe', target=self.dataset.default_target_attribute)
        
        self.x_train_size = int(np.ceil(self.X.shape[0] / self.n_folds * (self.n_folds-1))) # training size for one fold

        if self.task_type == "classification":
            self.num_classes = len(self.y.unique())
            self.imbalance_ratio = imbalance_ratio(self.y)
        else: 
            self.num_classes = None
            self.imbalance_ratio = None

        self.nan_ratio = self.X.isnull().mean().mean()

        # log a warning if there is a column with all values the same, because it doesn't make sense to keep it and it can cause problems with some algorithms (like PCA)
        cols_with_same_value = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                cols_with_same_value.append(col)
        if cols_with_same_value:
            logger.info(f"dataset {self.name} has {len(cols_with_same_value)} columns with all values the same: {cols_with_same_value}. Consider removing them.")            

        all_cols = np.array(attribute_names)
        categorical_indicator = np.array(categorical_indicator)
        
        cat_cols = [col for col in all_cols[categorical_indicator] ]
        num_cols = [col for col in all_cols[~categorical_indicator] ]

        # CAUTION: Marketing_Campaign has 1 non-numeric columns marked as numeric by OpenML: ['Dt_Customer']. Treating them as categorical.
        # Casting column Dt_Customer: num of distinct values: 663!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # OpenML sometimes flags date/text columns as numeric. Keep only real numeric columns here
        # so later mean-based imputation is applied exclusively to numeric data.
        non_numeric_num_cols = [c for c in num_cols if not pd.api.types.is_numeric_dtype(self.X[c])]
        if non_numeric_num_cols:
            logger.info(f"dataset {self.name} has {len(non_numeric_num_cols)} non-numeric columns marked as numeric by OpenML: {non_numeric_num_cols}. Treating them as categorical.")
            cat_cols.extend(non_numeric_num_cols)
            num_cols = [c for c in num_cols if c not in non_numeric_num_cols]
            for new_cat_col in non_numeric_num_cols:
                logger.info(f"Casting column {new_cat_col}: num of distinct values: {self.X[new_cat_col].nunique()}")

        bin_cols = [c for c in cat_cols if (self.X[c].dropna().nunique() <= 2) and (self.X[c].dropna().astype(str).str.lower().isin(['yes', 'no', 'true', 'false', 't', 'f', '1', '0'])).all()]
        cat_cols = [c for c in cat_cols if c not in bin_cols]

        # We assume there are no classes in test that are absent from train
        self.y = LabelEncoder().fit_transform(self.y.values)
        self.y = pd.Series(self.y, index=self.X.index)

        self.cat_cols = [str(c) for c in cat_cols] # np.str_ to str
        self.num_cols = [str(c) for c in num_cols]
        self.bin_cols = [str(c) for c in bin_cols]
    
    @staticmethod
    def _resolve_task_type(task):
        """
        return "classification", "regression", or "unknown_{id}" based on the task's type information.
        """
        raw_task_type = getattr(task, "task_type_id", None)
        if raw_task_type is None:
            raw_task_type = getattr(task, "task_type", None)

        task_type_id = -1

        # OpenML can expose task type as an int, a string, or a TaskType enum/object.
        try:
            task_type_id = int(raw_task_type)
        except (TypeError, ValueError):
            for attr in ("value", "id", "task_type_id"):
                attr_value = getattr(raw_task_type, attr, None)
                if attr_value is None:
                    continue
                try:
                    task_type_id = int(attr_value)
                    break
                except (TypeError, ValueError):
                    continue

        if task_type_id == -1 and raw_task_type is not None:
            task_type_name = str(getattr(raw_task_type, "name", raw_task_type)).lower()
            if "classification" in task_type_name:
                task_type_id = 1
            elif "regression" in task_type_name:
                task_type_id = 2

        if task_type_id == 1:
            return "classification"
        if task_type_id == 2:
            return "regression"
        return f"unknown_{task_type_id}"

    def get_data_of_fold(self, fold, numpy_arrays=False, subsample_to=None, subsample_random_state=42):
        self.fold = fold
        self.name_and_fold = f"{self.name}_fold{fold}"
        train_indices, test_indices = self.task.get_train_test_split_indices(fold=fold)
        x_train, y_train = (self.X.iloc[train_indices].copy(), self.y.iloc[train_indices].copy())
        x_test, y_test = (self.X.iloc[test_indices].copy(), self.y.iloc[test_indices].copy())
        
        # Subsample the training data if specified
        if subsample_to is not None and subsample_to < len(x_train):
            sampled_indices = x_train.sample(n=subsample_to, random_state=subsample_random_state).index
            x_train = x_train.loc[sampled_indices].reset_index(drop=True)
            y_train = y_train.loc[sampled_indices].reset_index(drop=True)
            self.name_and_fold += f"_subsample{subsample_to}_seed{subsample_random_state}" 
            self.name += f"_subsample{subsample_to}_seed{subsample_random_state}"

        # Some datasets have folds where test contains new categories absent from train. Example: dresses-sales Fold 0 col V8
        for col in self.cat_cols + self.bin_cols: 
            # Cast categorical and binary variables to pd.Categorical.
            # TabPFN2.5-Plus can handle text directly, but with v2.5 it must be converted to numeric categorical data. https://docs.priorlabs.ai/models
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
                dtype=float,
            )

            x_train[col] = encoder.fit_transform(x_train[[col]]).ravel()
            x_test[col] = encoder.transform(x_test[[col]]).ravel()

        if self.impute_nan:
            # TabPFN and TabClustPFN accept NaNs, but KMedoids does not.
            # The model that does not accept NaNs should decide whether to impute and how to do it.
            # FOR NOW I WILL KEEP NANS IMPUTED BECAUSE MAINTAINING IMPUTED AND NON-IMPUTED VERSIONS FOR DIFFERENT ALGORITHMS IS TOO MUCH WORK
            logger.info(f"Imputing NaN values in dataset {self.name} using mode for categorical and binary columns and mean for numerical columns.")
            for col in self.cat_cols + self.bin_cols:
                # for categorical and binary variables, fill with the train mode
                train_mode = x_train[col].mode()[0]
                x_train[col] = x_train[col].fillna(train_mode)
                x_test[col] = x_test[col].fillna(train_mode)
            for col in self.num_cols:
                # for numerical variables, fill with the train mean
                train_mean = x_train[col].mean()
                x_train[col] = x_train[col].fillna(train_mean)
                x_test[col] = x_test[col].fillna(train_mean)

        if numpy_arrays:
            x_train = x_train.values
            y_train = y_train.values
            x_test = x_test.values
            y_test = y_test.values
            
        return x_train, y_train, x_test, y_test


def get_benchmark_by_list_of_datasets(dataset_names, **benchmark_kwargs):
    """
    reurns the benchmark containing all the datasets in dataset_names. If there is no benchmark containing all the datasets, returns None.
    """
    b1 = Benchmark("OpenML-CC18", verbose=False, **benchmark_kwargs)
    b2 = Benchmark("tabarena-v0.1", verbose=False, **benchmark_kwargs)
    if set(dataset_names).issubset(set(b1.datasets_info['dataset_name'].values)):
        logger.info("Benchmark detected: OpenML-CC18")
        return b1
    elif set(dataset_names).issubset(set(b2.datasets_info['dataset_name'].values)):
        logger.info("Benchmark detected: tabarena-v0.1")
        return b2
    else:
        logger.warning("No benchmark found containing all specified datasets.")
        return None
    

if __name__ == "__main__":
    # Testing
    benchmark = Benchmark("OpenML-CC18")
    benchmark = Benchmark("tabarena-v0.1")
