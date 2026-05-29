import ast
import importlib
from importlib import resources
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from lucos.utils.paths import change_permissions_rw, datasets_folder, project_folder, results_folder


logger = logging.getLogger("lucos")


def parse_folds_arg(folds_arg: str) -> list[int]:
    """Parse folds list passed by CLI, e.g. "[0,1,2]" or "0,1,2"."""
    try:
        parsed = ast.literal_eval(folds_arg)
    except (ValueError, SyntaxError):
        parsed = [token.strip() for token in folds_arg.split(",") if token.strip()]

    if isinstance(parsed, int):
        parsed = [parsed]
    elif isinstance(parsed, (tuple, set)):
        parsed = list(parsed)

    if not isinstance(parsed, list):
        raise ValueError(f"Invalid folds format: {folds_arg}. Use e.g. '[0,1,2]' or '0,1,2'.")

    normalized = []
    seen = set()
    for fold in parsed:
        fold_int = int(fold)
        if fold_int < 0:
            raise ValueError(f"Fold index must be >= 0. Got {fold_int}.")
        if fold_int not in seen:
            normalized.append(fold_int)
            seen.add(fold_int)

    if not normalized:
        raise ValueError("At least one fold must be provided.")

    return normalized


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_yaml_config(config_name_or_path: str, package: str, *resource_parts: str, repo_relative_dir: str | Path | None = None):
    """Load a YAML config from an explicit path, the source tree, or package data."""
    requested_path = Path(config_name_or_path).expanduser()
    if requested_path.is_file():
        return OmegaConf.load(requested_path)

    if repo_relative_dir is not None and not requested_path.is_absolute():
        repo_path = project_folder / Path(repo_relative_dir) / config_name_or_path
        if repo_path.is_file():
            return OmegaConf.load(repo_path)

    if requested_path.is_absolute():
        raise FileNotFoundError(f"Config file not found: {requested_path}")

    resource = resources.files(package).joinpath(*resource_parts, config_name_or_path)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Config '{config_name_or_path}' was not found as a file or package resource "
            f"under {package}:{'/'.join(resource_parts)}."
        )

    with resource.open("r", encoding="utf-8") as handle:
        return OmegaConf.load(handle)


def load_config(config_file):
    config = load_yaml_config(
        config_file,
        "lucos.unsupervised_context_selection",
        "config",
        repo_relative_dir=Path("src") / "lucos" / "unsupervised_context_selection" / "config",
    )

    if config.experiment.exp_id == "now":
        config.experiment.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.experiment.device == "default":
        config.experiment.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if config.encoder.params.device == "default":
        config.encoder.params.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config.encoder.name = get_obj_from_str(config.encoder.target).get_name(config.encoder.params)
    config.results_basename = f"results_{config.encoder.name}_{config.experiment.sampling_method}_{config.experiment.exp_id}"

    config.project_folder = str(project_folder)
    config.results_folder = str(results_folder)
    config.datasets_folder = str(datasets_folder)
    if config.encoder.params is not None and "project_folder" in config.encoder.params:
        config.encoder.params.project_folder = str(project_folder)

    config.dataset.datasets_filename = str(datasets_folder / config.dataset.datasets_basename)
    config.results_filename = str(results_folder / config.results_basename)

    return config


def save_config(config):
    config_folder = Path(config.results_folder) / "yamls"
    os.makedirs(config_folder, exist_ok=True)
    config_path = config_folder / f"{config.results_basename}.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    change_permissions_rw(config_path)
    logger.info("Configuration saved to %s", config_path)
