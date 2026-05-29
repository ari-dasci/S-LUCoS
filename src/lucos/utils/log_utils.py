import logging
import logging.handlers
import getpass
from pathlib import Path
from lucos.utils.paths import log_folder, change_permissions_rw


def _add_user_to_filename(log_file: str) -> str:
    log_path = Path(log_file)
    username = getpass.getuser()

    if log_path.suffix:
        return f"{log_path.stem}_{username}{log_path.suffix}"
    return f"{log_path.name}_{username}"


def setup_logging(log_file="app.log", mp=False, log_level=logging.DEBUG):
    
    if mp:
        format = "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"
    else:
        format = "%(asctime)s | %(levelname)s | %(message)s"

    filename = log_folder / _add_user_to_filename(log_file)
    filename.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        str(filename), maxBytes=5_000_000, backupCount=5
    )

    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set root logger to WARNING to suppress external library logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Configure lucos logger to DEBUG and add handlers only to it
    logger = logging.getLogger("lucos")
    logger.setLevel(log_level)
    
    # Remove old handlers from lucos logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers only to lucos logger (not to root)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    change_permissions_rw(filename)
