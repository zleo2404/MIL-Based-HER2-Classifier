"""Consistent logging setup shared by all entry-point scripts."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(run_dir: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the 'her2_mil' logger with a console handler and, if
    `run_dir` is given, a file handler writing to `<run_dir>/run.log`."""
    logger = logging.getLogger("her2_mil")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("her2_mil")
