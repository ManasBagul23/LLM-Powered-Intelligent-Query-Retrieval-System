import logging
from typing import Optional


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    log_level = (level or "INFO").upper()
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("rag_platform")
    logger.setLevel(log_level)
    return logger
