import logging

from src.run_pipeline import run_pipeline
from utils.logger import setup_logging
from utils.paths import CONFIG
from omegaconf import OmegaConf


def main() -> int:
    # ------------------------------------------------------------------
    # 1) Load configuration
    # ------------------------------------------------------------------
    cfg = OmegaConf.load(CONFIG / "config.yaml")

    # ------------------------------------------------------------------
    # 2) Setup logging
    # ------------------------------------------------------------------
    level_str = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    setup_logging(level=level)

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded from %s", CONFIG)
    logger.info("Log level set to %s", logging.getLevelName(level))

    # ------------------------------------------------------------------
    # 3) Run pipeline
    # ------------------------------------------------------------------
    logger.info("Starting anomaly detection pipeline")
    run_pipeline(cfg)
    logger.info("Pipeline finished")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
