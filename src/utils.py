import logging
import logging.config


def setup_logging(config):
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")