import logging
import os
import sys

# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.


def setup_logger(
    log_name="pipeline",
    log_path="ml_pipeline/",
    level=logging.DEBUG,
    log_format="%(asctime)s: %(levelname)s: %(message)s @ %(pathname)s running method %(funcName)s on line: %(lineno)d",
    count_files=True,
    console_log_level=100,
    dummy_logger=False,
):
    """Basic logger.

    References
    ----------
    https://www.youtube.com/watch?v=-ARI4Cz-awo
    https://www.youtube.com/watch?v=jxmzY9soFXg

    https://docs.python.org/3.8/library/logging.html#logging-levels
    https://docs.python.org/3/library/logging.html#logrecord-attributes

    """

    if dummy_logger:
        return logging.getLogger("dummy")

    if count_files:
        no_of_files = len(
            [f for f in os.listdir(log_path) if f.endswith(".log") and os.path.isfile(os.path.join(log_path, f))]
        )
        log_file = log_path + log_name + str(no_of_files) + ".log"
    else:
        log_file = log_path + log_name + ".log"

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_log_level)

    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
