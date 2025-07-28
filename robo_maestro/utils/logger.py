import logging
import sys

# ANSI escape codes for coloring
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

# Setup the logger
color_logger = logging.getLogger("robo_maestro_colored_logger")
color_logger.setLevel(logging.DEBUG)

if not color_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    color_logger.addHandler(handler)
    color_logger.propagate = False

# Convenience functions
def log_info(msg): color_logger.info(f"{GREEN}{msg}{RESET}")
def log_warn(msg): color_logger.warning(f"{YELLOW}{msg}{RESET}")
def log_error(msg): color_logger.error(f"{RED}{msg}{RESET}")
def log_debug(msg): color_logger.debug(f"{CYAN}{msg}{RESET}")
