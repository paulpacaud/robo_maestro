import logging
import sys
import inspect


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

def get_call_chain(max_depth: int = None):
    """
    Walk the call stack, skipping frames inside this logger module,
    and return a string like:
      RunPolicyNode.run -> BaseEnv.reset -> Robot.reset -> Observer.wait_for_message
    """
    stack = inspect.stack()
    chain = []
    for frame_info in stack[2:]:   # skip the logger frames themselves
        module = inspect.getmodule(frame_info.frame)
        # skip any frames coming from this logger file
        if module and module.__name__.endswith("utils.logger"):
            continue

        func = frame_info.function
        # if this was a method, grab its class name
        cls = ""
        if 'self' in frame_info.frame.f_locals:
            cls = type(frame_info.frame.f_locals['self']).__name__ + "."

        chain.append(f"{cls}{func}")
        if max_depth and len(chain) >= max_depth:
            break

    # reverse so that the outermost caller is first
    chain.reverse()
    return " -> ".join(chain)

def log_info(msg):
    call_chain = get_call_chain()
    # chain uncolored, message in green
    color_logger.info(f"[{call_chain}] {GREEN}{msg}{RESET}")

def log_warn(msg):
    call_chain = get_call_chain()
    # chain uncolored, message in yellow
    color_logger.warning(f"[{call_chain}] {YELLOW}{msg}{RESET}")

def log_error(msg):
    call_chain = get_call_chain()
    # chain uncolored, message in red
    color_logger.error(f"[{call_chain}] {RED}{msg}{RESET}")

def log_debug(msg):
    call_chain = get_call_chain()
    # chain uncolored, message in cyan
    color_logger.debug(f"[{call_chain}] {CYAN}{msg}{RESET}")
