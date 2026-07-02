import time
import types

start = time.time_ns()
global_verbose = False

colorized = True
time_color = "\033[94m" if colorized else ""  # blue
class_color = "\033[93m" if colorized else ""  # yellow
reset_color = "\033[0m" if colorized else "" # white
error_color = "\033[31m" if colorized else "" # red


def print_verbose(obj, message, local_verbose: bool = False, error: bool = False):
    t = (time.time_ns() - start) / 1e9
    
    # 1. Dynamically extract the name based on what 'obj' is
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        # It's a standalone function or method
        name = obj.__name__
    elif hasattr(obj, "__class__") and obj.__class__.__name__ != "str":
        # It's a class instance
        name = obj.__class__.__name__
    else:
        # Fallback if a plain string was passed as the identifier
        name = str(obj)

    formatted_time = f"{time_color}{t:>8.3f}{reset_color}"
    formatted_name = f"{class_color}{name:<15}{reset_color}"
    
    message = f"{formatted_time} {formatted_name} {error_color if error else reset_color}{message}"

    # 2. Check all verbose conditions
    if global_verbose:
        print(message)
    elif local_verbose is True or error is True:
        print(message)
    elif hasattr(obj, "verbose") and isinstance(obj.verbose, bool) and obj.verbose:
        print(message)