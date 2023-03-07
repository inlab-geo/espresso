import errno
import os
import signal
import functools
import contextlib
import sys

# Thanks to: https://stackoverflow.com/a/2282656
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        if seconds is None:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

        return wrapper

    return decorator

# Thanks to: https://stackoverflow.com/a/25061573
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
