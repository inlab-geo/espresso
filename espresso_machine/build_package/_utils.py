import errno
import os
import signal
import functools
import contextlib
import sys
import tqdm

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

# Thanks to: 
# https://stackoverflow.com/a/25061573
# https://stackoverflow.com/a/72617364
@contextlib.contextmanager
def suppress_stdout():
    def tqdm_replacement(iterable_object,*args,**kwargs):
        return iterable_object

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        tqdm_copy = tqdm.tqdm # store it if you want to use it later
        tqdm.tqdm = tqdm_replacement
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            tqdm.tqdm = tqdm_copy
