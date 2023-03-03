import os, errno


def silent_remove(filename):
    r"""Remove a file if exists or do nothing otherwise

    Parameters
    ----------
    filename : Path | str
        The absoluate path of the file you want to remove
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
