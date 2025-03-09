# oneFEM/__main__.py

from ._systools.cli import _cli_main

if __name__ == "__main__":
    """
    Allow Python command line call as: 
    >>> python -m oneFEM --license
    >>> python -m oneFEM ./main.py
    """
    _cli_main()