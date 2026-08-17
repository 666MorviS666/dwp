"""Точка входа: `python -m dwp` открывает интерактивное меню."""

import sys

from .tui import main

if __name__ == "__main__":
    sys.exit(main())
