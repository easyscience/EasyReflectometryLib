# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Lightweight table rendering helpers."""

from __future__ import annotations


def in_notebook() -> bool:
    """Return ``True`` when running inside a Jupyter notebook or IPython environment.

    :return: Whether the current environment is a notebook.
    :rtype: bool
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell':
            return True
    except ImportError:
        pass
    return False


def render_table(rows: list[dict], columns: list[str]) -> object:
    """Render a list of dicts as a table.

    In notebooks, returns a pandas ``DataFrame`` for rich display.
    In terminals, prints a plain-text table.  If Rich is available a
    richer rendering is used; otherwise a simple aligned format is printed.

    :param rows: List of row dicts, one dict per row.
    :type rows: list[dict]
    :param columns: Ordered list of column names to include.
    :type columns: list[str]
    :return: A pandas ``DataFrame`` when in a notebook, otherwise ``None``.
    :rtype: object
    """
    if in_notebook():
        try:
            import pandas as pd

            return pd.DataFrame(rows, columns=columns)
        except ImportError:
            pass

    # Terminal fallback — print a formatted table
    _print_aligned_table(rows, columns)
    return None


def _print_aligned_table(rows: list[dict], columns: list[str]) -> None:
    """Print a plain-text aligned table.

    :param rows: List of row dicts.
    :param columns: Column names.
    """
    if not rows:
        print('(empty table)')
        return

    # Determine column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col, ''))
            widths[col] = max(widths[col], len(val))

    # Header
    header = '  '.join(col.ljust(widths[col]) for col in columns)
    print(header)
    print('  '.join('-' * widths[col] for col in columns))

    # Rows
    for row in rows:
        line = '  '.join(str(row.get(col, '')).ljust(widths[col]) for col in columns)
        print(line)
