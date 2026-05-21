# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for display table helpers."""

from __future__ import annotations

import sys
import types

from easyreflectometry.display import tables


def test_in_notebook_returns_false_without_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, 'IPython', None)

    assert tables.in_notebook() is False


def test_in_notebook_detects_zmq_shell(monkeypatch):
    class ZMQInteractiveShell:
        pass

    ipython = types.SimpleNamespace(get_ipython=lambda: ZMQInteractiveShell())
    monkeypatch.setitem(sys.modules, 'IPython', ipython)

    assert tables.in_notebook() is True


def test_render_table_prints_aligned_table(capsys):
    result = tables.render_table(
        [{'name': 'film', 'value': 12.5}],
        ['name', 'value'],
    )

    captured = capsys.readouterr()
    assert result is None
    assert 'name' in captured.out
    assert 'film' in captured.out
    assert '12.5' in captured.out


def test_render_table_prints_empty_table(capsys):
    result = tables.render_table([], ['name'])

    captured = capsys.readouterr()
    assert result is None
    assert '(empty table)' in captured.out


def test_render_table_returns_dataframe_in_notebook(monkeypatch):
    pd = pytest_importorskip_pandas()
    monkeypatch.setattr(tables, 'in_notebook', lambda: True)

    result = tables.render_table([{'name': 'film'}], ['name'])

    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, 'name'] == 'film'


def pytest_importorskip_pandas():
    import pytest

    return pytest.importorskip('pandas')
