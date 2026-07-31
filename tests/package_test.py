# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import easyreflectometry as pkg


def test_has_version():
    assert hasattr(pkg, '__version__')  # noqa S101
