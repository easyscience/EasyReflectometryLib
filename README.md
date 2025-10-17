![Logo](https://github.com/easyScience/EasyReflectometryLib/raw/master/docs/src/_static/logo.png)
[![CI badge](https://github.com/easyScience/EasyReflectometryLib/actions/workflows/python-ci.yml/badge.svg)](https://github.com/easyScience/easyReflectometryLib/actions/workflows/python-ci.yml)
[![PyPI badge](https://img.shields.io/pypi/v/easyreflectometry.svg)](https://pypi.python.org/pypi/easyreflectometry)
[![Quality badge](https://www.codefactor.io/repository/github/easyscience/easyreflectometrylib/badge)](https://www.codefactor.io/repository/github/easyscience/easyreflectometrylib)
[![Docs badge](https://img.shields.io/badge/docs-built-blue)](http://docs.easyreflectometry.org)

# About

A reflectometry python package and an application.

This repo and documentation is for the `easyreflectometry` Python package that is built on the `easyscience` [framework](https://easyscience.software).
To get more information about the application visit [`easyreflectometry.org`](https://easyreflectometry.org)

# Installation

## For Users

```sh
python -m pip install easyreflectometry
```

## For Developers (Pixi - Recommended)

This project now supports [Pixi](https://pixi.sh/) for reproducible development environments:

1. Install Pixi: https://pixi.sh/latest/
2. Clone this repository
3. Set up the development environment:

```sh
pixi install
pixi run dev-setup
```

4. Run tests:
```sh
pixi run test
```

See [PIXI_USAGE.md](PIXI_USAGE.md) for detailed Pixi usage instructions.

## For Developers (Traditional pip)

```sh
git clone https://github.com/easyScience/EasyReflectometryLib.git
cd EasyReflectometryLib
pip install -e '.[dev]'
```
