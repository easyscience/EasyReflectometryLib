# Pixi Usage Guide

This project has been converted to use [Pixi](https://pixi.sh/), a modern package manager for Python and other languages that provides reproducible environments.

## Installation

1. First, install Pixi by following the instructions at https://pixi.sh/latest/
2. Clone this repository
3. Navigate to the project directory
4. Install the project dependencies:

```bash
pixi install
```

## Development Setup

To set up the development environment:

```bash
pixi run dev-setup
```

This will install the package in development mode with all development dependencies.

## Available Commands

### Testing
- `pixi run test` - Run tests
- `pixi run test-cov` - Run tests with coverage
- `pixi run test-full` - Run tests with full coverage reporting

### Code Quality
- `pixi run lint` - Check code with ruff
- `pixi run lint-fix` - Fix code issues automatically
- `pixi run format` - Format code with ruff
- `pixi run format-check` - Check code formatting

### Documentation
- `pixi run docs-build` - Build documentation
- `pixi run docs-clean` - Clean documentation build
- `pixi run docs-serve` - Serve documentation locally

### Building
- `pixi run build` - Build both wheel and source distribution
- `pixi run build-wheel` - Build wheel only
- `pixi run build-sdist` - Build source distribution only

### Cleaning
- `pixi run clean` - Clean build artifacts
- `pixi run clean-pyc` - Clean Python cache files

### Coverage
- `pixi run coverage` - Generate coverage reports

## Environments

The project defines several environments:

- `default` - Development environment with all dev dependencies
- `docs` - Documentation building environment
- `prod` - Production environment with minimal dependencies

To use a specific environment:

```bash
pixi run -e docs docs-build
```

## Migration from pip/conda

If you were previously using pip or conda, you can now use the equivalent Pixi commands:

| Old Command | New Pixi Command |
|-------------|------------------|
| `pip install -e '.[dev]'` | `pixi run dev-setup` |
| `pytest` | `pixi run test` |
| `pytest --cov` | `pixi run test-cov` |
| `python -m build` | `pixi run build` |
| `ruff check .` | `pixi run lint` |
| `ruff check . --fix` | `pixi run lint-fix` |

## Benefits of Pixi

- **Reproducible environments**: Lock files ensure everyone has the same dependencies
- **Fast installs**: Conda-forge packages are pre-compiled
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Task management**: Built-in task runner for common development tasks
- **Multiple environments**: Easily switch between different dependency sets