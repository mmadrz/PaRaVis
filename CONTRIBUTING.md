# 🤝 Contributing to PaRaVis

Thank you for considering contributing to PaRaVis! We welcome contributions of all forms — bug reports, feature requests, documentation improvements, and code changes.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Checklist](#pull-request-checklist)
- [Reporting Issues](#reporting-issues)
- [Project Structure](#project-structure)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Be kind, be constructive, and assume good faith.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/paravis.git
   cd paravis
   ```
3. **Install** in development mode with all extras:
   ```bash
   pip install -e .[gui,gpu,dev]
   ```
4. Create a **branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Minimal (Headless)

If you only need to work on core computation or the API:

```bash
pip install -e .[dev]
```

### Full (GUI + GPU)

```bash
pip install -e .[gui,gpu,dev]
```

### Verify Installation

```bash
python -c "import paravis; print(paravis.__version__)"
```

---

## Running Tests

PaRaVis uses `pytest` with `pytest-cov` for coverage.

### All Tests

```bash
pytest tests/ -v
```

### With Coverage Report

```bash
pytest --cov=paravis --cov-report=term-missing tests/
```

### Core Computation Tests Only (No Qt Required)

```bash
pytest tests/test_core_*.py tests/test_raster_io.py tests/test_api.py tests/test_utils.py -v
```

### GUI Tests

Note: GUI tests require a running display server (X11/Wayland). They use heavy mocking to avoid CuPy thread-pool crashes:

```bash
pytest tests/test_gui_*.py tests/test_main_window.py -p no:cupyx -v
```

### Specific Test File

```bash
pytest tests/test_core_indices.py -v -k "test_compute"
```

### Code Coverage Goals

- **Core modules** (`paravis/core/`, `paravis/api/`, `paravis/utils/`): aim for ≥90%
- **GUI modules** (`paravis/gui/`, `paravis/workers/`): aim for ≥50% (limited by Qt/CuPy runtime conflicts)
- New code should include tests

---

## Code Style

### Python

- **PEP 8** — Follow standard Python style
- **NumPy-style docstrings** — Required for all public functions
- **Type hints** — Required for all function signatures
- **Line length** — 100 characters max

### Example

```python
def compute_index(
    raster_data: np.ndarray,
    bands: Dict[str, np.ndarray],
    index_name: str,
) -> np.ndarray:
    """Compute a single spectral index.

    Parameters
    ----------
    raster_data : np.ndarray
        3D array of shape (n_bands, height, width).
    bands : Dict[str, np.ndarray]
        Mapping from spectral code to band array.
    index_name : str
        Name of the index to compute.

    Returns
    -------
    np.ndarray
        2D array of computed index values.
    """
    ...
```

### Core Purity Rule

**`paravis/core/` must have zero `PySide6` imports.** This is the most important architectural constraint — it ensures the computation engine can run in headless environments (SSH, HPC clusters, CI runners without a display).

If you need to add Qt-dependent functionality, place it in `paravis/gui/` or `paravis/workers/`.

---

## Pull Request Checklist

Before submitting a pull request, please ensure:

- [ ] Code follows PEP 8 and project conventions
- [ ] NumPy-style docstrings and type hints added for new functions
- [ ] Tests added for new features or bug fixes
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Core modules remain Qt-free (no `from PySide6` imports in `paravis/core/`)
- [ ] Branch is up to date with `main`
- [ ] If adding a dependency, update `pyproject.toml` and the README

---

## Reporting Issues

When opening an issue, please include:

- **PaRaVis version** (`python -c "import paravis; print(paravis.__version__)"`)
- **Python version** and **operating system**
- **Steps to reproduce** with minimal code example
- **Expected vs actual behaviour**
- **Full error traceback** (if applicable)

---

## Project Structure

```
paravis/
├── paravis/
│   ├── api/           # Public API (convenience wrappers)
│   ├── core/          # Computation engine (no Qt!)
│   │   ├── indices/   # Spectral indices
│   │   ├── raoq/      # Rao's Q diversity
│   │   └── raster/    # Raster I/O
│   ├── gui/           # PySide6 desktop app
│   │   ├── widgets/   # Main panels
│   │   ├── dialogs/   # Dialog windows
│   │   ├── components/# Reusable widgets
│   │   └── models/    # Qt item models
│   ├── workers/       # Background QThread workers
│   └── utils/         # Shared utilities
├── tests/             # pytest test suite
├── examples/          # Ready-to-run example scripts
└── docs/              # Documentation
```

---

## Questions?

If you're unsure about anything, open a discussion or an issue. We're happy to help!

Thank you for contributing! 🎉
