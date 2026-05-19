# RetinAnalysis Master Development Plan

## Project Vision

retinanalysis is a Python analysis pipeline for multi-electrode array (MEA) and single-cell electrophysiology data from retinal ganglion cells. It manages experiment metadata via a DataJoint/MySQL database and provides classes for loading, organizing, and analyzing spike-sorted data (primarily from Kilosort 2.5 converted to Vision format). The package is designed for interactive use in Jupyter notebooks by neuroscience researchers.

This document outlines strategic priorities for improving the codebase's reliability, maintainability, usability, and extensibility.

---

## Current Milestone: v0.2 — Foundation & Reliability

*The focus of this milestone is addressing fundamental code quality issues, establishing a proper testing framework, and eliminating technical debt that makes the codebase fragile and hard to maintain.*

### 1. Active Priorities (In Order of Execution)

**Priority 1: Configuration & Environment Hardening**

* **Goal:** Eliminate user-specific paths from version control, make setup less error-prone, and ensure the package can be imported without a running database.
* **Key Changes:**
  * Move `config.ini` to `.gitignore` and provide `config.ini.template` with placeholder paths and clear comments.
  * Make database connection lazy — don't attempt connection at import time in `schema.py`. Instead, connect on first query. If connection fails, provide clear error message rather than crashing on import.
  * Consolidate hardcoded constants (frame rates, sample rates, date thresholds) into a single `constants.py` file.
  * Document the full setup procedure (Docker, conda, submodule, config) in a structured quickstart guide.
* **Files Modified:** `config/schema.py`, `config/settings.py`, `config/config.ini`, `__init__.py`

**Priority 2: Testing Infrastructure**

* **Goal:** Establish a real test suite with pytest that can run without the full data and database stack.
* **Key Changes:**
  * Set up `pytest` with `conftest.py` fixtures for mocking database connections and config paths.
  * Write unit tests for pure-logic functions: `ei_corr()`, `cluster_match()`, `check_frame_times()`, `get_ells()`, `get_timecourses()`, spike binning logic.
  * Write integration tests that use small fixture data files (subset of real Vision files if possible).
  * Add a `Makefile` or `justfile` with common commands (`make test`, `make lint`, `make populate-db`).
* **Files Modified:** `tests/`, `pyproject.toml`

**Priority 3: Code Deduplication & Separation of Concerns**

* **Goal:** Reduce copy-paste code and separate data loading from plotting.
* **Key Changes:**
  * Extract shared cell-type parsing logic from `AnalysisChunk.get_df()`, `MEAResponseBlock.add_cell_types()`, and `MEAResponseGroup.add_cell_types()` into a single utility function.
  * Extract shared filtering/querying logic (the repeated noise_ids/cell_types/typing_file pattern) into a reusable helper.
  * Move plotting methods out of data classes into dedicated `plotting/` module or keep as thin wrappers that delegate to standalone plotting functions.
  * Extract `EI loading + bad ID filtering` loop (duplicated in `AnalysisChunk.__init__` and `MEAResponseBlock.__init__`) into a shared utility.
* **Files Modified:** `classes/analysis_chunk.py`, `classes/response.py`, `classes/mea_pipeline.py`, `utils/vision_utils.py`

---

### 2. Future Priorities

**Priority 4: Error Handling & Logging**

* **Goal:** Replace bare `except:` blocks and `print()` statements with proper error handling and Python `logging`.
* **Key Changes:**
  * Replace all bare `except:` with specific exception types.
  * Replace `print()` calls with `logging.info()`, `logging.warning()`, `logging.error()` as appropriate.
  * Add a package-level logger configuration (default to `WARNING` level, users can enable `DEBUG`).
  * Add meaningful error messages that tell users what to check (e.g., "No EI for cell {id} — check that vision EI files exist in {path}").
* **Files Modified:** All files in `utils/` and `classes/`

**Priority 5: Documentation & API Reference**

* **Goal:** Make the package accessible to new lab members without requiring oral tradition.
* **Key Changes:**
  * Add docstrings to all public functions and methods (many already have good docstrings, but coverage is incomplete).
  * Generate API documentation (Sphinx or mkdocs).
  * Write a "Concepts" guide explaining the data hierarchy: Experiment → Sorting Chunk → Datafile → Cell IDs, and the relationship between noise chunks, protocol datafiles, and cluster matching.
  * Add annotated example notebooks showing common workflows (load experiment → create pipeline → plot RFs → get PSTHs).

**Priority 6: Dependency Cleanup**

* **Goal:** Reduce the dependency footprint and remove unused packages.
* **Key Changes:**
  * Audit whether `torch`, `torchaudio`, `torchvision` are actually used. If for future ML work, move to an `[ml]` optional dependency group.
  * Evaluate if `ipywidgets`, `ipympl`, `ipykernel` should be runtime dependencies or optional.
  * Pin dependency versions more carefully (currently only `scipy` and `numpy` are pinned).
  * Consider splitting `pyproject.toml` dependencies into `[core]` and `[dev]` groups.

**Priority 7: Database Improvements**

* **Goal:** Make database operations more robust and user-friendly.
* **Key Changes:**
  * Add migration support for schema changes (currently requires full DB rebuild).
  * Add `ra.check_database()` function that reports on DB health, missing experiments, stale typing files.
  * Improve `reload_celltypefiles()` to also update `SortedCellType` table.
  * Add ability to populate DB from a specific list of experiments rather than scanning all files.

**Priority 8: Performance & Caching**

* **Goal:** Speed up common operations that are currently slow.
* **Key Changes:**
  * Cache `get_exp_summary()` results (frequently called, always hits DB).
  * Lazy-load EIs and spatial maps (don't load in constructor unless requested).
  * Consider an on-disk cache (`.pkl` or HDF5) for cluster match results, which are expensive to compute and deterministic for the same inputs.
  * Profile common workflows to identify bottlenecks.

---

## Testing & Infrastructure Initiatives

* **CI Pipeline:** Set up GitHub Actions to run linting (`ruff` or `flake8`) and unit tests on push. Integration tests that require data/DB can be tagged and skipped in CI.
* **Type Checking:** Gradually add `mypy` or `pyright` checking. The codebase already has partial type hints — fill in the gaps and enforce in CI.
* **Code Formatting:** Adopt a formatter (`black` or `ruff format`) and enforce via pre-commit hook.

---

## Design Principles Going Forward

1. **Import should be fast and safe.** Importing `retinanalysis` should never crash even if Docker isn't running or paths don't exist. Errors should surface when the user actually tries to use the missing resource.
2. **Data classes should be data classes.** Keep `AnalysisChunk`, `MEAResponseBlock`, etc. focused on data loading and access. Move visualization to separate modules.
3. **Don't repeat yourself.** Any logic that appears in more than one place should be extracted into a shared utility.
4. **Fail loudly with helpful messages.** Replace silent error swallowing with specific exceptions that tell the user what went wrong and how to fix it.
5. **Test the math.** Core algorithms (EI correlation, cluster matching, spike binning, frame time correction) must have unit tests with known inputs and expected outputs.
