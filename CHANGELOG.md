# Changelog

## [1.2.0] - 2026-10-01

### Added

- `af3parallel build-json natural|denovo` — build AF3 input JSON from FASTA/PDB/manifest
  (natural: MSA omitted for data pipeline; denovo: empty MSA for designed proteins)
- `af3parallel msa` — parallel AF3 data pipeline with default `cpu_count//4` workers and
  gap filling; documents ColabFold as the GPU MSA alternative
- `af3parallel stats` — summarise `*_summary_confidences.json` (ligand interface or general)
- Updated overview figure (`docs/images/scheduling-overview.png`)

### Changed

- README / workflow docs cover build → MSA → run → stats
- Optional dependency: `tqdm` (progress bars for stats)

## [1.1.0] - 2026-06-10

### Added

- Installable Python package on PyPI: `pip install af3parallel`
- Unified CLI: `af3parallel run|profile|profile-ts|estimate-gpu|estimate-cpu|json|monitor`
- Individual console entry points (`af3parallel-run`, …)
- GitHub Actions workflow for PyPI publishing (`.github/workflows/publish-pypi.yml`)

### Changed

- Source moved from flat `*.py` scripts to `src/af3parallel/` package
- Documentation updated for PyPI installation

## [1.0.0] - 2026-06-09

- Initial public release as standalone scripts
