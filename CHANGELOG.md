# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Probabilistic ensemble evaluation metrics:
  - `crps()`: Continuous Ranked Probability Score for ensemble forecasts using the fair estimator.
  - `spread_skill_ratio()`: Ratio of ensemble spread to the RMSE of the ensemble mean.
  [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
- Plotting functions for ensemble verification:
  - `plot_rank_histogram()`: Generates Talagrand diagrams to evaluate ensemble calibration.
  [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
- Ensemble test fixtures (`da_ensemble_prediction_2d_utc`, `da_ensemble_prediction_2d_elapsed`) in `tests/unit/conftest.py` [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
- Full `groupby` support for all statistical metrics (including CRPS and SSR) to enable grouped verification in plotting pipelines [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
- Expanded unit tests for ensemble metrics and plotting functions, including unexpected input validation [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti

### Changed

- Standardized statistical function signatures to `(ds_reference, ds_prediction)` to match plotting conventions [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
- Improved plotting function robustness for `elapsed` and `UTC` time axes [\#4](https://github.com/mllam/mllam-verification/pull/4) @GiGiKoneti
