# Change Log

## v0.0.1.dev4 (08/09/2022)

### New examples

- `SimpleRegression`
- `XrayTomography`

### Changed examples

- `GravityDensity`

### Core and utilities

- Object oriented instead of a few functions for each problem
- `metadata` dict instead of a few class fields
- Utility functions including `absolute_path`, `loadtxt`
- Functions `list_problems`, `list_problem_names`
- Documentation construction
- `utils/` folder renamed to `tools/` to avoid ambiguity with `cofi_espresso.utils`
- Our own exception classes `EspressoError`, `InvalidExampleError`
- Fix relative path issue with utility scripts under `tools/`
