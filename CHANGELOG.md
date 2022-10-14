# Change Log

## v0.0.1.dev11 (14/10/2022)

### Changed examples

- `XrayTomography` compatibility for different PIL versions

## v0.0.1.dev10 (11/10/2022)

### Changed examples

- `XrayTomography`
    - allowed `kwargs` for plots
    - more data points for first example

## v0.0.1.dev9 (10/10/2022)

### Changed examples

- `FmmTomography` 
    - `kwargs` that can be passed down to waveTracker functions
    - `np.random.seed` instead of `random.seed`
- `XrayTomography`
    - changed cmap to be blue

## v0.0.1.dev8 (10/10/2022)

### Changed examples

- `FmmTomography` (`plot_model(model, with_paths=False, return_paths=False`))

## v0.0.1.dev7 (10/10/2022)

### Changed examples

- `FmmTomography` (return "paths" when plotting the model and `paths=True`)

## v0.0.1.dev6 (04/10/2022)

### Changed examples

- `FmmTomography` (minor changes; all in slowness space)
- `XrayTomography` (additional inlab-logo example)


## v0.0.1.dev5 (21/09/2022)

### New examples

- `FmmTomography`

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
