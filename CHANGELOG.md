# Change Log

## v0.3.10 (27/07/2023)

### Espresso core

- Change the function API `forward(model, return_jacobian)` instead of `forward(model, with_jacobian)`

## v0.3.9 (11/07/2023)

### Changed examples

- New examples (checkerboard and Australian shear wave velocities) and improvements 
  in `FmmTomography`

## v0.3.8 (19/06/2023)

### Changed examples

- ReceiverFunctionInversion: added a 3-layer example (4th example)

## v0.3.7 (07/06/2023)

### New examples

- `SurfaceWaveTomography` examples

### Changed examples

- Naming changes
    - `FmWavefrontTracker` -> `FmmTomography`
    - `XrayTracer` -> `XrayTomography`
    - `ReceiverFunction` -> `ReceiverFunctionInversion`
    - `GravityDensity` -> `GravityInversion`
    - `GreatCircleTracing` -> `SurfaceWaveTomography`

### Espresso core

- Template: Minor naming changes in templates
- Template: Return `matplotlib.axes.Axes` instead of `matplotlib.figures.Figure` in plotting methods

## v0.3.6 (19/05/2023)

### Changed examples

- Improve `ReceiverFunction.starting_model` and `ReceiverFunction.plot_model`

## v0.3.5 (10/05/2023)

- Fix `ReceiverFunction.log_prior` by adding more constraints

## v0.3.4 (02/05/2023)

### Changed examples

- Improve performance of `ReceiverFunction.log_likelihood` by pre computing covariance matrix

## v0.3.3 (01/05/2023)

### Changed examples

- Fix `log_likelihood` in `ReceiverFunction`

## v0.3.2 (01/05/2023)

### Espresso core

- `EspressoProblem.list_capabilities`

### Changed examples

- Fix `covariance_matrix` and `inverse_covariance_matrix` in `ReceiverFunction`


## v0.3.1 (27/04/2023)

### Espresso core

- Fix `list_capabilities` and enhance related documentation 

## v0.3.0 (24/04/2023)

First release.

## v0.3.0.dev0 (24/04/2023)

### Changed examples

- Fix installation of receiver function on Windows
- Fix documentation rendering of 1D MT problem


## v0.2.2.dev1 (18/04/2023)

### Changed examples

- Fix installation of receiver function under SKBUILD

## v0.2.2.dev0 (13/04/2023)

### New examples

- [#106](https://github.com/inlab-geo/espresso/pull/106) Receiver function

### Changed examples

- Naming changes
    - `FmmTomography` -> `FmWavefrontTracker`
    - `XrayTomography` -> `XrayTracer`

### Infrastructure

- Build and validation scripts to run only on specified contributions
- CI to test only changed contributions on pull request
- Write changed contributions to file
- Routinely check and update active problems list
- Timeout value defaults to 60 seconds for each method

## v0.2.1.dev1 (27/03/2023)

### Infrastructure

- [#118](https://github.com/inlab-geo/espresso/pull/118) Capability matrix on build
- [#121](https://github.com/inlab-geo/espresso/pull/121) Compile on build (instead of on import)
- Code refactor in `espresso_machine`

## v0.2.1.dev0 (06/03/2023)

### New examples

- [#100](https://github.com/inlab-geo/espresso/pull/100) 1D magnetotelluric example

## v0.2.0.dev0 (03/03/2023)

### Infrastructure

- Rename package from `cofi-espresso` to `geo-espresso`, module from `cofi_espresso`
  to `espresso`
- Rename `tools/` folder into `espresso_machine/`

## v0.1.0.dev0 (03/03/2023)

### New examples

- [#114](https://github.com/inlab-geo/espresso/pull/114) Hydrology examples - pumping test and slug test

### Infrastructure

- [#91](https://github.com/inlab-geo/espresso/issues/91) Use versioningit to generate dynamic version
- [#107](https://github.com/inlab-geo/espresso/issues/107) Documentation and structure update
- [#108](https://github.com/inlab-geo/espresso/issues/108) New Espresso machine structure

## v0.0.2.dev0 (17/01/2023)

### Infrastructure

- [#88](https://github.com/inlab-geo/espresso/issues/88) Rewrite Espresso installation process; delay compilation until it's needed and not found

## v0.0.1.dev12 (13/01/2023)

### Infrastructure

- [#91](https://github.com/inlab-geo/espresso/issues/91) Use `versioningit`

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
