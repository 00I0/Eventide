# Eventide Python

This directory contains:

- The Python package wrapping the C++ extension (`eventide/`)
- Analysis and plotting scripts for epidemic case studies

Recent updates: added support for alternating case and introduced `synthetic.py`.

## Structure

- `eventide/`: installable package (`setup.py`, wrappers, collectors, criteria, samplers)
- `plots/`: plotting helpers and figure builders
- Top-level `*.py` scripts: scenario-specific runs/analysis
- `robustness_outputs/`, `optimization_logs/`: generated outputs
- `map/`: GIS assets and notebook

## Environment Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas lmfit scikit-learn seaborn
```

Install the local package:

```bash
python -m pip install -e python/eventide
```

## Build the C++ Extension for Python

The extension module `_eventide` is built by top-level CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

After build, `_eventide` is placed under `python/eventide/` and the package is installed editable by the `python_install` target.

## Running Scripts

From repository root, with your virtual environment active:

```bash
python python/plot_final.py
```

Other common entry scripts are in `python/` (for example `plot_john_snow_cholera.py`, `plot_FMD.py`, `plot_TDK.py`, `robustness_appendix_analysis.py`).

## Notes

- Several scripts expect repository-root relative paths to CSV files.
- Some scripts are computationally heavy; start by reducing trajectory counts when iterating.

## Corrected alternating plots

`plot_alternating.py` reads saved optimizer CSVs, chooses the lowest finite
objective with at least 80 accepted histories, and simulates fresh posterior
histories using those window settings. It does not rerun the optimizer.
By default it reads `optimization_logs/alternating_quick` and plots every
available configured scenario and saved snapshot. Scenarios without saved
results are skipped; explicitly selecting one without results reports an error.

```bash
python python/plot_alternating.py --list
python python/plot_alternating.py
python python/plot_alternating.py --scenario dengue_florida_current
```

Each configured scenario produces three figures (PDF and PNG) in
`plots/alternating_optimized`: the latest snapshot's two-panel reproduction/timing
posterior, its cumulative host-incidence fit, and the calendar-date conditional
cessation curves for all saved snapshots. JSON summaries and compressed history caches
are saved alongside them. `--snapshots 2 8` selects observation counts;
`--accepted 30 --attempts 100000 --workers 2` provides a quick preview.
Use `--refresh` to simulate again. Insufficient acceptance is reported.
The previous `plots/alternating_fixed` figures remain untouched. Override
`--logs` and `--output` to use other directories, for example historical logs:
`--logs optimization_logs --output plots/alternating_historical`.
Rendering is ported directly from `aa0e51d:python/plot_sae.py` into
`plots/sae_alternating.py`: 90 mm publication columns, the original font/tick
sizes, single-blue latest-snapshot posteriors and cumulative fit, and the
original blue-gradient online extinction curves. Posterior panels have no
density-axis ticks; their medians and 95% interval endpoints use the original
tick formatting and blue interval labels. Cumulative shading uses SAE's three
nested tail-mean bands (`0.95, 0.8, 0.5`), not percentile credible intervals.
There are no added figure headings, panel letters, footers, or external legends.
Low acceptance remains reported in the console and JSON summaries.

The reproduction posterior is the requested geometric mean of the two
non-index reproduction means, `sqrt((R0_host*r_host)*(R0_vector*r_vector))`,
evaluated at the snapshot. The host-to-host timing density is the posterior
mixture of the convolution of the two Gamma generation intervals, with both
kernels evaluated at the snapshot. It is not a Gamma distribution fitted to
their sum, nor merely a histogram of posterior mean intervals. With later
controls it describes the local snapshot kernels, not a forecast across those
future controls.

The plotted cessation event means **no later host infection ever**, conditioned
on no new *host* infections during the quiet period. Curves are anchored to
their snapshot calendar dates. As in SAE, a solid segment ends at the next
observation, an open circle marks that observation, and its continuation is
dotted. These dotted portions are forecasts under continued silence, not
updated probabilities after the new case. The final snapshot is displayed for
40 further days by default (`--final-horizon`), including when only an earlier
snapshot has saved results. The extinction legend is fixed at the bottom right.
`--language hu` uses Hungarian throughout, matching SAE's Hungarian probability
and legend wording; `--language en` selects English. Cumulative
observations and snapshot anchors still include the complete final day.

Probabilities of **no later infection of either species** remain available in
the JSON summaries, but are not overlaid as a second dashed curve family.
Vector infections remain unobserved. Calculations retain index direct-offspring bounds, latent
vector histories, and each parent's parameters at its infection time. They
use Gamma survival functions and a ratio of posterior means in log space;
undefined conditioning events yield NaN. Finite-horizon host/vector integrals
use quadrature; increase `--quadrature-order` (default 48) to check convergence.
These events concern cessation after the snapshot, not eventual extinction
after allowing additional infections.

Acceptance retains ordinary cluster merging, includes the complete final
observation day, subtracts the deterministic host root from descendant targets,
and appends a separate local final-batch window using the optimized timing
uncertainty. That window requires at least the final batch locally, with the
ordinary upper tolerance; it is not exact final-day matching. Fractional-day
window and control boundaries are preserved. The new builder options are
opt-in; existing callers retain their original target construction.

Simulation stops at each snapshot rather than running a long future tail.
Future edges already scheduled by the simulator are ignored by the probability
calculator. Histories that hit the case cap cause an error requesting a larger
`--max-cases`, because they may be incomplete. Cache keys include model settings,
selected windows, simulation limits, builder source and compiled extension.
Probability curves are recalculated from cached histories, so changing plotting
horizons or quadrature order does not require new simulation.
Integration also uses the requested worker threads. In a local 64-history
snapshot check, eight threads reduced probability time from 2.79 s to 0.39 s
with identical output; increasing quadrature from 48 to 120 changed the host
curve by at most 0.000025 in that check. These timings and accuracy checks
describe that batch, not a guarantee for every model configuration.

### Historical scenario metadata

The expanded inventory contains **11 distinct observation examples / 13 runnable
configurations**, covering dengue, chikungunya and urban yellow fever. See
`ALTERNATING_EXAMPLES.md` for all locations, counts, source provenance and limits.
Recovered examples are not independently verified surveillance datasets; newly
registered locations use explicitly assigned disease-specific source priors,
not undocumented historical optimized priors.

The historical CSVs contain **only acceptance-window settings**, not observations,
priors, index bounds, or control scenarios. `alternating_scenarios.json` explicitly
records recovered inputs and their provenance; `--list` reports readiness. The supplied
`dengue_florida_current` entry uses the currently active optimizer settings (both
control scenarios are empty). Its figures are labeled as current-model reruns,
not reconstructions of undocumented historical controls. Entries still marked
incomplete require the original metadata before their scenarios can be plotted.

Recovery searched the other Git branch and retained Python 3.10/3.13 bytecode:

| Scenario | Recovery |
| --- | --- |
| Montpellier, uncontrolled | Optimizer bytecode explicitly names the saved log prefix and preserves observations, both priors, and index bounds `(1, 4)`. |
| Montpellier, controlled | Plotting bytecode preserves observations, priors, bounds `(1, 4)`, and controlled window settings matching the saved CSVs. The original controlled optimizer priors are not independently available. |
| Florida `uc_control` | Uses the active uncontrolled dengue plotting model and Florida observations in `aa0e51d:python/host_vector_chikungunya.py`. Association with this particular CSV is inferred and clearly labeled. |
| Original `host_only` | Unresolved: the CSV has no model identifier/snapshot; retained later examples contain several overwritten observation lists and different priors. No unique original model can be recovered. |

The recovered Montpellier control is encoded by post-index `r` priors, not
calendar-dated changes. Both recovered configurations therefore explicitly use
empty `changes` lists. Bytecode hashes and source identifiers are recorded in
the JSON `recovery` objects. These recovered values are configuration data;
running the plotting script does not require bytecode recovery tools.

### Quick alternating batch optimizer

```bash
.venv/bin/python python/optimize_alternating_acceptance_windows.py --dry-run
.venv/bin/python python/optimize_alternating_acceptance_windows.py
.venv/bin/python python/optimize_alternating_acceptance_windows.py --all-snapshots
.venv/bin/python python/plot_alternating.py --logs optimization_logs/alternating_quick
```

The default stopping tolerance is **5% relative L2 cumulative-fit error**:
`norm(mean_simulated_cumulative - observed_cumulative) / norm(observed_cumulative)
<= 0.05`. It is scale invariant, **not absolute MSE**, and is an aggregate curve
criterion, not a guarantee that every individual point is within 5%.
Override with `--relative-tolerance .03` (`--target-tol` is an alias).

The measles-style CMA-ES search uses normalized parameter coordinates, structural
window memoization, multithreaded C++ simulation, early stopping and warm starts
only from the same scenario. It requires `cmaes` in the project environment.
One persistent tqdm bar per scenario tracks completed snapshots, with live
generation/evaluation counts, best relative error and accepted-history count;
cached snapshots advance the same bar without per-evaluation console output.
The default schedule chooses at most six representative snapshots including the
first and final; `--all-snapshots`, `--snapshot-mode final`, or `--snapshots 2 4`
override it. `--scenario KEY` can be repeated. The unresolved `host_only` case
is skipped unless explicitly requested.

Defaults: 200 accepted histories per evaluation, at most 20 million attempted
trajectories, 30 generations, population 12, up to eight simulation workers.
Use `--min-accepted`, `--num-trajectories`, `--generations`, `--population` and
`--workers` to adjust cost and Monte Carlo precision. A 5% fit target is a
numerical stopping rule, not a confidence interval or calibration guarantee.
Infeasible or case-capped evaluations cannot satisfy it. A budget-exhausted
search saves its best result, marks `target_met: false`, and exits nonzero.

Best results are written after every completed evaluation to
`optimization_logs/alternating_quick/*_optimal.json`, alongside CSVs directly
readable by `plot_alternating.py --logs optimization_logs/alternating_quick`.
Matching `*_snapshot_N.checkpoint.json` files retain CMA-ES covariance, random
state and the partially evaluated population. With `--continue-unfinished`, restarting
resumes after the last completed evaluation; a simulation interrupted mid-evaluation
must be repeated. Generation limits are totals, so increase `--generations` to
continue beyond an exhausted limit. Changing population, seed, acceptance minimum,
attempt budget or the installed CMA-ES version starts a fresh search from saved
settings instead of restoring incompatible search state.

By default, any snapshot with saved parameters is reused directly, including
older cache fingerprints, even when its search missed the target or was
interrupted. No revalidation simulation or optimization is run. A changed
fingerprint is reported as `older cache reused`; these results have not been
validated against changed models or code. Their `target_met: false` status is
preserved; skipping does not mean they satisfied 5%. Use `--continue-unfinished`
to continue these searches, or `--force` to start fresh. Cache keys cover
simulation/acceptance semantics, not whole plotting
or progress-display files, so label/title/progress changes no longer restart all
searches. Use `--force` when you intentionally want to recompute old cached
results under the current model. Legacy CSVs can also supply a seed if no JSON
result exists. CSVs are appended for compatible resumes; replaced logs are preserved as
`.csv.previous-*` backups. `--force` explicitly starts fresh. Existing historical
logs are untouched. All evaluations retain end-of-final-day acceptance,
deterministic-index subtraction, fractional boundaries and the unmerged local
last-observation window. Missing/null index bounds impose no restriction.

Complete those entries or pass `--config my_scenarios.json`. Each scenario needs
`key`, `title`, `log_pattern`, daily-incidence `observations`, and
`host`/`vector` objects. `index_offspring_bounds` is optional: omitting it or
setting it to `null` disables the root-offspring acceptance criterion and the
matching prediction restriction. An explicit pair such as `[1, 4]` enables
both. Other observation-window acceptance criteria still apply.
Each species object contains `ranges` for
`R0, k, r, alpha, theta`, a `requirements` list of simulator expressions, and
an explicit `changes` list (empty means no control). Controls multiply one
parameter at a specified day relative to the first observation, for example:

```json
{"day": 8.5, "parameter": "r", "factor": 0.65}
```

These factors and dates are applied consistently in simulation and prediction.
Only host-root scenarios and multiplicative parameter changes are supported.
Other control laws require an explicit extension rather than silently using
constant reproduction parameters. Rerunning corrected acceptance can change
acceptance rates and incidence fits, so saved optimization scores do not describe
the newly simulated histories.

Regression and numerical checks can be run with:

```bash
python -m unittest discover -s python/tests -v
```

Alternating support is compiled into the regular extension; there is no separate
mode switch. Rebuild a stale extension with the project interpreter:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target _eventide -j 6
```

The alternating core also now gives host and vector samplers distinct RNG
streams. Previously both samplers received identical worker seeds, forcing
matching quantiles (and identical values for matching ranges) across species.
Recompilation and fresh simulation are required to obtain independent species
priors; old saved optimization windows are reused, but their old scores are
not estimates of the corrected model's fit or acceptance rate.
