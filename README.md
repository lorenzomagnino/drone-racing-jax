# drone-racing-jax

Checkpoint **3/5: receding-horizon controller**. Learning and the full paper
experiment remain pending. **No training has been run**, and no strategic robustness
is included.

Source: [Strategizing at Speed](https://github.com/andrejcarlo/ral26_strategizing_at_speed),
commit `4b3468deb2377197c3fa20549459f1c86c2e2eb1`.
The linked [paper](https://arxiv.org/abs/2602.06925) is by **Papuc et al.**, rather than Morra.
The source clone and inspected solver dependency sources live under ignored `reference/`.
To restore the original clone if needed:

```sh
git clone https://github.com/andrejcarlo/ral26_strategizing_at_speed.git reference
git -C reference checkout 4b3468deb2377197c3fa20549459f1c86c2e2eb1
```

## Install and run

```sh
make install
make test
make lint
make format
uv run pre-commit install
```

Enable float64 before building tracks or compiling functions, matching Julia Float64:

```python
import jax
import jax.numpy as jnp
from drone_racing_jax.game import default_context, rollout, trajectory_cost
from drone_racing_jax.tracks import make_track

jax.config.update("jax_enable_x64", True)
track = make_track("lemniscate")
x0 = jnp.zeros((2, 11))
us = jnp.zeros((15, 2, 4))
xs = jax.jit(rollout)(x0, us)
cost = jax.jit(trajectory_cost)(xs, us, default_context(), track)
print(xs.shape, cost)  # open-loop dynamics demonstration
```

To solve the original open-loop MPG at its standard starting positions:

```python
from drone_racing_jax.splines import evaluate
from drone_racing_jax.solver import solve_game

progress = jnp.array([track.length - 1.0, track.length - 2.5])
x0 = jnp.zeros((2, 11)).at[:, 9].set(progress)
x0 = x0.at[:, :3].set(jax.vmap(lambda s: evaluate(track.path, s))(progress))
solution = solve_game(x0, default_context(), track, horizon=15)
assert solution.converged, solution.residual
print(solution.us[0], solution.residual)

# Reverse-mode sensitivity of player 1's first jerk to the cost parameters:
gradient = jax.grad(lambda c: solve_game(x0, c, track).us[0, 0, 0])(default_context())
```

`make_track` and `periodic_spline` construct fixed data on the host using NumPy
(a JAX dependency). Dynamics, rollout, costs, constraints, and spline evaluation
are pure JAX and support `jit`, `vmap`, and differentiation. A joint state is `(2,11)`,
a joint input `(2,4)`, with time on the leading axis of trajectories.

Run the synchronous closed-loop simulation (15-step horizon, dt=0.05, no training):

```sh
uv run python -m drone_racing_jax.race --steps 100 --seed 42
```

It saves `trajectory.npz`, `summary.json`, and `trajectory.png` under ignored
`outputs/race/`. Starts use the original Cartesian uniform ±0.15 m offsets;
the standalone CLI uses NumPy's seeded generator, not Julia's Mersenne Twister.
Cross-language checks below instead give both implementations identical initial
states, so different random-number generators do not affect the comparison.

## Source correspondence

Paths below are relative to the original repository unless prefixed by a dependency.

| Python file | Julia source | Ported behavior |
| --- | --- | --- |
| `drone_racing_jax/game.py` | `DroneTrajectorySolver.jl/src/dynamics.jl` | `pmm_mpcc_jerk`, RK4 discretization |
| `drone_racing_jax/game.py` | `DroneTrajectorySolver.jl/src/game.jl` | `setup_dp_mpccg_game` / `setup_dp_lmpccg_game`; MPCCGB with nonzero blocking |
| `drone_racing_jax/game.py` | `DroneTrajectorySolver.jl/src/utils.jl`, `src/Server/server_utils.jl` | context order and medium-speed MPG defaults |
| `drone_racing_jax/game.py` | `src/environment.jl`, dependency `MCPTrajectoryGameSolver/src/solver_setup.jl` | world/state slacks, initial-state and transition equalities |
| `drone_racing_jax/splines.py` | `src/Spline/SplinesWithCachedDerivatives.jl`, DataInterpolations 8.3.1 | natural cubic values with wrapped domain and cached AD rules |
| `drone_racing_jax/tracks.py` | `src/trajectories.jl`, `src/Spline/spline.jl`, `src/utils.jl`, `src/Server/server_utils.jl` | four default tracks, arc-length fit, tangent approximation, gates, Gaussian contour weights |
| `drone_racing_jax/solver.py` | dependency `MCPTrajectoryGameSolver/src/solver_setup.jl`, `src/solve.jl` | coupled Nash stationarity and private complementarity, after eliminating linear dynamics |
| `drone_racing_jax/controller.py` | `src/Server/mcs_sync.jl`, `src/Server/referee.jl`, `src/trajectory_optimizer.jl`; Optim.jl 1.13.3 Brent | independent ego solves, measured-state progress projection/augmentation, role switching, shifted control warm starts |
| `drone_racing_jax/race.py` | `src/Server/server_utils.jl`, `src/Server/replay_buffer.jl` | seeded standard starts and a fixed-duration synchronous simulation |
| `scripts/reference_game.jl` | original game/dynamics/spline files | common Julia validation setup shared by both comparisons |
| `scripts/reference_open_loop.jl` | original game/dynamics/spline files and original MCP solver | isolated Julia validation without ROS/server/learning imports |
| `scripts/reference_closed_loop.jl` | original progress/augmentation functions and original MCP solver, synchronous loop rules | fixed-duration Julia check using the original unshifted primal warm start |
| `scripts/setup_reference.jl` | original `Manifest.toml` | reduced Julia environment with pinned direct package versions |
| `scripts/compare_open_loop.py` | Julia validation output | plan and warmed-timing comparisons |
| `scripts/compare_closed_loop.py` | Julia closed-loop output | physical trajectory, applied-input, and role-switch comparisons |
| `tests/` | formulas in the above files | RK4/cost checks, derivative caches, KKT conditions, implicit gradients, horizon indexing, and JAX transformations |

Here `src/` abbreviates `DroneTrajectorySolver.jl/src/`.

## Game semantics and fidelity choices

- State: `[px,py,pz,vx,vy,vz,ax,ay,az,theta,vtheta]`; input:
  `[jx,jy,jz,delta_vtheta]`. Default `dt=0.05`. The fourth input is integrated as
  path acceleration, including `0.5*dt**2` in theta. Mass is unused in this model.
- Context order: lag, contour amplitude, contour nominal, input, attacker ID,
  collision weight, collision radius, velocity weight, defender maximum speed,
  attacker maximum speed, progress weight, blocking weight. Attacker IDs remain
  **1 and 2**, even though Python array indices start at zero.
- Lag error is the vector `(tangent @ position_error) * tangent`, retaining the
  approximate tangent's length. Costs use squared lag/contour error, squared input,
  `4*max(speed_squared-max_speed_squared,0)**2`, the attacker's
  `max(radius_squared-separation_squared,0)**2`, and `-progress_weight*vtheta`.
  MPCCGB subtracts the defender's blocking-weighted collision term. Default blocking
  is zero. There is no time-step scaling or extra terminal cost.
  `default_context()` matches medium-speed MPG; the deployment LMPG context instead
  uses collision radius 1.25 (set context index 6 when porting that method).
- Hard bounds are `p_xyz in [-20,20]`, Cartesian velocity components and vtheta
  in `[-attacker_max_speed,attacker_max_speed]`. Progress, acceleration, and jerk are
  unbounded. No hard collision or track-corridor constraint exists in this game;
  the referee and safety controller are separate later-stage components.
  Inequality residuals are nonnegative when feasible; their order is documented
  by `state_constraints`, rather than the original LazySets polytope row order.
- **Horizon indexing is preserved:** the MPG dependency uses H states and H inputs,
  fixes state 0, and enforces only H−1 transitions. Its final input is costed but
  has no dynamics successor. LMPG's trajectory optimizer instead returns H+1 states
  and H inputs. The racing cost sums the first H state/input pairs, excluding the
  final state. `rollout` returns H+1 states; pass `xs[:-1]` for the MPG layout.
- Despite its name, Julia `PeriodicSpline` uses a **natural** cubic, overwrites its
  final value with its initial value, and wraps evaluation modulo track length.
  Its custom derivative is a separately fitted quadratic, followed by linear and
  constant caches. These are preserved, including the quadratic spline's omitted
  penultimate knot. Thus AD can differ from finite differences of cubic values.
  Only evaluation time is differentiable; spline data are fixed, as in the source.
- Track construction uses composite 32-point Gauss–Legendre quadrature on 32 panels
  instead of QuadGK. Bisection retains the source's `1e-5` tolerance, 30 segments,
  and 200 gate-weight samples. At checkpoint 2, 50 lemniscate samples of position,
  tangent, and contour weight match Julia within `3.9e-14`; other tracks have not
  yet been checked against Julia.
  NumPy performs this one-time setup; online evaluation is JAX.
- Only the default jerk model and two-player racing game are ported at this
  checkpoint. Unused alternative dynamics, toy games, and the single-player MPC
  baseline are outside this checkpoint. The solver uses JAX alone.

## Open-loop solver

`solve_game` solves each player's own stationarity equations jointly with
`min(slack, multiplier)=0`. Linear dynamics and the fixed initial state are
eliminated exactly, leaving H*8 input variables and (H−1)*28 inequality multipliers.
This is the original MPG KKT system with equality multipliers eliminated; it is
not a minimization of the sum of player costs. The returned plan has H states and
H inputs, retaining Julia's final unused input.

The numerical method is semismooth Newton with backtracking on squared KKT
residuals. A tiny primal diagonal regularizer and a least-squares safeguard handle
ill-conditioned steps. These alter the search direction only; success is measured
against the unregularized original equations. The default infinity-norm tolerance
is `1e-8`, with at most 100 iterations. `initial_guess=previous.us` supplies an
optional control warm start. Multipliers initially start at zero.

`Solution` contains `xs`, `us`, `multipliers`, `residual`, `iterations`, and
`converged`. Always check `converged` before using a plan. Nonconvergence is explicit;
reverse-mode differentiation of a failed solve produces NaNs. Successful implicit
gradients use an adjoint solve of the converged complementarity Jacobian, without
differentiating Newton iterations. Initial states and continuous context entries
are differentiable. Track data, numerical settings, and the hard velocity bound
are fixed; the attacker ID is discrete. Higher-order/forward-mode differentiation
of the solve is not provided.

There is a necessary distinction between preserving Julia's equations and its
linearization: `_stage_gradient` uses the original cached **first** spline derivatives
when forming stationarity. Newton and the implicit backward pass differentiate
the actual polynomial values in those equations. Julia recursively substitutes
higher derivative caches, which can differ from numerical differentiation.
The root equations remain the same; the exact numerical linearization makes
JAX's implicit gradients agree with finite differences of re-solved plans. Tests
also compare this stationarity assembly against AD of the original cost function.

As in the paper's PATH implementation, first-order KKT conditions establish a
stationary candidate, not a global Nash certificate. Sensitivities are local to
the selected solution branch and require a nonsingular Jacobian; at active-set
changes a unique derivative need not exist.

## Receding-horizon controller

At each measurement, `controller_step`:

1. Projects each measured position onto the track using the original bounded Brent
   search on `previous_progress ± 2.5`, with absolute/relative tolerance `1e-5`.
   Progress remains unwrapped across laps.
2. Rebuilds virtual theta and vtheta from the physical `(p,v,a)` measurement.
   As in Julia, vtheta is the **norm** of the velocity projection onto the tangent,
   including when the physical velocity points backwards.
3. Updates the attacker only after an overtake strictly exceeding 0.75 m in progress.
4. Solves two games, one per ego ordering. The second drone sees reversed player
   states and the reversed attacker ID. Each drone applies its own first input.
5. Shifts each full input plan by one step and appends zero for the next warm start.
   State trajectories are rebuilt from the next measurement; duals restart at zero.

The shift in item 5 is explicitly requested by this port. Julia instead reuses
unshifted previous primal variables and clears duals. The numerical comparisons
below retain that original behavior in Julia and the requested shift in JAX.

For an external measured-state stream:

```python
from drone_racing_jax.controller import initialize_controller, controller_step

memory = initialize_controller(progress, attacker=2, horizon=15)
step = controller_step(x0[:, :9], memory, default_context(), track)
assert step.plans.converged.all(), step.plans.residual
action, memory = step.us, step.controller  # action is (2,4); next call uses a fresh measurement
```

`controller_step` is JIT-compiled and keeps its state explicitly in a `ControllerState`.
`run_closed_loop` is a host simulation loop that measures the resulting physical
state after every dynamics step. It returns physical states `(T+1,2,9)`, the planning
states `(T,2,11)`, applied inputs, attacker IDs, and per-ego solver diagnostics.

The simulation has no computational delays, measurement noise, low-level tracking
controller, safety-filter QP, or disqualification/lap-termination logic. It executes
the requested number of steps rather than claiming a full race tournament. If either
ego solve fails, `controller_step` returns NaN actions and retains its old state;
the host loop raises **before** advancing the plant. This explicit failure behavior
differs from Julia's server fallback to an older plan. No such fallback was needed
in the validated runs.

## Findings for the next checkpoints

The source **does not learn racing-cost weights from demonstrations**. It learns
two neural input-reference generators by simultaneous gradient play. Each output
parameterizes a constrained trajectory optimization with objective
`sum(0.5*3*u**2 - u*reference)`; racing costs score the resulting joint trajectories.
The default training setup uses one candidate per player, with machinery for a
finite mixed-strategy game when there are multiple candidates. Its default gradient
stops derivatives through mixing probabilities. The original implicit derivative
uses the active-set KKT matrix, with active inequality multipliers above `1e-3`.

`train_policy.jl` invokes `Server/dp_replay_buffer.jl`: collect rollouts with
Bernoulli(0.5) reuse of previous plans and Normal(0,20) control noise, shuffle the
buffer, train each player, then swap players/attacker IDs for symmetric training.
The training state-box slack subtracts `t*(-0.125)`, relaxing bounds with horizon
depth. Deployment uses zero relaxation. The README/paper's high-level description
does not capture all these implementation details; the executable Julia code is
the behavior to reproduce.

Upstream reproduction details found during inspection:

- Julia 1.11.5 was installed locally under ignored `reference/_julia/` for validation.
- The root Manifest uses relative paths for both local Julia packages and a Git
  dependency for ParametricMCPs; preserve those when setting up the Julia environment.
- `trajectory_optimizer.jl` uses `original_domain` and `num_segments` as bare keyword
  arguments. The module does define both constants, so these are intentional.
- The training configuration's data-normalization file is supplied, but its named
  dataset and target checkpoint are absent. The active online replay-buffer trainer
  generates data itself and does not read that dataset, so its absence need not block
  training from scratch. Supplied pretrained checkpoints use different filenames.
- The baseline's warm-start copies previous primals and resets duals without shifting.
  The requested shifted control warm start is implemented in scope 3, as described above.

## Experiment commands and reproduction status

The original evaluation entry point is, from `reference/` after Julia setup:

```sh
./scripts/julia_server mcs
./scripts/julia_train params/train_parameters_medium_speed_lemniscate.toml lemniscate
```

The checked-in `startup.jl` enables LMPG vs MPC on the lemniscate; most tournament
jobs are commented out. A full-paper tournament therefore requires selecting jobs
and restoring sample counts, rather than treating current defaults as the full study.
The JAX simulation command above implements the synchronous MPG controller; it does
not yet run the full tournament or the learned controller.

The open-loop comparison can be reproduced using Julia 1.11.5:

The local validation binary is `reference/_julia/julia-1.11.5/bin/julia`; use that
path in place of `julia` below if Julia is not on your PATH.

```sh
julia scripts/setup_reference.jl
julia --project=reference/_parity --threads=4 scripts/reference_open_loop.jl outputs/reference_open_loop.json
uv run python scripts/compare_open_loop.py
```

The setup script pins direct package versions from the original Manifest, including
its ParametricMCPs fork. It uses a smaller environment so ROS, visualization, and
learning packages are not needed for this check. The Julia harness includes the
original game/dynamics/spline implementations; track construction runs independently
in Julia and Python. Generated plans and reports stay in ignored `outputs/`.

| Comparison | Julia run | JAX run | Gap |
| --- | --- | --- | --- |
| Closed-loop trajectories | 100-step seeded run + 10-step overtake | Same initial states | See checkpoint 3 below |
| Open-loop plans and solve times | Run, 4 cases | Run, 4 cases | See checkpoint 2 results below |
| Learned network parameters | Not run | Not implemented | Unmeasured |

### Checkpoint 2 results

Measured on macOS, CPU, float64, Julia 1.11.5 and JAX 0.12.0.1. Times are medians
of ten warmed solves with zero-input initial guesses, excluding game construction
and JIT compilation. The first JAX call for each new horizon took about 1.4 s.
These are local open-loop checks, not the paper's complete race tournament or its
hardware timing study. Timing varies with the environment and workload.

| Case | H | Max position gap (m) | Max input-component gap | Julia median (ms) | JAX median (ms) | JAX KKT residual |
| --- | --- | --- | --- | --- | --- | --- |
| Standard start | 4 | 8.68e-13 | 2.73e-9 | 14.13 | 1.08 | 5.94e-14 |
| Standard start | 15 | 7.63e-9 | 1.65e-7 | 87.54 | 13.54 | 6.47e-9 |
| Close interaction | 15 | 2.56e-10 | 2.84e-8 | 96.28 | 22.87 | 7.11e-14 |
| Moving state | 15 | 5.09e-8 | 1.01e-6 | 92.33 | 92.69 | 7.38e-13 |

All four original PATH runs returned `MCP_Solved`. Their KKT residuals ranged from
`9.43e-9` to `8.07e-7` with the original default solver tolerance. The maximum
difference across all state components was `5.19e-7`. Position, tangent, and
contour-weight differences at 50 independently constructed track samples were
`2.78e-14`, `2.77e-14`, and `3.82e-14`, respectively.

The moving state is a regression for a stalled plain Newton line search. Its
least-squares safeguard converges to the same plan as PATH. An additional seed-42
smoke check sampled 12 moving states (uniform progress on the lemniscate, separation
0.4–1.5 m in progress, speeds 1.8/2.5 m/s along the tangent, zero acceleration):

```text
Seed 42 moving-state checks: 12/12 converged; worst KKT residual: 1.7161834835188738e-09
Newton iterations: [13, 40, 9, 10, 11, 15, 11, 13, 41, 11, 8, 10]
```

The suite checks KKT residuals, feasibility, complementarity, original-cost
stationarity, active and inactive implicit gradients against finite differences,
initial-state sensitivity, warm starts, failure handling, and seeded `vmap` runs.
Checkpoint 3 adds seeded closed-loop and controller tests, reported below.

```text
$ make lint
uv run ruff check .
All checks passed!

$ make test
collected 32 items
tests/test_game.py ..........                                            [ 31%]
tests/test_solver.py .............                                       [ 71%]
tests/test_tracks.py .........                                           [100%]
============================= 32 passed in 39.73s ==============================
```

### Checkpoint 3 results

```sh
julia scripts/setup_reference.jl
julia --project=reference/_parity --threads=4 scripts/reference_closed_loop.jl outputs/reference_closed_loop.json
uv run python scripts/compare_closed_loop.py
```

The reference uses the original Julia progress and state-augmentation functions,
the original game and PATH solver, and the original unshifted primal warm start.
JAX independently projects/re-augments its measurements and shifts its input plans.
Both run two ego games per step. No training is involved.

| Case | Steps | Max position gap (m) | Max physical-state component gap | Max input-component gap | Role mismatches | JAX worst KKT residual |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 42 starts | 100 | 4.11e-9 | 7.73e-8 | 1.06e-6 | 0 | 9.74e-9 |
| Overtaking | 10 | 6.12e-10 | 1.47e-8 | 2.93e-7 | 0 | 5.83e-9 |

Both implementations switched roles once in the overtaking case, on the same step.
The seeded 100-step case had no role switch. Julia's worst KKT residuals were
`7.64e-7` and `4.39e-7`, respectively. Median warm full-step times (projection and
both ego solves) were JAX/Julia `49.14/173.75 ms` and `52.87/182.02 ms` on this
machine. These are synchronous controller checks, not real-time hardware claims.

The standalone NumPy-seed-42 CLI also completed 100 steps with worst residual
`8.34e-9`. Its generated plot was visually checked. The expanded suite tests
measurement feedback, unwrapped projection, vtheta augmentation, strict role-switch
thresholds, plan shifting, both ego orderings, seeded reproducibility, and failure
handling in addition to the game and solver checks:

```text
$ make lint
uv run ruff check .
All checks passed!

$ make test
collected 46 items
tests/test_controller.py ..............                                  [ 30%]
tests/test_game.py ..........                                            [ 52%]
tests/test_solver.py .............                                       [ 80%]
tests/test_tracks.py .........                                           [100%]
============================= 46 passed in 38.24s ==============================
```

Learning, long races, asynchronous delays, and the full paper tournament remain unverified.
