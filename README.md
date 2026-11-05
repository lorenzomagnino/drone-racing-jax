# drone-racing-jax

Checkpoint **1/5: game and dynamics**. Scopes 2–5 (Nash solver, receding-horizon
controller, learning, experiment reproduction) are deliberately pending the requested
checkpoint review. No strategic robustness is included.

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
print(xs.shape, cost)  # open-loop dynamics demonstration, not a Nash solution
```

`make_track` and `periodic_spline` construct fixed data on the host using NumPy
(a JAX dependency). Dynamics, rollout, costs, constraints, and spline evaluation
are pure JAX and support `jit`, `vmap`, and differentiation. A joint state is `(2,11)`,
a joint input `(2,4)`, with time on the leading axis of trajectories. There is no CLI
or Hydra configuration until there is an experiment to run.

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
| `tests/` | formulas in the above files | independent RK4 matrix/cost checks, derivative-cache checks, horizon and JAX transformations |

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
  and 200 gate-weight samples. Numerical parity against Julia remains unmeasured.
  NumPy performs this one-time setup; online evaluation is JAX.
- Only the default jerk model and two-player racing game are ported at this
  checkpoint. Unused alternative dynamics, toy games, and the single-player MPC
  baseline are outside this checkpoint. No solver library is needed yet.

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

Potential upstream reproduction obstacles found during inspection:

- Julia is not currently installed on this machine.
- The root Manifest uses relative paths for both local Julia packages and a Git
  dependency for ParametricMCPs; preserve those when setting up the Julia environment.
- `trajectory_optimizer.jl` uses `original_domain` and `num_segments` as bare keyword
  arguments. The module does define both constants, so these are intentional.
- The training configuration's data-normalization file is supplied, but its named
  dataset and target checkpoint are absent. The active online replay-buffer trainer
  generates data itself and does not read that dataset, so its absence need not block
  training from scratch. Supplied pretrained checkpoints use different filenames.
- The baseline's warm-start copies previous primals and resets duals without shifting.
  The requested shifted warm-start in scope 3 will be an explicit change to document.

## Experiment commands and reproduction status

The original evaluation entry point is, from `reference/` after Julia setup:

```sh
./scripts/julia_server mcs
./scripts/julia_train params/train_parameters_medium_speed_lemniscate.toml lemniscate
```

The checked-in `startup.jl` enables LMPG vs MPC on the lemniscate; most tournament
jobs are commented out. A full-paper tournament therefore requires selecting jobs
and restoring sample counts, rather than treating current defaults as the full study.
There is **no JAX main experiment yet** at checkpoint 1.

| Comparison | Julia run | JAX run | Gap |
| --- | --- | --- | --- |
| Closed-loop trajectories | Not run | Not implemented | Unmeasured |
| Nash solve times | Not run | Not implemented | Unmeasured |
| Learned network parameters | Not run | Not implemented | Unmeasured |

KKT residual, implicit-solver gradient, and seeded closed-loop tests belong to
scopes 2 and 3 and are not claimed here. Checkpoint checks on macOS, Python 3.12.13,
JAX 0.12.0.1, CPU, float64:

```text
$ make lint
uv run ruff check .
All checks passed!

$ make test
collected 19 items
tests/test_game.py ..........                                            [ 52%]
tests/test_tracks.py .........                                           [100%]
============================== 19 passed in 4.56s ==============================
```

No Julia execution or cross-language numerical agreement is claimed at this checkpoint.
