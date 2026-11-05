# Strategically robust drone racing

This extension implements the **nonlinear min–max objective** in the local draft
`../../rhg-paper/chapters/sr-olne.tex`, equations `eq:ball` and `eq:srolne`, read at
paper commit `d5bb816e0bdf610aa9d6778ad0cfc6fc56f91490`:

\[
u^{i*}\in\arg\min_{u^i\in U^i}\max_{v^{-i}\in U^{-i},\;
\|v^{-i}-u^{-i*}\|_2\le\varepsilon} J^i(u^i,v^{-i}).
\]

The norm is the unweighted Euclidean norm over the **entire input trajectory**.
It is not one independent ball per time step. Opponent deviations obey the original
world and velocity bounds and dynamics. Own dynamics, constraints, and the racing
cost coefficients are unchanged. Only the cost sees the adversarial trajectory.
The source's horizon indexing (H states, H inputs, H−1 transitions) is retained.

The draft's `epsilon * dual_norm(opponent_cost_gradient)` expression is exact for
affine opponent dependence under additional feasibility conditions. The racing
collision penalty is nonlinear, so **that approximation is not used here**.

## Reduction and numerical method

The default Julia racing game gives only the attacker a collision cost; the
defender's objective does not depend on the attacker. Consequently, its selected
nominal plan remains a best response in the robust game. The algorithm first
solves the nominal game, retains the defender's plan, then solves the attacker's
coupled minimization / adversarial maximization KKT equations.

The adversary can affect the attacker's cost only through Cartesian position.
Its virtual path-acceleration input and the last unused control can stay at their
nominal values without loss of worst-case value. The remaining deviation is
`delta_jerk = epsilon * w`, with one constraint `sum(w**2) <= 1`. The adversary's
unchanged vtheta bounds remain satisfied by the nominal center; its Cartesian
position/velocity constraints are enforced explicitly. This is an exact reduction
for the original default cost, not a reduced approximation of the uncertainty set.

Each local solve uses the shared damped Newton KKT implementation, three
deterministic adversarial starts directed toward likely close encounters, and
complementarity for both players' private bounds and the ambiguity ball. Candidate
adversaries must beat feasible geometric probes, including the nominal center.
Among converged candidates passing that check, the least-cost robust plan is
selected. Tests and the open-loop study independently re-solve the inner problem
with the selected plan held fixed. If a linear-dynamics reachability bound proves that the entire ball cannot
activate a collision penalty, the nominal plan is returned directly.

**These are local OL-SRNE candidates.** The collision game is nonconvex and the
inner objective is not globally concave. KKT residuals, multiple starts, and probe
checks do not certify a global inner maximum or a global Nash equilibrium. At a
degenerate active set or a branch switch, implicit derivatives may not exist.
The scalar radius is held fixed, as in the draft's receding-horizon definition;
the returned plan supports reverse-mode implicit gradients in continuous cost
parameters and the initial state on nondegenerate branches. No training is run.

The reduction supports the default **zero blocking weight**. A nonzero defender
blocking weight is rejected through `converged=False` for positive radius; treating
that case as an unchanged defender would be incorrect. Radius zero exactly returns
the nominal solver. Check `converged` before applying a plan.

## Files and commands

- `drone_racing_jax/robust.py`: coupled min–max solver and fixed-plan opponent oracle.
- `drone_racing_jax/solver.py`: shared Newton kernel; nominal equations are preserved.
- `drone_racing_jax/controller.py`: the same measured-state controller with fixed radius.
- `drone_racing_jax/robust_study.py`: paired ordinary/adversarial experiments.
- `tests/test_robust.py`: nominal limit, feasible ambiguity ball, analytic inner maxima,
  inner/outer KKT conditions, implicit gradients, and closed-loop reproducibility.

```sh
uv run python -m drone_racing_jax.race --radius 5 --steps 100 --seed 42 --output outputs/sr-race
uv run python -m drone_racing_jax.robust_study --steps 100 --seeds 0 1 2
```

The study writes incremental `results.json`, per-run state arrays, and
`comparison.png` under ignored `outputs/robust_study/`. It reports failures and
incomplete runs rather than counting them as good outcomes.

## Experiment protocol

The open-loop check uses progress `[4.2, 4.7]`, zero physical initial velocity and
acceleration, H=15, and the original medium-speed cost parameters. Nominal and SR
plans are evaluated by re-solving the same constrained inner problem at radii
0, 1, 5, and 10. Lower attacker cost is better; the cost can be negative because
of the progress reward.

The closed-loop check pairs exactly the same initial states across design radii
0, 5, and 10, for seeds 0, 1, 2 and 100 steps (5 s). The defender is initially ahead,
with the same Cartesian uniform ±0.15 m starting offsets as the nominal demo.
Two opponent modes are evaluated:

- **Ordinary play:** both drones use their selected nominal/SR plans.
- **Adversarial deviation, radius 5:** the current attacker selects its plan; the
  defender then applies the first input of a feasible radius-5 worst-response plan
  around the defender center predicted in that attacker's game. The entire deviation
  is checked before execution. The adversary re-plans at the next measurement.

This is the same adversarial policy and threat budget for every design radius,
not an adversary weakened for the robust controller. The actual input sequence
across multiple re-planning calls is not claimed to lie in a single initial ball.
The definition places a new ball around the current point prediction at each call.

Radii are in raw input L2 units, not meters. At H=15 and dt=0.05, the last-position
operator norm from jerk deviations is approximately 0.020469 s^3 before feasibility
constraints. Thus radius 5 permits at most about 0.102 m of predicted position
deviation at the end of this horizon, and radius 10 about 0.205 m.

Metrics are actual physical minimum separation, samples below the original 0.35 m
collision threshold, original realized attacking cost per tick, progress gain,
constraint/KKT diagnostics, and computation time. There is no hard collision
constraint added by robustness. Three short seeded runs are illustrative rather
than a statistical validation, and the experiment does not reproduce the full
paper tournament, asynchronous delays, or a learned policy.

## Results

All **18/18** paired runs completed all 100 steps. No run had a sample below the
0.35 m collision threshold, and none switched roles in these 5 s windows. Every
reported controller KKT residual was below `9.96e-9`. The largest applied opponent
plan deviation was `5.000000025` in L2 norm (floating-point feasibility tolerance),
and the smallest adversary state slack was `-1.78e-15`.

Open-loop attacker costs at the close-interaction state:

| Radius | Nominal plan, adversarial cost | SR plan, adversarial cost | SR plan, nominal cost |
| --- | --- | --- | --- |
| 0 | -30.582771 | -30.582771 | -30.582771 |
| 1 | -30.413918 | -30.414500 | -30.582057 |
| 5 | -29.683354 | -29.695098 | -30.569453 |
| 10 | -28.683025 | -28.713134 | -30.548175 |

At each positive radius, SR improved the computed worst-case cost, while giving up
some nominal open-loop performance. The improvement is small for these parameters;
negative numbers reflect the progress reward, not a sign error.

Closed-loop means over the three paired seeds:

| Opponent mode | Design radius | Minimum separation (m) | Attacking cost/tick (lower is better) | Attacker progress gain (m) |
| --- | --- | --- | --- | --- |
| Ordinary | 0 | 0.6913 | -3.4399 | 11.8827 |
| Ordinary | 5 | 0.7122 | -3.4783 | 11.8384 |
| Ordinary | 10 | 0.7230 | -3.5071 | 11.8010 |
| Adversarial, radius 5 | 0 | 0.5650 | -3.2832 | 11.7668 |
| Adversarial, radius 5 | 5 | 0.5842 | -3.3172 | 11.7202 |
| Adversarial, radius 5 | 10 | 0.5699 | -3.3386 | 11.7027 |

Here the attacker remained drone 2 throughout each run. Radius 5 increased average
minimum clearance under the bounded adversary by about **1.9 cm**, while giving up
about **4.7 cm** of progress over 5 s. Radius 10 improved the realized cost further,
but had **less clearance than radius 5** under the adversary and made less progress.
Thus larger radius was not uniformly better. Since even nominal play had zero
collision samples, these runs do not demonstrate a reduction in crashes.

Mean per-run median warm controller times in ordinary play were approximately
`52 / 165 / 195 ms` for radii `0 / 5 / 10`. With the adversary's additional oracle
solve included in the timed simulation step they were `85 / 190 / 186 ms`.
These CPU measurements include both ego games and projection, exclude the first
compiled step, and were collected while verification also ran on the machine.
They are indicative of overhead, not controlled hardware benchmarks. The SR solver
has a substantial cost and is not demonstrated to meet the original planning rate.

![Paired closed-loop results](outputs/robust_study/comparison.png)

The plot and raw arrays are generated artifacts kept out of Git. Re-run the study
command to recreate them. No model fitting, training data collection pipeline, or
learned policy is included in this extension.

## Verification

Tests include an independent outer stationarity calculation, exact geometric
one-transition inner maxima (with and without an active opponent velocity bound),
the full-horizon deviation budget, opponent/own feasibility, radius-zero identity,
the one-sided defender reduction, fixed-plan inner re-solves, reverse-mode implicit
gradients against finite differences, deterministic closed-loop execution, and the
adversarial simulation's pre-execution feasibility checks.

Final verification output:

```text
$ make lint
uv run ruff check .
All checks passed!

$ make test
collected 56 items
tests/test_controller.py ..............                                  [ 25%]
tests/test_game.py ..........                                            [ 42%]
tests/test_robust.py ..........                                          [ 60%]
tests/test_solver.py .............                                       [ 83%]
tests/test_tracks.py .........                                           [100%]
======================== 56 passed in 88.18s (0:01:28) ========================
```

Ruff and Black also pass the pre-commit checks. No new Python dependency was added.
