# Open-loop validation only: include the original game/dynamics without the ROS/server stack.
using JSON3, Statistics

reference = normpath(joinpath(@__DIR__, "..", "reference"))
license_match = match(r"ENV\[\"PATH_LICENSE_STRING\"\] = \"([^\"]+)\"",
    read(joinpath(reference, "startup.jl"), String))
if !isnothing(license_match)
    ENV["PATH_LICENSE_STRING"] = license_match.captures[1]
end

module DroneTrajectorySolver
using TrajectoryGamesBase: TrajectoryGamesBase, TrajectoryGame, TrajectoryGameCost,
    ProductDynamics, GeneralSumCostStructure, time_invariant_linear_dynamics,
    AbstractDynamics, state_dim, control_dim
using BlockArrays: blocks, mortar
using LinearAlgebra, StaticArrays, Random, TOML, QuadGK, ForwardDiff, Rotations
using LazySets: LazySets
using Symbolics: Symbolics
using SymbolicUtils: SymbolicUtils

source = joinpath(@__DIR__, "..", "reference", "DroneTrajectorySolver.jl", "src")
include(joinpath(source, "Spline", "SplinesWithCachedDerivatives.jl"))
using .SplinesWithCachedDerivatives: PeriodicSpline
include(joinpath(source, "Spline", "spline.jl"))
include(joinpath(source, "environment.jl"))
include(joinpath(source, "dynamics.jl"))
include(joinpath(source, "trajectories.jl"))
include(joinpath(source, "game.jl"))
end

using .DroneTrajectorySolver
using TrajectoryGamesBase: ProductDynamics, solve_trajectory_game!
using MCPTrajectoryGameSolver: Solver
using BlockArrays: mortar, blocks
using LinearAlgebra
using ParametricMCPs: ParametricMCPs

D = DroneTrajectorySolver
coeff = (; rot_angle_x=0.0, rot_angle_y=0.0, rot_angle_z=0.0,
    width_x=3.635, width_y=4.0, width_z=0.0,
    off_center_x=0.78, off_center_y=0.0, off_center_z=1.9,
    x_coeff=3.0, y_coeff=1.0, z_coeff=2.0)
original = D.lemniscate_traj(; track_coeff=coeff)
track = D.reparametrize_path(; traj=original)
gates = [track(original.length / 4), track(3 * original.length / 4)]
times = range(0, original.length, length=200)
# The exact Gaussian expression in utils.jl; no learned-model imports are needed.
qc = D.PeriodicSpline([sum(exp(-5 * sum(abs2, track(s) - g)) for g in gates) for s in times], times)
tangent = D.get_tangent_approximation(; trajectory=track)
state_bounds = (; lb=[fill(-Inf, 3); fill(-3.0, 3); fill(-Inf, 4); -3.0],
                  ub=[fill(Inf, 3); fill(3.0, 3); fill(Inf, 4); 3.0])
control_bounds = (; lb=fill(-Inf, 4), ub=fill(Inf, 4))
sp = D.pmm_mpcc_jerk(; dt=0.05, state_bounds, control_bounds)
game = D.setup_dp_mpccg_game(; dynamics=ProductDynamics([sp, sp]), trajectory=track, qc_of_s=qc)
context = [3.0, 1.5, 1.5, 0.001, 2.0, 1.5, 1.0, 0.075, 2.0, 3.0, 1.5, 0.0]

cases = []
solvers = Dict()
for (name, horizon, progress, speeds) in [
    ("start_h4", 4, [original.length - 1, original.length - 2.5], [0.0, 0.0]),
    ("start_h15", 15, [original.length - 1, original.length - 2.5], [0.0, 0.0]),
    ("interaction_h15", 15, [4.2, 4.7], [0.0, 0.0]),
    ("moving_h15", 15, [19.98427417, 18.81716934], [1.8, 2.5]),
]
    x0 = mortar([[track(s); tangent(s) * v; zeros(3); s; v] for (s, v) in zip(progress, speeds)])
    solver = get!(solvers, horizon) do
        Solver(game, horizon; context_dimension=12, compute_sensitivities=false)
    end
    strategy = solve_trajectory_game!(solver, game, x0; context)
    elapsed = Float64[]
    for _ in 1:10
        push!(elapsed, @elapsed solve_trajectory_game!(solver, game, x0; context))
    end
    raw = strategy.info.raw_solution
    problem = solver.mcp_problem_representation
    f = zeros(length(raw.z))
    problem.f!(f, raw.z, [x0; context; ones(2)])
    residual = maximum(abs, raw.z - clamp.(raw.z - f, problem.lower_bounds, problem.upper_bounds))
    @assert isfinite(residual) && residual < 1e-5
    xs = [[collect(ss.xs[t]) for ss in strategy.substrategies] for t in 1:horizon]
    us = [[collect(ss.us[t]) for ss in strategy.substrategies] for t in 1:horizon]
    push!(cases, (; name, horizon, x0=collect.(blocks(x0)), xs, us,
        residual, median_seconds=median(elapsed), status=string(raw.status)))
    println(name, " residual=", residual, " median_seconds=", median(elapsed))
    flush(stdout)
end

sample_s = collect(range(0, original.length, length=51))[1:end-1] .+ 0.001
samples = [(; s, path=track(s), tangent=tangent(s), qc=qc(s)) for s in sample_s]
output = (; source_commit="4b3468deb2377197c3fa20549459f1c86c2e2eb1",
    julia_version=string(VERSION), track_length=original.length, context, samples, cases)
mkpath(dirname(ARGS[1]))
write(ARGS[1], JSON3.write(output))
