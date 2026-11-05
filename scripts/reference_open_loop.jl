include(joinpath(@__DIR__, "reference_game.jl"))

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
