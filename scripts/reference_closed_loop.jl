# Original synchronous measurement/augmentation rules and unshifted primal warm starts.
include(joinpath(@__DIR__, "reference_game.jl"))

@eval D begin
    using Optim
    trajectory_rp = $track
    trajectory = $original
    include(joinpath(source, "Server", "referee.jl"))
    include(joinpath(source, "trajectory_optimizer.jl"))
end

horizon = 15
solver = Solver(game, horizon; context_dimension=12, compute_sensitivities=false)
cases = []
rng = D.Random.MersenneTwister(42)
for (name, steps, progress, speeds, randomize) in [
    ("seed42_h15", 100, [original.length - 1, original.length - 2.5], [0.0, 0.0], true),
    ("overtake_h15", 10, [4.2, 4.94], [1.8, 2.5], false),
]
    x0 = [[track(s) + (randomize ? rand(rng, 3) * 0.3 .- 0.15 : zeros(3));
        tangent(s) * v; zeros(3); s; v] for (s, v) in zip(progress, speeds)]
    physical = [x[1:9] for x in x0]
    progress = copy(progress)
    attacker = 2
    guesses = Any[nothing, nothing]
    states = [deepcopy(physical)]
    planning_states, controls, attackers, residuals, step_seconds = [], [], [], [], []
    for step in 1:steps
        start = time_ns()
        progress = [D.get_player_progress_initial_guess(physical[i][1:3]..., progress[i]) for i in 1:2]
        difference = progress[1] - progress[2]
        if attacker == 1 && difference > 0.75
            attacker = 2
        elseif attacker == 2 && difference < -0.75
            attacker = 1
        end
        augmented = [D.augment_mpcc_state(physical[i], i; progress_estimate=progress,
            tangent_approximation=tangent, attacker_nom_velocity=3.0) for i in 1:2]
        inputs, errors = [], []
        for ego in 1:2
            order = ego == 1 ? [1, 2] : [2, 1]
            initial = mortar(augmented[order])
            c = copy(context)
            c[5] = ego == 1 ? attacker : 3 - attacker
            strategy = solve_trajectory_game!(solver, game, initial; context=c,
                initial_guess=guesses[ego])
            raw = strategy.info.raw_solution
            n_primals = horizon * 30
            guesses[ego] = [raw.z[1:n_primals]; zeros(length(raw.z) - n_primals)]
            push!(inputs, copy(strategy.substrategies[1].us[1]))
            problem = solver.mcp_problem_representation
            f = zeros(length(raw.z))
            problem.f!(f, raw.z, [initial; c; ones(2)])
            error = maximum(abs, raw.z - clamp.(raw.z - f, problem.lower_bounds, problem.upper_bounds))
            @assert isfinite(error) && error < 1e-5
            push!(errors, error)
        end
        push!(step_seconds, (time_ns() - start) / 1e9)
        physical = [sp(augmented[i], inputs[i])[1:9] for i in 1:2]
        push!(states, deepcopy(physical))
        push!(planning_states, augmented)
        push!(controls, inputs)
        push!(attackers, attacker)
        push!(residuals, errors)
    end
    push!(cases, (; name, steps, horizon, x0, states, planning_states, us=controls,
        attackers, residuals, step_seconds))
    println(name, ": ", steps, " steps, switches=", sum(diff(attackers) .!= 0))
    flush(stdout)
end
output = (; source_commit="4b3468deb2377197c3fa20549459f1c86c2e2eb1",
    julia_version=string(VERSION), context, dt=0.05, cases)
mkpath(dirname(ARGS[1]))
write(ARGS[1], JSON3.write(output))
