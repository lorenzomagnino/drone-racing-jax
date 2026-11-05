# Reduced validation environment; versions come from the pinned original Manifest.
using Pkg, TOML

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
reference = normpath(joinpath(@__DIR__, "..", "reference"))
manifest = TOML.parsefile(joinpath(reference, "Manifest.toml"))["deps"]
Pkg.activate(joinpath(reference, "_parity"))
Pkg.add(PackageSpec(url="https://github.com/andrejcarlo/ParametricMCPs.jl.git",
    rev="f32f9ceafd92d5bb6cedb90c699b6ca06cdfa4c6"))
names = ["MCPTrajectoryGameSolver", "TrajectoryGamesBase", "DataInterpolations",
    "Symbolics", "SymbolicUtils", "SymbolicTracingUtils", "BlockArrays", "JSON3",
    "QuadGK", "ForwardDiff", "StaticArrays", "Rotations", "LazySets", "ChainRulesCore"]
Pkg.add([PackageSpec(name=name, version=manifest[name][1]["version"]) for name in names])
