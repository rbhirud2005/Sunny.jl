using Sunny, LinearAlgebra, Printf, Statistics, Random, Dates
using KernelAbstractions
using CUDA

const SEED      = parse(Int,     get(ENV, "SUNNY_BENCH_SEED",     "0"))
const N_REPS    = parse(Int,     get(ENV, "SUNNY_BENCH_REPS",     "5"))
const N_WARMUPS = parse(Int,     get(ENV, "SUNNY_BENCH_WARMUPS",  "2"))
const SLEEP_S   = parse(Float64, get(ENV, "SUNNY_BENCH_SLEEP",    "1.0"))
const NQ        = parse(Int,     get(ENV, "SUNNY_BENCH_NQ",       "150"))
const OUTDIR    =                get(ENV, "SUNNY_BENCH_OUTDIR",   joinpath(@__DIR__, "results"))

const SIZE_CONFIGS = Dict(
    "small"       => (label = "small  (9×9   → Na=81)",                 dims = (3,  3,  1), n_reps = nothing, n_warmups = nothing),
    "medium"      => (label = "medium (30×30  → Na=900)",                dims = (10, 10, 1), n_reps = nothing, n_warmups = nothing),
    "large"       => (label = "large  (99×99  → Na=9801)",               dims = (33, 33, 1), n_reps = nothing, n_warmups = nothing),
    "large_quick" => (label = "large  (99×99  → Na=9801) [quick 2+2]",   dims = (33, 33, 1), n_reps = 2,       n_warmups = 2),
)

const ALL_MODES = ["dipole", "SUN"]

function selected_sizes()
    raw = strip.(split(get(ENV, "SUNNY_BENCH_SIZES", "medium,large"), ","))
    unknown = setdiff(raw, collect(keys(SIZE_CONFIGS)))
    isempty(unknown) || error("Unknown SUNNY_BENCH_SIZES: $(join(unknown, ", "))")
    return [(key = s, SIZE_CONFIGS[s]...) for s in raw]
end

function selected_modes()
    raw = strip.(split(get(ENV, "SUNNY_BENCH_MODES", "dipole,SUN"), ","))
    unknown = setdiff(raw, ALL_MODES)
    isempty(unknown) || error("Unknown SUNNY_BENCH_MODES: $(join(unknown, ", "))")
    return raw
end

function robust_stats(times)
    sorted = sort(times)
    n = length(sorted)
    μ = mean(sorted)
    σ = n > 1 ? std(sorted) : 0.0
    return (
        n      = n,
        median = median(sorted),
        mean   = μ,
        std    = σ,
        min    = first(sorted),
        max    = last(sorted),
        q25    = quantile(sorted, 0.25),
        q75    = quantile(sorted, 0.75),
        cv     = μ == 0 ? NaN : σ / μ,
    )
end

function print_stats(prefix, stats)
    @printf("    %s  median %.4f s  mean %.4f s  std %.4f s  IQR [%.4f,%.4f]  CV %.1f%%\n",
            prefix, stats.median, stats.mean, stats.std,
            stats.q25, stats.q75, 100 * stats.cv)
end

function timed_trial_cpu(f; seed = SEED)
    GC.gc(true)
    Random.seed!(seed)
    return @elapsed f()
end

function timed_trial_gpu(f; seed = SEED)
    GC.gc(true)
    CUDA.synchronize()
    Random.seed!(seed)
    t = @elapsed begin
        f()
        CUDA.synchronize()
    end
    return t
end

function run_trials(label, trial_fn; n_warmup = N_WARMUPS, n_reps = N_REPS)
    println("    warmup ($n_warmup):")
    for i in 1:n_warmup
        t = trial_fn()
        @printf("      warmup %d: %.4f s\n", i, t)
    end

    println("    measured ($n_reps):")
    times = Float64[]
    for i in 1:n_reps
        t = trial_fn()
        push!(times, t)
        @printf("      run %d: %.4f s\n", i, t)
        sleep(SLEEP_S)
    end

    stats = robust_stats(times)
    print_stats(label, stats)
    return (times = times, stats = stats)
end

function cuda_free_mib()
    free, total = CUDA.memory_info()
    return free / 1024^2, total / 1024^2
end

function cuda_used_mib()
    free, total = CUDA.memory_info()
    return (total - free) / 1024^2
end

function build_dipole_system(dims)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
    cryst   = Crystal(latvecs, [[0, 0, 0]])
    sys     = System(cryst, [1 => Moment(s=1/2, g=1)], :dipole; dims=(3, 3, 1), seed=0)
    set_exchange!(sys, 1.0, Bond(1, 1, [1, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    sys2 = to_inhomogeneous(repeat_periodically(sys, dims))
    for (s1, s2, off) in symmetry_equivalent_bonds(sys2, Bond(1, 1, [1, 0, 0]))
        set_exchange_at!(sys2, 1.0 + randn(sys2.rng) / 3, s1, s2; offset = off)
    end
    minimize_energy!(sys2, maxiters = 2_000)
    return cryst, sys2
end

function build_sun_system(dims)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
    cryst   = Crystal(latvecs, [[0, 0, 0]])
    sys     = System(cryst, [1 => Moment(s=1, g=2)], :SUN; dims=(3, 3, 1), seed=0)
    set_exchange!(sys, -1.0, Bond(1, 1, [1, 0, 0]))
    polarize_spins!(sys, [0, 0, 1])
    minimize_energy!(sys)

    sys2 = repeat_periodically(sys, dims)
    minimize_energy!(sys2, maxiters = 2_000)
    return cryst, sys2
end

function write_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "mode,size,Na,L,twoL,Nq,N_chains,path,n,",
                    "median_s,mean_s,std_s,min_s,max_s,q25_s,q75_s,cv,",
                    "speedup,rel_err_vs_cpu,gpu_mib_used,raw_times_s")
        for row in rows
            raw = join((@sprintf("%.6f", t) for t in row.times), ";")
            @printf(io, "%s,%s,%d,%d,%d,%d,%d,%s,%d,",
                    row.mode, row.size, row.Na, row.L, row.twoL,
                    row.Nq, row.N_chains, row.path, row.stats.n)
            @printf(io, "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,",
                    row.stats.median, row.stats.mean, row.stats.std,
                    row.stats.min, row.stats.max, row.stats.q25, row.stats.q75,
                    row.stats.cv)
            @printf(io, "%.4f,%.3e,%.2f,%s\n",
                    row.speedup, row.rel_err_vs_cpu, row.gpu_mib, raw)
        end
    end
end

function run_one(mode, size_key, label, dims; n_reps = N_REPS, n_warmups = N_WARMUPS)
    is_sun = (mode == "SUN")
    @printf("  Mode: %s   Size: %s\n", mode, label)

    cryst, sys = is_sun ? build_sun_system(dims) : build_dipole_system(dims)
    Na = Sunny.nsites(sys)

    measure  = is_sun ? ssf_trace(sys) : ssf_perp(sys)
    Nf       = is_sun ? sys.Ns[1] - 1 : 1
    L        = Nf * Na
    twoL     = 2L
    Nobs     = size(measure.observables, 1)

    qs_path  = [[0, 0, 0], [1/3, 1/3, 0], [1/2, 0, 0], [0, 0, 0]]
    path     = q_space_path(cryst, qs_path, NQ)
    Nq       = length(path.qs)
    N_chains = Nq * Nobs
    ω_max    = is_sun ? 4.0 : 3.0
    energies = range(0.0, ω_max, NQ)
    broadening = lorentzian(fwhm = 0.4)

    @printf("  Na=%d  L=%d  twoL=%d  Nq=%d  Nobs=%d  N_chains=%d  Julia threads=%d\n",
            Na, L, twoL, Nq, Nobs, N_chains, Threads.nthreads())

    swt_kry = SpinWaveTheoryKPM(sys; measure, tol = 0.05, method = :lanczos)

    println("\n  [0] CPU reference (single call, untimed):")
    Random.seed!(SEED)
    t_ref = @elapsed res_cpu_ref = intensities(swt_kry, path; energies, kernel = broadening)
    maxref = maximum(abs, res_cpu_ref.data)
    @printf("    t=%.4f s   max|I|=%.3e\n", t_ref, maxref)

    println("\n  [1] CPU FP64 Lanczos:")
    cpu = run_trials("CPU FP64",
                     () -> timed_trial_cpu(() -> intensities(swt_kry, path;
                                                             energies, kernel = broadening));
                     n_warmup = n_warmups, n_reps = n_reps)

    println("\n  [2] GPU FP64 Lanczos (CUDA KA):")
    gpu        = nothing
    speedup    = NaN
    rel_err    = NaN
    gpu_mib    = NaN
    rows = Any[]

    try
        cuda_backend = CUDA.CUDABackend()

        used_before = cuda_used_mib()
        swt_d = Sunny.to_device_batched(swt_kry, cuda_backend; precision = Float64)
        used_after = cuda_used_mib()
        @printf("    to_device_batched: +%.1f MiB on device  (%.1f MiB total used)\n",
                used_after - used_before, used_after)

        gpu = run_trials("GPU FP64",
                         () -> timed_trial_gpu(
                             () -> Sunny.intensities(swt_d, path; energies, kernel = broadening));
                         n_warmup = n_warmups, n_reps = n_reps)

        Random.seed!(SEED)
        CUDA.synchronize()
        res_gpu = Sunny.intensities(swt_d, path; energies, kernel = broadening)
        CUDA.synchronize()
        rel_err = maximum(abs, res_gpu.data .- res_cpu_ref.data) / maxref
        gpu_mib = cuda_used_mib()
        speedup = cpu.stats.median / gpu.stats.median

        @printf("    rel err vs CPU FP64: %.3e\n", rel_err)
        @printf("    GPU memory in use:   %.1f MiB\n", gpu_mib)
        @printf("    Speedup (median):    %.2fx\n", speedup)

    catch e
        @printf("    GPU FAILED: %s\n", first(sprint(showerror, e), 300))
    end

    println()
    println("  " * "─" ^ 80)
    @printf("  %-20s  %10s  %10s  %10s  %12s  %s\n",
            "Path", "Median (s)", "Mean (s)", "Std (s)", "Speedup", "Rel err")
    println("  " * "─" ^ 80)
    @printf("  %-20s  %10.4f  %10.4f  %10.4f  %12s  reference\n",
            "CPU FP64 Lanczos", cpu.stats.median, cpu.stats.mean, cpu.stats.std, "1.00×")
    if !isnothing(gpu)
        @printf("  %-20s  %10.4f  %10.4f  %10.4f  %11.2f×  %.2e\n",
                "GPU FP64 KA CUDA", gpu.stats.median, gpu.stats.mean, gpu.stats.std,
                speedup, rel_err)
    else
        @printf("  %-20s  %10s  %10s  %10s  %12s  FAILED\n",
                "GPU FP64 KA CUDA", "─", "─", "─", "─")
    end
    println()

    push!(rows, (mode = mode, size = size_key,
                 Na = Na, L = L, twoL = twoL, Nq = Nq, N_chains = N_chains,
                 path = "cpu_fp64_lanczos",
                 stats = cpu.stats, times = cpu.times,
                 speedup = 1.0, rel_err_vs_cpu = 0.0, gpu_mib = NaN))
    if !isnothing(gpu)
        push!(rows, (mode = mode, size = size_key,
                     Na = Na, L = L, twoL = twoL, Nq = Nq, N_chains = N_chains,
                     path = "gpu_fp64_lanczos_cuda",
                     stats = gpu.stats, times = gpu.times,
                     speedup = speedup, rel_err_vs_cpu = rel_err, gpu_mib = gpu_mib))
    end

    return (mode = mode, size = size_key, label = label,
            Na = Na, L = L, twoL = twoL, Nq = Nq, N_chains = N_chains,
            cpu = cpu, gpu = gpu, speedup = speedup, rel_err = rel_err, rows = rows)
end

println("  Sunny KAExt — NVIDIA CUDA Benchmark")

CUDA.functional() || error("CUDA not functional on this system — check driver and CUDA.jl installation")

let dev = CUDA.device()
    free_mib, total_mib = cuda_free_mib()
    println("GPU:          ", CUDA.name(dev))
    @printf("VRAM:         %.2f GiB total  /  %.2f GiB free\n",
            total_mib / 1024, free_mib / 1024)
    @printf("CUDA:         %s\n", CUDA.runtime_version())
end
@printf("Julia:        %s   threads=%d   BLAS=%d\n",
        VERSION, Threads.nthreads(), BLAS.get_num_threads())
println()
@printf("Config:       SEED=%d  N_REPS=%d  N_WARMUPS=%d  SLEEP=%.1fs  NQ=%d\n",
        SEED, N_REPS, N_WARMUPS, SLEEP_S, NQ)
println()

sizes = selected_sizes()
modes = selected_modes()
results = Any[]

for mode in modes, sz in sizes
    try
        nr = something(sz.n_reps, N_REPS)
        nw = something(sz.n_warmups, N_WARMUPS)
        r = run_one(mode, sz.key, sz.label, sz.dims; n_reps = nr, n_warmups = nw)
        push!(results, r)
    catch e
        @printf("\nFAILED: mode=%s size=%s\n%s\n", mode, sz.key,
                first(sprint(showerror, e), 300))
    end
end

println()
println("SUMMARY")
@printf("%-8s  %-10s  %6s  %6s  %6s  %7s  %16s  %16s  %10s  %10s\n",
        "Mode", "Size", "Na", "L", "twoL", "N_ch",
        "CPU med (s)", "GPU med (s)", "Speedup", "Rel err")
for r in results
    if !isnothing(r.gpu)
        @printf("%-8s  %-10s  %6d  %6d  %6d  %7d  %16.4f  %16.4f  %9.2f×  %10.2e\n",
                r.mode, r.size, r.Na, r.L, r.twoL, r.N_chains,
                r.cpu.stats.median, r.gpu.stats.median, r.speedup, r.rel_err)
    else
        @printf("%-8s  %-10s  %6d  %6d  %6d  %7d  %16.4f  %16s  %10s  %10s\n",
                r.mode, r.size, r.Na, r.L, r.twoL, r.N_chains,
                r.cpu.stats.median, "FAILED", "─", "─")
    end
end
println()

let dev = CUDA.device()
    @printf("Machine: %s / Julia %s / %d Julia threads / %d BLAS threads\n",
            CUDA.name(dev), VERSION, Threads.nthreads(), BLAS.get_num_threads())
end

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
csv_path  = joinpath(OUTDIR, "nvidia_cuda_benchmark_$timestamp.csv")
write_csv(csv_path, reduce(vcat, (r.rows for r in results)))
