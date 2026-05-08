using Sunny, KernelAbstractions, CUDA, Printf, Statistics, Dates

backend = CUDABackend()
println("Per-Lanczos-Iteration Benchmark")
println("GPU:   $(CUDA.name(CUDA.device()))")
println("VRAM:  $(round(CUDA.total_memory()/1e9, digits=1)) GB")
println("Julia: $(VERSION)")
println()

function build_dipole(nx, ny)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
    cryst   = Crystal(latvecs, [[0, 0, 0]])
    sys     = System(cryst, [1 => Moment(s=1/2, g=2)], :dipole; dims=(nx, ny, 1), seed=0)
    set_exchange!(sys, -1.0, Bond(1, 1, [1, 0, 0]))
    polarize_spins!(sys, [0, 0, 1]); minimize_energy!(sys)
    return sys, cryst
end

function build_SUN(nx, ny)
    latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
    cryst   = Crystal(latvecs, [[0, 0, 0]])
    sys     = System(cryst, [1 => Moment(s=1, g=2)], :SUN; dims=(nx, ny, 1), seed=0)
    set_exchange!(sys, -1.0, Bond(1, 1, [1, 0, 0]))
    set_onsite_coupling!(sys, S -> -2.0*S[3]^2, 1)
    polarize_spins!(sys, [0, 0, 1]); minimize_energy!(sys)
    return sys, cryst
end

NITERS = 20
NQ     = 150

function run_size(mode, nx, ny; cpu_skip=false, gpu_skip=false)
    sys, cryst = mode == :dipole ? build_dipole(nx, ny) : build_SUN(nx, ny)
    Na   = length(sys.dipoles)
    Nf   = mode == :SUN ? 2 : 1
    twoL = 2 * Nf * Na

    qs       = [[0,0,0], [1/2,0,0], [1/3,1/3,0], [0,0,0]]
    path     = q_space_path(cryst, qs, NQ)
    energies = range(0, 4, 10)
    kernel   = lorentzian(fwhm=0.4)
    measure  = ssf_trace(sys)
    Nobs     = 1

    swt_kry = SpinWaveTheoryKPM(sys; measure, niters=NITERS)

    t_cpu = NaN
    if !cpu_skip
        intensities(swt_kry, path; energies, kernel)
        t_cpu = @elapsed intensities(swt_kry, path; energies, kernel)
    end

    t_gpu = NaN
    if !gpu_skip
        swt_d = Sunny.to_device_batched(swt_kry, backend)
        CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)
        CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)
        t_gpu = @elapsed CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)
    end

    return (Na=Na, twoL=twoL, Nobs=Nobs, t_cpu=t_cpu, t_gpu=t_gpu)
end

configs = [
    (:dipole,  10, 10, false, false),
    (:dipole,  20, 20, false, false),
    (:dipole,  30, 30, false, false),
    (:dipole,  50, 50, false, false),
    (:dipole,  99, 99, true,  false),
    (:SUN,     10, 10, false, false),
    (:SUN,     20, 20, false, false),
    (:SUN,     30, 30, false, false),
    (:SUN,     50, 50, true,  false),
    (:SUN,     99, 99, true,  false),
]

println("Fixed niters=$NITERS, Nq=$NQ, Nobs=1 (ssf_trace)")
println("Per-iter time = wall_time / (niters × Nobs × Nq) = wall_time / $(NITERS * 1 * NQ)")
@printf("%-8s  %7s  %6s  %6s  %10s  %10s  %12s  %12s  %8s\n",
        "Mode", "Size", "Na", "2L",
        "CPU (s)", "GPU (s)", "CPU μs/iter", "GPU μs/iter", "Speedup")

results = []
for (mode, nx, ny, cpu_skip, gpu_skip) in configs
    print("  Running $(mode) $(nx)×$(ny)...")
    r = run_size(mode, nx, ny; cpu_skip, gpu_skip)
    denom = NITERS * r.Nobs * NQ
    cpu_us = r.t_cpu * 1e6 / denom
    gpu_us = r.t_gpu * 1e6 / denom
    speedup = isnan(r.t_cpu) || isnan(r.t_gpu) ? NaN : r.t_cpu / r.t_gpu

    cpu_str     = cpu_skip  ? "     skip" : @sprintf("%9.3f", r.t_cpu)
    gpu_str     = gpu_skip  ? "     skip" : @sprintf("%9.3f", r.t_gpu)
    cpu_us_str  = cpu_skip  ? "        skip" : @sprintf("%12.2f", cpu_us)
    gpu_us_str  = gpu_skip  ? "        skip" : @sprintf("%12.2f", gpu_us)
    spd_str     = isnan(speedup) ? "    N/A" : @sprintf("%7.1fx", speedup)

    @printf("%-8s  %3dx%-3d  %6d  %6d  %10s  %10s  %12s  %12s  %8s\n",
            mode, nx, ny, r.Na, r.twoL,
            cpu_str, gpu_str, cpu_us_str, gpu_us_str, spd_str)
    println()
    push!(results, (mode, nx, ny, r, speedup))
end

println("\nKey metric: GPU μs/iter vs CPU μs/iter → per-matvec speedup")
