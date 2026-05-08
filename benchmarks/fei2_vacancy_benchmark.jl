using Sunny
using LinearAlgebra
using Statistics
using Printf
using Dates
using Random

has_cuda = try
    using CUDA
    CUDA.functional()
catch
    false
end

if has_cuda
    println("CUDA available: ", CUDA.name(CUDA.device()))
end

function build_fei2_base()
    a = b = 4.05012
    c = 6.75214
    latvecs = lattice_vectors(a, b, c, 90, 90, 120)
    positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    types = ["Fe", "I", "I"]
    cryst = Crystal(latvecs, positions; types)
    cryst = subcrystal(cryst, "Fe")

    sys = System(cryst, [1 => Moment(s=1, g=2)], :SUN; seed=2)

    J1pm   = -0.236
    J1pmpm = -0.161
    J1zpm  = -0.261
    J2pm   = 0.026
    J3pm   = 0.166
    J0pm   = 0.037
    J1pm_  = 0.013
    J2apm  = 0.068

    J1zz   = -0.236
    J2zz   = 0.113
    J3zz   = 0.211
    J0zz   = -0.036
    J1zz_  = 0.051
    J2azz  = 0.073

    J1xx = J1pm + J1pmpm
    J1yy = J1pm - J1pmpm
    J1yz = J1zpm

    set_exchange!(sys, [J1xx   0.0    0.0;
                        0.0    J1yy   J1yz;
                        0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
    set_exchange!(sys, [J2pm   0.0    0.0;
                        0.0    J2pm   0.0;
                        0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
    set_exchange!(sys, [J3pm   0.0    0.0;
                        0.0    J3pm   0.0;
                        0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
    set_exchange!(sys, [J0pm   0.0    0.0;
                        0.0    J0pm   0.0;
                        0.0    0.0    J0zz], Bond(1,1,[0,0,1]))
    set_exchange!(sys, [J1pm_  0.0    0.0;
                        0.0    J1pm_  0.0;
                        0.0    0.0    J1zz_], Bond(1,1,[1,0,1]))
    set_exchange!(sys, [J2apm  0.0    0.0;
                        0.0    J2apm  0.0;
                        0.0    0.0    J2azz], Bond(1,1,[1,2,1]))

    D = 2.165
    set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)

    return sys, cryst
end

function find_ground_state(sys)
    sys_large = resize_supercell(sys, (4, 4, 4))
    sys_min = reshape_supercell(sys_large, [1 0 0; 0 1 -2; 0 1 2])

    best_energy = Inf
    best_sys = nothing
    for trial in 1:50
        randomize_spins!(sys_min)
        minimize_energy!(sys_min)
        e = energy_per_site(sys_min)
        if e < best_energy
            best_energy = e
            best_sys = clone_system(sys_min)
        end
    end
    @printf("  Ground state energy/site: %.6f meV\n", best_energy)
    return best_sys
end

function build_vacancy_system(sys_min, nx, ny; vacancy_fraction=0.05)
    sys_big = repeat_periodically(sys_min, (nx, ny, 1))

    sys_inhom = to_inhomogeneous(sys_big)

    all_sites = collect(eachsite(sys_inhom))
    n_total = length(all_sites)
    n_vacancies = round(Int, n_total * vacancy_fraction)
    perm = randperm(sys_inhom.rng, n_total)
    for i in 1:n_vacancies
        set_vacancy_at!(sys_inhom, all_sites[perm[i]])
    end

    minimize_energy!(sys_inhom; maxiters=5_000)

    println("  System: $(nx)x$(ny)x1 supercell")
    println("  Total sites: $n_total")
    println("  Vacancies: $n_vacancies ($(round(100*vacancy_fraction, digits=1))%)")
    println("  Active sites: $(n_total - n_vacancies)")
    @printf("  Energy/site after vacancy doping: %.6f meV\n", energy_per_site(sys_inhom))

    return sys_inhom
end

function run_benchmark(sys_inhom, cryst; nq=200, tol=0.05)
    qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
    path = q_space_path(cryst, qs, nq)

    kernel = lorentzian(fwhm=0.3)
    energies = range(0, 10, 300)
    measure = ssf_perp(sys_inhom)

    println("  Building SpinWaveTheoryKPM (tol=$tol)...")
    swt_kry = SpinWaveTheoryKPM(sys_inhom; measure, tol, regularization=1e-4)

    println("\n  CPU benchmark...")
    small_path = q_space_path(cryst, [[0,0,0], [0.1,0,0]], 5)
    try
        intensities(swt_kry, small_path; energies=range(0,10,10), kernel)
    catch e
        println("  CPU warmup error: $e")
    end

    Na_total = nsites(sys_inhom)
    n_cpu_trials = Na_total > 1000 ? 1 : 3
    cpu_times = Float64[]
    local res_cpu
    for trial in 1:n_cpu_trials
        GC.gc()
        t = @elapsed begin
            res_cpu = intensities(swt_kry, path; energies, kernel)
        end
        push!(cpu_times, t)
        @printf("    CPU trial %d: %.3f s\n", trial, t)
        flush(stdout)
    end
    cpu_median = n_cpu_trials > 1 ? median(cpu_times) : cpu_times[1]
    @printf("  CPU median: %.3f s\n", cpu_median)
    flush(stdout)

    gpu_median = NaN
    local res_gpu = nothing
    if has_cuda
        println("\n  GPU benchmark...")
        swt_d = Sunny.to_device_batched(swt_kry, CUDABackend())
        kernel_d = kernel

        try
            CUDA.@sync intensities(swt_d, small_path; energies=range(0,10,10), kernel=kernel_d)
        catch e
            println("  GPU warmup error: $e")
        end

        gpu_times = Float64[]
        for trial in 1:5
            GC.gc(); CUDA.synchronize()
            t = @elapsed CUDA.@sync begin
                res_gpu = intensities(swt_d, path; energies, kernel=kernel_d)
            end
            push!(gpu_times, t)
            @printf("    GPU trial %d: %.3f s\n", trial, t)
            flush(stdout)
        end
        gpu_median = median(gpu_times)
        @printf("  GPU median: %.3f s\n", gpu_median)
        @printf("  Speedup: %.1fx\n", cpu_median / gpu_median)
        flush(stdout)

        println("\n  Validating GPU vs CPU...")
        cpu_data = res_cpu.data
        gpu_data = try
            Sunny.Intensities(res_gpu, cryst).data
        catch
            res_gpu.data
        end
        mask = abs.(cpu_data) .> 1e-10 * maximum(abs.(cpu_data))
        if any(mask)
            rel_err = maximum(abs.(cpu_data[mask] .- gpu_data[mask]) ./ abs.(cpu_data[mask]))
            mean_rel_err = mean(abs.(cpu_data[mask] .- gpu_data[mask]) ./ abs.(cpu_data[mask]))
            @printf("  Max relative error: %.2e\n", rel_err)
            @printf("  Mean relative error: %.2e\n", mean_rel_err)
        end
        flush(stdout)
    end

    return (cpu_time=cpu_median, gpu_time=gpu_median, res_cpu=res_cpu, res_gpu=res_gpu)
end

println("FeI₂ Vacancy-Doped SU(3) Benchmark")
if has_cuda
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("VRAM: $(round(CUDA.total_memory() / 1e9, digits=1)) GB")
end
println("Julia: $(VERSION)")
println()
flush(stdout)

println("Step 1: Building FeI₂ Hamiltonian (example 03 parameters)...")
sys, cryst = build_fei2_base()
flush(stdout)

println("Step 2: Finding ground state (50 random starts)...")
sys_min = find_ground_state(sys)
flush(stdout)

configs = [
    (10, 10),
    (20, 20),
    (30, 30),
]

results = []
for (nx, ny) in configs
    println("FeI₂ VACANCY-DOPED Benchmark: $(nx)x$(ny)x1, 5% vacancies")
    flush(stdout)

    sys_inhom = build_vacancy_system(sys_min, nx, ny; vacancy_fraction=0.05)
    flush(stdout)
    res = run_benchmark(sys_inhom, cryst; nq=200, tol=0.05)
    push!(results, (nx=nx, ny=ny, res...))
end

println("  System: inhomogeneous (to_inhomogeneous + set_vacancy_at!)")
println("  Vacancy fraction: 5%")
println("  Mode: SU(N) with N=3, D=2.165 meV single-ion anisotropy")
@printf("%-12s  %12s  %12s  %10s\n", "Size", "CPU (s)", "GPU (s)", "Speedup")
for r in results
    speedup = isnan(r.gpu_time) ? "N/A" : @sprintf("%.1fx", r.cpu_time / r.gpu_time)
    @printf("%-12s  %12.3f  %12.3f  %10s\n",
            "$(r.nx)x$(r.ny)x1", r.cpu_time, r.gpu_time, speedup)
end

flush(stdout)
