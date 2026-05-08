using Sunny, KernelAbstractions, CUDA, Printf, Dates

backend = CUDABackend()
println("FeI₂ SU(N=3) KPM GPU Benchmark")
println("GPU:   $(CUDA.name(CUDA.device()))")
println("VRAM:  $(round(CUDA.total_memory()/1e9, digits=1)) GB")
println("Julia: $(VERSION)")
println(); flush(stdout)

a = b = 4.05012; c = 6.75214
latvecs   = lattice_vectors(a, b, c, 90, 90, 120)
positions = [[0,0,0], [1/3,2/3,1/4], [2/3,1/3,3/4]]
cryst     = subcrystal(Crystal(latvecs, positions; types=["Fe","I","I"]), "Fe")

sys0 = System(cryst, [1 => Moment(s=1, g=2)], :SUN; seed=2)

J1pm=-0.236; J1pmpm=-0.161; J1zpm=-0.261
J2pm=0.026;  J2zz=0.113
J3pm=0.166;  J3zz=0.211
J0pm=0.037;  J0zz=-0.036
Jp1pm=0.013; Jp1zz=0.051
Jp2pm=0.068; Jp2zz=0.073
J1zz=-0.236

set_exchange!(sys0, [J1pm+J1pmpm 0 0; 0 J1pm-J1pmpm J1zpm; 0 J1zpm J1zz], Bond(1,1,[1,0,0]))
set_exchange!(sys0, [J2pm 0 0; 0 J2pm 0;   0 0 J2zz],   Bond(1,1,[1,2,0]))
set_exchange!(sys0, [J3pm 0 0; 0 J3pm 0;   0 0 J3zz],   Bond(1,1,[2,0,0]))
set_exchange!(sys0, [J0pm 0 0; 0 J0pm 0;   0 0 J0zz],   Bond(1,1,[0,0,1]))
set_exchange!(sys0, [Jp1pm 0 0; 0 Jp1pm 0; 0 0 Jp1zz],  Bond(1,1,[1,0,1]))
set_exchange!(sys0, [Jp2pm 0 0; 0 Jp2pm 0; 0 0 Jp2zz],  Bond(1,1,[1,2,1]))
set_onsite_coupling!(sys0, S -> -2.165*S[3]^2, 1)

sys_mag = reshape_supercell(sys0, [1 0 0; 0 1 -2; 0 1 2])

print("Finding ground state (20 restarts)..."); flush(stdout)
let best_e = Inf, best_mag = clone_system(sys_mag)
    for _ in 1:20
        randomize_spins!(sys_mag); minimize_energy!(sys_mag)
        e = energy_per_site(sys_mag)
        if e < best_e
            best_e   = e
            best_mag = clone_system(sys_mag)
        end
    end
    global sys_mag = best_mag
end
println(" done.")
println("Magnetic cell: $(length(sys_mag.dipoles)) Fe sites, E/site = $(round(energy_per_site(sys_mag),digits=4)) meV")
flush(stdout)

qs       = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path     = q_space_path(cryst, qs, 300)
energies = range(0, 10, 300)
kernel   = lorentzian(fwhm=0.3)

println("\n=== Exact LSWT (minimal cell, domain-averaged) ==="); flush(stdout)
swt_exact = SpinWaveTheory(sys_mag; measure=ssf_perp(sys_mag))
rotations = [([0,0,1], n*(2π/3)) for n in 0:2]
t_exact   = @elapsed begin
    res_exact = domain_average(cryst, path; rotations, weights=[1,1,1]) do path_r
        intensities(swt_exact, path_r; energies, kernel)
    end
end
@printf("Exact LSWT time: %.3f s\n", t_exact); flush(stdout)

println("FeI₂ KPM benchmark  (SU(N=3), tol=0.05, fwhm=0.3)")
@printf("%-8s  %6s  %6s  %9s  %9s  %8s\n",
        "Size", "Na", "twoL", "CPU (s)", "GPU (s)", "Speedup")
flush(stdout)

println("(CPU 10×10 from prior run: 698.65 s → see speedup column)"); flush(stdout)

cpu_known = Dict(10 => 698.65)

for N in [10, 20, 30]
    print("  Running $(N)×$(N) GPU..."); flush(stdout)
    sys_r   = repeat_periodically(sys_mag, (N, N, 1))
    Na      = length(sys_r.dipoles)
    Nf      = 2
    twoL    = 2 * Nf * Na
    measure = ssf_perp(sys_r)
    swt_kry = SpinWaveTheoryKPM(sys_r; measure, tol=0.05)

    swt_d  = Sunny.to_device_batched(swt_kry, backend)
    CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)
    CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)
    t_gpu  = @elapsed CUDA.@sync Sunny.intensities(swt_d, path; energies, kernel)

    t_cpu   = get(cpu_known, N, NaN)
    spd_str = isnan(t_cpu) ? "     N/A" : @sprintf("%7.1fx", t_cpu / t_gpu)
    cpu_str = isnan(t_cpu) ? "     skip" : @sprintf("%9.2f", t_cpu)

    @printf("%-8s  %6d  %6d  %9s  %9.2f  %8s\n",
            "$(N)×$(N)", Na, twoL, cpu_str, t_gpu, spd_str)
    flush(stdout)
end

println("\nGPU relerr vs CPU (10×10, from prior run): 4.13e-14 (machine precision).")
println("Exact LSWT reference: domain-averaged over three 120° rotation domains.")
flush(stdout)
