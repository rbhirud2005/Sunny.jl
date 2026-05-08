using Sunny, Printf

latvecs = lattice_vectors(1, 1, 10, 90, 90, 120)
cryst   = Crystal(latvecs, [[0, 0, 0]])

function build_sys(mode; D=2.0, seed=0)
    sys = System(cryst, [1 => Moment(s=1, g=2)], mode; dims=(4,4,1), seed=seed)
    set_exchange!(sys, -1.0, Bond(1,1,[1,0,0]))
    if D != 0.0
        set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)
    end
    randomize_spins!(sys)
    minimize_energy!(sys)
    return sys
end

qs   = [[0,0,0], [1/3,1/3,0], [1/2,0,0], [0,0,0]]
path = q_space_path(cryst, qs, 200)

for (label, D) in [("D=2.0", 2.0), ("D=0.0", 0.0)]
    println("Single-ion anisotropy: $label")

    sys_sun = build_sys(:SUN;   D=D)
    sys_dip = build_sys(:dipole; D=D)

    swt_sun = SpinWaveTheory(sys_sun; measure=ssf_perp(sys_sun))
    swt_dip = SpinWaveTheory(sys_dip; measure=ssf_perp(sys_dip))

    res_sun = intensities_bands(swt_sun, path)
    res_dip = intensities_bands(swt_dip, path)

    nbands_sun = length(res_sun.disp[:, 1])
    nbands_dip = length(res_dip.disp[:, 1])

    println("\nNumber of bands:  SU(N)=$(nbands_sun)   dipole=$(nbands_dip)")

    println("\n--- Gamma point (q=[0,0,0]) band energies ---")
    println("SU(N):")
    for b in 1:nbands_sun
        e   = res_sun.disp[b, 1]
        int = res_sun.data[b, 1]
        @printf("  band %2d: E = %8.4f meV,  intensity = %.4e\n", b, e, int)
    end
    println("Dipole:")
    for b in 1:nbands_dip
        e   = res_dip.disp[b, 1]
        int = res_dip.data[b, 1]
        @printf("  band %2d: E = %8.4f meV,  intensity = %.4e\n", b, e, int)
    end

    iq_zb = div(200, 3)
    q_zb  = path.qs[iq_zb]
    println("\n--- Zone-boundary point (q-index $iq_zb, q≈$(round.(q_zb, digits=3))) ---")
    println("SU(N):")
    for b in 1:nbands_sun
        e   = res_sun.disp[b, iq_zb]
        int = res_sun.data[b, iq_zb]
        @printf("  band %2d: E = %8.4f meV,  intensity = %.4e\n", b, e, int)
    end
    println("Dipole:")
    for b in 1:nbands_dip
        e   = res_dip.disp[b, iq_zb]
        int = res_dip.data[b, iq_zb]
        @printf("  band %2d: E = %8.4f meV,  intensity = %.4e\n", b, e, int)
    end

    n_zero_sun = count(abs.(res_sun.disp) .< 0.01)
    n_zero_dip = count(abs.(res_dip.disp) .< 0.01)
    println("\nZero-energy modes (|E|<0.01) across all q:  SU(N)=$n_zero_sun   dipole=$n_zero_dip")

    if nbands_sun > nbands_dip
        extra = (nbands_dip+1):nbands_sun
        max_int_extra = maximum(res_sun.data[extra, :])
        max_int_all   = maximum(res_sun.data)
        @printf("\nExtra SU(N) bands (bands %d–%d):  max intensity = %.4e  (%.1f%% of total max)\n",
                first(extra), last(extra), max_int_extra, 100*max_int_extra/max_int_all)
    end
end

println("\nDone.")
