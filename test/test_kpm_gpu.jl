
using Test
using Sunny
using CUDA
using LinearAlgebra
using Random

@assert CUDA.functional() "CUDA not functional; cannot run GPU regression test"
println("CUDA device:     ", CUDA.name(CUDA.device()))
println("CUDA capability: ", CUDA.capability(CUDA.device()))

@testset "GPU multiply_by_hamiltonian_dipole (matvec, Nq=1)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    @test Sunny.energy_per_site(sys_prim) ≈ -2J*(3/2)^2

    measure = ssf_perp(sys_prim)
    swt = SpinWaveTheory(sys_prim; measure)
    swt_d = Sunny.to_device(swt)

    L = Sunny.natoms(swt.sys.crystal)
    @test Sunny.natoms(swt.sys.crystal) == Sunny.nbands(swt)  # pins dipole-mode convention
    println("natoms(swt.sys.crystal) = L = ", L)
    println("nbands(swt) = ", Sunny.nbands(swt))
    @test L >= 2  # Need at least 2 atoms to exercise pair interactions

    Random.seed!(0)
    Nq = 1
    x_h = randn(ComplexF64, Nq, 2L)
    y_h_ref = zeros(ComplexF64, Nq, 2L)

    q_reshaped = Sunny.Vec3(0.123, 0.456, 0.789)
    qs_h = [q_reshaped]

    Sunny.mul_dynamical_matrix!(swt, y_h_ref, x_h, qs_h)

    x_d = CuArray(x_h)
    y_d = CUDA.zeros(ComplexF64, Nq, 2L)
    qs_d = CuVector(qs_h)

    CUDA.@sync Sunny.mul_dynamical_matrix!(swt_d, y_d, x_d, qs_d)
    y_d_host = Array(y_d)

    diff = y_d_host .- y_h_ref
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, y_h_ref)
    relerr = maxabs / maxref

    println("max abs diff:  ", maxabs)
    println("max ref value: ", maxref)
    println("max rel err:   ", relerr)

    @test relerr < 1e-10
end
@testset "GPU matvec, scalar biquadratic" begin
    latvecs = lattice_vectors(1.0, 1.0, 1.0, 90, 90, 90)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys = System(cryst, [1 => Moment(s=1, g=1)], :dipole; dims=(2, 1, 1), seed=0)

    J  = +0.7   # bilinear
    K  = +0.4   # scalar biquadratic — SpinWaveTheory builder promotes to Mat5
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]); biquad=K)

    randomize_spins!(sys)
    minimize_energy!(sys)

    measure = ssf_perp(sys)
    swt = SpinWaveTheory(sys; measure)
    swt_d = Sunny.to_device(swt)

    L = Sunny.natoms(swt.sys.crystal)
    @test L == 2  # 2x1x1 cubic = 2 atoms in the supercell unit
    println("biquad test: L = ", L)

    @test swt.sys.interactions_union[1].pair[1].biquad isa Sunny.Mat5

    Random.seed!(1)
    Nq = 1
    x_h = randn(ComplexF64, Nq, 2L)
    y_h_ref = zeros(ComplexF64, Nq, 2L)
    q_reshaped = Sunny.Vec3(0.211, 0.317, 0.433)
    qs_h = [q_reshaped]

    Sunny.mul_dynamical_matrix!(swt, y_h_ref, x_h, qs_h)

    x_d = CuArray(x_h)
    y_d = CUDA.zeros(ComplexF64, Nq, 2L)
    qs_d = CuVector(qs_h)
    CUDA.@sync Sunny.mul_dynamical_matrix!(swt_d, y_d, x_d, qs_d)
    y_d_host = Array(y_d)

    diff = y_d_host .- y_h_ref
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, y_h_ref)
    relerr = maxabs / maxref

    println("biquad test: max abs diff  = ", maxabs)
    println("biquad test: max ref value = ", maxref)
    println("biquad test: max rel err   = ", relerr)

    @test maxref > 1e-6  # not all-zero output

    @test relerr < 1e-14
end

@testset "GPU matvec, disordered (LSWT-reshaped inhomogeneous) system" begin
    latvecs = lattice_vectors(1.0, 1.0, 1.0, 90, 90, 90)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys_homo = System(cryst, [1 => Moment(s=1, g=1)], :dipole; dims=(1, 1, 1), seed=0)
    set_exchange!(sys_homo, +0.5, Bond(1, 1, [1, 0, 0]))

    sys = repeat_periodically(sys_homo, (2, 2, 1))
    sys_inhom = to_inhomogeneous(sys)

    @test Sunny.energy(sys_inhom) ≈ Sunny.energy(sys)

    set_exchange_at!(sys_inhom, 0.9, (1,1,1,1), (2,1,1,1); offset=(1, 0, 0))

    randomize_spins!(sys_inhom)
    minimize_energy!(sys_inhom; maxiters=500)

    measure = ssf_perp(sys_inhom)
    swt = SpinWaveTheory(sys_inhom; measure)
    swt_d = Sunny.to_device(swt)

    L = Sunny.natoms(swt.sys.crystal)
    println("inhomog test: L = ", L)
    @test L == 4  # 2x2x1 single-atom unit, reshaped → 4 sublattices

    bilin_magnitudes = Float64[]
    for sub in swt.sys.interactions_union
        for pc in sub.pair
            if !iszero(pc.bilin)
                push!(bilin_magnitudes, maximum(abs, pc.bilin))
            end
        end
    end
    @test length(unique(round.(bilin_magnitudes; digits=6))) >= 2  # at least 2 distinct values

    Random.seed!(2)
    Nq = 1
    x_h = randn(ComplexF64, Nq, 2L)
    y_h_ref = zeros(ComplexF64, Nq, 2L)
    q_reshaped = Sunny.Vec3(0.137, 0.241, 0.359)
    qs_h = [q_reshaped]

    Sunny.mul_dynamical_matrix!(swt, y_h_ref, x_h, qs_h)

    x_d = CuArray(x_h)
    y_d = CUDA.zeros(ComplexF64, Nq, 2L)
    qs_d = CuVector(qs_h)
    CUDA.@sync Sunny.mul_dynamical_matrix!(swt_d, y_d, x_d, qs_d)
    y_d_host = Array(y_d)

    diff = y_d_host .- y_h_ref
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, y_h_ref)
    relerr = maxabs / maxref

    println("inhomog test: max abs diff  = ", maxabs)
    println("inhomog test: max ref value = ", maxref)
    println("inhomog test: max rel err   = ", relerr)

    @test maxref > 1e-6  # not all-zero output

    @test relerr < 1e-14
end

@testset "GPU intensities_lanczos end-to-end (CoRh2O4 primitive)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)
    @test Sunny.energy_per_site(sys_prim) ≈ -2J*(3/2)^2

    measure = ssf_perp(sys_prim)
    swt_kry = SpinWaveTheoryKPM(sys_prim; measure, tol=0.05)

    swt_kry_d = Sunny.to_device(swt_kry)

    qs = [[0.0, 0.0, 0.0], [0.13, 0.27, 0.41], [0.5, 0.5, 0.5]]
    path = q_space_path(cryst, qs, 5)  # 5 points along the path

    kernel = lorentzian(fwhm=0.8)
    energies = range(0.0, 6.0, 20)

    res_cpu = intensities(swt_kry, path; energies, kernel)
    println("e2e CPU intensities computed: data shape = ", size(res_cpu.data))

    CUDA.@sync res_gpu = Sunny.intensities(swt_kry_d, path; energies, kernel)
    println("e2e GPU intensities computed: data shape = ", size(res_gpu.data))

    @test size(res_gpu.data) == size(res_cpu.data)

    diff = res_gpu.data .- res_cpu.data
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, res_cpu.data)
    relerr = maxref > 0 ? maxabs / maxref : maxabs

    println("e2e test: max abs diff  = ", maxabs)
    println("e2e test: max ref value = ", maxref)
    println("e2e test: max rel err   = ", relerr)

    @test maxref > 1e-12  # Sanity: not an all-zero output
    @test relerr < 1e-8
end
@testset "GPU batched matvec correctness vs per-q (4-site CoRh2O4)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    measure = ssf_perp(sys_prim)
    swt = SpinWaveTheory(sys_prim; measure)

    L = Sunny.natoms(swt.sys.crystal)
    twoL = 2 * L
    Nobs = size(measure.observables, 1)
    println("Batched matvec test: L=$L, Nobs=$Nobs")

    swt_d = Sunny.to_device(swt)
    cudaext = Base.get_extension(Sunny, :CUDAExt)
    incoming = cudaext.build_incoming_bond_csr(swt.sys)

    Nq = 3
    N_chains = Nq * Nobs
    qs_test_host = Sunny.Vec3[
        Sunny.Vec3(0.123, 0.456, 0.789),
        Sunny.Vec3(0.211, 0.317, 0.433),
        Sunny.Vec3(0.0, 0.5, 0.5),
    ]
    q_chain_host = Vector{Sunny.Vec3}(undef, N_chains)
    for iq in 1:Nq, ξ in 1:Nobs
        c = (iq - 1) * Nobs + ξ
        q_chain_host[c] = qs_test_host[iq]
    end
    q_chain_dev = CuVector(q_chain_host)

    Random.seed!(42)
    x_batch_host = randn(ComplexF64, N_chains, twoL)
    x_batch_dev  = CuArray(x_batch_host)
    y_batch_dev  = CUDA.zeros(ComplexF64, N_chains, twoL)

    CUDA.@sync cudaext.mul_dynamical_matrix_batched!(
        swt_d, incoming, y_batch_dev, x_batch_dev, q_chain_dev,
    )
    y_batch_host = Array(y_batch_dev)

    for c in 1:N_chains
        iq_for_c = ((c - 1) ÷ Nobs) + 1
        q_c = qs_test_host[iq_for_c]
        x_h_single = reshape(x_batch_host[c, :], 1, twoL)
        y_h_single_ref = zeros(ComplexF64, 1, twoL)

        Sunny.mul_dynamical_matrix!(swt, y_h_single_ref, x_h_single, [q_c])

        diff = vec(y_batch_host[c, :] .- vec(y_h_single_ref))
        maxabs = maximum(abs, diff)
        maxref = maximum(abs, y_h_single_ref)
        relerr_c = maxref > 0 ? maxabs / maxref : maxabs
        @test relerr_c < 1e-12
    end

    println("Batched matvec vs per-q: ALL $N_chains chains agreed at rtol < 1e-12")
end

@testset "GPU batched intensities vs per-q (CoRh2O4 primitive)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    measure = ssf_perp(sys_prim)
    swt_kry = SpinWaveTheoryKPM(sys_prim; measure, tol=0.05)

    qs = [[0.0, 0.0, 0.0], [0.13, 0.27, 0.41], [0.5, 0.5, 0.5]]
    path = q_space_path(cryst, qs, 5)
    kernel = lorentzian(fwhm=0.8)
    energies = range(0.0, 6.0, 20)

    swt_kry_d_perq = Sunny.to_device(swt_kry)
    CUDA.@sync res_perq = Sunny.intensities(swt_kry_d_perq, path; energies, kernel)

    swt_kry_d_batched = Sunny.to_device_batched(swt_kry)
    CUDA.@sync res_batched = Sunny.intensities(swt_kry_d_batched, path; energies, kernel)

    @test size(res_perq.data) == size(res_batched.data)

    diff = res_batched.data .- res_perq.data
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, res_perq.data)
    relerr = maxref > 0 ? maxabs / maxref : maxabs

    println("Batched vs per-q e2e: max abs diff = $maxabs, max ref = $maxref, rel err = $relerr")

    @test maxref > 1e-12
    @test relerr < 1e-8
end

@testset "GPU batched intensities vs CPU (CoRh2O4 primitive)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    measure = ssf_perp(sys_prim)
    swt_kry = SpinWaveTheoryKPM(sys_prim; measure, tol=0.05)

    qs = [[0.0, 0.0, 0.0], [0.13, 0.27, 0.41], [0.5, 0.5, 0.5]]
    path = q_space_path(cryst, qs, 5)
    kernel = lorentzian(fwhm=0.8)
    energies = range(0.0, 6.0, 20)

    res_cpu = intensities(swt_kry, path; energies, kernel)

    swt_kry_d = Sunny.to_device_batched(swt_kry)
    CUDA.@sync res_gpu = Sunny.intensities(swt_kry_d, path; energies, kernel)

    diff = res_gpu.data .- res_cpu.data
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, res_cpu.data)
    relerr = maxref > 0 ? maxabs / maxref : maxabs

    println("Batched vs CPU e2e: max abs diff = $maxabs, max ref = $maxref, rel err = $relerr")

    @test maxref > 1e-12
    @test relerr < 1e-8
end

@testset "GPU batched intensities vs per-q (CoRh2O4, different q path)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    measure = ssf_perp(sys_prim)
    swt_kry = SpinWaveTheoryKPM(sys_prim; measure, tol=0.05)

    qs = [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.3, 0.4, 0.1], [0.0, 0.0, 0.0]]
    path = q_space_path(cryst, qs, 6)
    kernel = lorentzian(fwhm=0.5)
    energies = range(0.0, 4.0, 15)

    swt_kry_d_perq = Sunny.to_device(swt_kry)
    CUDA.@sync res_perq = Sunny.intensities(swt_kry_d_perq, path; energies, kernel)

    swt_kry_d_batched = Sunny.to_device_batched(swt_kry)
    CUDA.@sync res_batched = Sunny.intensities(swt_kry_d_batched, path; energies, kernel)

    diff = res_batched.data .- res_perq.data
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, res_perq.data)
    relerr = maxref > 0 ? maxabs / maxref : maxabs

    println("Batched vs per-q alt path: max abs diff = $maxabs, max ref = $maxref, rel err = $relerr")

    @test maxref > 1e-12
    @test relerr < 1e-10
end

@testset "GPU batched off-diagonal corr_pairs validation (ssf_perp non-trivial g)" begin
    a = 8.5031
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[1/8, 1/8, 1/8]]
    cryst = Crystal(latvecs, positions, 227; types=["Co"])

    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    J = +0.63
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)

    shape = primitive_cell(cryst)
    sys_prim = reshape_supercell(sys, shape)

    measure = ssf_perp(sys_prim)
    swt_kry = SpinWaveTheoryKPM(sys_prim; measure, tol=0.05)

    has_off_diagonal = any(p -> p[1] != p[2], measure.corr_pairs)
    println("Off-diagonal corr_pairs in ssf_perp(CoRh2O4 primitive): $has_off_diagonal")
    println("corr_pairs = ", measure.corr_pairs)

    swt_kry_d_perq = Sunny.to_device(swt_kry)

    swt_kry_d_batched = Sunny.to_device_batched(swt_kry)

    qs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    path = q_space_path(cryst, qs, 3)
    kernel = lorentzian(fwhm=0.7)
    energies = range(0.0, 5.0, 10)

    CUDA.@sync res_perq = Sunny.intensities(swt_kry_d_perq, path; energies, kernel)
    CUDA.@sync res_batched = Sunny.intensities(swt_kry_d_batched, path; energies, kernel)

    diff = res_batched.data .- res_perq.data
    maxabs = maximum(abs, diff)
    maxref = maximum(abs, res_perq.data)
    relerr = maxref > 0 ? maxabs / maxref : maxabs

    println("Batched vs per-q (off-diagonal): max abs diff = $maxabs, rel err = $relerr")

    @test maxref > 1e-12
    @test relerr < 1e-8
end
