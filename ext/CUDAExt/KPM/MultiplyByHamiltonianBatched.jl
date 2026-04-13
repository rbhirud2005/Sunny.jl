
function multiply_by_hamiltonian_dipole_batched_kernel!(
    Y, X, sys, incoming, data,
    q_reshaped_chain, regularization, L,
)
    tid = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    N_chains = size(Y, 1)
    total = N_chains * L * 2
    if tid > total
        return
    end

    @inbounds begin
        chain_idx = ((tid - Int32(1)) % N_chains) + Int32(1)
        rest      =  (tid - Int32(1)) ÷ N_chains
        me        =  (rest % L) + Int32(1)
        sigma     =  (rest ÷ L) + Int32(1)

        q = q_reshaped_chain[chain_idx]

        (; local_rotations, stevens_coefs, sqrtS) = data
        (; extfield, pairs, indices, gs) = sys

        acc = ComplexF64(0)

        s_me = sqrtS[me]^2
        c2_me = stevens_coefs[me].c2
        c4_me = stevens_coefs[me].c4
        c6_me = stevens_coefs[me].c6
        A1 = -6*s_me*c2_me[3] - 80*s_me^3*c4_me[5] - 336*s_me^5*c6_me[7]
        A2 = 2*s_me*(c2_me[1] + im*c2_me[5]) + 12*s_me^3*(c4_me[3] + im*c4_me[7]) + 32*s_me^5*(c6_me[5] + im*c6_me[9])

        B = gs[1, 1, 1, me]' * extfield[1, 1, 1, me]
        Bprime = -dot(B, local_rotations[me][:, 3])

        if sigma == Int32(1)
            acc += (Bprime + A1) * X[chain_idx, me, 1]
            acc += A2 * X[chain_idx, me, 2]
        else
            acc += (Bprime + A1) * X[chain_idx, me, 2]
            acc += conj(A2) * X[chain_idx, me, 1]
        end

        for idx in indices[me]:indices[me + Int32(1)] - Int32(1)
            coupling = pairs[idx]
            (; isculled, bond) = coupling
            isculled && break
            j = bond.j  

            phase = exp(2π*im * dot(q, bond.n))

            si_local = sqrtS[me]^2  
            sj_local = sqrtS[j]^2   
            sij_local = sqrtS[me] * sqrtS[j]

            if !iszero(coupling.bilin)
                J = coupling.bilin
                P = 0.5 * sij_local * (J[1, 1] - J[2, 2] - im*J[1, 2] - im*J[2, 1])
                Q = 0.5 * sij_local * (J[1, 1] + J[2, 2] - im*J[1, 2] + im*J[2, 1])

                if sigma == Int32(1)
                    acc += Q * phase * X[chain_idx, j, 1]
                    acc += conj(P) * phase * X[chain_idx, j, 2]
                    acc -= sj_local * J[3, 3] * X[chain_idx, me, 1]
                else
                    acc += conj(Q) * phase * X[chain_idx, j, 2]
                    acc += P * phase * X[chain_idx, j, 1]
                    acc -= sj_local * J[3, 3] * X[chain_idx, me, 2]
                end
            end

            if !iszero(coupling.biquad)
                K = coupling.biquad
                Sj2Si = sj_local^2 * si_local
                Q = 0.5 * sij_local^3 * ( K[4, 4] + K[2, 2] - im*(-K[4, 2] + K[2, 4]))
                P = 0.5 * sij_local^3 * (-K[4, 4] + K[2, 2] - im*( K[4, 2] + K[2, 4]))

                if sigma == Int32(1)
                    acc += -12 * Sj2Si * K[3, 3] * X[chain_idx, me, 1]
                    acc += 4 * Sj2Si * (K[1, 3] + im*K[5, 3]) * X[chain_idx, me, 2]
                    acc += Q * phase * X[chain_idx, j, 1]
                    acc += conj(P) * phase * X[chain_idx, j, 2]
                else
                    acc += -12 * Sj2Si * K[3, 3] * X[chain_idx, me, 2]
                    acc += 4 * Sj2Si * (K[1, 3] - im*K[5, 3]) * X[chain_idx, me, 1]
                    acc += conj(Q) * phase * X[chain_idx, j, 2]
                    acc += P * phase * X[chain_idx, j, 1]
                end
            end
        end

        (; incoming_pairs, incoming_indices) = incoming
        for idx in incoming_indices[me]:incoming_indices[me + Int32(1)] - Int32(1)
            coupling = incoming_pairs[idx]
            bond = coupling.bond
            nbr = bond.i  

            phase = exp(2π*im * dot(q, bond.n))

            si_local = sqrtS[nbr]^2  
            sj_local = sqrtS[me]^2   
            sij_local = sqrtS[nbr] * sqrtS[me]

            if !iszero(coupling.bilin)
                J = coupling.bilin
                P = 0.5 * sij_local * (J[1, 1] - J[2, 2] - im*J[1, 2] - im*J[2, 1])
                Q = 0.5 * sij_local * (J[1, 1] + J[2, 2] - im*J[1, 2] + im*J[2, 1])

                if sigma == Int32(1)
                    acc += conj(P) * conj(phase) * X[chain_idx, nbr, 2]
                    acc += conj(Q) * conj(phase) * X[chain_idx, nbr, 1]
                    acc -= si_local * J[3, 3] * X[chain_idx, me, 1]
                else
                    acc += Q * conj(phase) * X[chain_idx, nbr, 2]
                    acc += P * conj(phase) * X[chain_idx, nbr, 1]
                    acc -= si_local * J[3, 3] * X[chain_idx, me, 2]
                end
            end

            if !iszero(coupling.biquad)
                K = coupling.biquad
                Si2Sj = si_local^2 * sj_local
                Q = 0.5 * sij_local^3 * ( K[4, 4] + K[2, 2] - im*(-K[4, 2] + K[2, 4]))
                P = 0.5 * sij_local^3 * (-K[4, 4] + K[2, 2] - im*( K[4, 2] + K[2, 4]))

                if sigma == Int32(1)
                    acc += -12 * Si2Sj * K[3, 3] * X[chain_idx, me, 1]
                    acc += 4 * Si2Sj * (K[3, 1] + im*K[3, 5]) * X[chain_idx, me, 2]
                    acc += conj(Q) * conj(phase) * X[chain_idx, nbr, 1]
                    acc += conj(P) * conj(phase) * X[chain_idx, nbr, 2]
                else
                    acc += -12 * Si2Sj * K[3, 3] * X[chain_idx, me, 2]
                    acc += 4 * Si2Sj * (K[3, 1] - im*K[3, 5]) * X[chain_idx, me, 1]
                    acc += Q * conj(phase) * X[chain_idx, nbr, 2]
                    acc += P * conj(phase) * X[chain_idx, nbr, 1]
                end
            end
        end

        acc += regularization * X[chain_idx, me, sigma]

        Y[chain_idx, me, sigma] = acc
    end

    return
end

function multiply_by_hamiltonian_dipole_batched!(
    Y::CUDA.CuMatrix{ComplexF64},                    
    X::CUDA.CuMatrix{ComplexF64},                    
    swt::SpinWaveTheoryDevice,
    incoming::IncomingBondCSR,
    q_reshaped_chain::CUDA.CuVector{Sunny.Vec3},     
)
    L = Sunny.natoms(swt.sys.crystal)
    N_chains = size(Y, 1)
    @assert size(Y) == size(X) == (N_chains, 2L) "Bogoliubov vector size mismatch"
    @assert length(q_reshaped_chain) == N_chains "q_reshaped_chain length must match N_chains"

    fill!(Y, ComplexF64(0))

    Y3 = reshape(Y, (N_chains, L, 2))
    X3 = reshape(X, (N_chains, L, 2))

    kernel = CUDA.@cuda launch=false multiply_by_hamiltonian_dipole_batched_kernel!(
        Y3, X3, swt.sys, incoming, swt.data,
        q_reshaped_chain, swt.regularization, L,
    )
    config = launch_configuration(kernel.fun)
    total = N_chains * L * 2
    threads = Base.min(total, config.threads)
    blocks = cld(total, threads)
    kernel(Y3, X3, swt.sys, incoming, swt.data,
           q_reshaped_chain, swt.regularization, L;
           threads, blocks)

    return Y
end

function mul_dynamical_matrix_batched!(
    swt::SpinWaveTheoryDevice,
    incoming::IncomingBondCSR,
    Y::CUDA.CuMatrix{ComplexF64},
    X::CUDA.CuMatrix{ComplexF64},
    q_reshaped_chain::CUDA.CuVector{Sunny.Vec3},
)
    @assert swt.sys.mode in (dipole, dipole_uncorrected) "GPU batched matvec only supports dipole / dipole_uncorrected mode"
    multiply_by_hamiltonian_dipole_batched!(Y, X, swt, incoming, q_reshaped_chain)
    return Y
end
