
using LinearAlgebra: SymTridiagonal

"""
    lanczos_device_batched(mulA!, mulS!,
                           v_batch, vp_batch, Sv_batch, Svp_batch, w_batch, Sw_batch,
                           lhs_dev, corr_dev, corr_host,
                           α_chains_dev, β²_chains_dev, β_chains_dev,
                           α_chains_host, β²_chains_host,
                           lhs_adj_Q_buf;
                           max_iters, min_iters, resolution, verbose=false)

Batched Lanczos driver. Mirrors `lanczos_device` from
`ext/CUDAExt/KPM/Lanczos.jl` but operates on N_chains independent
chains in lockstep using the batched matvec, batched dot,
permutedims-based transpose materialization, and cuBLAS
gemm_strided_batched! per-iteration projection.
`(αs_matrix, βs_matrix, lhs_adj_Q, n_iters_done_per_chain, c_norm_per_chain)` where:
- `αs_matrix :: Matrix{Float64}(max_iters, N_chains)`
- `βs_matrix :: Matrix{Float64}(max_iters - 1, N_chains)`
- `lhs_adj_Q :: Array{ComplexF64, 3}(Nobs, max_iters, N_chains)` (the
  same `lhs_adj_Q_buf` the caller passed in)
- `n_iters_done_per_chain :: Vector{Int}(N_chains)`
- `c_norm_per_chain :: Vector{Float64}(N_chains)` — the per-chain
  `c = sqrt(real(Sv' * v))` normalization factor, captured BEFORE the
  internal v normalization. The caller multiplies this into the
  projection (mirrors `src/KPM/SpinWaveTheoryKPM.jl:357`).
"""
function lanczos_device_batched(
    mulA!,
    mulS!,
    v_batch::CUDA.CuMatrix{ComplexF64},
    vp_batch::CUDA.CuMatrix{ComplexF64},
    Sv_batch::CUDA.CuMatrix{ComplexF64},
    Svp_batch::CUDA.CuMatrix{ComplexF64},
    w_batch::CUDA.CuMatrix{ComplexF64},
    Sw_batch::CUDA.CuMatrix{ComplexF64},
    lhs_dev::CUDA.CuArray{ComplexF64, 3},
    corr_dev::CUDA.CuArray{ComplexF64, 3},
    corr_host::Array{ComplexF64, 3},
    α_chains_dev::CUDA.CuVector{ComplexF64},
    β²_chains_dev::CUDA.CuVector{ComplexF64},
    β_chains_dev::CUDA.CuVector{Float64},
    α_chains_host::Vector{ComplexF64},
    β²_chains_host::Vector{ComplexF64},
    lhs_adj_Q_buf::Array{ComplexF64, 3};
    max_iters::Int,
    min_iters::Int,
    resolution::Float64,
    verbose::Bool = false,
)::Tuple{Matrix{Float64}, Matrix{Float64}, Array{ComplexF64, 3}, Vector{Int}, Vector{Float64}}

    N_chains = size(v_batch, 1)
    twoL = size(v_batch, 2)
    Nobs = size(lhs_dev, 2)
    Nq   = size(lhs_dev, 3)

    @assert size(v_batch) == size(vp_batch) == size(Sv_batch) == size(Svp_batch) == size(w_batch) == size(Sw_batch) "Recurrence buffers must all be (N_chains, 2L)"
    @assert size(lhs_dev, 1) == twoL "lhs_dev first axis must equal 2L"
    @assert Nobs * Nq == N_chains "lhs_dev's Nobs*Nq must equal N_chains"
    @assert size(corr_dev) == (Nobs, Nobs, Nq) "corr_dev must be (Nobs, Nobs, Nq)"
    @assert size(corr_host) == (Nobs, Nobs, Nq) "corr_host must be (Nobs, Nobs, Nq)"
    @assert length(α_chains_dev) == N_chains
    @assert length(β²_chains_dev) == N_chains
    @assert length(β_chains_dev) == N_chains
    @assert length(α_chains_host) == N_chains
    @assert length(β²_chains_host) == N_chains
    @assert size(lhs_adj_Q_buf) == (Nobs, max_iters, N_chains) "lhs_adj_Q_buf must be (Nobs, max_iters, N_chains)"
    @assert max_iters >= 1 "max_iters must be >= 1"

    αs_matrix = Matrix{Float64}(undef, max_iters, N_chains)
    βs_matrix = Matrix{Float64}(undef, max_iters - 1, N_chains)
    n_iters_done_per_chain = zeros(Int, N_chains)
    chain_done = falses(N_chains)
    c_norm_per_chain = Vector{Float64}(undef, N_chains)
    niters_eff = fill(max_iters, N_chains)

    mulS!(Sv_batch, v_batch)

    batched_dot_chains!(α_chains_dev, v_batch, Sv_batch)

    copyto!(α_chains_host, α_chains_dev)

    hermitian_atol = max(twoL, 1) * 1e-10
    @inbounds for c in 1:N_chains
        vSv_c = α_chains_host[c]
        if abs(imag(vSv_c)) > hermitian_atol
            error("Chain $c: S not Hermitian (imag(v†Sv) = $(imag(vSv_c)), |imag| = $(abs(imag(vSv_c))) > $hermitian_atol)")
        end
        if real(vSv_c) <= 0
            error("Chain $c: v†Sv ≤ 0 ($vSv_c); S not positive definite or v = 0")
        end
        c_norm_per_chain[c] = sqrt(real(vSv_c))
    end

    v_batch  ./= sqrt.(real.(reshape(α_chains_dev, N_chains, 1)))
    Sv_batch ./= sqrt.(real.(reshape(α_chains_dev, N_chains, 1)))

    mulA!(w_batch, Sv_batch)

    batched_dot_chains!(α_chains_dev, w_batch, Sv_batch)

    copyto!(α_chains_host, α_chains_dev)

    @inbounds for c in 1:N_chains
        αs_matrix[1, c] = real(α_chains_host[c])
    end

    w_batch .-= reshape(α_chains_dev, N_chains, 1) .* v_batch

    mulS!(Sw_batch, w_batch)

    batched_projection!(corr_dev, lhs_dev, v_batch)

    copyto!(corr_host, corr_dev)

    @inbounds for c in 1:N_chains
        iq = ((c - 1) ÷ Nobs) + 1
        ξ = ((c - 1) % Nobs) + 1
        for μ in 1:Nobs
            lhs_adj_Q_buf[μ, 1, c] = corr_host[μ, ξ, iq]
        end
        n_iters_done_per_chain[c] = 1
    end

    @inbounds for i in 1:max_iters - 1
        batched_dot_chains!(β²_chains_dev, w_batch, Sw_batch)

        copyto!(β²_chains_host, β²_chains_dev)

        all_done = true
        for c in 1:N_chains
            if !chain_done[c]
                β²_real = real(β²_chains_host[c])
                if β²_real <= 0
                    chain_done[c] = true
                end
            end
            if !chain_done[c]
                all_done = false
            end
        end

        if i == min_iters && isfinite(resolution)
            for c in 1:N_chains
                if !chain_done[c]
                    n_so_far = n_iters_done_per_chain[c]
                    if n_so_far >= 2
                        T_partial = SymTridiagonal(
                            αs_matrix[1:n_so_far, c],
                            βs_matrix[1:n_so_far-1, c],
                        )
                        Δϵ_c = eigmax(T_partial) - eigmin(T_partial)
                        niters_c = max(min_iters, fld(Δϵ_c, resolution))
                        niters_c += mod(niters_c, 2)
                        niters_eff[c] = min(Int(niters_c), max_iters)
                    end
                end
            end
            if verbose
                println("Per-chain niters_eff at min_iters=$min_iters: " *
                        "min=$(minimum(niters_eff)), max=$(maximum(niters_eff)), " *
                        "median=$(sort(niters_eff)[N_chains ÷ 2])")
            end
        end

        for c in 1:N_chains
            if !chain_done[c] && n_iters_done_per_chain[c] >= niters_eff[c]
                chain_done[c] = true
            end
        end

        all_done = all(chain_done)
        all_done && break

        β_chains_dev .= ifelse.(real.(β²_chains_dev) .> 0,
                                sqrt.(real.(β²_chains_dev)),
                                1.0)

        vp_batch  .= w_batch  ./ reshape(β_chains_dev, N_chains, 1)
        Svp_batch .= Sw_batch ./ reshape(β_chains_dev, N_chains, 1)

        mulA!(w_batch, Svp_batch)

        batched_dot_chains!(α_chains_dev, w_batch, Svp_batch)

        copyto!(α_chains_host, α_chains_dev)

        @inbounds for c in 1:N_chains
            if !chain_done[c]
                αs_matrix[i + 1, c] = real(α_chains_host[c])
                βs_matrix[i, c]     = sqrt(real(β²_chains_host[c]))
                n_iters_done_per_chain[c] = i + 1
            end
        end

        batched_w_update!(w_batch, α_chains_dev, vp_batch, β_chains_dev, v_batch)

        mulS!(Sw_batch, w_batch)

        copyto!(v_batch, vp_batch)

        batched_projection!(corr_dev, lhs_dev, v_batch)

        copyto!(corr_host, corr_dev)

        @inbounds for c in 1:N_chains
            if !chain_done[c]
                iq = ((c - 1) ÷ Nobs) + 1
                ξ = ((c - 1) % Nobs) + 1
                for μ in 1:Nobs
                    lhs_adj_Q_buf[μ, i + 1, c] = corr_host[μ, ξ, iq]
                end
            end
        end

        if verbose
            n_done = count(chain_done)
            println("lanczos_device_batched iter $i: $n_done/$N_chains chains done")
        end
    end

    return αs_matrix, βs_matrix, lhs_adj_Q_buf, n_iters_done_per_chain, c_norm_per_chain
end
