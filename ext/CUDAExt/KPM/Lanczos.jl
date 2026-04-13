
using LinearAlgebra: SymTridiagonal, dot, mul!, eigmax, eigmin

"""
    lanczos_device(mulA!, mulS!,
                   v, vp, Sv, Svp, w, Sw,
                   lhs, corr_dev, corr_host, lhs_adj_Q_buf;
                   min_iters, max_iters, resolution=Inf, verbose=false)

Device-side generalized Lanczos. Mirrors `Sunny.lanczos` from
`src/KPM/Lanczos.jl:94-167` but operates on caller-preallocated GPU
buffers and a caller-preallocated host `lhs_adj_Q` accumulator matrix.

Arguments:
- `mulA!`, `mulS!`: type-stable closures `(out, in) -> out`, both
  operating on `CuVector{ComplexF64}` of length `length(v)`. The caller
  is expected to construct these via `let` blocks for closure-capture
  hygiene.
- `v, vp, Sv, Svp, w, Sw :: CuVector{ComplexF64}`: preallocated Lanczos
  recurrence buffers, all of length `2L`. `v` is also the **input** —
  it carries the initial vector and gets overwritten in place.
- `lhs :: CuMatrix{ComplexF64}`: device "left-hand side" matrix of shape
  `(2L, size(lhs, 2))`. Used in `lhs' * v` per iteration. Lives on
  device for the duration of the call. In the per-q caller,
  `size(lhs, 2) == Nobs` (number of observables); the post-Lanczos
  Hermitian symmetrization in the host projection block maps this to
  `Ncorr` (number of correlation pairs).
- `corr_dev :: CuVector{ComplexF64}`: preallocated length-`size(lhs, 2)`
  device buffer for the per-iteration `lhs' * v` GEMV result. Written
  in place by `mul!(corr_dev, lhs', v)`.
- `corr_host :: Vector{ComplexF64}`: preallocated length-`size(lhs, 2)`
  host buffer for the per-iteration device→host pull of `corr_dev`.
  Written in place by `copyto!(corr_host, corr_dev)`.
- `lhs_adj_Q_buf :: Matrix{ComplexF64}`: host-side preallocated buffer
  of shape `(size(lhs, 2), max_iters)`, written column-wise.
- `min_iters :: Int`: lower bound on iteration count, used by the
  spectral-bandwidth-based termination heuristic.
- `max_iters :: Int`: upper bound on iteration count. The caller
  computes this via CPU `Sunny.eigbounds` + the `tol`/`fwhm` formula
  before allocating `lhs_adj_Q_buf`. Hard ceiling: `length(v) - 1`.
- `resolution :: Float64`: target resolution Δϵ/iteration; passed to the
  termination check at i == min_iters.
- `verbose :: Bool`: print diagnostics.

Returns `(T, lhs_adj_Q)` where:
- `T :: SymTridiagonal{Float64, Vector{Float64}}` — the Lanczos
  tridiagonal, host-side, ready for `eigen(T)`.
- `lhs_adj_Q :: Matrix{ComplexF64}` — host-side, of shape
  `(size(lhs, 2), n_iters_done)`. This is a copy of the populated
  portion of `lhs_adj_Q_buf`, returned independently so the caller can
  reuse the buffer for the next chain.
"""
function lanczos_device(
    mulA!,
    mulS!,
    v::CUDA.CuVector{ComplexF64},
    vp::CUDA.CuVector{ComplexF64},
    Sv::CUDA.CuVector{ComplexF64},
    Svp::CUDA.CuVector{ComplexF64},
    w::CUDA.CuVector{ComplexF64},
    Sw::CUDA.CuVector{ComplexF64},
    lhs::CUDA.CuMatrix{ComplexF64},
    corr_dev::CUDA.CuVector{ComplexF64},
    corr_host::Vector{ComplexF64},
    lhs_adj_Q_buf::Matrix{ComplexF64};
    min_iters::Int,
    max_iters::Int,
    resolution::Float64 = Inf,
    verbose::Bool = false,
)::Tuple{SymTridiagonal{Float64, Vector{Float64}}, Matrix{ComplexF64}}

    Ncorr = size(lhs, 2)
    @assert size(lhs_adj_Q_buf, 1) == Ncorr "lhs_adj_Q_buf row count must match size(lhs, 2)"
    @assert size(lhs_adj_Q_buf, 2) >= max_iters "lhs_adj_Q_buf column count too small"
    @assert length(corr_dev) == Ncorr "corr_dev length must match size(lhs, 2)"
    @assert length(corr_host) == Ncorr "corr_host length must match size(lhs, 2)"
    @assert length(v) == length(vp) == length(Sv) == length(Svp) == length(w) == length(Sw)
    @assert size(lhs, 1) == length(v)

    αs = Float64[]   
    βs = Float64[]   
    n_iters_done = 0

    mulS!(Sv, v)

    vSv = dot(v, Sv)

    nv2 = real(dot(v, v))             
    atol = nv2 * length(v) * 1e-12
    @assert isapprox(imag(vSv), 0; atol) "S not Hermitian (imag(v†Sv) ≠ 0)"
    @assert isapprox(real(vSv), 1; atol) "Initial v not normalized (real(v†Sv) ≠ 1)"

    norm_factor = sqrt(real(vSv))
    @. v  /= norm_factor
    @. Sv /= norm_factor

    mulA!(w, Sv)

    α = real(dot(w, Sv))

    @. w = w - α * v   
    mulS!(Sw, w)

    push!(αs, α)
    n_iters_done += 1

    mul!(corr_dev, lhs', v)
    copyto!(corr_host, corr_dev)
    lhs_adj_Q_buf[:, n_iters_done] .= corr_host

    niters_eff = max_iters
    @inbounds for i in 1:length(v)-1
        if i == min_iters
            T_partial = SymTridiagonal(αs, βs)
            Δϵ = eigmax(T_partial) - eigmin(T_partial)
            niters_from_resolution = max(min_iters, fld(Δϵ, resolution))
            niters_from_resolution += mod(niters_from_resolution, 2)
            niters_eff = min(max_iters, Int(niters_from_resolution))
            if verbose
                println("Δϵ=$Δϵ, niters_eff=$niters_eff (capped by max_iters=$max_iters)")
            end
        end

        i >= niters_eff && break

        β² = real(dot(w, Sw))
        iszero(β²) && break
        β² < 0 && error("S is not a positive definite measure (β² = $β² < 0)")

        β = sqrt(β²)

        @. vp  = w  / β
        @. Svp = Sw / β

        mulA!(w, Svp)
        α = real(dot(w, Svp))
        @. w = w - α * vp - β * v
        mulS!(Sw, w)

        @. v = vp

        push!(αs, α)
        push!(βs, β)
        n_iters_done += 1

        mul!(corr_dev, lhs', v)
        copyto!(corr_host, corr_dev)
        lhs_adj_Q_buf[:, n_iters_done] .= corr_host
    end

    T = SymTridiagonal(αs, βs)

    lhs_adj_Q = lhs_adj_Q_buf[:, 1:n_iters_done]

    return T, lhs_adj_Q
end
