
function batched_w_update_kernel!(w, α, vp, β, v)
    tid = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    N_chains = size(w, 1)
    twoL = size(w, 2)
    total = N_chains * twoL
    if tid > total
        return
    end
    @inbounds begin
        c = ((tid - Int32(1)) % N_chains) + Int32(1)
        k = ((tid - Int32(1)) ÷ N_chains) + Int32(1)
        w[c, k] = w[c, k] - α[c] * vp[c, k] - β[c] * v[c, k]
    end
    return
end

function batched_w_update!(
    w::CUDA.CuMatrix{ComplexF64},
    α::CUDA.CuVector{ComplexF64},
    vp::CUDA.CuMatrix{ComplexF64},
    β::CUDA.CuVector,                    # this is float64 or complexF64
    v::CUDA.CuMatrix{ComplexF64},
)
    total = length(w)
    kernel = CUDA.@cuda launch=false batched_w_update_kernel!(w, α, vp, β, v)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(w, α, vp, β, v; threads, blocks)
    return w
end
