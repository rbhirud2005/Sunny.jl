
function batched_projection_kernel!(corr, lhs, v_batch, Nobs, Nq, twoL)
    tid = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    total = Nobs * Nobs * Nq
    if tid > total
        return
    end

    @inbounds begin
        μ  = ((tid - Int32(1)) % Nobs) + Int32(1)
        rest = (tid - Int32(1)) ÷ Nobs
        ξ  = (rest % Nobs) + Int32(1)
        iq = (rest ÷ Nobs) + Int32(1)

        c = (iq - Int32(1)) * Nobs + ξ

        acc = ComplexF64(0)
        for k in 1:twoL
            acc += conj(lhs[k, μ, iq]) * v_batch[c, k]
        end
        corr[μ, ξ, iq] = acc
    end
    return
end

function batched_projection!(
    corr::CUDA.CuArray{ComplexF64, 3},    
    lhs::CUDA.CuArray{ComplexF64, 3},     
    v_batch::CUDA.CuMatrix{ComplexF64},   
)
    Nobs = size(lhs, 2)
    Nq = size(lhs, 3)
    twoL = size(lhs, 1)
    total = Nobs * Nobs * Nq

    kernel = CUDA.@cuda launch=false batched_projection_kernel!(corr, lhs, v_batch, Nobs, Nq, twoL)
    config = launch_configuration(kernel.fun)
    threads = min(total, config.threads)
    blocks = cld(total, threads)
    kernel(corr, lhs, v_batch, Nobs, Nq, twoL; threads, blocks)
    return corr
end
