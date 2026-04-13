
function batched_dot_chains_kernel!(result, a, b)
    tid = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    warp_id = (tid - Int32(1)) ÷ Int32(32) + Int32(1)   
    lane    = (tid - Int32(1)) % Int32(32)              

    N_chains = size(a, 1)
    if warp_id > N_chains
        return
    end

    @inbounds begin
        twoL = size(a, 2)
        c = warp_id  

        acc = ComplexF64(0)
        k = lane + Int32(1)
        while k <= twoL
            acc += conj(a[c, k]) * b[c, k]
            k += Int32(32)
        end

        mask = UInt32(0xFFFFFFFF)
        offset = Int32(16)
        while offset >= Int32(1)
            acc_re = real(acc)
            acc_im = imag(acc)
            acc_re += CUDA.shfl_down_sync(mask, acc_re, offset)
            acc_im += CUDA.shfl_down_sync(mask, acc_im, offset)
            acc = ComplexF64(acc_re, acc_im)
            offset ÷= Int32(2)
        end

        if lane == Int32(0)
            result[c] = acc
        end
    end

    return
end

function batched_dot_chains!(
    result::CUDA.CuVector{ComplexF64},   
    a::CUDA.CuMatrix{ComplexF64},        
    b::CUDA.CuMatrix{ComplexF64},        
)
    N_chains = size(a, 1)
    @assert size(a) == size(b) "batched_dot_chains! a and b must have the same shape"
    @assert length(result) == N_chains "batched_dot_chains! result length must match N_chains"
    @assert eltype(result) == eltype(a) == eltype(b) == ComplexF64 "batched_dot_chains! all buffers must be ComplexF64"

    warps_per_block = 8
    threads_per_block = warps_per_block * 32
    total_warps = N_chains
    blocks = cld(total_warps, warps_per_block)
    @cuda threads=threads_per_block blocks=blocks batched_dot_chains_kernel!(result, a, b)

    return result
end
