
struct IncomingBondCSR{TPairs, TIndices}
    incoming_pairs   :: TPairs    # CuVector{PairCouplingDevice} on host, CuDeviceVector{...} after Adapt
    incoming_indices :: TIndices  # CuVector{Int64} on host, CuDeviceVector{Int64} after Adapt
end

function Adapt.adapt_structure(to, csr::IncomingBondCSR)
    incoming_pairs   = Adapt.adapt_structure(to, csr.incoming_pairs)
    incoming_indices = Adapt.adapt_structure(to, csr.incoming_indices)
    IncomingBondCSR(incoming_pairs, incoming_indices)
end

function build_incoming_bond_csr(host_sys::Sunny.System)
    @assert host_sys.mode in (:dipole, :dipole_uncorrected) "IncomingBondCSR is dipole-only"
    @assert host_sys.interactions_union isa Vector "expected SpinWaveTheory-reshaped homogeneous Vector{Interactions}"
    @assert host_sys.dims == (1, 1, 1) "expected SWT-reshaped host_sys with dims=(1,1,1); got dims=$(host_sys.dims). Pass swt.sys to build_incoming_bond_csr, not the raw user system."

    Na = length(host_sys.interactions_union)

    counts = zeros(Int, Na)
    for int in host_sys.interactions_union
        for pair in int.pair
            pair.isculled && continue
            j = pair.bond.j
            @assert 1 <= j <= Na "Bond j index $(j) out of range [1, $Na]"
            counts[j] += 1
        end
    end

    incoming_indices_h = Vector{Int64}(undef, Na + 1)
    incoming_indices_h[1] = 1
    for j in 1:Na
        incoming_indices_h[j+1] = incoming_indices_h[j] + counts[j]
    end
    total_incoming = incoming_indices_h[Na+1] - 1

    incoming_pairs_h = Vector{PairCouplingDevice}(undef, total_incoming)

    write_pos = Vector{Int}(undef, Na)
    for j in 1:Na
        write_pos[j] = incoming_indices_h[j]
    end

    for int in host_sys.interactions_union
        for pair in int.pair
            pair.isculled && continue
            j = pair.bond.j
            incoming_pairs_h[write_pos[j]] = PairCouplingDevice(pair)
            write_pos[j] += 1
        end
    end

    @assert all(write_pos[j] == incoming_indices_h[j+1] for j in 1:Na) "IncomingBondCSR write_pos sanity check failed"

    return IncomingBondCSR(
        CUDA.CuVector(incoming_pairs_h),
        CUDA.CuVector(incoming_indices_h),
    )
end
