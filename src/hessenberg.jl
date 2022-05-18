import LinearAlgebra.ldiv!

struct FastHessenberg{T<:AbstractMatrix}
    H::T # H is assumed to be Hessenberg of size (m + 1) × m
end

@inline Base.size(H::FastHessenberg, args...) = size(H.H, args...)

"""
Solve Hy = rhs for a non-square Hessenberg matrix.
Note that `H` is also modified as is it converted
to an upper triangular matrix via Given's rotations
"""
function ldiv!(H::FastHessenberg, rhs)
    # Implicitly computes H = QR via Given's rotations
    # and then computes the least-squares solution y to
    # |Hy - rhs| = |QRy - rhs| = |Ry - Q'rhs|

    width = size(H, 2)

    # Hessenberg -> UpperTriangular; also apply to r.h.s.
    @inbounds for i = 1 : width
        c, s, _ = givensAlgorithm(H.H[i, i], H.H[i + 1, i])

        # Skip the first sub-diagonal since it'll be zero by design.
        H.H[i, i] = c * H.H[i, i] + s * H.H[i + 1, i]

        # Remaining columns
        @inbounds for j = i + 1 : width
            tmp = -conj(s) * H.H[i, j] + c * H.H[i + 1, j]
            H.H[i, j] = c * H.H[i, j] + s * H.H[i + 1, j]
            H.H[i + 1, j] = tmp
        end

        # Right hand side
        tmp = -conj(s) * rhs[i] + c * rhs[i + 1]
        rhs[i] = c * rhs[i] + s * rhs[i + 1]
        rhs[i + 1] = tmp
    end

    # Solve the upper triangular problem.
    U = UpperTriangular(view(H.H, 1 : width, 1 : width))
    ldiv!(U, view(rhs, 1 : width))
    nothing
end

# derived from LAPACK's dlartg
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function givensAlgorithm(f::T, g::T) where T
    onepar = one(T)
    twopar = 2one(T)
    T0 = typeof(onepar) # dimensionless
    zeropar = T0(zero(T)) # must be dimensionless

    # need both dimensionful and dimensionless versions of these:
    safmn2 = floatmin2(T0)
    safmn2u = floatmin2(T)
    safmx2 = one(T)/safmn2
    safmx2u = oneunit(T)/safmn2

    if g == 0
        cs = onepar
        sn = zeropar
        r = f
    elseif f == 0
        cs = zeropar
        sn = onepar
        r = g
    else
        f1 = f
        g1 = g
        scalepar = max(abs(f1), abs(g1))
        if scalepar >= safmx2u
            count = 0
            while true
                count += 1
                f1 *= safmn2
                g1 *= safmn2
                scalepar = max(abs(f1), abs(g1))
                if scalepar < safmx2u break end
            end
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
            for i = 1:count
                r *= safmx2
            end
        elseif scalepar <= safmn2u
            count = 0
            while true
                count += 1
                f1 *= safmx2
                g1 *= safmx2
                scalepar = max(abs(f1), abs(g1))
                if scalepar > safmn2u break end
            end
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
            for i = 1:count
                r *= safmn2
            end
        else
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
        end
        if abs(f) > abs(g) && cs < 0
            cs = -cs
            sn = -sn
            r = -r
        end
    end
    return cs, sn, r
end

floatmin2(::Type{Float64}) = reinterpret(Float64, 0x21a0000000000000)
floatmin2(::Any) = reinterpret(Float64, 0x21a0000000000000)