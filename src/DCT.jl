function matrix_broadcast(tens, mat)

    function slice_multiply(tens)
        if ndims(tens) == 2
            return transpose(transpose(tens) * mat)
        else 
            dim = ndims(tens)
            slices = collect(eachslice(tens, dims=dim))
            return cat([slice_multiply(slice) for slice in slices]..., dims=dim)
        end
    end

    return slice_multiply(tens)
end

function dct_1d_type_1(arr)
    N = length(arr)
    new_arr = []
    for k in 0:N - 1
        push!(new_arr, sum([(n == 0 || n == N - 1) ? (1/2)*arr[n + 1]*cos(pi/(N - 1)*n*k) : arr[n + 1]*cos(pi/(N - 1)*n*k) for n in 0:N-1]))
    end
    return 2*new_arr
end

function matrix_dct_nd_type_1(arr)
    degs = size(arr)
    dim = ndims(arr)
    for (dim, deg) in enumerate(degs)
        m = reshape(0:deg-1, (deg,1))
        W = cos.(pi/(deg - 1) * m * transpose(m))
        W[1, :] /= 2
        W[deg, :] /= 2
        order = vcat([dim - 1], collect(0:dim-2), collect(dim:ndims(arr)-1))
        backOrder = zeros(Int, ndims(arr))
        backOrder[order .+ 1] = collect(1:ndims(arr))
        arr = permutedims(matrix_broadcast(permutedims(arr, order .+ 1),W), backOrder)
    end
    return 2^dim*arr
end

function dct(arr)
    if ndims(arr) == 1
        return dct_1d_type_1(arr)
    else
        return matrix_dct_nd_type_1(arr)
    end
end

function dct_1d_type_1_arb(arr)
    N = length(arr)
    new_arr = []
    for k in 0:N - 1
        push!(new_arr, sum([(n == 0 || n == N - 1) ? type(.5)*arr[n + 1]*cos(type(pi)/type(N - 1)*n*k) : arr[n + 1]*cos(type(pi)/type(N - 1)*n*k) for n in 0:N-1]))
    end
    return type(2)*new_arr
end

function matrix_dct_nd_type_1_arb(arr)
    degs = size(arr)
    dim = ndims(arr)
    for (dim, deg) in enumerate(degs)
        m = reshape(0:deg-1, (deg,1))
        W = cos.(type(pi)/type(deg - 1) * m * transpose(m))
        W[1, :] /= type(2)
        W[deg, :] /= type(2)
        order = vcat([dim - 1], collect(0:dim-2), collect(dim:ndims(arr)-1))
        backOrder = zeros(Int, ndims(arr))
        backOrder[order .+ 1] = collect(1:ndims(arr))
        arr = permutedims(matrix_broadcast(permutedims(arr, order .+ 1),W), backOrder)
    end
    return type(2)^dim*arr
end

function dct_arb(arr)
    if ndims(arr) == 1
        return dct_1d_type_1_arb(arr)
    else
        return matrix_dct_nd_type_1_arb(arr)
    end
end