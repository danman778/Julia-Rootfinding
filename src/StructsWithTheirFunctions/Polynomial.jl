using RecursiveArrayTools

struct MultiPower
    """Contains the coeffs array for a MultiPower object"""
    coeff
    dim

    function MultiPower(coeff)
        dim = ndims(coeff)
        new(to_julia(coeff),dim)
    end
end

struct MultiCheb
    """Contains the coeffs array for a MultiCheb object"""
    coeff
    dim
    function MultiCheb(coeff)
        dim = ndims(coeff)
        new(to_julia(coeff),dim)
    end

end

function to_python(A)
    """" Converts A to an array who's indices match python indices
        So B[i,j,k,...] would return the same thing as P[i,j,k,...],
        where P is the python representation of the matrix A 
    """
    s = size(A)
    dim = length(s)
    B = permutedims(reshape(A,reverse(s)),dim:-1:1)
    return B
end

function to_julia(A)
    """" Converts A to an array who's indices match python indices
        So B[i,j,k,...] would return the same thing as P[i,j,k,...],
        where P is the python representation of the matrix A 
    """

    s = size(A)
    dim = length(s)
    dim_order = collect(dim:-1:1)
    dim_order[1] = dim-1
    dim_order[2] = dim
    # println(dim_order)
    B = permutedims(A,dim_order)
    return B
end


function eval_MultiPower(multiPower,points)
    function polyval(x, cc)
        cc = collect(eachslice(cc,dims=ndims(cc)))
        c0 = cc[end]
        for i in 1:(length(cc) - 1)
            c0 = c0.*x .+ cc[end-i]
        end
        return c0
    end
    
    if ndims(points) == 1
        if multiPower.dim > 1
            points = reshape(points,(size(points)[end],1))
        else
            points = reshape(points,(1,size(points)[end]))
        end
    end

    if size(points)[end-1] != multiPower.dim
        1/0
    end

    n = multiPower.dim
    c = permutedims(multiPower.coeff,(2,1,collect(3:n)...))
    cc = reshape(c,(ntuple(i->1, ndims(points))..., size(c)...))
    c = polyval(points[1,:],cc)
    for i in 2:n
        c = polyval(points[i,:],c)
    end
    if length(c) == 1
        return c[1]
    else
        return c
    end

end

function multipower_to_cheb(coeffs)
    """ Takes in a multipower coefficient matrix
        Returns the chebyshev coefficient matrix """
    function get_new_As(As)
        """ Finds the next transformation coefficients (Bs) from the previous ones (As).
            So if x^n = sum(As[i]*T_i(x)), x^(n+1) = sum(Bs[i]*T_i(x)).
        """
        n = length(As)
        if n == 0
            return [1.]
        end
        Bs = zeros(n+1)
        # Edge case if As has length 1
        if n == 1
            Bs[2] = As[1]
            return Bs
        end
        # Put in the first and last coeffs
        if n%2 == 0
            Bs[1] = As[2]/2
        end
        Bs[end] = As[end]/2
        # Put in the second coeff
        if n == 2
            Bs[2] = As[1]
            return Bs
        end
        if n%2 == 1
            Bs[2] = As[1] + As[3]/2
        end
        # Do all the middle coefficients, only editing the ones that shouldn't be 0.
        if n > 3
            Bs[3+n%2:2:end-1] = (As[2+n%2:2:end-1] + As[4+n%2:2:end])/2 
        end
        return Bs
    end
    function to_cheb1D(coeffs)
        """Transforms to chebyshev coeficcients along the first dimension of coeffs matrix"""
        cheb_coeffs = zero(coeffs)
        As = []
        # Update As, then take each slice of the coefficient matrix and matrix multiply 
        # by As, and add to the cheb_coeffs matrix.
        dims = length(size(coeffs))
        slices = []
        for i in 1:dims
            push!(slices,:)
        end
        slices[end] = 1:2
        for (i,coeff) in enumerate(eachslice(coeffs,dims=dims))
            As = get_new_As(As)
            slices[end] = 1:i
            # Invoke einsum to do the right matrix multiplication in n dimensions
            cheb_coeffs[slices...] += convert(Array,VectorOfArray([val*coeff for val in As]))
            #np.expand_dims(As,axis=1)@np.array([coeff])
        end
        return cheb_coeffs
    end
    function to_chebND(coeffs,dim)
        """Transforms to chebyshev coefficients along the dim axis of the coeffs matrix"""
        # Get the transopse order to make the desired dim first
        order = append!([dim],collect(1:dim-1),collect(dim+1:ndims(coeffs)))
        temp = order[end]
        order[end] = order[end-1]
        order[end-1] = temp
        order = reverse(order)
        # Then transpose with the inverted order after the transformation occurs.
        backOrder = zeros(Int64,ndims(coeffs))
        backOrder[order] = 1:ndims(coeffs)
        # Transpose coeffs, transform them along the first dimension, then transpose them back.
        return permutedims(to_cheb1D(permutedims(coeffs,order)),backOrder)
    end


    cheb_coeffs = coeffs
    for dim in 1:ndims(coeffs)
        # Go through each dimension and transform
        cheb_coeffs = to_chebND(cheb_coeffs,dim)
    end
    final_order = append!([2,1],collect(3:ndims(coeffs)))
    return permutedims(cheb_coeffs,final_order)
end


function chebval(x, cc)
    len = length(cc)
    if len == 1
        c0 = cc[1]
        c1 = zeros_like(c0)
    elseif len == 2
        c0 = cc[1]
        c1 = cc[2]
    else
        x2 = 2*x
        c0 = cc[end-1]
        c1 = cc[end]
        for i in 3:len
            tmp = c0
            c0 = cc[end-i+1] - c1
            c1 = tmp + c1*x2
        end
    end
    return c0 + c1*x
end


