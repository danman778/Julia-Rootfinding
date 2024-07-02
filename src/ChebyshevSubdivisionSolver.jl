# TODO: import from a library like this one instead of crowding our sourcecode with pre-written code https://github.com/JeffreySarnoff/ErrorfreeArithmetic.jl/blob/main/src/sum.jl
function twoSum(a,b)
    """Returns x,y such that a+b=x+y exactly, and a+b=x in floating point."""
    x = a+b
    z = x-a
    y = (a-(x-z)) + (b-z)
    return x, y
end

# TODO: import from a library instead
function split(a)
    """Returns x,y such that a = x+y exactly and a = x in floating point."""
    c = (2^27 + 1) * a
    x = c-(c-a)
    y = a-x
    return x,y
end

# TODO: import from a library instead
function twoProd(a,b)
    """Returns x,y such that a*b=x+y exactly and a*b=x in floating point."""
    x = a*b
    a1,a2 = split(a)
    b1,b2 = split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y
end

# TODO: import from a library instead
function TwoProdWithSplit(a,b,a1,a2)
    """Returns x,y such that a*b = x+y exactly and a*b = x in floating point but with a already split."""
    x = a*b
    b1,b2 = split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y
end 

function getLinearTerms(M)
    """Gets the linear terms of the Chebyshev coefficient tensor M.

    Uses the fact that the linear terms are located at
    M[(0,0, ... ,0,1)]
    M[(0,0, ... ,1,0)]
    ...
    M[(0,1, ... ,0,0)]
    M[(1,0, ... ,0,0)]
    which are indexes
    1, M.shape[-1], M.shape[-1]*M.shape[-2], ... when looking at M.ravel().

    Parameters
    ----------
    M : array
        The coefficient array to get the linear terms from

    Returns
    -------
    A: array
        An array with the linear terms of M
    """
    A = []
    spot = 1
    
    for i in size(M)
        push!(A, (i) == 1 ? 0 : reshape(M,(1,length(M)))[spot+1])
        spot *= (i)
    end

    return reverse(A) # Return linear terms in dimension order.
end

function linearCheck1(totalErrs,A,consts)
    """Takes A, the linear terms of each function approximation, and makes any possible reduction 
        in the interval based on the totalErrs.


    Parameters
    ----------
    totalErrs : array
        gives bounds for the function using error in our approximation and coefficients
    A : array 
        each row represents a function with the linear coefficients of each dimension as the columns
    consts : array
        constant terms for each function

    Returns
    -------
    a : array
        lower bound
    b : array
        lower bound
        
    """
    dim = length(A[1,:])
    a = -ones(dim) * Inf
    b = ones(dim) * Inf
    for row in 1:dim
        for col in 1:dim
            if A[col,row] != 0 #Don't bother running the check if the linear term is too small.
                v1 = totalErrs[row] / abs(A[col,row]) - 1
                v2 = 2 * consts[row] / A[col,row]
                if v2 >= 0
                    c = -v1
                    d = v1-v2
                else
                    c = -v2-v1
                    d = v1
                end
                a[col] = max(a[col], c)
                b[col] = min(b[col], d)
            end
        end
    end
    return a, b
end

# function ReduceSolvedDim(Ms, errors, trackedInterval, dim)
    
#     val = (trackedInterval.interval[dim+1,1] + trackedInterval.interval[dim+1,2]) / 2
#     # Get the matrix of the linear terms
#     A = [getLinearTerms(M) for M in Ms]
#     # Figure out which linear approximation is flattest in the dimension we want to reduce
#     dot_prods = transpose((transpose(A)/sqrt(sum(A^2, dims = 2))))[:,dim+1]
#     func_idx = argmax(dot_prods)
#     # Remove that polynomial from Ms and errors
#     deleteat!(Ms,func_idx)
#      """ HOW DO NP.DELETE """
#     new_errors = np.delete(errors,func_idx) 

#     # Evaluate other polynomials on the solved dimension
#     # Ms are already scaled, so we just want to evaluate the T_i(x_dim)'s at x_dim = 0
#     final_Ms = []
#     for M in Ms:
#         # Get the dimensions of each M
#         degs = size(M)
#         total_dim = len(degs)
#         # Make array [1,0,-1,0,1,...] representing values of Chebyshev polynomials at 0
#         x = np.zeros(degs[dim])
#         x[::2] = [(-1)**i for i in collect(0:(degs[dim]+1)//2-1)]
#         # Transpose M so we can use matrix multiplication to evaluate one dimension at a time
#         idxs = np.roll(np.arange(len(degs)),total_dim-dim-1)
#         new_M = np.transpose(M,idxs)@x
#         # Transpose the resulting matrix back to its original order
#         new_M = np.transpose(new_M,np.roll(np.arange(len(degs)-1),dim))
#         final_Ms.append(new_M)

#     # Remove the point dimension from the tracked interval
#     trackedInterval.ndim -= 1
#     trackedInterval.interval = np.delete(trackedInterval.interval,dim,axis=0)
#     trackedInterval.reducedDims.append(dim)
#     trackedInterval.solvedVals.append(val)

#     return final_Ms,new_errors,trackedInterval
# end