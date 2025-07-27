module LuxTestProject


"""
   AbstractLuxSurrogate

Abstract type for Lux based surrogate
"""
abstract type AbstractLuxSurrogate end


"""
    LuxSurrogate

Trained surrogate.
"""
mutable struct LuxSurrogate <: AbstractLuxSurrogate

end

"""
    LuxSurrogate(func, range)

Create and train a lux based surrogate for the function `func`.
Parameters
- `func`: function or callable object with signature `func(result, input)` where
   input is an vector of length n. The output of `func` is written into the vector `result`. 
- `range`: n-vector of 2-vectors descring the training rang

Return:
Trained neuronal network.
"""
function LuxSurrogate(
        func::TF,
        range::Vector{Vector{Float64}}
    ) where {TF}
end

"""
    luxsave(lux::LuxSurrogate, file::String)

Save LuxSurrogate with trained weights to file.
"""
function luxsave(lux::LuxSurrogate, file::String)

end

"""
   luxload(filename::String)

Load LuxSurrogate from file `filename` 

Return:
Trained neuronal network.
"""
function luxload(file::String)

end

"""
   luxeval!(result, lux::LuxSurrogate, input)

Evalute trained surrogate at n-vector `input`, result is writen to m-vector `result`
"""
function luxeval!(result, lux::LuxSurrogate, input)

end

end # module LuxTestProject
