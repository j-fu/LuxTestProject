module LuxTestProject

using Lux
using Random
using ComponentArrays
using Optimization
using OptimizationOptimJL
using JLD2
using LinearAlgebra
using Zygote  # Add Zygote for automatic differentiation

# Export the main types and functions
export AbstractLuxSurrogate, LuxSurrogate, luxsave, luxload

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
    model::Chain
    parameters::ComponentArray
    state::NamedTuple
    input_dim::Int
    output_dim::Int
    input_ranges::Vector{Vector{Float64}}
end

"""
    LuxSurrogate(func, range)

Create and train a lux based surrogate for the function `func`.
Parameters
- `func`: function or callable object with signature `output = func(input)` where
   input is a vector of length n and output is a vector of length m.
- `range`: n-vector of 2-vectors describing the training range

Return:
Trained neural network.
"""
function LuxSurrogate(
        func::TF,
        range::Vector{Vector{Float64}}
    ) where {TF}
    
    # Determine dimensions
    input_dim = length(range)
    
    # Create a test input to determine output dimension
    test_input = [0.5 * (r[1] + r[2]) for r in range]
    
    # Determine output dimension by calling the function
    test_output = func(test_input)
    output_dim = length(test_output)
    
    # Create neural network architecture (smaller for faster training)
    model = Chain(
        Dense(input_dim => 16, tanh),
        Dense(16 => 32, tanh),
        Dense(32 => 16, tanh),
        Dense(16 => output_dim)
    )
    
    # Initialize parameters and state
    rng = Random.default_rng()
    parameters, state = Lux.setup(rng, model)
    parameters = ComponentArray(parameters)
    
    # Generate training data (use Float32 for consistency)
    n_samples = 500  # Reduced for faster training
    X_train = zeros(Float32, input_dim, n_samples)
    Y_train = zeros(Float32, output_dim, n_samples)
    
    for i in 1:n_samples
        # Generate random input within the specified ranges
        input = [Float32(range[j][1] + rand() * (range[j][2] - range[j][1])) for j in 1:input_dim]
        
        try
            output = func(input)
            X_train[:, i] = Float32.(input)
            Y_train[:, i] = Float32.(output)  # Convert to Float32
        catch e
            # If function fails, skip this sample and try again
            i -= 1
            continue
        end
    end
    
    # Normalize inputs to [-1, 1] (keep as Float32)
    X_normalized = copy(X_train)
    for i in 1:input_dim
        min_val, max_val = Float32(range[i][1]), Float32(range[i][2])
        X_normalized[i, :] = 2.0f0 * (X_train[i, :] .- min_val) / (max_val - min_val) .- 1.0f0
    end
    
    # Define loss function
    function loss_function(params, data)
        X, Y = data
        Y_pred, _ = Lux.apply(model, X, params, state)
        return sum(abs2, Y_pred - Y) / size(Y, 2)
    end
    
    # Setup optimization
    data = (X_normalized, Y_train)
    optf = OptimizationFunction(loss_function, Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, parameters, data)
    
    # Train the model (fewer iterations for faster training)
    result = solve(optprob, BFGS(), maxiters=100)
    
    return LuxSurrogate(model, result.u, state, input_dim, output_dim, range)
end

"""
    luxsave(lux::LuxSurrogate, file::String)

Save LuxSurrogate with trained weights to file.
"""
function luxsave(lux::LuxSurrogate, file::String)
    # Create a dictionary with all necessary data
    data = Dict(
        "model_type" => "LuxSurrogate",
        "parameters" => lux.parameters,
        "state" => lux.state,
        "input_dim" => lux.input_dim,
        "output_dim" => lux.output_dim,
        "input_ranges" => lux.input_ranges,
        "model_layers" => _serialize_model(lux.model)
    )
    
    # Save to file using JLD2
    jldsave(file; surrogate_data=data)
end

# Helper function to serialize model architecture
function _serialize_model(model::Chain)
    layers = []
    for layer in model.layers
        if layer isa Dense
            push!(layers, Dict(
                "type" => "Dense",
                "in_dims" => layer.in_dims,
                "out_dims" => layer.out_dims,
                "activation" => string(layer.activation)
            ))
        end
    end
    return layers
end

"""
   luxload(filename::String)

Load LuxSurrogate from file `filename` 

Return:
Trained neuronal network.
"""
function luxload(file::String)
    # Load data from file
    data = load(file, "surrogate_data")
    
    # Reconstruct the model from serialized layers
    model = _deserialize_model(data["model_layers"])
    
    # Create and return the LuxSurrogate
    return LuxSurrogate(
        model,
        data["parameters"],
        data["state"],
        data["input_dim"],
        data["output_dim"],
        data["input_ranges"]
    )
end

# Helper function to deserialize model architecture
function _deserialize_model(layers_data)
    layers = []
    for layer_data in layers_data
        if layer_data["type"] == "Dense"
            activation = layer_data["activation"] == "tanh" ? tanh : identity
            push!(layers, Dense(layer_data["in_dims"] => layer_data["out_dims"], activation))
        end
    end
    return Chain(layers...)
end

"""
   (lux::LuxSurrogate)(input)

Make LuxSurrogate callable. Evaluate trained surrogate at n-vector `input`, returns m-vector result.
The output will have the same element type as the input.
"""
function (lux::LuxSurrogate)(input)
    T = eltype(input)  # Get the element type of input
    
    # Normalize input to the same range used during training (use Float32)
    normalized_input = zeros(Float32, lux.input_dim)
    for i in 1:lux.input_dim
        min_val, max_val = Float32(lux.input_ranges[i][1]), Float32(lux.input_ranges[i][2])
        # Clamp input to valid range and then normalize
        clamped_input = clamp(Float32(input[i]), min_val, max_val)
        normalized_input[i] = 2.0f0 * (clamped_input - min_val) / (max_val - min_val) - 1.0f0
    end
    
    # Evaluate the model
    output, _ = Lux.apply(lux.model, normalized_input, lux.parameters, lux.state)
    
    # Convert to the same type as input
    return T.(output)
end

end # module LuxTestProject
