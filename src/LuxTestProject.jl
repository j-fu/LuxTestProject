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


function LuxSurrogate(
        input::AbstractMatrix,
        Y_train::AbstractMatrix;
        range = extrema(input, dims = 2),
        X_train = similar(input),
        hidden_layers::Vector{Int} = [16, 32, 16],
        activation = tanh,
        optimizer = BFGS(),
        maxiters::Int = 100
    )
    input_dim = size(input, 1)
    output_dim = size(Y_train, 1)
    for i in 1:input_dim
        min_val, max_val = Float32(range[i][1]), Float32(range[i][2])
        @views X_train[i, :] = 2.0f0 * (input[i, :] .- min_val) / (max_val - min_val) .- 1.0f0
    end

    # Create neural network architecture using keyword arguments
    layers = []

    # Input layer
    if length(hidden_layers) > 0
        push!(layers, Dense(input_dim => hidden_layers[1], activation))

        # Hidden layers
        for i in 1:(length(hidden_layers) - 1)
            push!(layers, Dense(hidden_layers[i] => hidden_layers[i + 1], activation))
        end

        # Output layer
        push!(layers, Dense(hidden_layers[end] => output_dim))
    else
        # If no hidden layers, direct input to output
        push!(layers, Dense(input_dim => output_dim))
    end

    model = Chain(layers...)

    # Initialize parameters and state
    rng = Random.default_rng()
    parameters, state = Lux.setup(rng, model)
    parameters = ComponentArray(parameters)

    # Define loss function
    function loss_function(params, data)
        X, Y = data
        Y_pred, _ = Lux.apply(model, X, params, state)
        return sum(abs2, Y_pred - Y) / size(Y, 2)
    end

    # Setup optimization
    data = (X_train, Y_train)
    optf = OptimizationFunction(loss_function, Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, parameters, data)

    # Train the model
    result = solve(optprob, optimizer, maxiters = maxiters)

    return LuxSurrogate(model, result.u, state, input_dim, output_dim, range)
end

"""
    LuxSurrogate(func, range; n_samples=500, hidden_layers=[16, 32, 16], activation=tanh, optimizer=BFGS(), maxiters=100)

Create and train a lux based surrogate for the function `func`.
Parameters
- `func`: function or callable object with signature `output = func(input)` where
   input is a vector of length n and output is an AbstractVector of length m.
   The output can also be a tuple or scalar, which will be automatically converted to a vector.
- `range`: n-vector of 2-vectors describing the training range

Keyword Arguments
- `n_samples`: Number of training samples to generate (default: 500)
- `hidden_layers`: Vector of hidden layer sizes (default: [16, 32, 16])
- `activation`: Activation function for hidden layers (default: tanh)
- `optimizer`: Optimization algorithm (default: BFGS())
- `maxiters`: Maximum number of optimization iterations (default: 100)

Return:
Trained neural network.
"""
function LuxSurrogate(
        func::TF,
        range;
        n_samples::Int = 500,
        kwargs...
    ) where {TF}

    # Determine dimensions
    input_dim = length(range)

    # Create a test input to determine output dimension
    test_input = [0.5 * (r[1] + r[2]) for r in range]

    # Determine output dimension by calling the function
    test_output = func(test_input)
    output_dim = length(test_output)


    # Generate training data (use Float32 for consistency)
    X_train = zeros(Float32, input_dim, n_samples)
    Y_train = zeros(Float32, output_dim, n_samples)

    for i in 1:n_samples
        # Generate random input within the specified ranges
        for j in 1:input_dim
            X_train[j, i] = range[j][1] + rand() * (range[j][2] - range[j][1])
        end
        @views Y_train[:, i] = func(X_train[:, i])
    end

    return LuxSurrogate(
        X_train, Y_train;
        range,
        X_train,
        kwargs...
    )
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
    return jldsave(file; surrogate_data = data)
end

# Helper function to serialize model architecture
function _serialize_model(model::Chain)
    layers = []
    for layer in model.layers
        if layer isa Dense
            push!(
                layers, Dict(
                    "type" => "Dense",
                    "in_dims" => layer.in_dims,
                    "out_dims" => layer.out_dims,
                    "activation" => Symbol(layer.activation)
                )
            )
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
            activation = getproperty(Lux, layer_data["activation"])
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

    # Normalize input to the same range used during training
    normalized_input = zeros(T, lux.input_dim)
    for i in 1:lux.input_dim
        min_val, max_val = lux.input_ranges[i][1], lux.input_ranges[i][2]
        normalized_input[i] = 2 * (input[i] - min_val) / (max_val - min_val) - 1
    end

    # Evaluate the model
    output, _ = Lux.apply(lux.model, normalized_input, lux.parameters, lux.state)

    # Convert to the same type as input
    return output
end

end # module LuxTestProject
