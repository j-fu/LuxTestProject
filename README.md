LuxTestProject
=============

Test neural network training based on Lux.jl.

## Features

- **Pure Function Interface**: Functions should return vectors or scalars (scalars automatically converted to vectors)
- **Type Preservation**: Output element type matches input element type
- **Callable Surrogates**: LuxSurrogate objects can be called directly as `lux(input)`
- **Customizable Training**: Configure network architecture, optimizer, and training parameters
- **Save/Load**: Persist trained surrogates to disk

## Usage example

```julia
using LuxTestProject
using Test

function original(input)
  result1 = input[1]^2 + sin(input[2]^2)
  result2 = input[1]^3 + 1/input[2]
  return [result1, result2]  # Returns AbstractVector
end

lux = LuxSurrogate(original, [[-1.0, 1.0], [0.1, 5.0]])

luxsave(lux, "test.lux")

lux1 = luxload("test.lux")

xy = [0.5, 2.0]
oresult = original(xy)
luxresult = lux1(xy)  # LuxSurrogate is now callable!
@test norm(oresult - luxresult) < 0.1
```

## Supported Function Return Types

```julia
# Vector output (recommended)
function vector_func(input)
    return [input[1]^2, input[2]^2]
end

# Scalar output (automatically converted to 1-element vector)
function scalar_func(input)
    return input[1]^2 + input[2]^2
end

# Both work with LuxSurrogate
lux_vector = LuxSurrogate(vector_func, [[-1.0, 1.0], [-1.0, 1.0]])
lux_scalar = LuxSurrogate(scalar_func, [[-1.0, 1.0], [-1.0, 1.0]])
```

## Customization Options

```julia
# Custom network architecture and training
lux = LuxSurrogate(func, ranges;
    n_samples=1000,           # Number of training samples
    hidden_layers=[32, 64, 32], # Network architecture
    activation=relu,          # Activation function
    optimizer=LBFGS(),        # Optimization algorithm
    maxiters=200             # Training iterations
)
```
