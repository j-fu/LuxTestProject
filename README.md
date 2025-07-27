LuxTestProject
=============

Test neural network training based on Lux.jl.


## Usage example

```julia
using LuxTestProject
using Test

function original!(result, input)
  result[1]=input[1]^2+ sin(input[2]^2)
  result[2]=input[1]^3+ 1/input[2]
  return nothing
end

lux=LuxSurrogate(original, [ [-1,1.0], [0.1, 5]])

luxsave(lux, "test.lux")

lux1=luxload("test.lux")

xy=[1,2]
oresult= zeros(2)
luxresult= zeros(2)
original!(oresult, xy)
luxeval!(luxresult, lux1, xy)
@test norm(oresult-luxresult) < 1.0e-5
```
