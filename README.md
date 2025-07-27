LuxTestProject
=============

Test neural network training based on Lux.jl.


## Usage example

```julia
using LuxTestProject
using Test

function original(input)
  result1 = input[1]^2 + sin(input[2]^2)
  result2 = input[1]^3 + 1/input[2]
  return [result1, result2]
end

lux = LuxSurrogate(original, [[-1.0, 1.0], [0.1, 5.0]])

luxsave(lux, "test.lux")

lux1 = luxload("test.lux")

xy = [0.5, 2.0]
oresult = original(xy)
luxresult = luxeval(lux1, xy)
@test norm(oresult - luxresult) < 0.1
```
