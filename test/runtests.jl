using LuxTestProject
using Test
using LinearAlgebra
using OptimizationOptimJL
using Lux: relu, sigmoid

# Test function for the surrogate (simpler for faster training)
function test_func(input)
    result1 = input[1]^2 + input[2]
    result2 = input[1] + input[2]^2
    return [result1, result2]
end

# More complex function for README test
function original(input)
    result1 = input[1]^2 + sin(input[2]^2)
    result2 = input[1]^3 + 1/input[2]
    return [result1, result2]
end

@testset "LuxTestProject.jl" begin
    
    # Include the README example test
    include("readme_example_test.jl")
    
    @testset "Basic Surrogate Creation and Training" begin
        # Test surrogate creation with simpler function (faster parameters)
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=100, maxiters=20)
        
        @test lux isa LuxSurrogate
        @test lux.input_dim == 2
        @test lux.output_dim == 2
        @test length(lux.input_ranges) == 2
        @test lux.input_ranges[1] == [-1.0, 1.0]
        @test lux.input_ranges[2] == [-1.0, 1.0]
    end
    
    @testset "Save and Load Functionality" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=50, maxiters=10)
        
        # Test saving
        test_file = "test_surrogate.lux"
        luxsave(lux, test_file)
        @test isfile(test_file)
        
        # Test loading
        lux_loaded = luxload(test_file)
        @test lux_loaded isa LuxSurrogate
        @test lux_loaded.input_dim == lux.input_dim
        @test lux_loaded.output_dim == lux.output_dim
        @test lux_loaded.input_ranges == lux.input_ranges
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Surrogate Evaluation Accuracy" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=50, maxiters=15)
        
        # Test fewer points for faster execution
        test_points = [
            [0.0, 0.0],
            [0.5, 0.5], 
            [-0.5, 0.3]
        ]
        
        for xy in test_points
            oresult = test_func(xy)
            luxresult = lux(xy)  # Using callable interface
            
            error_norm = norm(oresult - luxresult)
            
            # Neural networks should approximate reasonably well
            @test error_norm < 0.5  # More lenient for faster training
            
            # Test that we get finite results
            @test all(isfinite.(luxresult))
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=40, maxiters=10)
        
        # Test evaluation with points outside training range (should be clamped)
        xy_outside = [2.0, 2.0]  # Outside the training ranges
        
        # Should not throw an error due to clamping
        result = lux(xy_outside)  # Using callable interface
        @test_nowarn result = lux(xy_outside)
        @test all(isfinite.(result))
        
        # Test with minimum valid input
        xy_min = [-1.0, -1.0]
        result_min = lux(xy_min)  # Using callable interface
        @test_nowarn result_min = lux(xy_min)
        
        # Test with maximum valid input  
        xy_max = [1.0, 1.0]
        result_max = lux(xy_max)  # Using callable interface
        @test_nowarn result_max = lux(xy_max)
    end
    
    @testset "Consistency Between Original and Loaded Surrogate" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=30, maxiters=8)
        
        test_file = "consistency_test.lux"
        luxsave(lux, test_file)
        lux_loaded = luxload(test_file)
        
        # Test that original and loaded surrogates give same results
        xy = [0.3, 0.3]
        result1 = lux(xy)  # Using callable interface
        result2 = lux_loaded(xy)  # Using callable interface
        
        @test norm(result1 - result2) < 1e-10  # Should be essentially identical
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Different Function Output Dimensions" begin
        # Test with single output function (simpler)
        function single_output(input)
            return [input[1]^2 + input[2]^2]
        end
        
        lux_single = LuxSurrogate(single_output, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=30, maxiters=8)
        @test lux_single.output_dim == 1  # Should correctly detect 1 output dimension
        
        result = lux_single([0.5, 0.5])  # Using callable interface
        @test length(result) == 1
        @test isfinite(result[1])
        
        # Test with 3-output function
        function triple_output(input)
            return [input[1]^2, input[2]^2, input[1] * input[2]]
        end
        
        lux_triple = LuxSurrogate(triple_output, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=30, maxiters=8)
        @test lux_triple.output_dim == 3  # Should correctly detect 3 output dimensions
        
        result_triple = lux_triple([0.5, 0.5])  # Using callable interface
        @test length(result_triple) == 3
        @test all(isfinite.(result_triple))
    end
    
    @testset "Type Preservation Tests" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=30, maxiters=8)
        
        # Test with Float64 input
        xy_f64 = [0.5, 0.5]
        result_f64 = lux(xy_f64)
        @test eltype(result_f64) == Float64
        @test eltype(xy_f64) == eltype(result_f64)
        
        # Test with Float32 input
        xy_f32 = Float32[0.5f0, 0.5f0]
        result_f32 = lux(xy_f32)
        @test eltype(result_f32) == Float32
        @test eltype(xy_f32) == eltype(result_f32)
        
        # Test with BigFloat input (if supported)
        xy_big = BigFloat[0.5, 0.5]
        result_big = lux(xy_big)
        @test eltype(result_big) == BigFloat
        @test eltype(xy_big) == eltype(result_big)
    end
    
    @testset "Keyword Arguments Tests" begin
        # Test with custom n_samples (smaller for speed)
        lux_samples = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; n_samples=50, maxiters=10)
        @test lux_samples isa LuxSurrogate
        result_samples = lux_samples([0.5, 0.5])
        @test all(isfinite.(result_samples))
        @test length(result_samples) == 2
        
        # Test with smaller hidden layers
        lux_small = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; hidden_layers=[5], n_samples=30, maxiters=5)
        @test lux_small isa LuxSurrogate
        result_small = lux_small([0.5, 0.5])
        @test all(isfinite.(result_small))
        @test length(result_small) == 2
        
        # Test with no hidden layers (linear model)
        lux_linear = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; hidden_layers=Int[], n_samples=20, maxiters=5)
        @test lux_linear isa LuxSurrogate
        result_linear = lux_linear([0.5, 0.5])
        @test all(isfinite.(result_linear))
        @test length(result_linear) == 2
        
        # Test with custom activation function (keep it simple)
        lux_relu = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; activation=relu, hidden_layers=[4], n_samples=25, maxiters=5)
        @test lux_relu isa LuxSurrogate
        result_relu = lux_relu([0.5, 0.5])
        @test all(isfinite.(result_relu))
        @test length(result_relu) == 2
        
        # Test with custom optimizer (LBFGS - faster for small problems)
        lux_lbfgs = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]]; optimizer=LBFGS(), n_samples=20, maxiters=3)
        @test lux_lbfgs isa LuxSurrogate
        result_lbfgs = lux_lbfgs([0.5, 0.5])
        @test all(isfinite.(result_lbfgs))
        @test length(result_lbfgs) == 2
        
        # Test type preservation with keyword arguments
        xy_f32 = Float32[0.5f0, 0.5f0]
        @test eltype(lux_samples(xy_f32)) == Float32
        @test eltype(lux_small(xy_f32)) == Float32
        @test eltype(lux_linear(xy_f32)) == Float32
    end
end
