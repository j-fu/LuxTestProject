using LuxTestProject
using Test
using LinearAlgebra

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
        # Test surrogate creation with simpler function
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        
        @test lux isa LuxSurrogate
        @test lux.input_dim == 2
        @test lux.output_dim == 2
        @test length(lux.input_ranges) == 2
        @test lux.input_ranges[1] == [-1.0, 1.0]
        @test lux.input_ranges[2] == [-1.0, 1.0]
    end
    
    @testset "Save and Load Functionality" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        
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
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        
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
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        
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
        
        # Test the mutating version for backward compatibility
        result_mut = zeros(2)
        @test_nowarn luxeval!(result_mut, lux, xy_outside)
        @test all(isfinite.(result_mut))
    end
    
    @testset "Consistency Between Original and Loaded Surrogate" begin
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        
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
        
        lux_single = LuxSurrogate(single_output, [[-1.0, 1.0], [-1.0, 1.0]])
        @test lux_single.output_dim == 1  # Should correctly detect 1 output dimension
        
        result = lux_single([0.5, 0.5])  # Using callable interface
        @test length(result) == 1
        @test isfinite(result[1])
        
        # Test with 3-output function
        function triple_output(input)
            return [input[1]^2, input[2]^2, input[1] * input[2]]
        end
        
        lux_triple = LuxSurrogate(triple_output, [[-1.0, 1.0], [-1.0, 1.0]])
        @test lux_triple.output_dim == 3  # Should correctly detect 3 output dimensions
        
        result_triple = lux_triple([0.5, 0.5])  # Using callable interface
        @test length(result_triple) == 3
        @test all(isfinite.(result_triple))
    end
    
    @testset "Backward Compatibility Tests" begin
        # Test that luxeval and luxeval! still work for backward compatibility
        lux = LuxSurrogate(test_func, [[-1.0, 1.0], [-1.0, 1.0]])
        xy = [0.5, 0.5]
        
        # Test luxeval function
        result_luxeval = luxeval(lux, xy)
        result_callable = lux(xy)
        @test norm(result_luxeval - result_callable) < 1e-10
        
        # Test luxeval! function
        result_mut = zeros(2)
        luxeval!(result_mut, lux, xy)
        @test norm(result_mut - result_callable) < 1e-10
    end
end
