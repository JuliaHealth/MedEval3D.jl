
using CUDA ,BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

 
function teEx()
    z= 0
    grid_handle = this_grid()

    for j in 1:1000
        sync_grid(grid_handle)
        z+=1
    end 

end

#no grid sync 189.548 μs

@benchmark CUDA.@sync @cuda threads=(32,20) blocks=40  cooperative=true teEx()