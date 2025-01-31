using Revise, Parameters, Logging, Test
using CUDA
includet("../../src/structs/BasicStructs.jl")
includet("../../src/utils/CUDAGpuUtils.jl")
includet("../../src/utils/IterationUtils.jl")
includet("../../src/utils/ReductionUtils.jl")
includet("../../src/utils/MemoryUtils.jl")
includet("../../src/distanceMetrics/MeansMahalinobis.jl")
includet("../../src/distanceMetrics/Mahalanobis.jl")
using ..CUDAGpuUtils , ..MeansMahalinobis, ..IterationUtils,..ReductionUtils , ..MemoryUtils
nx=512 ; ny=512 ; nz=317
#first we initialize the metrics on CPU so we will modify them easier
goldBoolCPU= zeros(Float32,nx,ny,nz); #mimicks gold standard mask

cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
goldBoolCPU[cartTrueGold].=Float32(1.0);

numberToLooFor= Float32(1.0);
goldBoolGPU = CuArray(goldBoolCPU);

Float64(goldBoolCPU[5,5,5])

#we will fill it after we work with launch configuration
loopXdim = UInt32(1);
loopYdim = UInt32(1) ;
loopZdim = UInt32(1) ;
sizz = size(goldBoolCPU);
maxX = UInt32(sizz[1])
maxY = UInt32(sizz[2])
maxZ = UInt32(sizz[3])

resList= CUDA.zeros(UInt32, length(goldBoolGPU) );
resListCounter= CUDA.ones(UInt32,1)

intermediateresCheck=100;
intermidiateResLength=UInt16(1);
warpNumber= UInt16(1)

totalX= CuArray([0]);
totalY= CuArray([0]);
totalZ= CuArray([0]);
totalCount= CuArray([0]);
blocks=UInt32(1)
debugArr = CUDA.zeros(600);
dynamicMemoryLength= 100

args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxX, maxY,maxZ
,resList,resListCounter,intermediateresCheck
,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)


    # calculate the amount of dynamic shared memory for a 2D block size
    get_shmem(threads) = (sizeof(UInt32)*3*4)
    
    function get_threads(threads)
        threads_x = 32
        threads_y = cld(threads,threads_x )
        return (threads_x, threads_y)
    end

    kernel = @cuda launch=false MeansMahalinobis.meansMahalinobisKernel(args...)
   
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(get_threads(threads)))

   # convert to 2D block size and figure out appropriate grid size
    threads = get_threads(config.threads)
    blocks = UInt32(config.blocks)
    loopXdim = UInt32(cld(maxX, threads[1]))
    loopYdim = UInt32(cld(maxY, threads[2])) 
    loopZdim = UInt32(cld(maxZ,blocks )) 


    totalX= CuArray([0]);
    totalY= CuArray([0]);
    totalZ= CuArray([0]);
    totalCount= CuArray([0]);
    dynamicMemoryLength = sum(threads)+intermediateresCheck

    resListCounter= CUDA.zeros(UInt32,1)
    resList= CUDA.zeros(UInt32, length(goldBoolGPU) );

    args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxX, maxY,maxZ
    ,resList,resListCounter,intermediateresCheck
    ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)

    @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)
    @test totalCount[1]==45
    @test totalX[1]== sum(map(ind->ind[1],cartTrueGold))
    @test totalY[1]== sum(map(ind->ind[2],cartTrueGold))
    @test totalZ[1]== sum(map(ind->ind[3],cartTrueGold))

# using StaticArrays
# using LinearAlgebra


# ones= CUDA.ones(Float16,4,4)
# fragA= CUDA.ones(Float16,4,4)
# fragB= CUDA.ones(Float16,4,4)
# d_out= CUDA.ones(Float16,4,4)
# dataShmem = CuArray([ 1  2  3 0;
#                       11  22  33  0])
#     @cuda threads=32 WMMAkernel(dataShmem,ones,d_out,fragA,fragB)
#     d = Array(d_out)
#     fragAa= Array(fragA)
#     fragBa= Array(fragB)

#     fragAa*fragBa+ Base.ones(Float16,4,4)


# fragA = CuArray(Float16.([ 1.0 -11.0  -1.0   11.0;
#   2.0 -22.0  -2.0   22.0;
#   3.0 -33.0  -3.0   33.0;
#   0.0  0.0    0.0   0.0
# ]))

# fragB = CuArray(Float16.([1.0 2.0 3.0 0.0;
#  1.0 2.0 3.0 0.0;
#  11.0 22.0 33.0 0.0;
#  11.0 22.0 33.0 0.0
# ]))

# Array(fragA*fragB+fragC)

# fragC = CUDA.ones(Float16,4,4)
# d_out= CUDA.ones(Float16,4,4)

# function testKernel(fragA,fragB,fragC, d_out)
#     conf = WMMA.Config{16, 16, 16, Float16}
#     a_frag = WMMA.load_a(pointer(fragA), 16, WMMA.ColMajor, conf)
#     b_frag = WMMA.load_b(pointer(fragB), 16, WMMA.ColMajor, conf)
#     c_frag = WMMA.load_c(pointer(fragC), 16, WMMA.ColMajor, conf)
#    d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
#    WMMA.store_d(pointer(d_out), d_frag, 16, WMMA.ColMajor, conf)
# end
# @cuda threads=32 blocks=1 testKernel(fragA,fragB,fragC, d_out)
# Array(d_out)
# dataShmem = [ 1  2  3  0;
#              11  22  33 0 ]

# aFrag = zeros(4,4);
# bFrag = zeros(4,4);

# for i in 1:16
#     div,remm = divrem(i-1,4)
#     # aFrag[fld(div+1,2)+1,rem+1 ] = dataShmem[rem+1,fld(div+1,2)+1 ]

#     a =  ((i-1) & (3))+1
#     b = ~((i+3)>>2 & 1)+3
#     c = ((i-1)>>2 )+1
#     aFrag[a,c] = dataShmem[b,a]*( ((i>4 && i<13)*-2)+1 )
# @info "i$(i)   a $(a)   b $(b)   c $(c)"

# end    
# aFrag

# 5>>1

# using  Statistics
# arr = [1 2 3; 
#        2 5 6; 
#        6 7 62; 
#        24 53 61; 
#        6 8 11]
# cov(arr)


#     @test all(isapprox.(a * b + 0.5 * c, d; rtol=0.01))    




#     dataShmem = [ 1  2  3  0;
#     11  22  33 0 ]

# aFrag = zeros(4,4);
# bFrag = zeros(4,4);

# for i in 1:32
    
#     a =  ((i-1) >>3)
#     b = ((i-1)>>4 )+1
#     c = ((i-1) & (2^4 - 1))+1
#   @info "i $(i) a $(a)  b $(b) c $(c) "
# end    


# bFrag