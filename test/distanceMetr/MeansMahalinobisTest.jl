using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")

using Cthulhu
using Main.BasicPreds, Main.CUDAGpuUtils , Main.MeansMahalinobis, Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils
using Cthulhu
nx=512 ; ny=512 ; nz=317
#first we initialize the metrics on CPU so we will modify them easier
goldBoolCPU= zeros(Float32,nx,ny,nz); #mimicks gold standard mask
segmBoolCPU= zeros(Float32,nx,ny,nz); #mimicks other  mask

cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
goldBoolCPU[cartTrueGold].=Float32(1.0);

cartTrueSegm =  CartesianIndices(zeros(150,150,150) ).+CartesianIndex(100,100,100);

segmBoolCPU[cartTrueSegm].=Float32(1.0);

numberToLooFor= Float32(1.0);
goldBoolGPU = CuArray(goldBoolCPU);
segmBoolGPU = CuArray(segmBoolCPU);



#we will fill it after we work with launch configuration
loopXdim = UInt32(1);loopYdim = UInt32(1) ;loopZdim = UInt32(1) ;
sizz = size(goldBoolCPU);maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
#gold
totalXGold= CuArray([0]);
totalYGold= CuArray([0]);
totalZGold= CuArray([0]);
totalCountGold= CuArray([0]);
#segm
totalXSegm= CuArray([0]);
totalYSegm= CuArray([0]);
totalZSegm= CuArray([0]);
totalCountSegm= CuArray([0]);

args = (goldBoolGPU,segmBoolGPU,numberToLooFor
,loopYdim,loopXdim,loopZdim
,(maxX, maxY,maxZ)
,totalXGold,totalYGold,totalZGold,totalCountGold
,totalXSegm,totalYSegm,totalZSegm,totalCountSegm)


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



#gold
totalXGold= CuArray([0]);
totalYGold= CuArray([0]);
totalZGold= CuArray([0]);
totalCountGold= CuArray([0]);
#segm
totalXSegm= CuArray([0]);
totalYSegm= CuArray([0]);
totalZSegm= CuArray([0]);
totalCountSegm= CuArray([0]);

args = (goldBoolGPU,segmBoolGPU,numberToLooFor
,loopYdim,loopXdim,loopZdim
,(maxX, maxY,maxZ)
,totalXGold,totalYGold,totalZGold,totalCountGold
,totalXSegm,totalYSegm,totalZSegm,totalCountSegm)
    @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)
    @test totalCountGold[1]==45
    @test totalXGold[1]== sum(map(ind->ind[1],cartTrueGold))
    @test totalYGold[1]== sum(map(ind->ind[2],cartTrueGold))
    @test totalZGold[1]== sum(map(ind->ind[3],cartTrueGold))

    @test totalXSegm[1]== sum(map(ind->ind[1],cartTrueSegm))
    @test totalYSegm[1]== sum(map(ind->ind[2],cartTrueSegm))
    @test totalZSegm[1]== sum(map(ind->ind[3],cartTrueSegm))



    # @device_code_warntype interactive=true @cuda MeansMahalinobis.meansMahalinobisKernel(args...)












#     args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,(maxX, maxY,maxZ)
#     ,resList,resListCounter,intermediateresCheck
#     ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)
    
    
#         # calculate the amount of dynamic shared memory for a 2D block size
#         get_shmem(threads) = (sizeof(UInt32)*3*4)
        
#         function get_threads(threads)
#             threads_x = 32
#             threads_y = cld(threads,threads_x )
#             return (threads_x, threads_y)
#         end
    
#         kernel = @cuda launch=false MeansMahalinobis.meansMahalinobisKernelB(args...)
       
#         config = launch_configuration(kernel.fun, shmem=threads->get_shmem(get_threads(threads)))
    
#        # convert to 2D block size and figure out appropriate grid size
#         threads = get_threads(config.threads)
#         blocks = UInt32(config.blocks)
#         loopXdim = UInt32(cld(maxX, threads[1]))
#         loopYdim = UInt32(cld(maxY, threads[2])) 
#         loopZdim = UInt32(cld(maxZ,blocks )) 
    
    
    
    
#         totalX= CuArray([0]);
#     totalY= CuArray([0]);
#     totalZ= CuArray([0]);
#     totalCount= CuArray([0]);
#         dynamicMemoryLength = sum(threads)+intermediateresCheck
    
#         resListCounter= CUDA.zeros(UInt32,1)
#         resList= CUDA.zeros(UInt32, length(goldBoolGPU) );
    
#         args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,(maxX, maxY,maxZ)
#         ,resList,resListCounter,intermediateresCheck
#         ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)

#         @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernelB(args...)
#         @test totalCount[1]==45
#         @test totalX[1]== sum(map(ind->ind[1],cartTrueGold))
#         @test totalY[1]== sum(map(ind->ind[2],cartTrueGold))
#         @test totalZ[1]== sum(map(ind->ind[3],cartTrueGold))
    

        

        
# using BenchmarkTools
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
# BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
# BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


# function toBenchA()
  
#     totalX= CuArray([0]);
#     totalY= CuArray([0]);
#     totalZ= CuArray([0]);
#     totalCount= CuArray([0]);
#         dynamicMemoryLength = sum(threads)+intermediateresCheck
    
#         resListCounter= CUDA.zeros(UInt32,1)
#         resList= CUDA.zeros(UInt32, length(goldBoolGPU) );
    
#         args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxX, maxY,maxZ
#         ,resList,resListCounter,intermediateresCheck
#         ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)


#     CUDA.@sync   @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)

#                         end

# function toBenchB()
  
#     totalX= CuArray([0]);
#     totalY= CuArray([0]);
#     totalZ= CuArray([0]);
#     totalCount= CuArray([0]);
#         dynamicMemoryLength = sum(threads)+intermediateresCheck
    
#         resListCounter= CUDA.zeros(UInt32,1)
#         resList= CUDA.zeros(UInt32, length(goldBoolGPU) );
    
#         args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,(maxX, maxY,maxZ)
#         ,resList,resListCounter,intermediateresCheck
#         ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)


#     CUDA.@sync   @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernelB(args...)

#                         end

# bbA = @benchmark toBenchA() 
# bbB = @benchmark toBenchB()
# #bbA = @benchmark toBench()  setup=(goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools())
