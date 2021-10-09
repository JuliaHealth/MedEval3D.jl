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

#cartTrueGold =  collect(vec(CartesianIndices(zeros(11,4,16) ).+CartesianIndex(70,71,78)))
cartTrueGold =  unique(vcat(collect(vec(CartesianIndices(zeros(8,15,5) ).+CartesianIndex(5,5,5)))
         ,collect(vec(CartesianIndices(zeros(14,19,25) ).+CartesianIndex(100,99,88)))
         ,collect(vec(CartesianIndices(zeros(81,87,84) ).+CartesianIndex(100,99,210)))
         )) ;
goldBoolCPU[cartTrueGold].=Float32(1.0);

cartTrueSegm =  CartesianIndices(zeros(34,235,76) ).+CartesianIndex(100,100,100);

segmBoolCPU[cartTrueSegm].=Float32(1.0);

numberToLooFor= Float32(1.0);
goldBoolGPU = CuArray(goldBoolCPU);
segmBoolGPU = CuArray(segmBoolCPU);



#we will fill it after we work with launch configuration
loopXdim = UInt32(1);loopYdim = UInt32(1) ;loopZdim = UInt32(1) ;
sizz = size(goldBoolCPU);maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
#gold
totalXGold= CuArray([0.0]);
totalYGold= CuArray([0.0]);
totalZGold= CuArray([0.0]);
totalCountGold= CuArray([0]);
#segm
totalXSegm= CuArray([0.0]);
totalYSegm= CuArray([0.0]);
totalZSegm= CuArray([0.0]);

varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);
varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);

totalCountSegm= CuArray([0]);
totalCountGold
#countPerZGold= CUDA.zeros(Float32,sizz[3]+1);
#countPerZSegm= CUDA.zeros(Float32,sizz[3]+1);
countPerZGold= CUDA.zeros(Float32,500);
countPerZSegm= CUDA.zeros(Float32,500);

#covariancesSliceWise= CUDA.zeros(Float32,12,sizz[3]+1);
covariancesSliceWiseGold= CUDA.zeros(Float32,6,500);
covariancesSliceWiseSegm= CUDA.zeros(Float32,6,500);
covarianceGlobal= CUDA.zeros(Float32,12,1);

mahalanobisResGlobal= CUDA.zeros(1);
mahalanobisResSliceWise= CUDA.zeros(500);
#mahalanobisResSliceWise= CUDA.zeros(sizz[3]);

args = (goldBoolGPU,segmBoolGPU,numberToLooFor
,loopYdim,loopXdim,loopZdim
,(maxX, maxY,maxZ)
,totalXGold,totalYGold,totalZGold,totalCountGold
,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,countPerZGold
, countPerZSegm,covariancesSliceWiseGold, covariancesSliceWiseSegm,
varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
    ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
    ,mahalanobisResGlobal, mahalanobisResSliceWise)


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

    Int64(blocks*loopZdim+blo)

#covariancesSliceWise= CUDA.zeros(Float32,12,sizz[3]+1);
covariancesSliceWiseGold= CUDA.zeros(Float32,6,500);
covariancesSliceWiseSegm= CUDA.zeros(Float32,6,500);

#gold
totalXGold= CuArray([0.0]);
totalYGold= CuArray([0.0]);
totalZGold= CuArray([0.0]);
totalCountGold= CuArray([0]);
#segm
totalXSegm= CuArray([0.0]);
totalYSegm= CuArray([0.0]);
totalZSegm= CuArray([0.0]);
totalCountSegm= CuArray([0]);



varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);
varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);


args = (goldBoolGPU,segmBoolGPU,numberToLooFor
,loopYdim,loopXdim,loopZdim
,(maxX, maxY,maxZ)
,totalXGold,totalYGold,totalZGold,totalCountGold
,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,countPerZGold
, countPerZSegm,covariancesSliceWiseGold, covariancesSliceWiseSegm,
varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
    ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
    ,mahalanobisResGlobal, mahalanobisResSliceWise)

    @cuda cooperative=true threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)
  
 

@testset " mahalinobis tests " begin
      @test totalCountGold[1]==length(cartTrueGold)
    @test totalXGold[1]== sum(map(ind->ind[1],cartTrueGold))
    @test totalYGold[1]== sum(map(ind->ind[2],cartTrueGold))
    @test totalZGold[1]== sum(map(ind->ind[3],cartTrueGold))

    @test totalXSegm[1]== sum(map(ind->ind[1],cartTrueSegm))
    @test totalYSegm[1]== sum(map(ind->ind[2],cartTrueSegm))
    @test totalZSegm[1]== sum(map(ind->ind[3],cartTrueSegm))
    @test totalCountSegm[1]== length(cartTrueSegm)
    @test sum(countPerZGold)==length(cartTrueGold)
    @test sum(countPerZSegm)==length(cartTrueSegm)
    #@test countPerZSegm[100+1]==150*150
    @test countPerZGold[5+1]==8*15

    meanX = totalXGold[1]/totalCountGold[1]
    meanY = totalYGold[1]/totalCountGold[1]
    meanZ = totalZGold[1]/totalCountGold[1]
    @test isapprox(varianceXGlobalGold[1],sum(map(ind->( ind[1] -meanX )^2 ,cartTrueGold)); atol = 50)

    @test isapprox(covarianceXYGlobalGold[1],sum(map(ind->(( ind[1] -meanX )*( ind[2] -meanY )) ,cartTrueGold)); atol = 50)
    @test isapprox(covarianceXZGlobalGold[1],sum(map(ind->(( ind[1] -meanX )*( ind[3] -meanZ )) ,cartTrueGold)); atol = 50)
    
   Int64(round( covarianceXZGlobalGold[1]- sum(map(ind->(( ind[1] -meanX )*( ind[3] -meanZ )) ,cartTrueGold))))

    @test isapprox(varianceYGlobalGold[1],sum(map(ind->(( ind[2] -meanY )^2) ,cartTrueGold)); atol = 50)

    @test isapprox(varianceZGlobalGold[1], sum(map(ind->(( ind[3] -meanZ )^2) ,cartTrueGold)); atol = 50) 
    
    @test isapprox(covarianceYZGlobalGold[1],sum(map(ind->(( ind[2] -meanY )*(ind[3] -meanZ  )) ,cartTrueGold)); atol = 50)
      
    meanX = totalXSegm[1]/totalCountSegm[1]
    @test isapprox(varianceXGlobalSegm[1],sum(map(ind->( ind[1] -meanX )^2 ,cartTrueSegm)); atol = 50)

    meanY = totalYSegm[1]/totalCountSegm[1]
    @test isapprox(covarianceXYGlobalSegm[1],sum(map(ind->(( ind[1] -meanX )*( ind[2] -meanY )) ,cartTrueSegm)); atol = 50)
    meanZ = totalZSegm[1]/totalCountSegm[1]
    @test isapprox(covarianceXZGlobalSegm[1],sum(map(ind->(( ind[1] -meanX )*( ind[3] -meanZ )) ,cartTrueSegm)); atol =50)
    
    @test isapprox(varianceYGlobalSegm[1],sum(map(ind->(( ind[2] -meanY )^2) ,cartTrueSegm)); atol =50)

    @test isapprox(varianceZGlobalSegm[1],sum(map(ind->(( ind[3] -meanZ )^2) ,cartTrueSegm)); atol = 50)
    
    Int64(round(varianceZGlobalSegm[1]-sum(map(ind->(( ind[3] -meanZ )^2) ,cartTrueSegm))))
    
    @test isapprox(covarianceYZGlobalSegm[1],sum(map(ind->(( ind[2] -meanY )*(ind[3] -meanZ  )) ,cartTrueSegm)); atol = 50)
      
end





    
    # Int64(varianceXGlobalGold[1])-sum(map(ind->( ind[1] -meanX )^2 ,cartTrueGold))

    


    # @device_code_warntype interactive=true @cuda MeansMahalinobis.meansMahalinobisKernel(args...)


