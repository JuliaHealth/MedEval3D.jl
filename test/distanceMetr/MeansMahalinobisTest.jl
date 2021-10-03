using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")


using Main.BasicPreds, Main.CUDAGpuUtils , Main.MeansMahalinobis




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
maxX, maxY,maxZ = size(goldBoolCPU);

resList= CUDA.zeros(UInt32, length(goldBoolGPU) );
resListCounter= CuArray([0]);

intermediateresCheck=UInt16(100);
intermidiateResLength=UInt16(1);
warpNumber= UInt16(1)

totalX= CuArray([0]);
totalY= CuArray([0]);
totalZ= CuArray([0]);
totalCount= CuArray([0]);

args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxX, maxY,maxZ
,resList,resListCounter,intermediateresCheck
,totalX,totalY,totalZ,totalCount)

    # calculate the amount of dynamic shared memory for a 2D block size
    get_shmem(threads) = (( threads[1] *threads[2]*(cld(maxX,threads[1] )))+intermediateresCheck * sizeof(UInt16)) + (sizeof(UInt32)*33*4)+ (sizeof(UInt32)*33*4)
    
    function get_threads(threads)
        threads_x = 32
        threads_y = cld(threads,threads_x )
        return (threads_x, threads_y)
    end

    kernel = @cuda launch=false MeansMahalinobis.meansMahalinobisKernel(args...)
   
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(get_threads(threads)))

   # convert to 2D block size and figure out appropriate grid size
    threads = get_threads(config.threads)
    blocks = config.blocks

    maxX, maxY,maxZ 
    loopXdim = UInt32(cld(maxX, threads[1]))
    loopYdim = UInt32(cld(maxY, threads[2])) 
    loopZdim = UInt32(cld(maxZ,blocks )) 


    args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxX, maxY,maxZ
    ,resList,resListCounter,intermediateresCheck
    ,totalX,totalY,totalZ,totalCount)

    @cuda threads=(16,16) blocks=1 MeansMahalinobis.meansMahalinobisKernel(args...)
    resListCounter[1]
    Int64(maximum(resList))

