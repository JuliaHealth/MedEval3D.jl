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

    # @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernel(args...)
    # @test totalCount[1]==45
    # @test totalX[1]== sum(map(ind->ind[1],cartTrueGold))
    # @test totalY[1]== sum(map(ind->ind[2],cartTrueGold))
    # @test totalZ[1]== sum(map(ind->ind[3],cartTrueGold))



    # @device_code_warntype interactive=true @cuda MeansMahalinobis.meansMahalinobisKernel(args...)












    args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,(maxX, maxY,maxZ)
    ,resList,resListCounter,intermediateresCheck
    ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)
    
    
        # calculate the amount of dynamic shared memory for a 2D block size
        get_shmem(threads) = (sizeof(UInt32)*3*4)
        
        function get_threads(threads)
            threads_x = 32
            threads_y = cld(threads,threads_x )
            return (threads_x, threads_y)
        end
    
        kernel = @cuda launch=false MeansMahalinobis.meansMahalinobisKernelB(args...)
       
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
    
        args = (goldBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,(maxX, maxY,maxZ)
        ,resList,resListCounter,intermediateresCheck
        ,totalX,totalY,totalZ,totalCount,blocks,debugArr,dynamicMemoryLength)

        @cuda threads=threads blocks=blocks MeansMahalinobis.meansMahalinobisKernelB(args...)
        @test totalCount[1]==45
        @test totalX[1]== sum(map(ind->ind[1],cartTrueGold))
        @test totalY[1]== sum(map(ind->ind[2],cartTrueGold))
        @test totalZ[1]== sum(map(ind->ind[3],cartTrueGold))
    

        a=2
        b=3
        offIt = 1 
        @redWitAct(offIt,1,a,+,b,-)
       dump( @redWitAct(offIt,1,a,+,b,-))






        dump( @redWitAct(offIt,1,a,+,b,-) )

        a = quote
            while 3 < 4
            1+1
            end
            end
     
dump(a)
