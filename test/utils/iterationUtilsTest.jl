using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")

using Main.BasicPreds, Main.CUDAGpuUtils , Main.MeansMahalinobis, Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils


function getExampleKernelArgs()
    nx=512 ; ny=512 ; nz=317
    #first we initialize the metrics on CPU so we will modify them easier
    goldBoolCPU= zeros(Float32,nx,ny,nz); #mimicks gold standard mask
    segmBoolCPU= ones(Float32,nx,ny,nz); #mimicks other  mask
    cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
    cartTrueSegm =  CartesianIndices(zeros(150,150,150) ).+CartesianIndex(100,100,100);

    goldBoolCPU[cartTrueGold].=Float32(1.0);
    segmBoolCPU[cartTrueSegm].=Float32(1.0);

    numberToLooFor= Float32(1.0);
    goldBoolGPU = CuArray(goldBoolCPU);
    segmBoolGPU = CuArray(segmBoolCPU);
    #we will fill it after we work with launch configuration
    loopXdim = UInt32(1);loopYdim = UInt32(1) ;loopZdim = UInt32(1) ;
    sizz = size(goldBoolCPU);maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
    threads = (32,20)
    blocks = 50
    loopXdim = UInt32(fld(maxX, threads[1]))
    loopYdim = UInt32(fld(maxY, threads[2])) 
    loopZdim = UInt32(fld(maxZ,blocks )) 

    totalCountGold= CuArray([0]);
    totalCountSegm= CuArray([0]);

    return (goldBoolGPU,segmBoolGPU,numberToLooFor
    ,loopYdim,loopXdim,loopZdim
    ,(maxX, maxY,maxZ)
    ,totalCountGold
    ,totalCountSegm)
end




# @testset "iter3d" begin

    args = getExampleKernelArgs()
    goldBoolGPU,segmBoolGPU,numberToLooFor,loopYdim,loopXdim,loopZdim,maxes,totalCountGold,totalCountSegm= args

    function testLoopKernelA(goldArr,segmArr
        ,numberToLooFor
        ,loopYdim::UInt32
        ,loopXdim::UInt32
        ,loopZdim::UInt32
        ,arrDims::Tuple{UInt32,UInt32,UInt32}
        ,totalCountGold,totalCountSegm)

        countGold::UInt32 = UInt32(0)
        countSegm::UInt32 = UInt32(0)
        shmemSum= @cuStaticSharedMem(UInt32, (32,2))   
        clearSharedMemWarpLong(shmemSum, UInt8(2))

        @iter3d(arrDims,  loopXdim,loopYdim,loopZdim,if(  @inbounds(goldArr[x,y,z])  ==numberToLooFor)
            #updating variables needed to calculate means
            countGold+=UInt32(1)
        end)#if bool in arr  
        
        sync_threads()

        @iter3dWithVal(segmArr,arrDims, loopXdim,loopYdim,loopZdim, if(  value  ==numberToLooFor)
            #updating variables needed to calculate means
            countSegm+=UInt32(1)
        end)#if bool in arr
        

        offsetIter = UInt8(1)
        @redWitAct(offsetIter,shmemSum,  countGold,+, countSegm,+  )
        @addAtomic(shmemSum,totalCountGold, totalCountSegm)

        return
    end

    @cuda threads=(32,20) blocks=50 testLoopKernelA(args...)

    @test totalCountGold[1]==3*3*5
    nx=512 ; ny=512 ; nz=317
    @test totalCountSegm[1]==(nx*ny*nz)










    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.BasicStructs
    using Shuffle,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates
    using Main.MainOverlap, Main.RandIndex , Main.ProbabilisticMetrics , Main.VolumeMetric ,Main.InformationTheorhetic
    using Main.CUDAAtomicUtils, Main.TpfpfnKernel, Main.InterClassCorrKernel
    

    dimx = 529
    dimy = 556
    dimz = 339
    # dimx = 333
    # dimy = 33
    # dimz = 33
    arrGold = CUDA.ones(UInt8,(dimx,dimy,dimz))
    arrAlgo =  CUDA.ones(UInt8,(dimx,dimy,dimz))
    sizz= size(arrGold)
    ##### load tests ...
    conf = ConfigurtationStruct(trues(12)...)
    numberToLooFor = UInt8(1)
    
    args,threads,blocks,metricsTuplGlobal= TpfpfnKernel.prepareForconfusionTableMetricsNoSliceWise(numberToLooFor  ,conf)
    
    # iterLoop= args[9]
    # pixPerSlice = Int64(args[10]) 
    singleSum= CUDA.zeros(UInt64,1)
    function testIter(goldGPU#goldBoolGPU
        , segmGPU#segmBoolGPU
        ,sliceMetricsTupl
        ,tp,tn,fp,fn#tp,tn,fp,fn
        ,arrDims,totalNumbOfVoxels,iterLoop,pixPerSlice
        ,numberToLooFor#numberToLooFor
        #,metricsTuplGlobal
        ,conf,singleSum)
        @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,begin 
            @atomic singleSum[1]+=1
            #CUDA.@cuprint "i $(i) \n"
        end)
    
    return 
    end
    
    @cuda threads=threads blocks=blocks testIter(arrGold, arrAlgo,args..., singleSum) #args[8][3]  is number of slices ...
    
    @test Int64(singleSum[1]) ==dimx*dimy*dimz
    
    Int64(singleSum[1])-dimx*dimy*dimz
    





    
# end#testset