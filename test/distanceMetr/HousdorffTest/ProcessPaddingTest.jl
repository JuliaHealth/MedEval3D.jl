
using  Test, Revise,CUDA 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessPadding.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu


using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu

@testset "processPaddingTest" begin 

    testArrInCPU= falses(60,60,60);
    testArrInCPU[CartesianIndex(1,1,1)]= true;
    testArrInCPU[CartesianIndex(5,5,5)]= true;
    testArrIn = CuArray(testArrInCPU);
    referenceArray= CUDA.ones(Bool,32,32,32);
    resArray = CUDA.zeros(UInt32,32,32,32);
    resArraysCounter=CUDA.zeros(Int32,1);
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0)
    currBlockX,currBlockY,currBlockZ =UInt8(2),UInt8(2),UInt8(2)
    isPassGold = true;
    metaDataCPU = falses(10,10,10,4); # all neighbouring set to be inactive and not full
    metadata= CuArray(metaDataCPU);
    metadataDims = size(metaData);
    mainQuesCounter=CUDA.zeros(Int32,1);
    mainWorkQueue = CUDA.zeros(Bool,10,4)#simulating place for 10 blocks
    iterationNumber= UInt32(2)
function testKernelForPaddingAnalysis(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resArray,resArraysCounter
                                    ,isPassGold,currBlockX,currBlockY,currBlockZ,metadata,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        ProcessMainData.executeDataIterFirstPassWithPadding(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter,currBlockX,currBlockY,currBlockZ,isPassGold,metadata,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber)
        #clearMainShmem(shmem)
        #now we need to deal with padding in shmem res
        # processAllPaddingPlanes(blockBeginingX,blockBeginingY,blockBeginingZ,resShmem
        #                     ,currBlockX,currBlockY,currBlockZ
        #                     ,analyzedArr,refAray,resArray
        #                     ,metaData,metadataDims
        #                     ,isPassGold)
        return
end


@cuda threads=(32,32) blocks=1 testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,
  resArraysCounter,isPassGold,currBlockX,currBlockY,currBlockZ,metadata,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber) 
resArraysCounter[1]

@device_code_warntype interactive=true @cuda testprocessMaskData(testArrIn,resArray,referenceArray,resArraysCounter)

end