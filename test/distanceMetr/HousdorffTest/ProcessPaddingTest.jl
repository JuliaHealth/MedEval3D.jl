
using  Test, Revise,CUDA 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessPadding.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu


using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu

#@testset "processPaddingTest" begin 

    testArrInCPU= falses(100,100,100);
    #testArrInCPU[CartesianIndex(1,1,1)]= true;
    testArrInCPU[CartesianIndex(1,1,1)]= true
    testArrInCPU[CartesianIndex(5,5,5)]= true
    testArrIn = CuArray(testArrInCPU);

    metaDataCPU = falses(15,15,15,4); # all neighbouring set to be inactive and not full
    metaData= CuArray(metaDataCPU);
    debugArr= CUDA.zeros(Bool,9);
    resArraysCounter=CUDA.zeros(Int32,1);


    referenceArray= CUDA.ones(Bool,100,100,100);
    resArray = CUDA.zeros(UInt16,50,50,50);
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(33),UInt8(33),UInt8(33);
    currBlockX,currBlockY,currBlockZ =UInt8(2),UInt8(2),UInt8(2);
    isPassGold = true;
    #metaDataCPU[2,2,3,:].= true

    metadataDims = size(metaData);
    mainQuesCounter=CUDA.zeros(Int32,1);
    mainWorkQueue = CUDA.zeros(UInt8,10,4);#simulating place for 10 blocks
    iterationNumber= UInt32(2);
function testKernelForPaddingAnalysis(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resArray,resArraysCounter
                                    ,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        ProcessMainData.executeDataIterFirstPassWithPadding(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter,currBlockX,currBlockY,currBlockZ,isPassGold,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)
        #clearMainShmem(shmem)
        #now we need to deal with padding in shmem res
        # processAllPaddingPlanes(blockBeginingX,blockBeginingY,blockBeginingZ,resShmem
        #                     ,currBlockX,currBlockY,currBlockZ
        #                     ,analyzedArr,refAray,resArray
        #                     ,metaData,metadataDims
        #                     ,isPassGold)


        return
end


@cuda threads=(16,16) blocks=1 testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,
  resArraysCounter,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr) 
  resArraysCounter[1]
  Int64(maximum(mainWorkQueue))



  indicies = CartesianIndices(mainWorkQueue);
  filtered = filter(ind-> mainWorkQueue[ind]>0,indicies  )


debugArr[5]
Int64(mainWorkQueue[2,4])

Int64(maximum(mainWorkQueue))

indicies = CartesianIndices(mainWorkQueue);
filtered = filter(ind-> mainWorkQueue[ind]>0,indicies  )

metaData[2, 2, 3, 2]==true


debugArr[1]
debugArr[2]
debugArr[3]
debugArr[4]
debugArr[5]
debugArr[6]











testArrInCPU= falses(100,100,100);
#testArrInCPU[CartesianIndex(1,1,1)]= true;
testArrInCPU[CartesianIndex(5,5,5)]= true;
testArrInCPU[5,5,32]=true

testArrIn = CuArray(testArrInCPU);
referenceArray= CUDA.ones(Bool,100,100,100);
resArray = CUDA.zeros(UInt32,32,32,32);
blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0);
currBlockX,currBlockY,currBlockZ =UInt8(2),UInt8(2),UInt8(2);
isPassGold = true;
metaDataCPU = falses(15,15,15,4); # all neighbouring set to be inactive and not full
#metaDataCPU[2,2,3,:].= true

metaData= CuArray(metaDataCPU);
metadataDims = size(metaData);
mainQuesCounter=CUDA.zeros(Int32,1);
mainWorkQueue = CUDA.zeros(Bool,10,4);#simulating place for 10 blocks
iterationNumber= UInt32(2);

@device_code_warntype interactive=true @cuda testKernelForPaddingAnalysis(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ,  resArray,resArraysCounter,isPassGold,currBlockX,currBlockY,currBlockZ,metaData,metadataDims,mainQuesCounter,mainWorkQueue,iterationNumber,debugArr)

#end