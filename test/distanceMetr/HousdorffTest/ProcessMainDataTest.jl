using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\WorkQueueUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\ProcessMainDataVerB.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\MetadataAnalyzePass.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils



##### iter data block
indices = CUDA.zeros(900,15);
threads=(32,5);
blocks =17;
mainArrDims= (67,177,90);
datBdim = (43,21,17);
mainArrCPU= falses(mainArrDims);
mainArrCPU[5,5,5]= true;
mainArrGPU = CuArray(mainArrCPU);
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);
metaDataDims= size(metaData);
inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    );
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (1,1)#(metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )

resShmem = CUDA.zeros(Bool,(datBdim[1]+2,datBdim[2]+2,datBdim[3]+2 ));
sourceShmem = CUDA.zeros(Bool,(datBdim));

function processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
            maskBool=mainArrGPU[x,y,z]
            ProcessMainDataVerB.@processMaskData( maskBool)

    end)end)
    
    return
end


@cuda threads=threads blocks=blocks processDataKernel(mainArrGPU,sourceShmem,resShmem,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
@test  resShmem[5+1,5+1,5+1]==false
@test  resShmem[7,6,6]==true
@test  resShmem[4+1,5+1,5+1]==true
@test  resShmem[5+1,4+1,5+1]==true
@test  resShmem[5+1,6+1,5+1]==true
@test  resShmem[5+1,5+1,4+1]==true
@test  resShmem[5+1,5+1,6+1]==true

sum(resShmem)














# @testset "processMaskData" begin 


    ####### first it should be zeros 
    testArrIn = CUDA.zeros(Bool,34,34,34);
    testArrOut = CUDA.zeros(Bool,34,34,34);


    
    function testKernprocessMaskData(testArrInn,testArrOut)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        clearMainShmem(resShmem)
        blockBeginingX=1
        blockBeginingY=1
        blockBeginingZ=1
        isMaskFull= zeros(MVector{1,Bool})
        isMaskEmpty= ones(MVector{1,Bool}) 
        #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
        locArr= UInt32(0)
        
        @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
           processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem)
        end
        fillGlobalFromShmem(testArrOut,resShmem)

        return
    end    
    @cuda threads=(32,32) blocks=1 testKernprocessMaskData(testArrIn,testArrOut) 
    @test  sum(testArrOut)==0


    ########### now  we get one arbitrary point it should lead to 6 in output
    testArrIn = CUDA.zeros(Bool,32,32,32);
    testArrOut = CUDA.zeros(Bool,34,34,34);

    testArrIn[5,5,5]=true
    function testKernprocessMaskDataB(testArrInn,testArrOut)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        clearMainShmem(resShmem)
        blockBeginingX=0
        blockBeginingY=0
        blockBeginingZ=0
        isMaskFull= zeros(MVector{1,Bool})
        isMaskEmpty= ones(MVector{1,Bool}) 
        #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
        locArr= UInt32(0)
        
        @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
           processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem)
        end
        sync_threads()
        fillGlobalFromShmem(testArrOut,resShmem)

        return
    end    
    @cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 
   
    @test  sum(testArrOut)==6
    @test  testArrOut[5,5,5]==false
    @test  testArrOut[7,6,6]==true
    @test  testArrOut[4+1,5+1,5+1]==true
    @test  testArrOut[5+1,4+1,5+1]==true
    @test  testArrOut[5+1,6+1,5+1]==true
    @test  testArrOut[5+1,5+1,4+1]==true
    @test  testArrOut[5+1,5+1,6+1]==true

    getIndiciesWithTrue(testArrOut)

    ########### checking corner cases
    testArrIn = CUDA.zeros(Bool,32,32,32);
    testArrOut = CUDA.zeros(Bool,34,34,34);
    testLocArr = CUDA.zeros(UInt32,32,32,32);


    testArrIn[1,1,1]=true
    function testKernprocessMaskDataBB(testArrInn,testArrOut)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        clearMainShmem(resShmem)
        blockBeginingX=UInt8(0)
        blockBeginingY=UInt8(0)
        blockBeginingZ=UInt8(0)
        isMaskFull= false
        isMaskEmpty= false 
        #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
        locArr= UInt32(0)
        
        @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
            #locArr|= true << zIter
            processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem)
            # testLocArr[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=locArr
        end
        sync_threads()
        fillGlobalFromShmem(testArrOut,resShmem)

        return
    end    
    @cuda threads=(32,32) blocks=1 testKernprocessMaskDataBB(testArrIn,testArrOut) 
    @test  sum(testArrOut)==6

    @test  testArrOut[1,1,1]==false
    @test  testArrOut[2,2,1]==true
    #locArr|= true << zIter

    CUDA.reclaim()# just to destroy from gpu our dummy data


### stress test with ones data

testArrIn = CUDA.ones(Bool,34,34,34);
testArrOut = CUDA.zeros(Bool,34,34,34);

function testKernprocessMaskDataB(testArrInn,testArrOut)
    resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
    clearMainShmem(resShmem)
    blockBeginingX=0
    blockBeginingY=0
    blockBeginingZ=0

    #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
    locArr= UInt32(0)
    
    @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
       local locBool = testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
       # locArr|= locBool << zIter
       processMaskData( locBool, zIter, resShmem)
       CUDA.unsafe_free!(locBool)
    end
    sync_threads()
    #CUDA.@cuprint "locArr $(Int64(locArr)) \n" 

    fillGlobalFromShmem(testArrOut,resShmem)

    return
end    
@cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 
@test  (length(testArrOut)-(sum(testArrOut)+(4*34)+(8*32)  )) ==0

end#processMaskData





using  Test, Revise,CUDA 

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")


using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu


using Main.CUDAGpuUtils, Main.HFUtils,Cthulhu


@testset "executeDataIterFirstPass" begin 

    testArrInCPU= falses(60,60,60)
    testArrInCPU[CartesianIndex(1,1,1)]= true
    testArrInCPU[CartesianIndex(5,5,5)]= true
    testArrIn = CuArray(testArrInCPU);
    referenceArray= CUDA.ones(Bool,32,32,32);
    resArray = CUDA.zeros(UInt16,32,32,32);
    resArraysCounter=CUDA.zeros(Int32,1);
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0)

function testKernelForExecuteMainPass(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resArray,resArraysCounter)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        ProcessMainData.executeDataIterFirstPass(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter)
    return
end

@cuda threads=(16,16) blocks=1 testKernelForExecuteMainPass(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ ,resArray,resArraysCounter) 
@test resArraysCounter[1]==9

@test resArray[5,5,5]==false
@test resArray[5,6,5]==true
@test resArray[6,5,5]==true
@test resArray[4,5,5]==true
@test resArray[5,4,5]==true
@test resArray[5,6,5]==true
@test resArray[5,5,4]==true
@test resArray[5,5,6]==true

# @test resArray[1,1,2]==true
# @test resArray[1,2,1]==true
# @test resArray[2,1,1]==true

end


@testset "processMaskData" begin 

    testArrIn = CUDA.ones(Bool,32,32,32);
    referenceArray= CUDA.ones(Bool,32,32,32);
    resArray = CUDA.zeros(UInt16,32,32,32);
    resArraysCounter=CUDA.zeros(Int32,1);

    function testprocessMaskData(testArrInn::CuDeviceArray{Bool, 3, 1},resArray,referenceArray,resArraysCounter)
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0)

    resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
    locArr = Int32(0)
    isMaskFull= false
    isMaskEmpty = false
    #locArr.x |= true << UInt8(2)

    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locBool::Bool = @inbounds testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
        locArr|= locBool << zIter
        ProcessMainData.processMaskData( locBool, zIter, resShmem)
    end#for 
   sync_threads() #we should have in resShmem what we need 

    @unroll for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locVal::Bool = @inbounds  (locArr>>zIter & Int32(1))==Int32(1)
        shmemVal::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
        locValOrShmem = (locVal | shmemVal)
        isMaskFull= locValOrShmem & isMaskFull
        isMaskEmpty = ~locValOrShmem & isMaskEmpty
     
       if(!locVal && shmemVal)
         # setting value in global memory
         @inbounds  testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]= true
         # if we are here we have some voxel that was false in a primary mask and is becoming now true - if it is additionaly true in reference we need to add it to result
         isInReferencaArr::Bool= @inbounds referenceArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
         if(isInReferencaArr)
            @inbounds  resArray[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]=UInt16(1)
            #CUDA.atomic_inc!(pointer(resArraysCounter), Int32(1))

            atomicallyAddOneInt(resArraysCounter)
         end#if
       end#if

       

         end#for

    
  # IterToValidate()

    return

end#testprocessMaskData

@cuda threads=(32,32) blocks=1 testprocessMaskData(testArrIn,resArray,referenceArray,resArraysCounter) 
resArraysCounter


end







# #   nv-nsight-cu-cli --mode=launch julia 
# using  Test, Revise 

# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\aPrfofiling\\profilingProcessMaskData.jl")

# CUDA.@profile wrapForProfile()


# using  Test, Revise 

# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

# using Main.HFUtils, Main.ProcessMainData,CUDA,Main.CUDAGpuUtils,StaticArrays


# function wrapForProfile()

#     testArrIn = CUDA.ones(Bool,34,34,34);
#     testArrOut = CUDA.zeros(Bool,34,34,34);
    
#     function testKernprocessMaskDataB(testArrInn,testArrOut)
#         resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
#         clearMainShmem(resShmem)
#         blockBeginingX=UInt8(0)
#         blockBeginingY=UInt8(0)
#         blockBeginingZ=UInt8(0)
#         isMaskFull= zeros(MVector{1,Bool})
#         isMaskEmpty= ones(MVector{1,Bool}) 
#         #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
#         locArr= zeros(MVector{32,Bool})
        
#         @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
#            processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem,locArr)
#         end
#         sync_threads()
    
#         fillGlobalFromShmem(testArrOut,resShmem)
    
#         return
#     end    
#     @cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 
# end    
# CUDA.@profile wrapForProfile()



#     CUDA.reclaim()# just to destroy from gpu our dummy data

# end # processMaskData


