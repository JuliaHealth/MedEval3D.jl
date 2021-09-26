using  Test, Revise 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

using Main.HFUtils
using Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA, StaticArrays

using Main.HFUtils, Main.ProcessMainData,CUDA,Main.CUDAGpuUtils,StaticArrays
using Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA, StaticArrays


@testset "processMaskData" begin 


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


using Main.CUDAGpuUtils, Main.HFUtils






@testset "processMaskData" begin 

    testArrIn = CUDA.ones(Bool,32,32,32);
    referenceArray= CUDA.ones(Bool,32,32,32);
    resArray = CUDA.zeros(UInt16,32,32,32);
    resArraysCounter=CUDA.ones(UInt32,1);
    isMaskFull = Ref(false)
    isMaskEmpty = Ref(false)
    locArr = Ref(Int32(0))


    function testprocessMaskData(testArrInn::CuDeviceArray{Bool, 3, 1},resArray,referenceArray,resArraysCounter,isMaskFull::CUDA.CuRefValue{Bool},isMaskEmpty::CUDA.CuRefValue{Bool},locArr)
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(0),UInt8(0),UInt8(0)
    resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
    #locArr = Int32(0)
    #locArr.x |= true << UInt8(2)

   for zIter::UInt8 in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
        locBool::Bool = @inbounds testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
        #ProcessMainData.myIncreaseBitt(locBool, zIter,locArr)
        #locArr.x|= locBool << zIter
        locArr.x|= true << 4
        #locArr[] |= locBool << UInt8(zIter)
        processMaskData( locBool, zIter, resShmem)
    end#for 

    sync_threads() #we should have in resShmem what we need 
    
    # @unroll for zIter::UInt8  in UInt8(1):UInt8(32) # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
    #       local locBoolRegister::Bool = (locArr.x>>zIter & 1)==1
    #       local locBoolShmem::Bool = @inbounds resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
    #       #validataDataFirstPass(locBoolRegister,locBoolShmem,resShmem,isMaskFull,isMaskEmpty,blockBeginingX,blockBeginingY,blockBeginingZ,testArrInn, referenceArray,resArray,resArraysCounter,zIter)
    #     #CUDA.unsafe_free!(locBoolRegister)
    #     # CUDA.unsafe_free!(locBoolShmem)
    #  end#for

    return

end#testprocessMaskData

@cuda threads=(32,32) blocks=1 testprocessMaskData(testArrIn,resArray,referenceArray,resArraysCounter,isMaskFull,isMaskEmpty,locArr) 
isMaskFull

@device_code_warntype interactive=true @cuda testprocessMaskData(testArrIn,resArray,referenceArray,resArraysCounter,isMaskFull,isMaskEmpty,locArr)


outB= false
function modBool(bb)
    bb=true
end    
modBool(outB)
outB


rr = Ref(false)

function modBoolB(rrr)
    rrr[]=true
end   
modBoolB(rr)

rr[]





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


