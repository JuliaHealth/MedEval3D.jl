using  Test, Revise 
includet("./src/utils/CUDAGpuUtils.jl")
includet("./src/distanceMetrics/Housdorff/mainHouseDorffKernel/HFUtils.jl")
includet("./src/distanceMetrics/Housdorff/mainHouseDorffKernel/ProcessMainData.jl")
includet("./test/GPUtestUtils.jl")

using ..HFUtils
using ..CUDAGpuUtils,BenchmarkTools , CUDA, StaticArrays

using ..HFUtils, ..ProcessMainData,CUDA,..CUDAGpuUtils,StaticArrays
using ..CUDAGpuUtils,BenchmarkTools , CUDA, StaticArrays

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
    locArr.x|= locBool << zIter
    #locArr[] |= locBool << UInt8(zIter)
    ProcessMainData.processMaskData( locBool, zIter, resShmem)
end#for 

sync_threads() #we should have in resShmem what we need 

# @unroll for zIter::UInt8  in UInt8(1):UInt8(32) # most outer loop is responsible for z dimension - importnant in this loop we ignore padding we will deal with it separately
#       local locBoolRegister::Bool = (locArr[]>>zIter & UInt32(1))==UInt32(1)
#       local locBoolShmem::Bool = resShmem[threadIdxX()+1,threadIdxY()+1,zIter+1]
#       ProcessMainData.validataDataFirstPass(locBoolRegister,locBoolShmem,resShmem,isMaskFull,isMaskEmpty,blockBeginingX,blockBeginingY,blockBeginingZ,testArrInn, referenceArray,resArray,resArraysCounter,zIter)
#     #CUDA.unsafe_free!(locBoolRegister)
#     # CUDA.unsafe_free!(locBoolShmem)
#  end#for

return

end#testprocessMaskData

@cuda threads=(32,32) blocks=1 testprocessMaskData(testArrIn,resArray,referenceArray,resArraysCounter,isMaskFull,isMaskEmpty,locArr) 