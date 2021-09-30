
 #nv-nsight-cu-cli --mode=launch julia 
#  using  Test, Revise ,CUDA
 
#  includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\aPrfofiling\\profilingProcessMaskData.jl")
 
#  CUDA.@profile wrapForProfile()




using  Test, Revise,CUDA 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")


using Main.CUDAGpuUtils, Main.HFUtils



function testKernelForExecuteMainPass(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resArray,resArraysCounter)
    resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
    ProcessMainData.executeDataIterFirstPass(analyzedArr, refAray,blockBeginingX,blockBeginingY,blockBeginingZ,resShmem,resArray,resArraysCounter)
return
end





function wrapForProfile()

    testArrInCPU= rand(Bool,200,200,200)
    #testArrInCPU[CartesianIndex(1,1,1)]= true
    testArrInCPU[CartesianIndex(5,5,5)]= true
    testArrIn = CuArray(testArrInCPU);
    referenceArray= CUDA.ones(Bool,200,200,200);
    resArray = CUDA.zeros(UInt16,200,200,200);
    resArraysCounter=CUDA.zeros(Int32,1);
    blockBeginingX,blockBeginingY,blockBeginingZ =UInt8(32),UInt8(32),UInt8(32)

    @cuda threads=(32,32) blocks=1 testKernelForExecuteMainPass(testArrIn,referenceArray,blockBeginingX,blockBeginingY,blockBeginingZ ,resArray,resArraysCounter) 

end    


# testArrIn = CUDA.ones(Bool,32,32,32);
# testArrOut = CUDA.zeros(Bool,34,34,34);

# function testKernprocessMaskDataB(testArrInn,testArrOut)
#     resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
#     clearMainShmem(resShmem)
#     blockBeginingX=UInt8(0)
#     blockBeginingY=UInt8(0)
#     blockBeginingZ=UInt8(0)
#     isMaskFull= false
#     isMaskEmpty= false
#     #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
#     locArr= UInt32(0)
    
#     @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
#         processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem,locArr)
#     end
#     sync_threads()
#         #fillGlobalFromShmem(testArrOut,resShmem)

#     return
# end    


# @cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 










# numb = UInt32(0)
# #settingcorrectly
# numb |= true << UInt8(2)
# numb


# numb = UInt32(0)
# #settingcorrectly
# numb |= 1 << 1
# numb |= 1 << 2
# numb |= 1 << 0
# numb |= 1 << 0
# numb |= 1 << 5
# numb


#reading...
# numb>>1 & UInt32(1) 
# numb>>2 & UInt32(1) 
# numb>>3 & UInt32(1) 
# numb>>4 & UInt32(1) 
# numb>>5 & UInt32(1) 


# #processMaskDataB( testArrInn[threadIdxY(),1,1], zIter, resShmem ) # coalesced