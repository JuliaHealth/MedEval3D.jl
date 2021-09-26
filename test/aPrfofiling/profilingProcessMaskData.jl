
 #nv-nsight-cu-cli --mode=launch julia 
#  using  Test, Revise 
 
#  includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\aPrfofiling\\profilingProcessMaskData.jl")
 
#  CUDA.@profile wrapForProfile()




using  Test, Revise 

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\ProcessMainData.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")

using Main.HFUtils, Main.ProcessMainData,CUDA,Main.CUDAGpuUtils,StaticArrays




function processMaskDataB(maskBool::Bool
    ,zIter::UInt8
    ,resShmem
    ,locArr::UInt32 )
# save it to registers - we will need it later

locArr|= maskBool << zIter

#now we are saving results evrywhere we are intrested in so around without diagonals (we use supremum norm instead of euclidean)
if(maskBool)
resShmem[threadIdxX()+1,threadIdxY()+1,zIter]=true #up
resShmem[threadIdxX()+1,threadIdxY()+1,zIter+2]=true #down

resShmem[threadIdxX(),threadIdxY()+1,zIter+1]=true #left
resShmem[threadIdxX()+2,threadIdxY()+1,zIter+1]=true #right

resShmem[threadIdxX()+1,threadIdxY()+2,zIter+1]=true #front
resShmem[threadIdxX()+1,threadIdxY(),zIter+1]=true #back
end#if    
return true
end#processMaskData



function wrapForProfile()

    testArrIn = CUDA.ones(Bool,32,32,32);
    testArrOut = CUDA.zeros(Bool,34,34,34);
    
    function testKernprocessMaskDataB(testArrInn,testArrOut)
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
            local locBool = testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)]
            locArr|= locBool << zIter
           processMaskData( locBool, zIter, resShmem)
            end
        sync_threads()
            #fillGlobalFromShmem(testArrOut,resShmem)
    
        return
    end    
   
    @cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 
   
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