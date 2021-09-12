using Revise, Parameters, Logging
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernels\\TpfpfnKernel.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\overLap\\MainOverlap.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernels\\InterClassCorrKernel.jl")
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA
using Main.MainOverlap, Main.TpfpfnKernel, Main.InterClassCorrKernel

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();




sizz= size(goldBoolGPU)
goldBoolGPU

sumOfGold= CuArray([0.0]);
sumOfSegm= CuArray([0.0]);
sswTotal= CuArray([0.0]);
SsbTotalGlobal = CuArray([0.0]);

sumOfGoldPartials = CuArray(zeros(Float32,nz));
sumOfSegmPartials = CuArray(zeros(Float32,nz));
sswPartials = CuArray(zeros(Float32,nz));
ssbPartialsperSlice = CuArray(zeros(Float32,nz));
maxSlicesPerBlock,slicesPerBlockMatrix,numberOfBlocks = InterClassCorrKernel.assignWorkToCooperativeBlocks( sizz[3])

loopNumb, indexCorr = getKernelContants(256,sizz[1])

args =  (FlattGoldGPU
        ,FlattSegGPU
        ,loopNumb
        ,indexCorr
        ,sizz[1]*sizz[1]
        ,length(goldBoolGPU)
        ,sumOfGold
        ,sumOfSegm
        ,sumOfGoldPartials
        ,sumOfSegmPartials
        ,sswPartials
        ,sswTotal
        ,ssbPartialsperSlice
        ,SsbTotalGlobal
        ,slicesPerBlockMatrix
        ,maxSlicesPerBlock)




@cuda cooperative=true threads=256 blocks=numberOfBlocks InterClassCorrKernel.kernel_InterClassCorr(args...)



sumOfGold[1] ==sum(FlattGoldGPU)==sum(sumOfGoldPartials)
sumOfSegm[1] ==sum(FlattSegGPU)==sum(sumOfSegmPartials)



using Test
z = mean(FlattGoldGPU)
typeof(z)
sizz= size(goldBoolGPU)

flattG, flattSeg
512*512


pixelsNum = sizz[1]*sizz[1]
ll = length(goldBoolGPU)
slicePixelNumb=pixelsNum 
threadId=256
sliceNumb= sizz[3] #350# sizz[3]

ress = ((threadId-1)* indexCorr) + (slicePixelNumb*(sliceNumb-1))+1# used as a basis to get data we want from global memory

ress/slicePixelNumb
ll/slicePixelNumb
diff = ll - ress


   """
   returning the data  from a kernel that  calclulate number of true positives,
   true negatives, false positives and negatives par image and per slice in given data 
   goldBoolGPU - array holding data of gold standard bollean array
   segmBoolGPU - boolean array with the data we want to compare with gold standard
   tp,tn,fp,fn - holding single values for true positive, true negative, false positive and false negative
   intermediateResTp, intermediateResFp, intermediateResFn - arrays holding slice wise results for true positive ...
   threadNumPerBlock = threadNumber per block defoult is 512
   IMPORTANT - in the ned of the goldBoolGPU and segmBoolGPU one need  to add some  additional number of 0=falses - number needs to be the same as indexCorr
   IMPORTANT - currently block sizes of 512 are supported only
   """
   function getTpfpfnData!(goldBoolGPU
       , segmBoolGPU
       ,tp,tn,fp,fn
       ,intermediateResTp
       ,intermediateResFp
       ,intermediateResFn
       ,sliceEdgeLength::Int64
       ,numberOfSlices::Int64
       ,threadNumPerBlock::Int64 = 512)
   
   blockNum, loopNumb, indexCorr = getKernelContants(threadNumPerBlock,numberOfSlices,sliceEdgeLength)
   args = (goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn, loopNumb, indexCorr,sliceEdgeLength,Int64(round(threadNumPerBlock/32)))
   @cuda threads=threadNumPerBlock blocks=blockNum getBlockTpFpFn(args...) 
   
   end#getTpfpfnData






"""
creates shared memory and initializes it to 0
wid - the number of the warp in the block
"""
function createAndInitializeShmem(wid, threadId,sliceEdgeLength,amountOfWarps)
   #shared memory for  stroing intermidiate data per lane  
   shmem = @cuStaticSharedMem(UInt8, (513,3))
   #for storing results from warp reductions
   shmemSum = @cuStaticSharedMem(UInt16, (33,3))
    #setting shared memory to 0 
    shmem[threadId, 2]=0
    shmem[threadId, 1]=0
    shmemSum[wid,1]=0
    shmemSum[wid,2]=0
return (shmem,shmemSum )

end#createAndInitializeShmem








