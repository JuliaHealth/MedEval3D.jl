
 
#  #nv-nsight-cu-cli --mode=launch julia 
# #  using  Test, Revise ,CUDA
 
# #  includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\aPrfofiling\\profilingProcessMaskData.jl")
 
# #  CUDA.@profile wrapForProfile()




# using  Test, Revise,CUDA 
# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")


# using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..BasicStructs
# using Shuffle,..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
# using ..MainOverlap, ..RandIndex , ..ProbabilisticMetrics , ..VolumeMetric ,..InformationTheorhetic
# using ..CUDAAtomicUtils, ..TpfpfnKernel, ..InterClassCorrKernel,..MeansMahalinobis
# using HDF5

# pathToHd5= "C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5"

# const g = h5open(pathToHd5, "r+")
# not_translated=read(g["not_translated"])
# translated=read(g["translated"])

# arrGold = CuArray(not_translated)
# arrAlgo = CuArray(translated)

# sizz= size(not_translated)
# conf = ConfigurtationStruct(dice=true)
# numberToLooFor = UInt8(1)
# args,threads,blocks,metricsTuplGlobal= TpfpfnKernel.prepareForconfusionTableMetrics(arrGold    , arrAlgo    ,numberToLooFor  ,conf)

# argsB = TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal) 
# dice = metricsTuplGlobal[4][1]



# function getBlockTpFpFnProf(goldGPU#goldBoolGPU
#     , segmGPU#segmBoolGPU
#     #,sliceMetricsTupl
#     ,tp,tn,fp,fn#tp,tn,fp,fn
#     ,arrDims,totalNumbOfVoxels,iterLoop,pixPerSlice
#     ,numberToLooFor#numberToLooFor
#     #,metricsTuplGlobal
#     ,conf)

#     shmemSum = @cuStaticSharedMem(UInt32, (33,3))  
#     shmemblockData = @cuStaticSharedMem(UInt32,(32, 32 ,3))

#     # locArr= zeros(MVector{3,UInt16})
#     @iterateLinearlyMultipleBlocks(iterLoop,pixPerSlice,totalNumbOfVoxels,
#     #inner expression
#     begin
#         # CUDA.@cuprint "i $(i)  val $(goldGPU[i])"
#         #updating variables needed to calculate means
#         boolGold = goldGPU[i]==numberToLooFor  
#         boolSegm = segmGPU[i]==numberToLooFor 
#         @inbounds  shmemblockData[threadIdxX(),threadIdxY(), (boolGold & boolSegm + boolSegm +1)]+=(boolGold | boolSegm)
#         #    locArr[1]=boolGold    
#         #   @inbounds locArr[ (boolGold & boolSegm + boolSegm +1) ]+=(boolGold | boolSegm)
#     end) 

#     # tell what variables are to be reduced and by what operation
#     @redWitAct(offsetIter,shmemSum, shmemblockData[threadIdxX(),threadIdxY(),1],+,    shmemblockData[threadIdxX(),threadIdxY(),2],+,     shmemblockData[threadIdxX(),threadIdxY(),3],+   )
#     sync_threads()
#     @addAtomic(shmemSum, fn, fp,tp)

# return  
# end


# function getTpfpfnDataProf(goldGPU
#     , segmGPU
#     ,args,threads,blocks,metricsTuplGlobal) 

#     for i in  1:4   
#         CUDA.fill!(args[i],0)
#     end   

#     for i in 1:length(metricsTuplGlobal)    
#         metricsTuplGlobal[i]=0
#         #CUDA.fill!(args[11][i],0)
#     end   


#     #get tp,fp,fna and slicewise results if required
#     @cuda threads=threads blocks=blocks getBlockTpFpFnProf(goldGPU, segmGPU,args...) #args[8][3]  is number of slices ...
#     #TpfpfnKernel.getMetricsCPU(args[1][1],args[3][1], args[4][1],(args[6]-(args[1][1] +args[3][1]+ args[4][1] )) ,metricsTuplGlobal,args[10],1 )
#     return args
# end#getTpfpfnData


# function wrapForProfile()
  
#     getTpfpfnDataProf(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal)
#     return metricsTuplGlobal[4][1]
# end    


# # testArrIn = CUDA.ones(Bool,32,32,32);
# # testArrOut = CUDA.zeros(Bool,34,34,34);

# # function testKernprocessMaskDataB(testArrInn,testArrOut)
# #     resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
# #     clearMainShmem(resShmem)
# #     blockBeginingX=UInt8(0)
# #     blockBeginingY=UInt8(0)
# #     blockBeginingZ=UInt8(0)
# #     isMaskFull= false
# #     isMaskEmpty= false
# #     #here we will store in registers data uploaded from mask for later verification wheather we should send it or not
# #     locArr= UInt32(0)
    
# #     @unroll for zIter in UInt8(1):UInt8(32)# most outer loop is responsible for z dimension
# #         processMaskData( testArrInn[(blockBeginingX+threadIdxX()),(blockBeginingY +threadIdxY()),(blockBeginingZ+zIter)], zIter, resShmem,locArr)
# #     end
# #     sync_threads()
# #         #fillGlobalFromShmem(testArrOut,resShmem)

# #     return
# # end    


# # @cuda threads=(32,32) blocks=1 testKernprocessMaskDataB(testArrIn,testArrOut) 










# # numb = UInt32(0)
# # #settingcorrectly
# # numb |= true << UInt8(2)
# # numb


# # numb = UInt32(0)
# # #settingcorrectly
# # numb |= 1 << 1
# # numb |= 1 << 2
# # numb |= 1 << 0
# # numb |= 1 << 0
# # numb |= 1 << 5
# # numb


# #reading...
# # numb>>1 & UInt32(1) 
# # numb>>2 & UInt32(1) 
# # numb>>3 & UInt32(1) 
# # numb>>4 & UInt32(1) 
# # numb>>5 & UInt32(1) 


# # #processMaskDataB( testArrInn[threadIdxY(),1,1], zIter, resShmem ) # coalesced