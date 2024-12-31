

using Revise, Parameters, Logging
using CUDA
# includet("./src/kernelEvolutions.jl")
includet("../../src/utils/CUDAGpuUtils.jl")
includet("../../src/structs/BasicStructs.jl")

includet("../../src/kernels/TpfpfnKernel.jl")
includet("../../src/overLap/MainOverlap.jl")
using ..BasicPreds, ..CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA
using ..MainOverlap, ..TpfpfnKernel

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();


sizz= size(goldBoolGPU)


TpfpfnKernel.getTpfpfnData!(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn,sizz[1],sizz[1]*sizz[1],sizz[3],Float32(1))
tp[1]
tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue #tn[1] == tnTotalTrue && 




using CUDA



#@benchmark CUDA.@sync 
# we are adding false in the end to make indexing easier
blockss = Int64(round((length(FlattGoldGPU)/512)/16))


#@benchmark CUDA.@sync 
args = (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn, loopNumb, indexCorr)
 @cuda threads=512 blocks=blockNum-1 getBlockTpFpFn(args...) 
tp[1]
fp[1]

tp[1] ==tpTotalTrue && fp[1] ==fpTotalTrue && fn[1] ==fnTotalTrue #tn[1] == tnTotalTrue && 

tp[1]
fp[1]
fn[1]

sum(intermediateResTp)
sum(intermediateResFp)
sum(intermediateResFn)

sum(FlattGoldGPU)

tpNew = (9*9*9)*2 +18

  @device_code_warntype interactive=true @cuda getBlockTpFpFn(FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)

## using occupancy API to calculate  threads number, block number etc...
#args = (FlattGoldGPU,FlattSegGPU,tp,tn,fp,fn, intermediateResTp,intermediateResFp,intermediateResFn)
calcBlocks,valcThreads,maxBlocks = computeBlocksFromOccupancy(args,3 )