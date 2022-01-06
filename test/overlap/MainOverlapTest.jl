"""
a lot of evaluation metrics can be found in 
https://pymia.readthedocs.io/en/latest/pymia.evaluation.metric.html?highlight=dice
"""


module MainOverlapTest



using Distances, Test, Revise 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\overLap\\MainOverlap.jl")
using ..BasicStructs
using ..BasicPreds, ..CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA, ..MainOverlap

goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,flattG, flattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn = getSmallTestBools();

#tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue
@testset "jaccard test" begin
    dicee = dice(tpTotalTrue,fpTotalTrue, fnTotalTrue)
    x =  (dicee)/(2-dicee)
    jaccardOut = 1- evaluate(Jaccard(), flattG, flattSeg)
    @test jaccardOut ≈ jaccard(tpTotalTrue,fpTotalTrue, fnTotalTrue )
    @test jaccardOut ≈ x
   
end;

@testset "global consistency error test" begin
    local_refinement_error( flattG, flattSeg)
end;



end# MainOverlapTest

