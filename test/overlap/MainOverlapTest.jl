
module MainOverlapTest

using Distances, Test, Revise 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\overLap\\MainOverlap.jl")
using Main.BasicStructs
using Main.BasicPreds, Main.GPUutils,Cthulhu,BenchmarkTools , CUDA, Main.MainOverlap

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
https://github.com/hushell/drnn/blob/master/RNN/code/tcut/compare_segmentations.m

end;
end# MainOverlapTest

