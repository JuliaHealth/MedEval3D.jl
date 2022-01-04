
using HDF5
using CUDA ,BenchmarkTools
using Revise, Parameters, Logging, Test
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.BasicStructs
using Shuffle,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates
using Main.MainOverlap, Main.RandIndex , Main.ProbabilisticMetrics , Main.VolumeMetric ,Main.InformationTheorhetic
using Main.CUDAAtomicUtils, Main.TpfpfnKernel, Main.InterClassCorrKernel,Main.MeansMahalinobis
using Conda
using PyCall
using Pkg
Conda.pip_interop(true)
Conda.pip("install", "gspread")
gspread= pyimport("gspread")
gspread= pyimport("gspread")
json=pyimport("json")







BenchmarkTools.DEFAULT_PARAMETERS.samples = 500
BenchmarkTools.DEFAULT_PARAMETERS.seconds =600
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

pathToHd5= "C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5"

const g = h5open(pathToHd5, "r+")
not_translated=read(g["not_translated"])
translated=read(g["translated"])
onlyLungs=read(g["onlyLungs"])
onlyBladder=read(g["onlyBladder"])
onlyBladder

sum(onlyBladder)

########## first non distance metrics
arrGold = CuArray(not_translated)
arrAlgo = CuArray(translated)

sizz= size(not_translated)
conf = ConfigurtationStruct(dice=true)
numberToLooFor = UInt8(1)
args,threads,blocks,metricsTuplGlobal= TpfpfnKernel.prepareForconfusionTableMetrics(arrGold    , arrAlgo    ,numberToLooFor  ,conf)

argsB = TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal) 
dice = metricsTuplGlobal[4][1]
dice # should be 0.757
@benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )



# includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\aPrfofiling\\profilingProcessMaskData.jl")
 
# CUDA.@profile wrapForProfile()

# function clearFunction()
#     for i in  1:4   
#         CUDA.fill!(args[i],0)
#     end   
    
#     for i in 1:length(metricsTuplGlobal)    
#         metricsTuplGlobal[i]=0
#         #CUDA.fill!(args[11][i],0)
#     end   
#     sleep(2)
#     return (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal)
# end


# function toBench(data)
#     TpfpfnKernel.getTpfpfnData!(data...)
# end

# @benchmark toBench(data) setup=(data=clearFunction() )


@benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
# @btime CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal) 
#all 57 ms



# f = h5py.File("mytestfile.hdf5", "w")
# f.create_dataset("not_translated", data=  not_translated.cpu().detach().numpy())
# f.create_dataset("translated", data=  translated.cpu().detach().numpy())
# f.create_dataset("onlyLungs", data=  onlyLungs.cpu().detach().numpy())
# f.create_dataset("onlyBladder", data=  onlyBladder.cpu().detach().numpy())

# ```@doc
# getting example study in a form of 3 dimensional array
# ```
# function getExample(typ::Type{Tt}) ::Array{Tt, 3} where Tt
#      read(g["trainingScans/liver-orig006.mhd"]["liver-orig006.mhd"])
# end
# function teEx()
#     z= 0
#     grid_handle = this_grid()

#     for j in 1:1000
#         sync_grid(grid_handle)
#         z+=1
#     end 

# end

# #no grid sync 189.548 μs

# @benchmark CUDA.@sync @cuda threads=(32,20) blocks=40  cooperative=true teEx()