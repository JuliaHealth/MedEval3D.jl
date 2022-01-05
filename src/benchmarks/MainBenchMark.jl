
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
using Statistics
using BenchmarkTools

Conda.pip_interop(true)
Conda.pip("install", "gspread")
gspread= pyimport("gspread")
gspread= pyimport("gspread")
# using PyCall
# @pyimport pip
# pip.main(["install","google-api-python-client","google-auth-httplib2","google-auth-oauthlib"])

acc= gspread.service_account(filename="C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\credentials.json.txt")
sh= acc.open_by_url("https://docs.google.com/spreadsheets/d/1YBKQ70ghpEN-OQdRLoWAHl5EetDzBoCa6ViNQ1D7zYg/edit#gid=0")
worksheet = sh.get_worksheet(0)

# print(sh.sheet1.get("A1"))


BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds =30
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

pathToHd5= "C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5"

const g = h5open(pathToHd5, "r+")
not_translated=read(g["not_translated"])
translated=read(g["translated"])
onlyLungs=read(g["onlyLungs"])
onlyBladder=read(g["onlyBladder"])
onlyBladder
sizz= size(not_translated)

sum(onlyBladder)

########## first non distance metrics
arrGold = CuArray((not_translated))
arrAlgo = CuArray((translated))
conf = ConfigurtationStruct(dice=true)
cellStr= "B2"
"""
given particular configuration struct it prepares and executes the benchmark also save result to given cell in worksheet
"""
# function getBenchmarkInMilisForConfigurationStruct(conf,cellStr,arrGold,arrAlgo,worksheet)
    numberToLooFor = UInt8(1)
    argss,threads,blocks,metricsTuplGlobal= TpfpfnKernel.prepareForconfusionTableMetrics(arrGold    , arrAlgo    ,numberToLooFor  ,conf)
    
    argsB = TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argss,threads,blocks,metricsTuplGlobal) 
    dice = metricsTuplGlobal[4][1]
    # dice # should be 0.757
    bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
    res= Statistics.median(bench).time/1000000
    worksheet.update(cellStr, res)
    bench
# end

# getBenchmarkInMilisForConfigurationStruct(ConfigurtationStruct(dice=true),"B2",arrGold,arrAlgo,worksheet)

conf = ConfigurtationStruct(jaccard=true)
cellStr= "B3"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)


conf = ConfigurtationStruct(gce=true)
cellStr= "B4"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(vol=true)
cellStr= "B5"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(randInd=true)
cellStr= "B6"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(ic=true)
cellStr= "B7"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(kc=true)
cellStr= "B8"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(mi=true)
cellStr= "B9"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

conf = ConfigurtationStruct(vi=true)
cellStr= "B10"
bench = @benchmark CUDA.@sync TpfpfnKernel.getTpfpfnData!(arrGold ,arrAlgo   ,argsB,threads,blocks,metricsTuplGlobal)# setup = (arrGold ,arrAlgo   ,args,threads,blocks,metricsTuplGlobal = clearFunction()   )
res= Statistics.median(bench).time/1000000
worksheet.update(cellStr, res)

bench
 #in miliseconds



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