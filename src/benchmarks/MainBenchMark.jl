
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
using BenchmarkTools, Main.MainAbstractions

#!!!!!!!!!!!!! below important if we do not set google sheets we need to set it as false
isTobeSavedToGoogle= true
worksheet=0


Conda.pip_interop(true)
Conda.pip("install", "gspread")
gspread= pyimport("gspread")
gspread= pyimport("gspread")
# using PyCall
# @pyimport pip
# pip.main(["install","google-api-python-client","google-auth-httplib2","google-auth-oauthlib"])

if(isTobeSavedToGoogle)
    acc= gspread.service_account(filename="C:\\Users\\1\\PycharmProjects\\credentials.json.txt")
    sh= acc.open_by_url("https://docs.google.com/spreadsheets/d/1YBKQ70ghpEN-OQdRLoWAHl5EetDzBoCa6ViNQ1D7zYg/edit#gid=0")
    worksheet = sh.get_worksheet(0)
end
# print(sh.sheet1.get("A1"))


BenchmarkTools.DEFAULT_PARAMETERS.samples = 300
BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
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
conf = ConfigurtationStruct(false,trues(11)...)
numberToLooFor = UInt8(1)
preparedDict=MainAbstractions.prepareMetrics(conf)

typeof(:dice)

conf = ConfigurtationStruct(dice=true)
cellStr= "B2"
"""
given particular configuration struct it prepares and executes the benchmark also save result to given cell in worksheet
isTobeSavedToGoogle- if true and we will supply valid worksheet object benchmark time and calculated value of metric will be added to the becnchmark table in google sheets
"""
# function getBenchmarkInMilisForConfigurationStruct(conf,cellStr,arrGold,arrAlgo,worksheet)
macro  benchmarkking(confStr::Symbol, cellString::String,field::Symbol, fieldCellString::String, isTobeSavedToGoogle::Symbol)
    return esc(quote
    # dice # should be 0.757
    resultSingle = calcMetricGlobal(preparedDict,$confStr,arrGold,arrAlgo,numberToLooFor)
    println(resultSingle)
    bench = @benchmark(CUDA.@sync calcMetricGlobal(preparedDict,($confStr),arrGold,arrAlgo,numberToLooFor))
    res= Statistics.median(bench).time/1000000
    if(isTobeSavedToGoogle)
        worksheet.update($cellString, res)
        worksheet.update($fieldCellString, getfield(resultSingle,field ))
    end
    end)    
end  

confStr= ConfigurtationStruct(dice=true)
field=Symbol("dice")
@benchmarkking(confStr, "B2",field , "G2",isTobeSavedToGoogle)

    # # dice # should be 0.757
    # bench = @benchmark CUDA.@sync calcMetricGlobal(preparedDict,conf,arrGold,arrAlgo,numberToLooFor)
    # res= Statistics.median(bench).time/1000000
    # worksheet.update(cellStr, res)
    # bench

# end

# getBenchmarkInMilisForConfigurationStruct(ConfigurtationStruct(dice=true),"B2",arrGold,arrAlgo,worksheet)


confStr= ConfigurtationStruct(jaccard=true)
field=Symbol("jaccard")
@benchmarkking(confStr, "B3",field , "G3",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(gce=true)
field=Symbol("gce")
@benchmarkking(confStr, "B4",field , "G4",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(vol=true)
field=Symbol("vol")
@benchmarkking(confStr, "B5",field , "G5",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(randInd=true)
field=Symbol("randInd")
@benchmarkking(confStr, "B6",field , "G6",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(kc=true)
field=Symbol("kc")
@benchmarkking(confStr, "B8",field , "G8",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(mi=true)
field=Symbol("mi")
@benchmarkking(confStr, "B9",field , "G9",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(vi=true)
field=Symbol("vi")
@benchmarkking(confStr, "B10",field , "G10",isTobeSavedToGoogle)

confStr= ConfigurtationStruct(ic=true)
field=Symbol("ic")
@benchmarkking(confStr, "B7",field , "G7",isTobeSavedToGoogle)
#all confusionmatrix metrics
confStr= ConfigurtationStruct(dice=true,jaccard=true, gce=true,vol=true,randInd=true,kc=true,mi=true,vi=true)
field=Symbol("dice")
@benchmarkking(confStr, "B11",field , "G11",isTobeSavedToGoogle)



using Main.MeansMahalinobis
arrGold = CuArray((onlyLungs))
arrAlgo = CuArray((onlyBladder))

confStr= ConfigurtationStruct(md=true)
field=Symbol("md")
@benchmarkking(confStr, "B12",field , "G12",isTobeSavedToGoogle)


