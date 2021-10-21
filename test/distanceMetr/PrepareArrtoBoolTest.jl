using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool
using CUDA



@testset "getIndexOfQueue" begin 
    """
1)   Left FP  
2)   Left FN  
3)   Right FP  
4)   Right FN  
5)   Posterior FP  
6)   Posterior FN  
7)   Anterior FP  
8)   Anterior FN  
9)   Top FP  
10)   Top FN  
11)   Bottom FP  
12)   Bottom FN  
13)   Total block Fp  
14)   Total block Fn  

"""
datBdim= (5,5,5)
@test getIndexOfQueue(1,1,1, (5,5,5),false)==1
@test getIndexOfQueue(1,1,1, (5,5,5),true)==2

@test getIndexOfQueue(5,1,1, (5,5,5),false)==3
@test getIndexOfQueue(5,1,1, (5,5,5),true)==4

@test getIndexOfQueue(2,1,1, (5,5,5),false)==5
@test getIndexOfQueue(2,1,1, (5,5,5),true)==6

@test getIndexOfQueue(2,5,1, (5,5,5),false)==7
@test getIndexOfQueue(2,5,1, (5,5,5),true)==8

@test getIndexOfQueue(2,2,1, (5,5,5),false)==9
@test getIndexOfQueue(2,2,1, (5,5,5),true)==10

@test getIndexOfQueue(2,2,5, (5,5,5),false)==11
@test getIndexOfQueue(2,2,5, (5,5,5),true)==12

@test getIndexOfQueue(5,5,5, (5,5,5),false)==3
@test getIndexOfQueue(5,5,5, (5,5,5),true)==4

@test getIndexOfQueue(2,2,2, (5,5,5),false)==13
@test getIndexOfQueue(2,2,2, (5,5,5),true)==14

end


##### iter3dOuter
singleVal = CUDA.zeros(14)

threads=(32,20)


threads=(32,5)
blocks =2
mainArrDims= (7,15,13)
datBdim = (2,3,2)

# blocks =7
# metaDataDims= (8,9,7)
# datBdim = (33,40,37)
mainArrDims= (7,15,13)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
datBdim

function iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    PrepareArrtoBool.@iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin
    @ifXY 1 1  @atomic singleVal[]+=1
    @ifXY 1 1    CUDA.@cuprint "zMeta $(zMeta)  yMeta$(yMeta) xMeta$(xMeta)  idX $(blockIdxX()) \n"   

end)
    
    return
end
@cuda threads=threads blocks=blocks iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]






using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool
using CUDA



##### iter data block
singleVal = CUDA.zeros(Int64,14)
indices = CUDA.zeros(900,15)
threads=(32,5)
blocks =1
# mainArrDims= (3,2,3)
# datBdim = (2,1,2)
# blocks =7
#mainArrDims= (317,268,239)
#datBdim = (8,8,8)

mainArrDims= (178,345,327)
datBdim = (42,19,17)

# mainArrDims= (178,345,327)
# datBdim = (42,4,2)
# mainArrDims= (3,3,3)
# datBdim = (2,2,2)
# mainArrDims= (7,15,13)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]

function iterDataBlocksKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
    PrepareArrtoBool.@iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        PrepareArrtoBool.@iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
         @atomic singleVal[1]+=1
    end)end)
    
    return
end
@cuda threads=threads blocks=blocks iterDataBlocksKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,indices)
@test singleVal[1]==mainArrDims[1]*mainArrDims[2]*mainArrDims[3]
Int64(singleVal[1])


##### uploadLocalfpFNCounters



##### iter data block
singleVal = CUDA.zeros(Int64,14)
indices = CUDA.zeros(900,15)
threads=(32,5)
blocks =1
# mainArrDims= (3,2,3)
# datBdim = (2,1,2)
# blocks =7
#mainArrDims= (317,268,239)
#datBdim = (8,8,8)

mainArrDims= (178,345,327)
datBdim = (42,19,17)

# mainArrDims= (178,345,327)
# datBdim = (42,4,2)
# mainArrDims= (3,3,3)
# datBdim = (2,2,2)
# mainArrDims= (7,15,13)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]

function iterDataBlocksKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
    PrepareArrtoBool.@iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        PrepareArrtoBool.@iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
            if(blockIdx.x=4)#4 as arbitrary number
                     PrepareArrtoBool.@uploadDataToMetaData() 
                
        end #if
    end) end)    
    
    return
end
@cuda threads=threads blocks=blocks iterDataBlocksKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,indices)
@test singleVal[1]==mainArrDims[1]*mainArrDims[2]*mainArrDims[3]
Int64(singleVal[1])

















# #@testset "getBoolCubeKernel" begin 
#     sliceW = 512
#     sliceH = 512
#     sliceNumb = 91
#     arrGold = zeros(Int64,sliceW,sliceH,sliceNumb);
#     arrSegm = zeros(Int64,sliceW,sliceH,sliceNumb);

# arrDims = size(arrSegm)
#     # so we should get all mins as 2 and all maxes as 5
 
#     arrGold[9,2,5]= 2
#     arrGold[3,6,4]= 2
#     arrGold[3,3,4]= 2
    
#     arrGold[4,4,4]= 2
#     arrSegm[4,4,4]= 2

#     arrSegm[3,18,4]= 2
#     arrSegm[5,2,11]= 2

#      fn=CuArray([UInt32(0)]);     fp=CuArray([UInt32(0)]);     minxRes=CuArray([UInt32(100000)]);     maxxRes=CuArray([UInt32(0)]);
#      minyRes=CuArray([UInt32(10000)]);     maxyRes=CuArray([UInt32(0)]);     minZres=CuArray([UInt32(100000)]);     maxZres=CuArray([UInt32(0)]);

#      reducedGoldA=CuArray(falses(arrDims));
#      reducedSegmA=CuArray(falses(arrDims));
#      reducedGoldB=CuArray(falses(arrDims));
#      reducedSegmB=CuArray(falses(arrDims));


#     Main.PrepareArrtoBool.getBoolCube!(
#      CuArray(arrGold)
#     ,CuArray(arrSegm)
#     , sliceNumb
#     ,fn    ,fp    ,minxRes
#     ,maxxRes    ,minyRes
#     ,maxyRes    ,minZres
#     ,maxZres    ,2
#     , CuArray([UInt32(0)])
#     , reducedGoldA
#     , reducedSegmA
#     , reducedGoldB
#     ,reducedSegmB
#       );

#       Int64( maxxRes[])

#       @test Int64(fn[])==3
#       @test Int64(fp[])==2
#       @test Int64(minxRes[])==3
#       @test Int64( maxxRes[])==9
#       @test Int64(minyRes[])==2
#       @test  Int64(maxyRes[])==18
#       @test  Int64( minZres[])==4
#       @test  Int64( maxZres[])==11

#      vv =  collect(view(reducedGoldA,Int64(minxRes[]) : Int64( maxxRes[]) , Int64(minyRes[]) : Int64(maxyRes[]),Int64( minZres[]) :Int64( maxZres[]) ))
# typeof(vv)
# size(vv)


#       @test  reducedGoldA[9,2,5]==true
#       @test   reducedGoldA[3,6,4]==true
#       @test   reducedGoldA[3,3,4]==true
#       @test   reducedGoldA[4,4,4]==true
#       @test   reducedGoldA[5,5,5]==false
#       @test   reducedGoldA[5,2,5]==false
#       @test   reducedGoldA[5,2,90]==false

#       @test  reducedGoldB[9,2,5]==true
#       @test   reducedGoldB[3,6,4]==true
#       @test   reducedGoldB[3,3,4]==true
#       @test   reducedGoldB[4,4,4]==true
#       @test   reducedGoldB[5,5,5]==false
#       @test   reducedGoldB[5,2,5]==false
#       @test   reducedGoldB[5,2,90]==false

#       @test  reducedSegmA[3,18,4]==true
#       @test   reducedSegmA[5,2,11]==true
#       @test   reducedSegmA[4,4,4]==true
#       @test   reducedSegmA[5,5,5]==false
#       @test   reducedSegmA[5,2,5]==false
#       @test   reducedSegmA[5,2,90]==false

#       @test  reducedSegmB[3,18,4]==true
#       @test   reducedSegmB[5,2,11]==true
#       @test   reducedSegmB[4,4,4]==true
#       @test   reducedSegmB[5,5,5]==false
#       @test   reducedSegmB[5,2,5]==false
#       @test   reducedSegmB[5,2,90]==false


# #end#testset

