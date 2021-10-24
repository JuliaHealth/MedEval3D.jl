using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool,Main.HFUtils
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


using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool,Main.HFUtils
using CUDA


singleVal = CUDA.zeros(14)

# threads=(32,20)


threads=(32,5)
blocks =8

mainArrDims= (67,177,90)
datBdim = (43,21,17)

# blocks =7
# metaDataDims= (8,9,7)
# datBdim = (33,40,37)
#mainArrDims= (7,15,13)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
datBdim

function iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ )
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin
    @ifXY 1 1  @atomic singleVal[]+=1
    @ifXY 1 1    CUDA.@cuprint "zMeta $(zMeta)  yMeta$(yMeta) xMeta$(xMeta)  idX $(blockIdxX()) \n"   

end)
    
    return
end
@cuda threads=threads blocks=blocks iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]



################ iter data block


using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool,Main.HFUtils
using CUDA



##### iter data block
singleVal = CUDA.zeros(Int64,14)
indices = CUDA.zeros(900,15)
threads=(32,5)
blocks =17
# mainArrDims= (3,2,3)
# datBdim = (2,1,2)
# blocks =7
#mainArrDims= (317,268,239)
#datBdim = (8,8,8)

# mainArrDims= (178,345,327)
# datBdim = (42,19,17)


blocks =8

mainArrDims= (67,177,90)
datBdim = (43,21,17)

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
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
         @atomic singleVal[1]+=1
    end)end)
    
    return
end
@cuda threads=threads blocks=blocks iterDataBlocksKernel(mainArrDims,singleVal,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,indices)
@test singleVal[1]==mainArrDims[1]*mainArrDims[2]*mainArrDims[3]
Int64(singleVal[1])


##### @uploadLocalfpFNCounters
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool,Main.HFUtils
using CUDA

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =3

mainArrDims= (67,78,90)
datBdim = (17,7,12)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]



function uploadLocalfpFNCountersKernel(mainArrDims,localQuesValues,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
    # localQuesValues= @cuStaticSharedMem(Float32, 14)   

    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 
        #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
                    boolGold=true
                    boolSegm=false
                    #coord=PrepareArrtoBool.getIndexOfQueue((xdim * blockDimX())+threadIdxX() ,(ydim * blockDimY())+threadIdxY(),(zdim+1),datBdim,false)
                    PrepareArrtoBool.@uploadLocalfpFNCounters()
                    # CUDA.@cuprint " spot $(coord)  x $((xdim * blockDimX())+threadIdxX()  )) y $((ydim * blockDimY())+threadIdxY()  )) z $((zdim+1)) datBdim $(datBdim[1]),$(datBdim[2]),$(datBdim[3]) \n"    
                    #CUDA.atomic_add!(pointer(localQuesValues, coord),Float32(1))
                    

                #@inbounds @atomic localQuesValues[coord]+=Float32(1)
                #CUDAAtomicUtils.atomicallyAddToSpot(Float32,localQuesValues,getIndexOfQueue(x,y,z,datBdim,boolSegm),1)
                #PrepareArrtoBool.@uploadLocalfpFNCounters() 
                
                end)end)
    
    return
end



@cuda threads=threads blocks=blocks uploadLocalfpFNCountersKernel(mainArrDims,localQuesValues,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
@test sum(localQuesValues)>0

@test localQuesValues[1]==datBdim[2]*datBdim[3]
@test localQuesValues[3]==datBdim[2]*datBdim[3]

@test localQuesValues[5]==datBdim[1]*datBdim[3]-2*datBdim[3]
@test localQuesValues[7]==datBdim[1]*datBdim[3]-2*datBdim[3]

@test localQuesValues[9]==(datBdim[1]-2)*(datBdim[2]-2) 
@test localQuesValues[11]==(datBdim[1]-2)*(datBdim[2]-2) 

@test localQuesValues[13]==datBdim[1]*datBdim[2]*datBdim[3] - sum(localQuesValues[1:12])


########### uploadDataToMetaData


using Revise, Parameters, Logging, Test
using CUDA,Main.HFUtils
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =8

mainArrDims= (634,521,632)
datBdim = (43,21,17)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);

function uploadDataToMetaDataKernel(mainArrDims,localQuesValuesB,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
    localQuesValues= @cuStaticSharedMem(UInt32, 14)   
    localBool=false
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
                    boolGold=true
                    boolSegm=true
                    localBool=true
                    if(xMeta==2 && yMeta==2 && zMeta==4)

                        PrepareArrtoBool.@uploadLocalfpFNCounters()
                    end    
                
                end)
                sync_threads()

                isAnyPositive= true
                    PrepareArrtoBool.@uploadDataToMetaData()
                    sync_threads()

                     sync_threads()
                    #threadfence()
                    @ifY 1 if(threadIdxX()<15)
                         localQuesValues[threadIdxX()]=0
                    end
            end)
    
    return
end



@cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(mainArrDims,localQuesValues,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)


@test metaData[3,3,5,getBeginingOfFpFNcounts()+1]==datBdim[2]*datBdim[3]
@test metaData[3,3,5,getBeginingOfFpFNcounts()+3]==datBdim[2]*datBdim[3]

@test metaData[3,3,5,getBeginingOfFpFNcounts()+5]==datBdim[1]*datBdim[3]-2*datBdim[3]
@test metaData[3,3,5,getBeginingOfFpFNcounts()+7]==datBdim[1]*datBdim[3]-2*datBdim[3]

@test metaData[3,3,5,getBeginingOfFpFNcounts()+9]==(datBdim[1]-2)*(datBdim[2]-2) 
@test metaData[3,3,5,getBeginingOfFpFNcounts()+11]==(datBdim[1]-2)*(datBdim[2]-2) 

@test metaData[3,3,5,getBeginingOfFpFNcounts()+13]==datBdim[1]*datBdim[2]*datBdim[3] - sum(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+12])

@test metaData[3,3,5,getBeginingOfFpFNcounts()+15]==datBdim[1]*datBdim[2]*datBdim[3] 

sum(Array(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+13]))


metaData[3,3,5,getBeginingOfFpFNcounts()+1]+ metaData[3,3,5,getBeginingOfFpFNcounts()+3]+ metaData[3,3,5,getBeginingOfFpFNcounts()+5]+ metaData[3,3,5,getBeginingOfFpFNcounts()+7]+ metaData[3,3,5,getBeginingOfFpFNcounts()+9]+ metaData[3,3,5,getBeginingOfFpFNcounts()+11]+ metaData[3,3,5,getBeginingOfFpFNcounts()+13]




########### uploadDataToMetaData 2


using Revise, Parameters, Logging, Test
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA,Main.HFUtils

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =8

mainArrDims= (634,521,632)
datBdim = (43,21,17)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);
metaDataDims
function uploadDataToMetaDataKernel(mainArrDims,localQuesValuesB,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
    localQuesValues= @cuStaticSharedMem(UInt32, 14)   
    localBool=false
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
                    boolGold=false
                    boolSegm=false
                    localBool=true
                    if(xMeta==2 && yMeta==2 && zMeta==4)

                        PrepareArrtoBool.@uploadLocalfpFNCounters()
                    end    
                
                end)
                sync_threads()

                isAnyPositive= true
                    PrepareArrtoBool.@uploadDataToMetaData()
                    sync_threads()

                     sync_threads()
                    #threadfence()
                    @ifY 1 if(threadIdxX()<15)
                         localQuesValues[threadIdxX()]=0
                    end
            end)
    
    return
end



@cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(mainArrDims,localQuesValues,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)


@test metaData[3,3,5,getBeginingOfFpFNcounts()+2]==datBdim[2]*datBdim[3]
@test metaData[3,3,5,getBeginingOfFpFNcounts()+4]==datBdim[2]*datBdim[3]

@test metaData[3,3,5,getBeginingOfFpFNcounts()+6]==datBdim[1]*datBdim[3]-2*datBdim[3]
@test metaData[3,3,5,getBeginingOfFpFNcounts()+8]==datBdim[1]*datBdim[3]-2*datBdim[3]

@test metaData[3,3,5,getBeginingOfFpFNcounts()+10]==(datBdim[1]-2)*(datBdim[2]-2) 
@test metaData[3,3,5,getBeginingOfFpFNcounts()+12]==(datBdim[1]-2)*(datBdim[2]-2) 

@test metaData[3,3,5,getBeginingOfFpFNcounts()+14]==datBdim[1]*datBdim[2]*datBdim[3] - sum(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+12])

@test metaData[3,3,5,getBeginingOfFpFNcounts()+16]==datBdim[1]*datBdim[2]*datBdim[3] 

sum(Array(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+13]))


metaData[3,3,5,getBeginingOfFpFNcounts()+1]+ metaData[3,3,5,getBeginingOfFpFNcounts()+3]+ metaData[3,3,5,getBeginingOfFpFNcounts()+5]+ metaData[3,3,5,getBeginingOfFpFNcounts()+7]+ metaData[3,3,5,getBeginingOfFpFNcounts()+9]+ metaData[3,3,5,getBeginingOfFpFNcounts()+11]+ metaData[3,3,5,getBeginingOfFpFNcounts()+13]






#################   uploadMinMaxesToShmem



using Revise, Parameters, Logging, Test
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA,Main.HFUtils

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =17

mainArrDims= (634,521,632)
datBdim = (17,7,12)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);


minX= CuArray([Float32(1110.0) ])
maxX= CuArray([Float32(0.0)])
minY= CuArray([Float32(1110.0)])
maxY= CuArray([Float32(0.0)    ])
minZ= CuArray([Float32(1110.0)])
maxZ= CuArray([Float32(0.0) ])


function uploadDataToMetaDataKernel(minX, maxX, minY,maxY,minZ,maxZ,mainArrDims,localQuesValuesB,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
    localQuesValues= @cuStaticSharedMem(UInt32, 14)   
    isAnyPositive=true
    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
    begin 

        @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
        begin
                   
                    if(xMeta>=2 && yMeta>=1 && zMeta>=3 && xMeta<=4 && yMeta<=5 && zMeta<=6  )
                        sync_threads()
                        PrepareArrtoBool.@uploadDataToMetaData()

                        PrepareArrtoBool.@uploadMinMaxesToShmem()
                        sync_threads()

                    end    
                
                end)

            end)
    
    return
end



@cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(minX, maxX, minY,maxY,minZ,maxZ,mainArrDims,localQuesValues,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
@test minX[1]== 2+1
@test maxX[1]== 4+1
@test minY[1]== 1+1
@test maxY[1]== 5+1
@test minZ[1]== 3+1
@test maxZ[1]== 6+1


###########################   getBoolCubeKernel

using Revise, Parameters, Logging, Test
using CUDA,Main.HFUtils
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =8

mainArrDims= (634,521,632)
datBdim = (2,2,2)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);


    arrGold = zeros(Int32,mainArrDims);
    arrSegm = zeros(Int32,mainArrDims);

    # so we should get all mins as 2 and all maxes as 5
 
    arrGold[9,2,5]= 2
    arrGold[3,6,4]= 2
    arrGold[3,3,4]= 2
    
    arrGold[4,4,4]= 2
    arrSegm[4,4,4]= 2

    arrSegm[3,18,4]= 2
    arrSegm[5,2,11]= 2

     fn=CuArray([UInt32(0)]);     fp=CuArray([UInt32(0)]);   
     
     minX= CuArray([Float32(1110.0) ])
     maxX= CuArray([Float32(0.0)])
     minY= CuArray([Float32(1110.0)])
     maxY= CuArray([Float32(0.0)    ])
     minZ= CuArray([Float32(1110.0)])
     maxZ= CuArray([Float32(0.0) ])

     reducedGoldA=CuArray(falses(mainArrDims));
     reducedSegmA=CuArray(falses(mainArrDims));
     reducedGoldB=CuArray(falses(mainArrDims));
     reducedSegmB=CuArray(falses(mainArrDims));

 


      @cuda threads=threads blocks=blocks Main.PrepareArrtoBool.getBoolCubeKernel(  CuArray(arrGold)
      ,CuArray(arrSegm)
      ,Int32(2)
      , reducedGoldA
      , reducedSegmA
      , reducedGoldB
      ,reducedSegmB
      ,fn    
      ,fp 
      ,minX
      ,maxX    
      ,minY
      ,maxY    
      ,minZ
      ,maxZ            
      ,datBdim
      ,metaData
      ,metaDataDims
      ,mainArrDims
      ,loopXMeta
      ,loopYZMeta
      ,inBlockLoopX
      ,inBlockLoopY
      ,inBlockLoopZ
          );

          @test  Int64( maxZ[])==6

      @test Int64(fn[])==3
      @test Int64(fp[])==2
      
      @test Int64(minX[])==2
      @test Int64( maxX[])==5
      @test Int64(minY[])==1
      @test  Int64(maxY[])==9
      @test  Int64( minZ[])==2
      @test  Int64( maxZ[])==6





      @test  reducedGoldA[9,2,5]==true
      @test   reducedGoldA[3,6,4]==true
      @test   reducedGoldA[3,3,4]==true
      @test   reducedGoldA[4,4,4]==true
      @test   reducedGoldA[5,5,5]==false
      @test   reducedGoldA[5,2,5]==false
      @test   reducedGoldA[5,2,90]==false

      @test  reducedGoldB[9,2,5]==true
      @test   reducedGoldB[3,6,4]==true
      @test   reducedGoldB[3,3,4]==true
      @test   reducedGoldB[4,4,4]==true
      @test   reducedGoldB[5,5,5]==false
      @test   reducedGoldB[5,2,5]==false
      @test   reducedGoldB[5,2,90]==false

      @test  reducedSegmA[3,18,4]==true
      @test   reducedSegmA[5,2,11]==true
      @test   reducedSegmA[4,4,4]==true
      @test   reducedSegmA[5,5,5]==false
      @test   reducedSegmA[5,2,5]==false
      @test   reducedSegmA[5,2,90]==false

      @test  reducedSegmB[3,18,4]==true
      @test   reducedSegmB[5,2,11]==true
      @test   reducedSegmB[4,4,4]==true
      @test   reducedSegmB[5,5,5]==false
      @test   reducedSegmB[5,2,5]==false
      @test   reducedSegmB[5,2,90]==false



###########################   getBoolCubeKernel  2

using Revise, Parameters, Logging, Test
using CUDA,Main.HFUtils
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAAtomicUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\structs\\BasicStructs.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\IterationUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\ReductionUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\MemoryUtils.jl")
includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/MetaDataUtils.jl")
#includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\MeansMahalinobis.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\verB\\HFUtils.jl")

includet("C:/GitHub/GitHub/NuclearMedEval/src/distanceMetrics/Housdorff/verB/PrepareArrtoBool.jl")

using Main.PrepareArrtoBool, Main.CUDAGpuUtils, Main.PrepareArrtoBool, Main.CUDAAtomicUtils, Main.MetaDataUtils
using CUDA

localQuesValues = CUDA.zeros(Float32,14)
threads=(32,5)
blocks =7

mainArrDims= (634,521,632)
datBdim = (10,10,10)

metaDataDims= (cld(mainArrDims[1],datBdim[1] ),cld(mainArrDims[2],datBdim[2]),cld(mainArrDims[3],datBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(datBdim[1] ,threads[1]),fld(datBdim[2] ,threads[2]),datBdim[3]    )
#we are iterating here block by block sequentially
loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
yTimesZmeta= metaDataDims[2]*metaDataDims[3]
metaData = MetaDataUtils.allocateMetadata(mainArrDims,datBdim);


    arrGold = zeros(Int32,mainArrDims);
    arrSegm = zeros(Int32,mainArrDims);

    # so we should get all mins as 2 and all maxes as 5
    
    # 1)   Left FP  
    # 2)   Left FN  
    arrGold[11,13,13]= 2
    arrSegm[11,13,15]= 2


   # 3)   Right FP 
   # 4)   Right FN  
    arrGold[20,13,13]= 2
    arrSegm[20,13,15]= 2

    arrGold[20,14,13]= 2
    arrSegm[20,14,15]= 2


 # 5)   Posterior FP  

    # 6)   Posterior FN  
 
    arrGold[13,11,14]= 2
    arrSegm[14,11,13]= 2

    arrGold[15,11,13]= 2
    arrSegm[13,11,15]= 2

    arrGold[16,11,13]= 2
    arrSegm[13,11,16]= 2


    # 7)   Anterior FP  
    # 8)   Anterior FN  
    arrGold[14,20,13]= 2
    arrSegm[13,20,14]= 2

    arrGold[15,20,13]= 2
    arrSegm[14,20,16]= 2

    arrGold[17,20,13]= 2
    arrSegm[15,20,17]= 2
    
    arrGold[18,20,17]= 2
    arrSegm[16,20,18]= 2
   

    # 9)   Top FP  
    # 10)   Top FN  
    arrGold[14,15,11]= 2
    arrSegm[14,16,11]= 2

    arrGold[17,15,11]= 2
    arrSegm[17,16,11]= 2

    arrGold[15,15,11]= 2
    arrSegm[15,16,11]= 2
    
    arrGold[16,15,11]= 2
    arrSegm[16,16,11]= 2

    arrGold[18,15,11]= 2
    arrSegm[18,16,11]= 2


    # 11)   Bottom FP  
    # 12)   Bottom FN  

    arrGold[14,15,20]= 2
    arrSegm[14,16,20]= 2

    arrGold[17,15,20]= 2
    arrSegm[17,16,20]= 2

    arrGold[15,15,20]= 2
    arrSegm[15,16,20]= 2
    
    arrGold[16,15,20]= 2
    arrSegm[16,16,20]= 2

    arrGold[18,15,20]= 2
    arrSegm[18,16,20]= 2

    arrGold[19,15,20]= 2
    arrSegm[19,16,20]= 2


    # 13)   main block Fp  
    # 14)   main block Fn  
    
    arrGold[14,15,14]= 2
    arrSegm[14,16,16]= 2

    arrGold[17,15,17]= 2
    arrSegm[17,16,13]= 2

    arrGold[15,15,13]= 2
    arrSegm[15,16,13]= 2
    
    arrGold[16,15,13]= 2
    arrSegm[16,16,13]= 2

    arrGold[18,15,13]= 2
    arrSegm[18,16,13]= 2

    arrGold[19,15,13]= 2
    arrSegm[19,16,13]= 2

    arrGold[19,12,13]= 2
    arrSegm[19,12,13]= 2

    arrGold[12,12,17]= 2
    arrSegm[12,12,16]= 2

########### some noise around 


arrGold[22,15,13]= 2
arrSegm[18,22,13]= 2

arrGold[19,22,13]= 2
arrSegm[22,16,13]= 2

arrGold[19,22,13]= 2
arrSegm[19,12,22]= 2

arrGold[12,12,17]= 2
arrSegm[12,12,22]= 2    

arrGold[9,15,13]= 2
arrSegm[18,9,13]= 2

arrGold[19,22,9]= 2
arrSegm[9,16,13]= 2

arrGold[19,9,13]= 2
arrSegm[19,12,9]= 2

arrGold[9,12,17]= 2
arrSegm[12,9,22]= 2

   
   
    fn=CuArray([UInt32(0)]);     fp=CuArray([UInt32(0)]);   
     
     minX= CuArray([Float32(1110.0) ])
     maxX= CuArray([Float32(0.0)])
     minY= CuArray([Float32(1110.0)])
     maxY= CuArray([Float32(0.0)    ])
     minZ= CuArray([Float32(1110.0)])
     maxZ= CuArray([Float32(0.0) ])

     reducedGoldA=CuArray(falses(mainArrDims));
     reducedSegmA=CuArray(falses(mainArrDims));
     reducedGoldB=CuArray(falses(mainArrDims));
     reducedSegmB=CuArray(falses(mainArrDims));

 


      @cuda threads=threads blocks=blocks Main.PrepareArrtoBool.getBoolCubeKernel(  CuArray(arrGold)
      ,CuArray(arrSegm)
      ,Int32(2)
      , reducedGoldA
      , reducedSegmA
      , reducedGoldB
      ,reducedSegmB
      ,fn    
      ,fp 
      ,minX
      ,maxX    
      ,minY
      ,maxY    
      ,minZ
      ,maxZ            
      ,datBdim
      ,metaData
      ,metaDataDims
      ,mainArrDims
      ,loopXMeta
      ,loopYZMeta
      ,inBlockLoopX
      ,inBlockLoopY
      ,inBlockLoopZ
          );
          @test   metaData[2,2,2,getBeginingOfFpFNcounts()+1]==2
          @test   metaData[2,3,2,getBeginingOfFpFNcounts()+2]==3
          @test   metaData[2,2,4,getBeginingOfFpFNcounts()+3]==4


        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+1])==1
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+2])==1
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+3])==2
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+4])==2
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+5])==3
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+6])==3
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+7])==4
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+8])==4
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+9])==5
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+10])==5
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+11])==6
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+12])==6
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+13])==7
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+14])==7
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+15])== 1+2+3+4+5+6+7
        @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+16])== 1+2+3+4+5+6+7

        # @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+16])== 1+2+3+4+5+6+7
        # @test Int64(metaData[2,2,2,getBeginingOfFpFNcounts()+17])== 1+2+3+4+5+6+7

        #   2)   Left FN  
        #   3)   Right FP  
        #   4)   Right FN  
        #   5)   Posterior FP  
        #   6)   Posterior FN  
        #   7)   Anterior FP  
        #   8)   Anterior FN  
        #   9)   Top FP  
        #   10)   Top FN  
        #   11)   Bottom FP  
        #   12)   Bottom FN  
        #   13)   main block Fp  
        #   14)   main block Fn  
          
        #   15)   total block Fp  
        #   16)   total block Fn  