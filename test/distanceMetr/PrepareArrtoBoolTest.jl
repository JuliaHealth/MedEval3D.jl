
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..PrepareArrtoBool
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates

@testset "getBoolKernel" begin

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
    dataBdim= (5,5,5)
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


# metadat= zeros(Int64,100,50,40,15);
# metaDataDims= size(metadat)
# for i in 1:length(metadat)
#     metadat[i]=i
# end 
# linIdexMetaPrim= (100*50*3)+303
# linIdexMeta= (-1)+ (100*50*3)+303
# xMeta= mod(linIdexMeta,metaDataDims[1])
# zMeta= fld((linIdexMeta),metaDataDims[1]*metaDataDims[2])
# yMeta= fld((linIdexMeta-((zMeta*metaDataDims[1]*metaDataDims[2] ) + xMeta )),metaDataDims[1])
# qn= 7
# metadat[linIdexMetaPrim+(qn-1)*metaDataDims[1]*metaDataDims[2]*metaDataDims[3] ]
# metadat[xMeta+1,yMeta+1,zMeta+1,qn]


##### iter3dOuter


@testset "iter3dOuter" begin 



        using Revise, Parameters, Logging, Test
        using CUDA
        includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
        using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
        using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates
        
    singleVal = CUDA.zeros(14)

    # threads=(32,20)


    threads=(32,8)
    # blocks =2

    # mainArrDims= (7,7,7)
    # dataBdim = (2,2,2)

    blocks =136
    dataBdim = (33,32,32)
    mainArrDims= (512, 512, 500)

    metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))
    #we are iterating here block by block sequentially
    metaDataDims= (16, 16, 16,90)
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
    loopMeta= fld(metaDataLength,blocks )


    # metaDataDims (16, 16, 16, 90)  loopMeta 409 metaDataLength 4096
    function iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,dataBdim, metaDataLength, loopMeta )
        @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
        begin
        @ifXY 1 1 CUDA.@atomic singleVal[]+=1
     @ifXY 1 1    CUDA.@cuprint " linIdexMeta $(linIdexMeta)  zMeta $(zMeta)  yMeta$(yMeta) xMeta$(xMeta)  idX $(blockIdxX()) \n"   

    end)
        
        return
    end
    @cuda threads=threads blocks=blocks iter3dOuterKernel(mainArrDims,singleVal,metaDataDims,dataBdim, metaDataLength, loopMeta )
    @test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]


end #"iter3dOuter" 




################ iter data block
@testset "iter data block" begin 

    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
    using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates



    ##### iter data block
    singleVal = CUDA.zeros(Int64,14)
    threads=(32,5)

    blocks =17
    mainArrDims= (67,177,90)
    dataBdim = (43,21,17)
    indices = CUDA.zeros(Bool,mainArrDims)

    metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))

    inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    )
    #we are iterating here block by block sequentially
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
    loopMeta= fld(metaDataLength,blocks )

    function iterDataBlocksKernel(loopMeta,metaDataLength,mainArrDims,singleVal,metaDataDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
        @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
        begin 
            @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
            begin
           CUDA.@atomic singleVal[1]+=1
            indices[x,y,z]=true
            #CUDA.@cuprint "x $(x) y $(y) z $(z) zMeta $(zMeta)  yMeta $(yMeta) xMeta $(xMeta)  idX $(blockIdxX()) \n"   

        end)end)
        
        return
    end
    @cuda threads=threads blocks=blocks iterDataBlocksKernel(loopMeta,metaDataLength,mainArrDims,singleVal,metaDataDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
    @test singleVal[1]==mainArrDims[1]*mainArrDims[2]*mainArrDims[3]
    Int64(singleVal[1])
    7*7*7
    inddd = Array(indices);
    cartt = CartesianIndices(inddd);
    filtered = filter(cc-> !inddd[cc]  ,cartt)
    sum(indices)



    








end #iter data block



##### @uploadLocalfpFNCounters
@testset "uploadLocalfpFNCounters" begin 


    localQuesValues = CUDA.zeros(UInt32,14)
    threads=(32,5)
    blocks =1

    mainArrDims= (99,99,99)
    dataBdim = (17,7,12)

    metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))

    inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    )
    #we are iterating here block by block sequentially
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
    loopMeta= fld(metaDataLength,blocks )



    function uploadLocalfpFNCountersKernel(mainArrDims,localQuesValues,metaDataDims,loopMeta,metaDataLength,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
        # localQuesValues= @cuStaticSharedMem(Float32, 14)   

        # @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
        # begin 
            #@ifXY 1 1    CUDA.@cuprint "  xMeta $(xMeta) yMeta $(yMeta)  zMeta $(zMeta) \n"   
            xMeta= 2
            zMeta= 2
            yMeta= 2

            @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
            begin
                        boolGold=true
                        boolSegm=false
                        #coord=PrepareArrtoBool.getIndexOfQueue((xdim * blockDimX())+threadIdxX() ,(ydim * blockDimY())+threadIdxY(),(zdim+1),dataBdim,false)
                        PrepareArrtoBool.@uploadLocalfpFNCounters()
                        #CUDA.@cuprint " spot $(coord)  x $((xdim * blockDimX())+threadIdxX()  )) y $((ydim * blockDimY())+threadIdxY()  )) z $((zdim+1)) dataBdim $(dataBdim[1]),$(dataBdim[2]),$(dataBdim[3]) \n"    
                        #CUDA.atomic_add!(pointer(localQuesValues, coord),Float32(1))
                        

                    #@inboundsCUDA.@atomic localQuesValues[coord]+=Float32(1)
                    #CUDAAtomicUtils.atomicallyAddToSpot(Float32,localQuesValues,getIndexOfQueue(x,y,z,dataBdim,boolSegm),1)
                    #PrepareArrtoBool.@uploadLocalfpFNCounters() 
                    
                    end)#end)
        
        return
    end



    @cuda threads=threads blocks=blocks uploadLocalfpFNCountersKernel(mainArrDims,localQuesValues,metaDataDims,loopMeta,metaDataLength,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ)
    @test sum(localQuesValues)>0

    @test Int64(localQuesValues[1])==dataBdim[2]*dataBdim[3]
    @test Int64(localQuesValues[3])==dataBdim[2]*dataBdim[3]

    @test Int64(localQuesValues[5])==dataBdim[1]*dataBdim[3]-2*dataBdim[3]
    @test Int64(localQuesValues[7])==dataBdim[1]*dataBdim[3]-2*dataBdim[3]

    @test Int64(localQuesValues[9])==(dataBdim[1]-2)*(dataBdim[2]-2) 
    @test Int64(localQuesValues[11])==(dataBdim[1]-2)*(dataBdim[2]-2) 

    @test Int64(localQuesValues[13])==dataBdim[1]*dataBdim[2]*dataBdim[3] - sum(localQuesValues[1:12])

end #"uploadLocalfpFNCounters" 



using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates


# ########### uploadDataToMetaData
#  @testset "uploadDataToMetaData 1 " begin
   
#     localQuesValues = CUDA.zeros(Float32,14)
#     threads=(32,5)
#     blocks =8
#     mainArrDims= (634,521,632)
#     dataBdim = (43,21,17)

#     metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))
#     loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)

#     #we are iterating here block by block sequentially
#     metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
#     metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
#     loopMeta= fld(metaDataLength,blocks )
#     function uploadDataToMetaDataKernel(loopMeta,metaDataLength,mainArrDims,metaDataDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
#         localQuesValues= @cuStaticSharedMem(UInt32, 14)   
#         localBool=false
#         anyPositive=false

#         @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
#         begin 

#             @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
#             begin
#                         boolGold=true
#                         boolSegm=false
#                         localBool=true
#                         if(xMeta==2 && yMeta==2 && zMeta==4)

#                             PrepareArrtoBool.@uploadLocalfpFNCounters()
#                         end    
                    
#                     end)
#                     sync_threads()

#                     isAnyPositive= true
#                         PrepareArrtoBool.@uploadDataToMetaData()
#                         sync_threads()

#                         sync_threads()
#                         #threadfence()
#                         @ifY 1 if(threadIdxX()<15)
#                             localQuesValues[threadIdxX()]=0
#                         end
#                 end)
        
#         return
#     end



#     @cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(loopMeta,metaDataLength,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)

#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+1]==dataBdim[2]*dataBdim[3]
#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+3]==dataBdim[2]*dataBdim[3]

#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+5]==dataBdim[1]*dataBdim[3]-2*dataBdim[3]
#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+7]==dataBdim[1]*dataBdim[3]-2*dataBdim[3]

#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+9]==(dataBdim[1]-2)*(dataBdim[2]-2) 
#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+11]==(dataBdim[1]-2)*(dataBdim[2]-2) 

#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+13]==dataBdim[1]*dataBdim[2]*dataBdim[3] - sum(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+12])

#     @test metaData[3,3,5,getBeginingOfFpFNcounts()+15]==dataBdim[1]*dataBdim[2]*dataBdim[3] 

#     sum(Array(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+13]))


#     metaData[3,3,5,getBeginingOfFpFNcounts()+1]+ metaData[3,3,5,getBeginingOfFpFNcounts()+3]+ metaData[3,3,5,getBeginingOfFpFNcounts()+5]+ metaData[3,3,5,getBeginingOfFpFNcounts()+7]+ metaData[3,3,5,getBeginingOfFpFNcounts()+9]+ metaData[3,3,5,getBeginingOfFpFNcounts()+11]+ metaData[3,3,5,getBeginingOfFpFNcounts()+13]


# end #test set 
# ########### uploadDataToMetaData 2 
# @testset "uploadDataToMetaData2" begin


#     # localQuesValues = CUDA.zeros(Float32,14)
#     # threads=(32,5)
#     # blocks =8

#     # mainArrDims= (634,521,632)
#     # dataBdim = (43,21,17)

#     # metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))
#     # metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);

#     # inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    )
#     # #we are iterating here block by block sequentially
#     # loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
#     # yTimesZmeta= metaDataDims[2]*metaDataDims[3]
#     # metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
#     # loopMeta= fld(metaDataLength,blocks )
#     # function uploadDataToMetaDataKernel(loopMeta,metaDataLength,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)

#     #     localQuesValues= @cuStaticSharedMem(UInt32, 14)   
#     #     localBool=false
#     #     @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
#     #     begin 

#     #         @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,
#     #         begin
#     #                     boolGold=false
#     #                     boolSegm=true
#     #                     localBool=true
#     #                     if(xMeta==2 && yMeta==2 && zMeta==4)

#     #                         PrepareArrtoBool.@uploadLocalfpFNCounters()
#     #                     end    
                    
#     #                 end)
#     #                 sync_threads()

#     #                 isAnyPositive= true
#     #                     PrepareArrtoBool.@uploadDataToMetaData()
#     #                     sync_threads()

#     #                     sync_threads()
#     #                     #threadfence()
#     #                     @ifY 1 if(threadIdxX()<15)
#     #                         localQuesValues[threadIdxX()]=0
#     #                     end
#     #             end)
        
#     #     return
#     # end



#     # @cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(loopMeta,metaDataLength,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)


#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+2]==dataBdim[2]*dataBdim[3]
#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+4]==dataBdim[2]*dataBdim[3]

#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+6]==dataBdim[1]*dataBdim[3]-2*dataBdim[3]
#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+8]==dataBdim[1]*dataBdim[3]-2*dataBdim[3]

#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+10]==(dataBdim[1]-2)*(dataBdim[2]-2) 
#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+12]==(dataBdim[1]-2)*(dataBdim[2]-2) 

#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+14]==dataBdim[1]*dataBdim[2]*dataBdim[3] - sum(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+12])

#     # @test metaData[3,3,5,getBeginingOfFpFNcounts()+16]==dataBdim[1]*dataBdim[2]*dataBdim[3] 

#     # sum(Array(metaData[3,3,5,getBeginingOfFpFNcounts():getBeginingOfFpFNcounts()+13]))





# end#testset



#################   uploadMinMaxesToShmem

@testset "uploadMinMaxesToShmem" begin 
    localQuesValues = CUDA.zeros(Float32,14)
    threads=(32,5)
    blocks =17

    mainArrDims= (634,521,632)
    dataBdim = (17,7,12)

    metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))

    inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    )
    #we are iterating here block by block sequentially
    loopXMeta,loopYZMeta= (metaDataDims[1],fld(metaDataDims[2]*metaDataDims[3] ,blocks)  )
    yTimesZmeta= metaDataDims[2]*metaDataDims[3]
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);


    minX= CuArray([Float32(1110.0) ])
    maxX= CuArray([Float32(0.0)])
    minY= CuArray([Float32(1110.0)])
    maxY= CuArray([Float32(0.0)    ])
    minZ= CuArray([Float32(1110.0)])
    maxZ= CuArray([Float32(0.0) ])
    metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
    loopMeta= fld(metaDataLength,blocks )


    function uploadDataToMetaDataKernel(loopMeta,metaDataLength,minX, maxX, minY,maxY,minZ,maxZ,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
        localQuesValues= @cuStaticSharedMem(UInt32, 14)   
        isAnyPositive=true
        @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
        begin 

            @iterDataBlock(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
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



    @cuda threads=threads blocks=blocks uploadDataToMetaDataKernel(loopMeta,metaDataLength,minX, maxX, minY,maxY,minZ,maxZ,mainArrDims,metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaData)
    @test minX[1]== 2+1
    @test maxX[1]== 4+1
    @test minY[1]== 1+1
    @test maxY[1]== 5+1
    @test minZ[1]== 3+1
    @test maxZ[1]== 6+1
end #test set 

###########################   getBoolCubeKernel
@testset "getBoolCubeKernel" begin





    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..PrepareArrtoBool, ..BitWiseUtils
    using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates, ..Housdorff
    

    threads=(32,5)
    blocks =8
    mainArrDims= (512,512,155)
    dataBdim = (2,2,2)
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData)
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


        gold3d,segm3d = CuArray(arrGold), CuArray(arrSegm)
        numberToLooFor =Int32(2)
        robustnessPercent = 0.9

        boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain=    preparehousedorfKernel(gold3d,segm3d,robustnessPercent,numberToLooFor)
        mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=boolKernelArgs
   
   
        function locForKernel(goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,inBlockLoopXZIterWithPadding,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
     
        @getBoolCubeKernel()

        return
    end

        @cuda threads=threadsBoolKern blocks=blocksBoolKern locForKernel(gold3d,segm3d,boolKernelArgs...)
            @test  Int64( maxzRes[])==6

        @test Int64(fn[])==3
        @test Int64(fp[])==2
        
        @test Int64(minxRes[])==2
        @test Int64( maxxRes[])==5
        @test Int64(minyRes[])==1
        @test  Int64(maxyRes[])==9
        @test  Int64( minzRes[])==2
        @test  Int64( maxzRes[])==6




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
end#test set 


###########################   getBoolCubeKernel  2

using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils, ..PrepareArrtoBool
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates

 @testset "getBoolCubeKernel 2 " begin
    localQuesValues = CUDA.zeros(Float32,14)
    threads=(32,5)
    blocks =7

    mainArrDims= (512,512,400)
    dataBdim = (10,10,10)
    metaData = MetaDataUtils.allocateMetadata(mainArrDims,dataBdim);
    metaDataDims= size(metaData)
    arrGold = zeros(Int32,mainArrDims);
    arrSegm = zeros(Int32,mainArrDims);
    loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength = calculateLoopsIter(dataBdim,threads[1],threads[2],metaDataDims,blocks)
    minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp  =getSmallForBoolKernel();
    reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB=  getLargeForBoolKernel(mainArrDims,dataBdim);


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

    
    gold3d,segm3d = CuArray(arrGold), CuArray(arrSegm)
    numberToLooFor =Int32(2)

    function locForKernelB(mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,gold3d,segm3d,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)

    @getBoolCubeKernel()

    return
    end

    @cuda threads=threads blocks=blocks locForKernelB(mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,gold3d,segm3d,numberToLooFor,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)



            #   @test   metaData[2,2,2,getBeginingOfFpFNcounts()+1]==2
            #   @test   metaData[2,3,2,getBeginingOfFpFNcounts()+2]==3
            #   @test   metaData[2,2,4,getBeginingOfFpFNcounts()+3]==4


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
end# test set 

end#total test set






using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using ..CUDAGpuUtils ,..IterationUtils,..ReductionUtils , ..MemoryUtils,..CUDAAtomicUtils
using ..ResultListUtils, ..MetadataAnalyzePass,..MetaDataUtils,..WorkQueueUtils,..ProcessMainDataVerB,..HFUtils, ..ScanForDuplicates



##### iter data block
singleVal = CUDA.zeros(Int64,14)
threads=(32,5)

blocks =17
mainArrDims= (67,177,90)
dataBdim = (43,21,17)
indices = CUDA.zeros(Bool,mainArrDims)

metaDataDims= (cld(mainArrDims[1],dataBdim[1] ),cld(mainArrDims[2],dataBdim[2]),cld(mainArrDims[3],dataBdim[3]))

inBlockLoopX,inBlockLoopY,inBlockLoopZ= (fld(dataBdim[1] ,threads[1]),fld(dataBdim[2] ,threads[2]),dataBdim[3]    )
#we are iterating here block by block sequentially
metaDataLength= metaDataDims[1]*metaDataDims[2]*metaDataDims[3]
loopMeta= fld(metaDataLength,blocks )

function iterDataBlocksKernel(loopMeta,metaDataLength,mainArrDims,singleVal,metaDataDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
    @iter3dOuter(metaDataDims,loopMeta,metaDataLength,
    begin 
        @iterDataBlockZdeepest(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta,
        begin
       CUDA.@atomic singleVal[1]+=1
        indices[x,y,z]=true
        #CUDA.@cuprint "x $(x) y $(y) z $(z) zMeta $(zMeta)  yMeta $(yMeta) xMeta $(xMeta)  idX $(blockIdxX()) \n"   

    end,begin indices[1]=true
                 end )end)
    
    return
end
@cuda threads=threads blocks=blocks iterDataBlocksKernel(loopMeta,metaDataLength,mainArrDims,singleVal,metaDataDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ ,indices)
@test singleVal[1]==mainArrDims[1]*mainArrDims[2]*mainArrDims[3]
Int64(singleVal[1])
7*7*7
inddd = Array(indices);
cartt = CartesianIndices(inddd);
filtered = filter(cc-> !inddd[cc]  ,cartt)
sum(indices)
