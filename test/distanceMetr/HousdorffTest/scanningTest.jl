



"""
after previous sync threads we already have the number of how much we increased number of results  relative to previous dilatation step
now we need to go through  those numbers and in case some of the border queues were incremented we need to analyze those added entries to establish is there 
any duplicate in case there will be we need to decrement counter and set the corresponding duplicated entry to 0 
"""
macro scanForDuplicates(oldCount, countDiff) 
    return esc(quote
    @unroll for resQueueNumb in 1:12 #we have diffrent result queues
        @exOnWarp resQueueNumb begin
            @unroll for threadNumber in 1:32 # we need to analyze all thread id x 
                if( resShmem[warpNumb+1,resQueueNumb+1,3]) # futher actions necessary only if counter diffrence is bigger than 0 
                    if( threadIdxX()==threadNumber ) #now we need some  values that are in the registers  of the associated thread 
                        #those will be needed to know what we need to iterate over 
                        shmemSum[33,resQueueNumb]=  $oldCount 
                        shmemSum[34,resQueueNumb]=  $countDiff
                        shmemSum[35,resQueueNumb]=  linIndex
                    end# if ( threadIdxX()==warpNumb )
                    
                    sync_warp()# now we should have the required number for scanning of new values for duplicates
                    @scanForDuplicatesMainPart()
                end# resShmem[warpNumb+1,i+1,3]
            end # for warp number  
        end #@exOnWarp
    end#for 
end )
end

singleVal = CUDA.zeros(14)

threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
 metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:17,2:18,4:10,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )

function metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)

  
    MetadataAnalyzePass.@scanForDuplicates()
    
    return
end
@cuda threads=threads blocks=blocks metaDataWarpIterKernel(singleVal,metaDataDims,loopXMeta,loopYZMeta)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]



