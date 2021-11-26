"""
this kernel will prepare da
"""
module PrepareArrtoBool
using CUDA, Logging,Main.CUDAGpuUtils, Logging,StaticArrays, Main.IterationUtils,Main.BitWiseUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils,Main.HFUtils
export @planeIter,getLargeForBoolKernel,getSmallForBoolKernel,@getBoolCubeKernel,@localAllocations,@uploadLocalfpFNCounters,@uploadMinMaxesToShmem,@uploadDataToMetaData,@finalGlobalSet


"""
allocates in the local, register in shared memory
"""
macro localAllocations()

    return esc(quote
    anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
    #creates shared memory and initializes it to 0
    # shmemSum = @cuStaticSharedMem(Float32,(32,2))
    locFps= UInt32(0)
    locFns= UInt32(0)
    offsetIter= UInt32(0)
    #storing data about block in a forrmat where each Int32 number is representing a part of data block with constant x and y and varia ble z position
    # shmemblockData = @cuDynamicSharedMem(Float32,dataBdim[1], dataBdim[2])
    shmemblockData = @cuDynamicSharedMem(UInt32,(dataBdim[1], dataBdim[2]))

   
    ######## needed for establishing min and max values of blocks that are intresting us 
     minX =@cuStaticSharedMem(Float32, 1)
     maxX= @cuStaticSharedMem(Float32, 1)
     minY = @cuStaticSharedMem(Float32, 1)
     maxY= @cuStaticSharedMem(Float32, 1)
     minZ = @cuStaticSharedMem(Float32, 1)
     maxZ= @cuStaticSharedMem(Float32, 1)      
     
     #resetting
     minX[1]= Float32(1110.0)
     maxX[1]= Float32(0.0)
     minY[1]= Float32(1110.0)
     maxY[1]= Float32(0.0)    
     minZ[1]= Float32(1110.0)
     maxZ[1]= Float32(0.0) 
     #in shared memory
 
#####needed for fp fn sums
     #1 - false negative; 2- false positive
     #locArr= (UInt32(0.0), UInt32(0.0))# for global fp fn sums
     #locArrB= (Float32(0.0), Float32(0.0))# for local fp fn sums
    #  1)   Left FP  
    #  2)   Left FN  
    #  3)   Right FP  
    #  4)   Right FN  
    #  5)   Posterior FP  
    #  6)   Posterior FN  
    #  7)   Anterior FP  
    #  8)   Anterior FN  
    #  9)   Top FP  
    #  10)   Top FN  
    #  11)   Bottom FP  
    #  12)   Bottom FN  
    #13)   main part FP  
    #14)   main Part FN  
    localQuesValues= @cuStaticSharedMem(UInt32, 14)   
  

    #making sure they are initialized all to zeros

     
     sync_threads()
end)
end



"""
invoked on each lane and on the basis of its position will update the number of fp or fn in given queue
"""
macro uploadLocalfpFNCounters()
   return esc(quote
   atomicallyAddToSpot(localQuesValues,getIndexOfQueue(xpos,ypos,zpos,dataBdim,boolSegm),1)
    end)
end   

"""
invoked after we gone through data block and now we save data into shared memory
"""
macro uploadMinMaxesToShmem()
    return  esc(quote
    
    # if( anyPositive)
    #     @ifXY 1 1  CUDA.@cuprint "xMeta+1 $(xMeta+1) yMeta+1 $(yMeta+1) zMeta+1 $(zMeta+1) anyPositive $(anyPositive) \n"
    #     @ifXY 1 1  CUDA.@cuprint "xMeta+1 $(xMeta+1) yMeta+1 $(yMeta+1) zMeta+1 $(zMeta+1) anyPositive $(anyPositive) \n"
    # end
    # @ifXY 1 1 if(anyPositive) CUDA.@cuprint "aaaaaa minX[1] $(minX[1]) xMeta+1 $(xMeta+1)  " end


        @ifXY 1 1 if(anyPositive) minX[1]= min(minX[1],xMeta+1) end
        @ifXY 1 2 if(anyPositive) maxX[1]= max(maxX[1],xMeta+1) end
        @ifXY 2 1 if(anyPositive) minY[1]= min(minY[1],yMeta+1) end
        @ifXY 2 2 if(anyPositive) maxY[1]= max(maxY[1],yMeta+1) end
        @ifXY 3 1 if(anyPositive) minZ[1]= min(minZ[1],zMeta+1) end
        @ifXY 3 2 if(anyPositive) maxZ[1]= max(maxZ[1],zMeta+1) end 
    end)

end

"""
invoked after we gone through data block and now we save data into appropriate spots in metadata of this metadata block
"""
macro uploadDataToMetaData()
    esc(quote
    #now we should also add the total value by adding all fp or fn values required   
    @ifY 1 if(threadIdxX()<15 && anyPositive)
        @setMeta(getBeginingOfFpFNcounts()+ threadIdxX(),localQuesValues[threadIdxX()])
        
    end
    @ifXY 15 1 if(anyPositive)
        # metaData[xMeta+1,yMeta+1,zMeta+1,(getBeginingOfFpFNcounts()+ 15)]= (localQuesValues[1]+localQuesValues[3]+localQuesValues[5]+localQuesValues[7] +localQuesValues[9]+localQuesValues[11]+localQuesValues[13] ) 
        @setMeta(getBeginingOfFpFNcounts()+ 15, (localQuesValues[1]+localQuesValues[3]+localQuesValues[5]+localQuesValues[7] +localQuesValues[9]+localQuesValues[11]+localQuesValues[13] )  )
        end
    @ifXY 16 1 if(anyPositive) 
        # metaData[xMeta+1,yMeta+1,zMeta+1,(getBeginingOfFpFNcounts()+ 16)]= (localQuesValues[2]+localQuesValues[4]+localQuesValues[6]+localQuesValues[8]+localQuesValues[10]+localQuesValues[12]+localQuesValues[14]) 
        @setMeta(getBeginingOfFpFNcounts()+ 16, (localQuesValues[2]+localQuesValues[4]+localQuesValues[6]+localQuesValues[8]+localQuesValues[10]+localQuesValues[12]+localQuesValues[14]) )
        end  

    end)

end#uploadDataToMetaData

"""
invoked after all of the data was scanned so after we will do atomics between blocks we will know 
    the minimal and maximal in each dimensions
"""
macro  finalGlobalSet()
    esc(quote
        offsetIter=1
        @redWitAct(offsetIter,shmemblockData,  locFns,+,     locFps,+   )
        @addAtomic(shmemblockData,fn,fp)
        if(minX[1]<100)
            # CUDA.@cuprint "aaaaaaaa  minxRes[1] $(minxRes[1])  minX $(minX[1])\n"
        end
        @ifXY 1 1 atomicMinSet(minxRes,minX[1])
        @ifXY 1 2 atomicMaxSet(maxxRes,maxX[1])

        @ifXY 2 1 atomicMinSet(minyRes,minY[1])
        @ifXY 2 2 atomicMaxSet(maxyRes,maxY[1])

        @ifXY 3 1 atomicMinSet(minzRes,minZ[1])
        @ifXY 3 2 atomicMaxSet(maxzRes,maxZ[1])
    end)
end


"""
we need to give back number of false positive and false negatives and min,max x,y,x of block containing all data 
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldGPU - array holding 3 dimensional data of gold standard bollean array
segmGPU - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres
dataBdim - dimensions of data block
metaDataDims - dimensions of the metadata
loopXMeta,loopYZMeta- indicates how many times we need to iterate over the metadata
inBlockLoopX,inBlockLoopY,inBlockLoopZ - indicates how many times we need to iterate over the data block using our size of thread block
                                          basically data block size will be established by the thread block size of main kernel  
"""
# function getBoolCubeKernel(goldGPU
#         ,segmGPU
#         ,numberToLooFor::T
#         ,reducedGoldA
#         ,reducedSegmA
#         ,reducedGoldB
#         ,reducedSegmB
#         ,fn
#         ,fp
#         ,minxRes
#         ,maxxRes
#         ,minyRes
#         ,maxyRes
#         ,minzRes
#         ,maxzRes
#         ,dataBdim
#         ,metaData
#         ,metaDataDims
#         ,mainArrDims
#         ,loopXMeta,loopYZMeta
#         ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
# ) where T

macro getBoolCubeKernel()
 return esc(quote
    @localAllocations()
    #we need nested x,y,z iterations so we will iterate over the matadata and on its basis over the  data in the main arrays 
    #first loop over the metadata 


    HFUtils.@iter3dOuter(metaDataDims,loopMeta,metaDataLength,
         begin
        # inner loop is over the data indicated by metadata
        @iterDataBlockZdeepest(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta
                         ,begin 
                                boolGold=goldGPU[x,y,z]==numberToLooFor
                                boolSegm=segmGPU[x,y,z]==numberToLooFor    
                                #we set all bits so we do not need to reset 
                                # CUDA.@cuprint "ypos $(ypos)   "

                                @setBitTo(offsetIter,zpos,boolGold)

                                # @setBitTo((shmemblockData[xpos,ypos]),zpos,boolGold)
                                #we need to also collect data about how many fp and fn we have in main part and borders
                                #important in case of corners we will first analyze z and y dims and z dim on last resort only !
  
                                #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
                                if(boolGold  || boolSegm)  
                                    #if(boolGold) @setBitTo1(offsetIter,zpos) end   
                        # # if(offsetIter>0)
                        #     CUDA.@cuprint "x $(x) y $(y) (z) $(z) offsetIter $(Int64(offsetIter))\n"
                        # # end    
                                        anyPositive=true
                                        if((boolGold  ⊻ boolSegm))
                                            @uploadLocalfpFNCounters()
                                            locFps+=boolSegm
                                            locFns+=boolGold
                                        end# if (boolGold  ⊻ boolSegm)
                                    #now                        

                                end#if boolGold  || boolSegm
                            end#ex
                          #additional after X
                            ,begin 
                        #here we iterated over all z dimension so offsetIter is ready to be uploaded to global memory
                        # if(offsetIter>0)
                        #     CUDA.@cuprint "x $(x) y $(y) (zMeta+1) $((zMeta+1)) \n"
                        # end  
                        if(offsetIter>0)  
                            @inbounds reducedGoldA[x,y,(zMeta+1)]=offsetIter
                        end
                        end)                
                  # #now we are just after we iterated over a single data block  we need to we save data about border data blocks 
                  anyPositive = sync_threads_or(anyPositive) 

                #   if(anyPositive)
                #     @ifXY 1 1  CUDA.@cuprint "aaaaaa  xMeta+1 $(xMeta+1) yMeta+1 $(yMeta+1) zMeta+1 $(zMeta+1) anyPositive $(anyPositive) \n"

                # end  
                 @uploadMinMaxesToShmem()   



                 #in order to reduce used shared memory we are setting values of output array separately
                 @iterDataBlockZdeepest(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta
                 ,begin 
                    if((segmGPU[x,y,z]==numberToLooFor)) @setBitTo1(offsetIter,zpos) end   
                                @setBitTo(offsetIter,zpos,(segmGPU[x,y,z]==numberToLooFor))

                       #we set all bits so we do not need to reset 
                    # @setBitTo(shmemblockData[xpos,ypos],zpos,(segmGPU[x,y,z]==numberToLooFor))
                end,begin 
                #here we iterated over all z dimension so offsetIter is ready to be uploaded to global memory
                if(offsetIter>0)  
                    @inbounds reducedSegmA[x,y,(zMeta+1)]=offsetIter
                end
                end)     
                sync_threads()            

         

                # if(anyPositive)
                #     CUDA.@cuprint "xMeta+1 $(xMeta+1) anyPositive $(anyPositive) \n"
                # end
                    #we want to invoke this only once per data block
                    #save the data about number of fp and fn of this block and accumulate also this sum for global sum 
                    #doing all on first warp
                    @uploadDataToMetaData() 
                    # if(anyPositive)           
                    #     @ifXY 4 1  CUDA.@cuprint "xMeta $(xMeta+1)  ,yMeta $(yMeta+1),zMeta $(zMeta+1) \n"
                    # end    
                    #invoked after we gone through data block and now we save data into shared memory

                    sync_threads()
                    #resetting
                    anyPositive= false  #reset                   
                    
                    @ifY 2 if(threadIdxX()<15)
                        localQuesValues[threadIdxX()]=0
                    end
               sync_threads()

            end) #outer loop        
    #             #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 
                
        # outer loop expession  )
    #clear 
    # @ifY 1 shmemblockData[threadIdxX(),1]=0
    # @ifY 1 shmemblockData[threadIdxX(),2]=0
    
    # sync_threads()

    @finalGlobalSet()


   return  
end)#quote
   end
   
"""
creates small memory footprint GPU variables for getBoolCubeKernel
  return  minX,maxX,minY,maxY,minZ,maxZ,fn,fp
"""
function getSmallForBoolKernel()
    return (CuArray([Float32(1110.0) ])
    , CuArray([Float32(0.0)])
    , CuArray([Float32(1110.0)])
    , CuArray([Float32(0.0)    ])
    , CuArray([Float32(1110.0)])
    , CuArray([Float32(0.0) ])
    ,CuArray([UInt32(0)])
    ,CuArray([UInt32(0)]))
end    


"""
creates large memory footprint GPU variables for getBoolCubeKernel
    return reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB
"""
function getLargeForBoolKernel(mainArrDims,dataBdim)
    #this is in order to be sure that array is divisible by data block so we reduce necessity of boundary checks
    xDim= cld(mainArrDims[1],dataBdim[1])*dataBdim[1]
    yDim = cld(mainArrDims[2],dataBdim[2])*dataBdim[2]
    zDim = cld(mainArrDims[3],dataBdim[3])*dataBdim[3]
    newDims = (xDim,yDim,zDim)
return (
    CUDA.zeros(UInt32,(newDims)),CUDA.zeros(UInt32,(newDims))
    )

end

"""
iterating over shmemblockData
"""
macro planeIter(loopXinPlane,loopYinPlane,maxXdim, maxYdim,ex)
    mainExp = generalizedItermultiDim(
    arrDims=:()
    ,loopXdim=loopXinPlane
    ,loopYdim=loopYinPlane
    ,yCheck = :(y <=$maxYdim)
    ,xCheck = :(x <=$maxXdim )
    ,is3d = false
    , ex = ex)
      return esc(:( $mainExp))
end


end#module


