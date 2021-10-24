"""
this kernel will prepare da
"""
module PrepareArrtoBool
export getIndexOfQueue
using CUDA, Logging,Main.CUDAGpuUtils, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils,Main.MetaDataUtils,Main.HFUtils



"""
allocates in the local, register in shared memory
"""
macro localAllocations()

    return esc(quote
    anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
    #creates shared memory and initializes it to 0
    shmemSum = @cuStaticSharedMem(Float32,(32,2))
    locFps= UInt32(0)
    locFns= UInt32(0)
    offsetIter= UInt16(0)


    ######## needed for establishing min and max values of blocks that are intresting us 
     minX =@cuStaticSharedMem(Float32, 1)
     maxX= @cuStaticSharedMem(Float32, 1)
     minY = @cuStaticSharedMem(Float32, 1)
     maxY= @cuStaticSharedMem(Float32, 1)
     minZ = @cuStaticSharedMem(Float32, 1)
     maxZ= @cuStaticSharedMem(Float32, 1)      
     
     true  ⊻ false

     isAnyPositive = @cuStaticSharedMem(Bool, 1)
     #resetting
     minX[1]= Float32(1110.0)
     maxX[1]= Float32(0.0)
     minY[1]= Float32(1110.0)
     maxY[1]= Float32(0.0)    
     minZ[1]= Float32(1110.0)
     maxZ[1]= Float32(0.0) 
     @ifXY 1 1 isAnyPositive[1]= false
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
    localQuesValues= @cuStaticSharedMem(Float32, 14)   
  

    #making sure they are initialized all to zeros

     
     sync_threads()
end)
end


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

xpos,ypos,zpos -current  position in x,y,z dimension 
datBdim - dimensions of the data block

on the basis of the data it should give the index from 1 to 14 - to the appropriate queue
"""
function getIndexOfQueue(xpos,ypos,zpos, datBdim,boolGold)
    #we need to do so many != in order to deal with corners ...
    return (
     (xpos==1)*1
    +(xpos==datBdim[1])*3
    +(ypos==1 && xpos!=1 && xpos!=datBdim[1] )*5
    +(ypos==datBdim[2] && xpos!=1 && xpos!=datBdim[1]  )*7
    +(zpos==1  && xpos!=1 && xpos!=datBdim[1]  && ypos!=1 && ypos!=datBdim[2] )*9
    +(zpos==datBdim[3] && xpos!=1 && xpos!=datBdim[1]  && ypos!=1 && ypos!=datBdim[2])*11
    +(xpos>1 && xpos<datBdim[1] &&  ypos>1 && ypos<datBdim[2] && zpos>1 && zpos<datBdim[3])*13
    )+boolGold# in that way we will get odd for fp an even for fn

end

"""
invoked on each lane and on the basis of its position will update the number of fp or fn in given queue
"""
macro uploadLocalfpFNCounters()
   return esc(quote
   coord=PrepareArrtoBool.getIndexOfQueue((xdim * blockDimX())+threadIdxX() ,(ydim * blockDimY())+threadIdxY(),(zdim+1),datBdim,boolGold)
   atomicallyAddToSpot(Float32,localQuesValues,coord,1)
    end)
end   


"""
invoked after we gone through data block and now we save data into shared memory
"""
macro uploadMinMaxesToShmem()
    return  esc(quote
        @ifXY 1 1 if(isAnyPositive[1]) minX[1]= min(minX[1],xMeta+1) end
        @ifXY 1 2 if(isAnyPositive[1]) maxX[1]= max(maxX[1],xMeta+1) end
        @ifXY 2 1 if(isAnyPositive[1]) minY[1]= min(minY[1],yMeta+1) end
        @ifXY 2 2 if(isAnyPositive[1]) maxY[1]= max(maxY[1],yMeta+1) end
        @ifXY 3 1 if(isAnyPositive[1]) minZ[1]= min(minZ[1],zMeta+1) end
        @ifXY 3 2 if(isAnyPositive[1]) maxZ[1]= max(maxZ[1],zMeta+1) end 
    end)

end

"""
invoked after we gone through data block and now we save data into appropriate spots in metadata of this metadata block
"""
macro uploadDataToMetaData()
    esc(quote
    
    #now we should also add the total value by adding all fp or fn values required

    
    @ifY 1 if(threadIdxX()<15 && isAnyPositive[1])
        if(isAnyPositive[1]) metaData[xMeta+1,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+ threadIdxX()]=localQuesValues[threadIdxX()]
       # localQuesValues[threadIdxX()]=0
        end
    end

    @ifXY 15 1 if(isAnyPositive[1])
        metaData[xMeta+1,yMeta+1,zMeta+1,(getBeginingOfFpFNcounts()+ 15)]= (localQuesValues[1]+localQuesValues[3]+localQuesValues[5]+localQuesValues[7] +localQuesValues[9]+localQuesValues[11]+localQuesValues[13] ) 
        end
    @ifXY 16 1 if(isAnyPositive[1]) 
        metaData[xMeta+1,yMeta+1,zMeta+1,(getBeginingOfFpFNcounts()+ 16)]= (localQuesValues[2]+localQuesValues[4]+localQuesValues[6]+localQuesValues[8]+localQuesValues[10]+localQuesValues[12]+localQuesValues[14]) 
        end
    
            #as all was done on first warp we just need to synchronize this and clear localQuesValues
    #now we need to clear localQuesValues

    # vote_ballot_sync(FULL_MASK, localBool)
    # threadfence()

    # if(threadIdxX()<15)
    #     localQuesValues[threadIdxX()]=Float32(0.0)
    # end

    end)

end#uploadDataToMetaData

"""
invoked after all of the data was scanned so after we will do atomics between blocks we will know 
    the minimal and maximal in each dimensions
"""
macro  finalGlobalSet()
    esc(quote
        @redWitAct(offsetIter,shmemSum,  locFns,+,     locFps,+   )
        @addAtomic(shmemSum,fn,fp)
        # @ifXY 1 1 CUDA.@cuprint  """ minxRes $(minxRes)  minX[1] $(minX[1])  maxxRes $(maxxRes) maxX[1] $(maxX[1])
        # minyRes $(minyRes)  minY[1] $(minY[1])  maxyRes $(maxyRes)  maxY[1] $(maxY[1])  
        # minzRes $(minzRes) minZ[1] $(minZ[1])  maxzRes $(maxzRes) maxZ[1] $(maxZ[1])
        # \n"""
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
goldBoolGPU3d - array holding 3 dimensional data of gold standard bollean array
segmBoolGPU3d - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres
datBdim - dimensions of data block
metaDataDims - dimensions of the metadata
loopXMeta,loopYZMeta- indicates how many times we need to iterate over the metadata
inBlockLoopX,inBlockLoopY,inBlockLoopZ - indicates how many times we need to iterate over the data block using our size of thread block
                                          basically data block size will be established by the thread block size of main kernel  
"""
function getBoolCubeKernel(goldBoolGPU3d
        ,segmBoolGPU3d
        ,numberToLooFor::T
        ,reducedGoldA
        ,reducedSegmA
        ,reducedGoldB
        ,reducedSegmB
        ,fn
        ,fp
        ,minxRes
        ,maxxRes
        ,minyRes
        ,maxyRes
        ,minzRes
        ,maxzRes
        ,datBdim
        ,metaData
        ,metaDataDims
        ,mainArrDims
        ,loopXMeta,loopYZMeta
        ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
) where T
    @localAllocations()
    #we need nested x,y,z iterations so we will iterate over the matadata and on its basis over the  data in the main arrays 
    #first loop over the metadata 


    @iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta,
         begin
         #inner loop is over the data indicated by metadata
         @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ
                         ,begin 
                                boolGold=goldBoolGPU3d[x,y,z]==numberToLooFor
                                boolSegm=segmBoolGPU3d[x,y,z]==numberToLooFor    
     
                                #we need to also collect data about how many fp and fn we have in main part and borders
                                #important in case of corners we will first analyze z and y dims and z dim on last resort only !

                                #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
                                if(boolGold  || boolSegm)
                                        if((boolGold  ⊻ boolSegm))
                                            @uploadLocalfpFNCounters()

                                            locFps+=boolSegm
                                            locFns+=boolGold
                                            isAnyPositive[1]= true #- we just mark that there was some fp or fn in this block 
                                        end# if (boolGold  ⊻ boolSegm)
                                    #passing data to new arrays needed for running final algorithm
                                    reducedGoldA[x,y,z]=boolGold    
                                    reducedSegmA[x,y,z]=boolSegm    
                                    reducedGoldB[x,y,z]=boolGold    
                                    reducedSegmB[x,y,z]=boolSegm 
                                end#if boolGold  || boolSegm
                            end)#ex                
                # #now we are just after we iterated over a single data block  we need to we save data about border data blocks 
                sync_threads()

                    #we want to invoke this only once per data block
                    #save the data about number of fp and fn of this block and accumulate also this sum for global sum 
                    #doing all on first warp
                    @uploadDataToMetaData() 
                    # if(isAnyPositive[1])           
                    #     @ifXY 4 1  CUDA.@cuprint "xMeta $(xMeta+1)  ,yMeta $(yMeta+1),zMeta $(zMeta+1) \n"
                    # end    
                    #invoked after we gone through data block and now we save data into shared memory
                    #doing all on second warp
                    @uploadMinMaxesToShmem()            

                    sync_threads()
                    ### set x y z needed in case we will use later linear indexing
                    @ifXY 2 1 metaData[xMeta+1,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+1]= xMeta+1
                    @ifXY 3 1 metaData[xMeta+1,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+2]= yMeta+1
                    @ifXY 4 1 metaData[xMeta+1,yMeta+1,zMeta+1,getBeginingOfFpFNcounts()+3]= zMeta+1


                    #resetting
                    @ifXY 1 1 isAnyPositive[1]= false  #reset   

                 
                    
                    @ifY 2 if(threadIdxX()<15)
                        localQuesValues[threadIdxX()]=0
                    end
               sync_threads()

            end) #outer loop        
    #             #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 
                
        # outer loop expession  )
    #in order to have global data 
    sync_threads()

    @finalGlobalSet()


   return  

   end

end


