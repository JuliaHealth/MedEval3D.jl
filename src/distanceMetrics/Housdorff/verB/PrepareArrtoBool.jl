"""
this kernel will prepare da
"""
module PrepareArrtoBool
export getIndexOfQueue
using CUDA, Logging,Main.CUDAGpuUtils, Logging,StaticArrays, Main.IterationUtils, Main.ReductionUtils, Main.CUDAAtomicUtils



"""
allocates in the local, register in shared memory
"""
macro localAllocations()

    return esc(quote
    anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
    #creates shared memory and initializes it to 0
    shmemSum = @cuStaticSharedMem(Float32,(32,2))
    

    ######## needed for establishing min and max values of blocks that are intresting us 
     minX =@cuStaticSharedMem(Float32, 1)
     maxX= @cuStaticSharedMem(Float32, 1)
     minY = @cuStaticSharedMem(Float32, 1)
     maxY= @cuStaticSharedMem(Float32, 1)
     minZ = @cuStaticSharedMem(Float32, 1)
     maxZ= @cuStaticSharedMem(Float32, 1)      
     
     
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
     locArr= (Float32(0.0), Float32(0.0))# for global fp fn sums
     locArrB= (Float32(0.0), Float32(0.0))# for local fp fn sums
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
specialization of 2 dim iteration  for iterating over 3 dimensional metadata
    we join y and z iterations in order to increase occupancy for potentially small arrays
"""
macro iter3dOuter(metaDataDims,loopXMeta,loopYZMeta,yTimesZmeta, ex)
    mainExp = generalizedItermultiDim(;xname=:(xMeta)
    ,yname= :(yzSpot)
    ,arrDims=metaDataDims
    ,loopXdim=loopXMeta 
    ,loopYdim=loopYZMeta
    ,isFullBoundaryCheckY=false
    ,isFullBoundaryCheckX =false
    ,yOffset = :(ydim*gridDim().x)
    ,yAdd=  :(blockIdxX()-1) 
    ,additionalActionBeforeY= :( yMeta= rem(yzSpot,$metaDataDims[2]) ; zMeta= fld(yzSpot,$metaDataDims[2]) )
    ,yCheck = :(yMeta < $metaDataDims[2] && zMeta<$metaDataDims[3] )
    ,xCheck = :(xMeta < $metaDataDims[1])
    #so each block will iterate over all xses
    ,xOffset= :(0)
    ,xAdd= :(xdim)
    ,is3d = false
    , ex = ex)  
    return esc(:( $mainExp))

end #iter3dOuter

"""
will enable iterating over the data of data block
"""
macro iterDataBlock(mainArrDims,dataBlockDims,loopXdim ,loopYdim,loopZdim,ex)
    mainExp = generalizedItermultiDim(;arrDims=mainArrDims
    ,loopXdim
    ,loopYdim
    ,loopZdim
    # ,xCheck = :((xMeta* $dataBlockDims[1]+x)<=$mainArrDims[1] )
    # ,yCheck = :((yMeta* $dataBlockDims[2]+y)<=$mainArrDims[2])
    # ,zCheck = :( (zMeta* $dataBlockDims[3]+z)<=$mainArrDims[3])
    ,xCheck = :(((xdim * blockDimX())+threadIdxX()  )<= $dataBlockDims[1] && x<=$mainArrDims[1] )
    ,yCheck = :(((ydim * blockDimY())+threadIdxY()  )<= $dataBlockDims[2] &&  y<=$mainArrDims[2])
    ,zCheck = :((zdim+1)<= $dataBlockDims[3]  &&   z<=$mainArrDims[3])
    ,zOffset= :(zMeta* ( ($dataBlockDims[3])  ) )
    ,zAdd =:(zdim+1)
   ,yOffset = :(ydim* blockDimY()+yMeta* $dataBlockDims[2])
   ,yAdd= :(threadIdxY())
   ,xOffset= :( (xdim * blockDimX()) +xMeta* $dataBlockDims[1])
    ,xAdd= :(threadIdxX())
    ,isFullBoundaryCheckX =true
    , isFullBoundaryCheckY=true
    , isFullBoundaryCheckZ=true
    , ex = ex)  
    return esc(:( $mainExp))

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
function getIndexOfQueue(xpos,ypos,zpos, datBdim,boolSegm)
    # @info " xpos = $(xpos) ; ypos = $(ypos) ; zpos = $(zpos) "
    #we need to do so many != in order to deal with corners ...
    return (
     (xpos==1)*1
    +(xpos==datBdim[1])*3
    +(ypos==1 && xpos!=1 && xpos!=datBdim[1] )*5
    +(ypos==datBdim[2] && xpos!=1 && xpos!=datBdim[1]  )*7
    +(zpos==1  && xpos!=1 && xpos!=datBdim[1]  && ypos!=1 && ypos!=datBdim[2] )*9
    +(zpos==datBdim[3] && xpos!=1 && xpos!=datBdim[1]  && ypos!=1 && ypos!=datBdim[2])*11
    +(xpos>1 && xpos<datBdim[1] &&  ypos>1 && ypos<datBdim[2] && zpos>1 && zpos<datBdim[3])*13
    )+boolSegm# in that way we will get odd for fp an even for fn

end

"""
invoked on each lane and on the basis of its position will update the number of fp or fn in given queue
"""
macro uploadLocalfpFNCounters()
   return esc(quote
   coord=PrepareArrtoBool.getIndexOfQueue((xdim * blockDimX())+threadIdxX() ,(ydim * blockDimY())+threadIdxY(),(zdim+1),datBdim,boolSegm)
   atomicallyAddToSpot(Float32,localQuesValues,coord,1)
    end)
end   


"""
invoked after we gone through data block and now we save data into shared memory
"""
macro uploadMinMaxesToShmem()
    return  esc(quote
        @ifXY 3 1 if(isAnyPositive[1]) minX[1]= min(minX[1],xMeta) end
        @ifXY 4 1 if(isAnyPositive[1]) maxX[1]= max(maxX[1],xMeta) end
        @ifXY 5 1 if(isAnyPositive[1]) minY[1]= min(minY[1],yMeta) end
        @ifXY 6 1 if(isAnyPositive[1]) maxY[1]= max(maxY[1],yMeta) end
        @ifXY 7 1 if(isAnyPositive[1]) minZ[1]= min(minZ[1],zMeta) end
        @ifXY 8 1 if(isAnyPositive[1]) maxZ[1]= max(maxZ[1],zMeta) end 
    end)

end

"""
invoked after we gone through data block and now we save data into appropriate spots in metadata of this metadata block
"""
macro uploadDataToMetaData()
    esc(quote
    @ifXY 1 1 if(isAnyPositive[1]) setMetaDataFpCount(locArrB[2], xMeta,yMeta,zMeta) end   
    @ifXY 2 1 if(isAnyPositive[1]) setMetaDataFnCount(locArrB[1], xMeta,yMeta,zMeta) end
    end)

end#uploadDataToMetaData

"""
invoked after all of the data was scanned so after we will do atomics between blocks we will know 
    the minimal and maximal in each dimensions
"""
macro  finalGlobalSet()
    esc(quote
        @redWitAct(offsetIter,shmemSum,  locArr[1],+,     locArr[2],+   )
        @addAtomic(shmemSum,fn,fp)

        @ifXY 1 1 atomicMinSet(minxRes[1],minX[1])
        @ifXY 2 2 atomicMaxSet(maxxRes[1],maxX[1])

        @ifXY 3 3 atomicMinSet(minyRes[1],minY[1])
        @ifXY 4 4 atomicMaxSet(maxyRes[1],maxY[1])

        @ifXY 5 5 atomicMinSet(minzRes[1],minY[1])
        @ifXY 6 6 atomicMaxSet(maxzRes[1],maxZ[1])
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
loopXMeta,loopYMeta,loopZmeta - indicates how many times we need to iterate over the metadata
inBlockLoopX,inBlockLoopY,inBlockLoopZ - indicates how many times we need to iterate over the data block using our size of thread block
                                          basically data block size will be established by the thread block size of main kernel  
"""
function getBoolCubeKernel(goldBoolGPU3d
        ,segmBoolGPU3d
        ,reducedGoldA
        ,reducedSegmA
        ,reducedGoldB
        ,reducedSegmB
        ,numberToLooFor::T
        ,fn::CuDeviceVector{UInt32, 1}
        ,fp::CuDeviceVector{UInt32, 1}
        ,minxRes::CuDeviceVector{UInt32, 1}
        ,maxxRes::CuDeviceVector{UInt32, 1}
        ,minyRes::CuDeviceVector{UInt32, 1}
        ,maxyRes::CuDeviceVector{UInt32, 1}
        ,minzRes::CuDeviceVector{UInt32, 1}
        ,maxzRes::CuDeviceVector{UInt32, 1}
        ,datBdim
        ,metaDataDims
        ,mainArrDims
        ,loopXMeta,loopYMeta,yTimesZmeta
        ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
) where T
    @localAllocations()
    #we need nested x,y,z iterations so we will iterate over the matadata and on its basis over the  data in the main arrays 
    #first loop over the metadata 
    #datBdim - indicats dimensions of data blocks
    @iter3dOuter(metaDataDims, loopXMeta,loopYMeta,loopZmeta,
         begin
         #inner loop is over the data indicated by metadata
         @iterDataBlock(mainArrDims,datBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ
                         ,begin 
                                boolGold=goldBoolGPU3d[x,y,z]==numberToLooFor
                                boolSegm=segmBoolGPU3d[x,y,z]==numberToLooFor                                    
                                @inbounds locArr[boolGold+ boolSegm+ boolSegm]+=(boolGold  ⊻ boolSegm)
                                @inbounds locArrB[boolGold+ boolSegm+ boolSegm]+=(boolGold  ⊻ boolSegm)
                                #we need to also collect data about how many fp and fn we have in main part and borders
                                #important in case of corners we will first analyze z and y dims and z dim on last resort only !
                                @uploadLocalfpFNCounters()
                    
                                #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
                                if(boolGold  || boolSegm)
                                        if((boolGold  ⊻ boolSegm))
                                            isAnyPositive[1]= true #- we just mark that there was some fp or fn in this block 
                                        end# if (boolGold  ⊻ boolSegm)
                                    #passing data to new arrays needed for running final algorithm
                                    reducedGoldA[x,y,z]=boolGold    
                                    reducedSegmA[x,y,z]=boolSegm    
                                    reducedGoldB[x,y,z]=boolGold    
                                    reducedSegmB[x,y,z]=boolSegm 
                                end#if boolGold  || boolSegm
                            end)#ex                
                #now we are just after we iterated over a single data block  we need to we save data about border data blocks 
                sync_threads()
                #we want to invoke this only once per data block
                #save the data about number of fp and fn of this block and accumulate also this sum for global sum 
                @uploadDataToMetaData()            

                #invoked after we gone through data block and now we save data into shared memory
                @uploadMinMaxesToShmem()            

                
                sync_warp()
                #resetting
                @ifXY 1 1 isAnyPositive[1]= false  #reset     
                @ifXY 2 1  locArrB[1]= false  #reset     
                @ifXY 3 1  locArrB[2]= false  #reset     
     end) #outer loop        
                #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 
                
        # outer loop expession  )
    #in order to have global data 


    @finalGlobalSet()


   return  
   end




# """
# add value to the shared memory in the position i, x where x is 1 ,2 or 3 and is calculated as described below
# boolGold & boolSegm + boolGold +1 will evaluate to 
#     ⊻- xor gate 
#     1 in case of false negative
#     2 in case of false positive
# x,y,z - the coordinates we are currently in 

# """
# @inline function incr_locArr(boolGold::Bool
#                             ,boolSegm::Bool
#                             ,locArr::MVector{6, UInt16}
#                             ,x,y,z
#                             ,reducedGoldA
#                             ,reducedSegmA
#                             ,reducedGoldB
#                             ,reducedSegmB
#                             ,anyPositive)
#     #first we need the flase positives and false negatives - this will write also true positive - but later we will 
#     @inbounds locArr[boolGold+ boolSegm+ boolSegm]+=(boolGold  ⊻ boolSegm)
#     #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
#     if(boolGold  || boolSegm)
#     #locArr  0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
#     locArr[3]= min(locArr[3],x)
#     locArr[4]= max(locArr[4],x)
#     locArr[5]= min(locArr[5],y)
#     locArr[6]= max(locArr[6],y)
    
#     #CUDA.@cuprint " locArr A $(locArr[1]) B $(locArr[2])  C $(locArr[3]) D $(locArr[4]) E $(locArr[5]) F $(locArr[6])  x $x y $y z $z  \n"    

#     #passing data to new arrays needed for running final algorithm
#     reducedGoldA[x,y,z]=boolGold    
#     reducedSegmA[x,y,z]=boolSegm    
#     reducedGoldB[x,y,z]=boolGold    
#     reducedSegmB[x,y,z]=boolSegm    
    
#     #anyPositive[1]=true
#     end    
   
#     return true
# end
# """
# get which warp it is in a block and which lane in warp 
# """
# function getWidAndLane(threadIdx)::Tuple{UInt8, UInt8}
#       return fldmod1(threadIdx,32)
# end

# """
# creates shared memory and initializes it to 0
# wid - the number of the warp in the block
# """
# function createAndInitializeShmem(wid, threadId,lane)
#    #for storing results from warp reductions
#    shmemSum = @cuStaticSharedMem(UInt16, (33,6))

#     if(wid==1)
#         shmemSum[lane,1]=0
#     elseif(wid==2)
#         shmemSum[lane,2]=0
#     elseif(wid==3)
#         shmemSum[lane,3]=10000 # in case of minimum we must start high    
#     elseif(wid==4)
#         shmemSum[lane,4]=0       
#     elseif(wid==5)
#         shmemSum[lane,5]=10000 # in case of minimum we must start high            
#     elseif(wid==6)
#         shmemSum[lane,6]=0
#     end            

# return shmemSum

# end#createAndInitializeShmem


# """
# reduction across the warp and adding to appropriate spots in the  shared memory
# """
# function firstReduce(locArr,shmemSum,wid)
#     #locArr  0 - false negative; 1- false positive; 2 -minx; 3 max x; 4 miny; 5 maxy
#     @inbounds shmemSum[wid,1] = reduce_warp(locArr[1],32)
#     @inbounds shmemSum[wid,2] = reduce_warp(locArr[2],32)

#     @inbounds shmemSum[wid,3] = reduce_warp_min(locArr[3],32)
#     @inbounds shmemSum[wid,4] = reduce_warp_max(locArr[4],32)
#     @inbounds shmemSum[wid,5] = reduce_warp_min(locArr[5],32)
#     @inbounds shmemSum[wid,6] = reduce_warp_max(locArr[6],32)

# #    CUDA.@cuprint " shmemSum[wid,3] $(shmemSum[wid,3])  shmemSum[wid,4] $(shmemSum[wid,4] ) shmemSum[wid,5]  $(shmemSum[wid,5])  shmemSum[wid,6]  $(shmemSum[wid,6])   \n"


# end#firstReduce

# """
# sets the final block amount of true positives, false positives and false negatives and saves it
# to the  array representing each slice, 
# wid - the warp in a block we want to use
# numb - number associated with constant - used to access shared memory for example
# chosenWid - on which block we want to make a reduction to happen
# intermediateRes - array with intermediate -  slice wise results
# singleREs - the final  constant holding image witde values (usefull for example for debugging)
# shmemSum - shared memory where we get the  results to be reduced now and to which we will also save the output
# blockId - number related to block we are currently in 
# lane - the lane in the warp
# """
# function getSecondBlockReduceSum(chosenWid,numb,wid, singleREs,shmemSum,blockId,lane)
#     if(wid==chosenWid )
#         shmemSum[33,numb] = reduce_warp(shmemSum[lane,numb],32 )
        
#       #probably we do not need to sync warp as shfl dow do it for us         
#       if(lane==1)
#           @inbounds @atomic singleREs[]+=shmemSum[33,numb]    
#       end    
#     #   if(lane==3)
#     #     #ovewriting the value 
#     #     @inbounds shmemSum[1,numb]=vall
#     #   end     

#   end  


# end#getSecondBlockReduce
# function getSecondBlockReduceMin(chosenWid,numb,wid, singleREs::CuDeviceVector{UInt32, 1},shmemSum,blockId,lane)
#     if(wid==chosenWid )
#         shmemSum[33,numb] = reduce_warp_min(shmemSum[lane,numb],32 )
        

#       #probably we do not need to sync warp as shfl dow do it for us         
#       if(lane==1)
#         @inbounds CUDA.atomic_min!(pointer(singleREs),UInt32(shmemSum[33,numb]))    
#       end    
#     #   if(lane==3)
#     #     #ovewriting the value 
#     #     @inbounds shmemSum[1,numb]=vall
#     #   end     

#   end  

# end#getSecondBlockReduce
# function getSecondBlockReduceMax(chosenWid,numb,wid, singleREs::CuDeviceVector{UInt32, 1},shmemSum,blockId,lane,singleREsMin::CuDeviceVector{UInt32, 1},singleREsMax::CuDeviceVector{UInt32, 1})
#     if(wid==chosenWid )
#         shmemSum[33,numb] = reduce_warp_max(shmemSum[lane,numb],32 )
        
#       #probably we do not need to sync warp as shfl dow do it for us         
#       if(lane==1)
#         @inbounds CUDA.atomic_max!(pointer(singleREs),UInt32(shmemSum[33,numb]))   
#       end    
#       if(lane==3 && shmemSum[33,numb]>0)
#         @inbounds CUDA.atomic_min!(pointer(singleREsMin),UInt32(blockId))   
#       end  
#       if(lane==4&& shmemSum[33,numb]>0)
#         @inbounds CUDA.atomic_max!(pointer(singleREsMax),UInt32(blockId))   
#       end 
      
#     #   if(lane==3)
#     #     #ovewriting the value 
#     #     @inbounds shmemSum[1,numb]=vall
#     #   end     

#   end  

# end#getSecondBlockReduce

# function getSecondBlockReduceForZ(chosenWid,numb,wid, singleREsMin::CuDeviceVector{UInt32, 1},singleREsMax::CuDeviceVector{UInt32, 1},value,lane)
#     if(wid==chosenWid )

#       if(lane==3)
#         @inbounds CUDA.atomic_min!(pointer(singleREsMin),UInt32(blockId))   
#       end  
#       if(lane==4)
#         @inbounds CUDA.atomic_max!(pointer(singleREsMax),UInt32(blockId))   
#       end    
#     #   if(lane==3)
#     #     #ovewriting the value 
#     #     @inbounds shmemSum[1,numb]=vall
#     #   end     

#   end  

# end#getSecondBlockReduce





end#TpfpfnKernel



########### version with cooperative groups

# function getBlockTpFpFn(goldBoolGPU
#     , segmBoolGPU
#     ,tp,tn,fp,fn
#     ,intermediateResTp
#     ,intermediateResFp
#     ,intermediateResFn
#     ,loopNumb::Int64
#     ,indexCorr::Int64
#     ,amountOfWarps::Int64
#     ,pixelNumberPerSlice::Int64
#     ,numberToLooFor::T
#     ,IndexesArray
#     ,maxSlicesPerBlock::Int64
#     ,slicesPerBlockMatrix
#     ,numberOfBlocks::Int64) where T
# # we multiply thread id as we are covering now 2 places using one lane - hence after all lanes gone through we will cover 2 blocks - hence second multiply    
# correctedIdx = (threadIdxX()-1)* indexCorr+1
# i= correctedIdx
# #i = correctedIdx + ((blockIdx().x - 1) *indexCorr) * (blockDimX())# used as a basis to get data we want from global memory
# wid, lane = fldmod1(threadIdxX(),32)
# #creates shared memory and initializes it to 0
# shmem,shmemSum = createAndInitializeShmem()
# shmem[513,1]= numberToLooFor
# ##### in this outer loop we are iterating over all slices that this block is responsible for
# @unroll for blockRef in 1:maxSlicesPerBlock    
#     sliceNumb= slicesPerBlockMatrix[blockIdx().x,blockRef]
#         if(sliceNumb>0)
#             i = correctedIdx + (pixelNumberPerSlice*(sliceNumb-1))# used as a basis to get data we want from global memory
#             setShmemTo0(wid,threadIdxX(),lane,shmem,shmemSum)           
#             # incrementing appropriate number of times 
        
#         @unroll for k in 0:loopNumb
#                 if(correctedIdx+k<=pixelNumberPerSlice)
#                     incr_shmem(threadIdxX(),goldBoolGPU[i+k]==shmem[513,1],segmBoolGPU[i+k]==shmem[513,1],shmem)
#                 end#if
#             end#for   
#         #reducing across the warp
#         firstReduce(shmem,shmemSum,wid,threadIdxX(),lane,IndexesArray,i)
        
        
#         sync_threads()
#         #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory 
#         getSecondBlockReduce( 1,3,wid,intermediateResTp,tp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 2,2,wid,intermediateResFp,fp,shmemSum,blockIdx().x,lane)
#         getSecondBlockReduce( 3,1,wid,intermediateResFn,fn,shmemSum,blockIdx().x,lane)
#     end#if     
# end#for

# return  
# end


# """
# This will prepare data for more complex distance metrics - we need to change input data type into boolean and find smallest possible cube that hold all necessery data

# returning the data  from a kernel that  calclulate number of true positives,
# true negatives, false positives and negatives par image and per slice in given data 
# goldBoolGPU - array holding data of gold standard bollean array
# segmBoolGPU - array with the data we want to compare with gold standard
# we have a and b - becouse the Housdorff distance is defined as 2 pass algorithm
# reducedGold a nd b - the smallest boolean block (3 dim array) that contains all positive entris from both masks
# reducedSegm a nd b- the smallest boolean block (3 dim array) that contains all positive entris from both masks
# numberToLooFor - number we will analyze whether is the same between two sets
# metaData - sets some basic informations to metaData
# cuda arrays holding just single value wit atomically reduced result
# ,fn,fp
# ,minxRes,maxxRes
# ,minyRes,maxyRes
# ,minZres,maxZres
# """
# function getBoolCube!(goldBoolGPU3d
#     ,segmBoolGPU3d
#     ,numberOfSlices::Int64
#     ,fn
#     ,fp
#     ,minxRes
#     ,maxxRes
#     ,minyRes
#     ,maxyRes
#     ,minZres
#     ,maxZres
#     ,numberToLooFor::T
#     ,IndexesArray
#     ,reducedGoldA
#     ,reducedSegmA
#     ,reducedGoldB
#     ,reducedSegmB
#     ,metaData) where T

# # we prepare the boolean array of dimensions at the begining the same as the gold standard array - later we will work only on view of it

# goldDims=size(goldBoolGPU3d) 

# #biggest divisible by 32 number to cover the x dimension
# warpNumb = cld(goldDims[1],32)
# threadNumb = min(1024,warpNumb*32)

# args = (goldBoolGPU3d
#         ,segmBoolGPU3d
#         ,reducedGoldA
#         ,reducedSegmA
#         ,reducedGoldB
#         ,reducedSegmB
#         ,UInt16(goldDims[2])
#         ,UInt16(goldDims[1])
#         ,UInt16(cld(goldDims[1],threadNumb))
#         ,numberToLooFor
#         ,IndexesArray
#         ,fn
#         ,fp
#         ,minxRes
#         ,maxxRes
#         ,minyRes
#         ,maxyRes
#         ,minZres
#         ,maxZres
#         ,warpNumb
#         )
# #getMaxBlocksPerMultiproc(args, getBlockTpFpFn) -- evaluates to 3

# @cuda threads=threadNumb blocks=numberOfSlices getBoolCubeKernel(args...) 
# return args
# end#getTpfpfnData