"""
we need to give back number of false positive and false negatives and min,max x,y,x of block containing all data 
IMPORTANT - we require at least 7 in y dim and 32 in x dim of block thread
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldBoolGPU3d - array holding 3 dimensional data of gold standard bollean array
segmBoolGPU3d - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
loopNumbYdim - number of times the single lane needs to loop in order to get all needed data - in this kernel it will be exactly a y dimension of a slice
xdim - length in x direction of source array 
loopNumbXdim - in case the x dim will be bigger than number of threads we will create second inner loop
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres

"""
function getBoolCubeKernel(goldBoolGPU3d
        ,segmBoolGPU3d
        ,reducedGoldA
        ,reducedSegmA
        ,reducedGoldB
        ,reducedSegmB
        ,loopNumbYdim::UInt16
        ,xdim::UInt16
        ,loopNumbXdim::UInt16
        ,numberToLooFor::T
        ,fn::CuDeviceVector{UInt32, 1}
        ,fp::CuDeviceVector{UInt32, 1}
        ,minxRes::CuDeviceVector{UInt32, 1}
        ,maxxRes::CuDeviceVector{UInt32, 1}
        ,minyRes::CuDeviceVector{UInt32, 1}
        ,maxyRes::CuDeviceVector{UInt32, 1}
        ,minzRes::CuDeviceVector{UInt32, 1}
        ,maxzRes::CuDeviceVector{UInt32, 1}
        ,dataBdim
) where T

   anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
   #creates shared memory and initializes it to 0
   shmemSum = @cuStaticSharedMem(Float32,(32,2))
   interMediateFPFN=@cuStaticSharedMem(UInt32,(2))
   #incrementing appropriate number of times 
   
    
    #1 - false negative; 2- false positive
    locArr= (Float32(0.0), Float32(0.0))
    offsetIter= UInt8(1)
    #needed to get the borders in metadata terms - so blocks that have sth of our intrest
    minX =@cuStaticSharedMem(Float32, 1)
    maxX= @cuStaticSharedMem(Float32, 1)
    minY = @cuStaticSharedMem(Float32, 1)
    maxY= @cuStaticSharedMem(Float32, 1)
    minZ = @cuStaticSharedMem(Float32, 1)
    maxZ= @cuStaticSharedMem(Float32, 1) 
    #those will be used in order to establish how much results we have in each border queue
    #will be accesses=d atomically from threads - to be experimented upon wheather we have better way
    topFP =@cuStaticSharedMem(UInt16, 1)
    topFN =@cuStaticSharedMem(UInt16, 1)

    bottomFP =@cuStaticSharedMem(UInt16, 1)
    bottomFN =@cuStaticSharedMem(UInt16, 1)

    leftFP =@cuStaticSharedMem(UInt16, 1)
    leftFN =@cuStaticSharedMem(UInt16, 1)

    rightFP =@cuStaticSharedMem(UInt16, 1)
    rightFN =@cuStaticSharedMem(UInt16, 1)

    anteriorFP =@cuStaticSharedMem(UInt16, 1)
    anteriorFN =@cuStaticSharedMem(UInt16, 1)

    posteriorFP =@cuStaticSharedMem(UInt16, 1)
    posteriorFN =@cuStaticSharedMem(UInt16, 1)

    isAnyPositive = @cuStaticSharedMem(Bool, 1)
    #resetting
    minX[1]= Float32(1110.0)
    maxX[1]= Float32(0.0)
    minY[1]= Float32(1110.0)
    maxY[1]= Float32(0.0)    
    minZ[1]= Float32(1110.0)
    maxZ[1]= Float32(0.0) 

    topFP = UInt16(0)
    topFN = UInt16(0)
    bottomFP= UInt16(0)
    bottomFN= UInt16(0)
    leftFP= UInt16(0)
    leftFN= UInt16(0)
    rightFP= UInt16(0)
    rightFN= UInt16(0)
    anteriorFP= UInt16(0)
    anteriorFN= UInt16(0)
    posteriorFP= UInt16(0)
    posteriorFN= UInt16(0)

    isAnyPositive[1]= false


    sync_threads()
    #we need nested x,y,z iterations so we will iterate over the matadata and on its basis over the  data in the main arrays 
    #first loop over the metadata 
    #dataBdim - indicats dimensions of data blocks
    @iter3dOuter(xname,yName , zName, loopXMeta,loopYMeta,loopZmeta,
         begin
         #inner loop is over the data indicated by metadata
         @iter3dInMeta(xOuter*dataBdim[1] ,yOuter*dataBdim[2], zzOuter*dataBdim[3],true,zdim ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
                         ,begin 
                                boolGold=    goldBoolGPU3d[x,y,z]==numberToLooFor
                                boolSegm=    segmBoolGPU3d[x,y,z]==numberToLooFor
                                fpXOrFn = (boolGold  ⊻ boolSegm)    
                                #we need to also collect data about how many fp and fn we have in main part and borders
                                #important in case of corners we will first analyze z and y dims and z dim on last resort only !
                                if(fpXOrFn)
                                    #here we store total count first false nehative, second false positive
                                    @inbounds locArr[boolGold+ boolSegm*2]+=1                                      

                                    if(xdim ==1) #left
                                        incrementAtomFPifTrue(boolSegm,leftFP,leftFN)                                
                                    elseif(xdim == inBlockLoopDims[1] )  # right   
                                        incrementAtomFPifTrue(boolSegm,rightFP, rightFN)  
                                    elseif(ydim == 0 )  # posterior   
                                        incrementAtomFPifTrue(boolSegm,posteriorFP , posteriorFN)  
                                    elseif(ydim == inBlockLoopDims[2] )  # anterior                           
                                        incrementAtomFPifTrue(boolSegm,anteriorFP, anteriorFN)  
                                    elseif(zdim == 0 )  # top  
                                        incrementAtomFPifTrue(boolSegm,topFP,topFN)  
                                    elseif(zdim == inBlockLoopDims[3] )  # bottom 
                                        incrementAtomFPifTrue(boolSegm,bottomFP, bottomFN)  
                                    end
                                    isAnyPositive[1]= true #- we just mark that there was some fp or fn in this block 
                                end#if fpXOrFn       

                                #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
                                if(boolGold  || boolSegm)
                                    #passing data to new arrays needed for running final algorithm
                                    reducedGoldA[x,y,z]=boolGold    
                                    reducedSegmA[x,y,z]=boolSegm    
                                    reducedGoldB[x,y,z]=boolGold    
                                    reducedSegmB[x,y,z]=boolSegm 
                                end#if boolGold  || boolSegm
                            end)#ex
                
                #now we are just after we iterated over a single data block  we need to update the data about block metadata

                #we save data about border data blocks 
                sync_threads()
                @redWitAct(offsetIter,shmemSum,  locArr[1],+,     locArr[2],+ )
                sync_threads()
             
                #now we need to reduce locArr - to get total amount of fp and fn            


               #save the data about number of fp and fn of this block and accumulate also this sum for global sum 
                @ifXY 1 1 if(isAnyPositive[1]) minX[1]= min(minX[1],xOuter) end
                @ifXY 2 1 if(isAnyPositive[1]) maxX[1]= max(maxX[1],xOuter) end
                @ifXY 3 1 if(isAnyPositive[1]) minY[1]= min(minY[1],yOuter) end
                @ifXY 4 1 if(isAnyPositive[1]) maxY[1]= max(maxY[1],yOuter) end
                @ifXY 5 1 if(isAnyPositive[1]) minZ[1]= min(minZ[1],zOuter) end
                @ifXY 6 1 if(isAnyPositive[1]) maxZ[1]= max(maxZ[1],zOuter) end
                @ifXY 7 1 if(isAnyPositive[1]) isAnyPositive[1]= false end #reset 
                
                
                @ifXY 1 2 if(isAnyPositive[1]) setMetaLeftFP(metaData,leftFP[1]) end  
                @ifXY 2 2 if(isAnyPositive[1]) setMetaLeftFN(metaData,leftFN[1]) end  

                @ifXY 3 2 if(isAnyPositive[1]) setMetaRightFP(metaData,rightFP[1]) end  
                @ifXY 4 2 if(isAnyPositive[1]) setMetaRightFN(metaData,rightFN[1]) end  

                @ifXY 5 2 if(isAnyPositive[1]) setMetaPosteriorFP(metaData,posteriorFP[1]) end  
                @ifXY 6 2 if(isAnyPositive[1]) setMetaPosteriorFN(metaData,posteriorFN[1]) end  

                @ifXY 7 2 if(isAnyPositive[1]) setMetaAnteriorFP(metaData,anteriorFP[1]) end  
                @ifXY 8 2 if(isAnyPositive[1]) setMetaAnteriorFN(metaData,anteriorFN[1]) end  

                @ifXY 9 2 if(isAnyPositive[1]) setMetaTopFP(metaData,topFP[1]) end  
                @ifXY 10 2 if(isAnyPositive[1]) setMetaTopFN(metaData,topFN[1]) end  

                @ifXY 11 2 if(isAnyPositive[1]) setMetaBottomFP(metaData,bottomFP[1]) end  
                @ifXY 12 2 if(isAnyPositive[1])  setMetaBottomFN(metaData,bottomFN[1]) end   


                @ifXY 1 3 if(isAnyPositive[1]) setMetaDataTotalFpCount(metaData,shmemSum[1,2], xOuter,yOuter,zOuter) end   
                @ifXY 2 3 if(isAnyPositive[1]) setMetaDataTotalFnCount(metaData,shmemSum[1,1], xOuter,yOuter,zOuter) end
                #now in order to get total count  of fp and fn per block  we will do this subtraction              

                @ifXY 3 3 if(isAnyPositive[1]) setMetaDataMainFpCount(metaData,shmemSum[1,2] -leftFP[1] - rightFP[1]- posteriorFP[1] -anteriorFP[1]- topFP[1] -bottomFP[1] , xOuter,yOuter,zOuter) end   
                @ifXY 4 3 if(isAnyPositive[1]) setMetaDataMainFnCount(metaData,shmemSum[1,1] -leftFN[1] - rightFN[1] -posteriorFN[1]- anteriorFN[1]- topFN[1] -bottomFN[1]   , xOuter,yOuter,zOuter) end
                @ifXY 4 4 setMetaDataXYZ(metaData, xOuter,yOuter,zOuter  )

                #adding the total fp and fn of a block to total result per thread block - later it will be pushed as total amiunt of fp and fn 
                @ifXY 5 3 interMediateFPFN[1] +=shmemSum[1,1] 
                @ifXY 6 3 interMediateFPFN[2] +=shmemSum[1,2] 
                sync_threads()

                #cleaning
                clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
                locArr= (Float32(0.0), Float32(0.0))

                #set the x,y,z coordinates - so we will able to query it efficiently also with linear index
                #what is important later as we will use only part of meta data this indicies will need to be updated

     end) #outer loop        
                #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 
                
        # outer loop expession  )

    #in order to have global data 

    @ifXY 1 1 atomicMinSet(minxRes[],minX[1])
    @ifXY 2 1 atomicMaxSet(maxxRes[],maxX[1])

    @ifXY 3 1 atomicMinSet(minyRes[],minY[1])
    @ifXY 4 1 atomicMaxSet(maxyRes[],maxY[1])

    @ifXY 5 1 atomicMinSet(minzRes[],minY[1])
    @ifXY 6 1 atomicMaxSet(maxzRes[],maxZ[1])

    @ifXY 7 1 atomicAdd(fn[],interMediateFPFN[2])
    @ifXY 8 1 atomicAdd(fp[],interMediateFPFN[1])
   return  
   end





   """
   atomically increment by one fp if true otherwise increment fn by 1
   """
   function incrementAtomFPifTrue(bool,fp,fn)
        if(bool)
            @atomic @inbounds fp[]+=1
        end        
        @atomic @inbounds fn[]+=1

   end