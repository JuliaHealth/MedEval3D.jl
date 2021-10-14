


"""
For first metadata pass we need already cut the area o boolean arrays that we are intrested in and portion of metadata array that we are intrested in 
this way indexing will be simpler we will start from 0 and we will reduce memor usage


3) first metadata pass we add to the metadata offset where each block will put its main results and padding results - so all will be stored in result quueue but in diffrent spots
    we need to make ques for paddings longer than number of possible results becouse of possible modifications from neighbouring blocks that can happen simultanously
    we will establish this offsets using atomics- at this pass we will also prepare first work queue with indicies of metadata blocks and booleans indicating is it related to gold pass dilatation step or other pass

5) we do the metadata pass we analyze only those blocks that are in the borders of intrests - max min x y z was specified in step 1 , we check weather block is set to be activated
 and in case it is not full we will make it active , if the block is set as active we just add it to work queue  that is appropriate to next iteration
  - we need also to scan  the border result ques if there is any duplicate result - if so we set it to zeros and we reduce the border result counter
        of course we check it only in case new counter is bigger than old counter 


        so 1) we check metadata when metadata comply with our predicate described above we atomically increase local shared memory work queue counter
we synchronize
        -we proceed if local workqueue counter is greater than 0 
        2)  we add the local workqueue counter to global keep old value as an offset in shared memory 
        3) sync threads,  add to the work queue the data from registers of those threads that met the predicate
"""

module  MetadataAnalyzePass     


"""
we analyze metadate as described above 
minX, minY,minZ - minimal indexes of metadata that holds all of the data that is of intrest to us
maxX,maxY,maxZ  - maximal indexes of metadata that holds all of the data that is of intrest to us
metaData - global memory data structure that we analyze
"""
macro analyzeMetadataFirstPass(minX, minY,minZ, maxX,maxY,maxZ, metaData )
        # we need to iterate over all metadata blocks with checks so the blocks can not be full outside the area of intrest defined by  minX, minY,minZ and maxX,maxY,maxZ
        @metaDataWarpIter  begin
            #now we upload all data related to amount of data that is of our intrest 
            @ifXY 1 idY locArr= getMetaLeftFP(metadat)
            @ifXY 2 idY locArr= getMetaLeftFN(metadat)
            @ifXY 3 idY locArr=  getMetaRightFP(metadat)
            @ifXY 4 idY locArr=  getMetaRightFN(metadat)
            @ifXY 5 idY locArr=  getMetaPosteriorFP(metadat)
            @ifXY 6 idY locArr=  getMetaPosteriorFN(metadat)
            @ifXY 7 idY locArr=  getMetaAnteriorFP(metadat)
            @ifXY 8 idY locArr=   getMetaAnteriorFN(metadat)
            @ifXY 9 idY locArr=  getMetaTopFP(metadat)
            @ifXY 10 idY locArr=  getMetaTopFN(metadat)
            @ifXY 11 idY locArr=  getMetaBottomFP(metadat)
            @ifXY 12 idY locArr= getMetaBottomFN(metadat)
            
            #now we will calculate and writ

         
         #gets  count of fp, fn in main part
         
         getMetaDataMainFpCount
         getMetaDataMainFnCount
            
         #now we want to set offsets        
    
    
         #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 
                
        end# outer loop expession  )
        
        
end      

"""
this will enable iteration of metadata blocks where each warp will be responsible for single block
we will use the linear indexing in order to     
"""

macro metaDataWarpIter()
    linIndex - linear index
    idY -  thread id of warp that is responsible for this iteration
    metadat= metaData[x]
end





HFUtils.clearMainShmem(resShmem)
        # first we check weather next block is viable for processing
        @unroll for zIter in 1:6
 
          ----------- what is crucial those actions will be happening on diffrent threads hence when we will reduce it we will know results from all        
     
            #we will iterate over all padding planes below way to calculate the next block in all dimensions not counting oblique directions
            @ifXY 1 zIter isMaskOkForProcessing = ((currBlockX+UInt8(zIter==1)-UInt8(zIter==2))>0)
            @ifXY 2 zIter @inbounds isMaskOkForProcessing = (currBlockX+UInt8(zIter==1)-UInt8(zIter==2))<=metadataDims[1]
            @ifXY 3 zIter @inbounds isMaskOkForProcessing = (currBlockY+UInt8(zIter==3)-UInt8(zIter==4))>0
            @ifXY 4 zIter @inbounds isMaskOkForProcessing = (currBlockY+UInt8(zIter==3)-UInt8(zIter==4))<=metadataDims[2]
            @ifXY 5 zIter @inbounds isMaskOkForProcessing = (currBlockZ+UInt8(zIter==5)-UInt8(zIter==6))>0
            @ifXY 6 zIter @inbounds isMaskOkForProcessing = (currBlockZ+UInt8(zIter==5)-UInt8(zIter==6))<=metadataDims[3]
            @ifXY 7 zIter @inbounds isMaskOkForProcessing = !metaData[currBlockX+UInt8(zIter==1)-UInt8(zIter==2)
                                                            ,(currBlockY+UInt8(zIter==3)-UInt8(zIter==4))
                                                            ,(currBlockZ+UInt8(zIter==5)-UInt8(zIter==6)),isPassGold+3]#then we need to check weather mask is already full - in this case we can not activate it 
            #now we check are all true 
                 ----------- this can be done by one of the reduction macros    

           offset = UInt8(1)
            @ifY zIter begin 
                while(offset <UInt8(8)) 
                    @inbounds isMaskOkForProcessing =  isMaskOkForProcessing & shfl_down_sync(FULL_MASK, isMaskOkForProcessing, offset)
                    offset<<= 1
                end #while
            end# @ifY 
        #here is the information wheather we want to process next block
        @ifXY 1 zIter @inbounds resShmem[2,zIter+1,2] = isMaskOkForProcessing
         end#for zIter   
                
         sync_threads()#now we should know wheather we are intrested in blocks around
       
   
            
            
        # ################################################################################################################################ 
        #checking is there anything in the padding plane - so we basically do (most of reductions)
        #values stroing in local registers is there anything in padding associated # becouse we will store it in one int we can pass it at one move of registers
        locArr=0 #reset for reuse
               ----------- this was created for cubic 32x32x32 block where one plane of threads can analyze all paddings 
                   ----------- as in variable size thread blocks some of threads when processing padding will have nothing to do we can think so it will work in this time on the  isMaskForProcessing from above
        locArr|= resShmem[ 34 ,threadIdxX() , threadIdxY() ] << 1 #RIGHT
        locArr|= resShmem[1 ,threadIdxX() , threadIdxY()] << 2 #LEFT
        locArr|= resShmem[threadIdxX() ,34 ,threadIdxY() ] << 3 #ANTERIOR
        locArr|=  resShmem[ threadIdxX(),1 , threadIdxY()] << 4 #POSTERIOR
        locArr|= resShmem[ threadIdxX() , threadIdxY() ,1] << 5 #TOP
        locArr|= resShmem[ threadIdxX() , threadIdxY() ,34] << 6 #BOTTOM

   ----------- this reduction can be done probably together with reduction from step above        
                #we need to reduce now  the values  of padding vals to establish weather there is any true there if yes we put the neighbour block to be active 
                    #reduction                   
                    offset = UInt8(1)
                    while(offset <32) 
                        #we load first value from nearby thread 
                        shuffled = shfl_down_sync(FULL_MASK, locArr, offset)
                        #we loop over  bits and updating we are intrested weather there is any positive so we use or
                        @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
                            locArr|= @inbounds ((shuffled>>zIter & 1) | @inbounds  (locArr>>zIter & 1) ) <<zIter
                        end#for    
                        #isMaskOkForProcessing = (isMaskOkForProcessing | 
                        offset<<= 1
                    end

                    @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
                        @ifX 1  resShmem[zIter+1,threadIdxY()+1,3]=  @inbounds  (locArr>>zIter & 1)
                        #@ifX 1 CUDA.@cuprint " resShmem[zIter+1,threadIdxX()+1,3]   $(resShmem[zIter+1,threadIdxX()+1,3] )   locArr $(locArr) \n" 
                    end#for  
                             
             sync_threads()#now we have partially reduced values marking wheather we have any true in padding         
                  #  # we get full reductions
            @unroll for zIter::UInt8 in UInt8(1):UInt8(6)
                if(resShmem[2,zIter+1,2] )
                offset = UInt8(1)
                if(UInt8(threadIdxY())==zIter)
                    while(offset <32)                        
                        @inbounds  resShmem[zIter+1,threadIdxX()+1,3] = (resShmem[zIter+1,threadIdxX()+1,3] | shfl_down_sync(FULL_MASK,resShmem[zIter+1,threadIdxX()+1,3], offset))
                        offset<<= 1
                    end#while
                end#if    
                end#if                          
            end#for

            sync_threads()#now we have fully reduced in resShmem[zIter+1,1+1,3]= resShmem[zIter+1,2,3]
    
                
                
                
                
                    #updating metadata
    if(resShmem[2,primaryZiter+1,2] && resShmem[primaryZiter+1,2,3] )   
        @ifXY 2 primaryZiter @inbounds  metaData[(currBlockX+(primaryZiter==1)-(primaryZiter==2)),(currBlockY+(primaryZiter==3)-(primaryZiter==4)),(currBlockZ+(primaryZiter==5)-(primaryZiter==6)),isPassGold+1]= true
    end#if
    sync_warp()


    
    
  end #MetadataAnalyzePass
