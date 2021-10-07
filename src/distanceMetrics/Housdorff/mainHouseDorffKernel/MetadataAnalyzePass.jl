"""
here we will analyze the matadata and on this basis establish is block should be acrivated or it should reamin active 
so we got set in each block is it full and also from padding of surrounding blocks is it to be activated
Hence when conditions are met so it is not full and active or not full and to be activated - we set it to active
when we know that block is active it is also needed to add it to work queue - we will do it after each iteration 
        so 1) we check metadata when metadata comply with our predicate described above we atomically increase local shared memory work queue counter
we synchronize
        -we proceed if local workqueue counter is greater than 0 
        2)  we add the local workqueue counter to global keep old value as an offset in shared memory 
        3) sync threads,  add to the work queue the data from registers of those threads that met the predicate
"""

module  MetadataAnalyzePass     


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
