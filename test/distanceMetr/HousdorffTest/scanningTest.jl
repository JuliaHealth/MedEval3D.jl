


macro loadAndScanForDuplicates()
   @unroll for outerWarpLoop::Uint8 in 0:iterThrougWarNumb     
        innerWarpNumb = (threadidY()+ outerWarpLoop*blockimY()
           #now we will load the diffrence between old and current counter
        if(( innerWarpNumb)<13)
            @ifY innerWarpNumb begin
                #store result in registers
                #store result in registers (we are reusing some variables)
                #old count
                $locArr = getOldCount(numb, mataData,linIndex)
                #diffrence new - old 
                offsetIter= geNewCount(numb, mataData,linIndex)- $locArr
                #queue result offset
                localResOffset = metaData[xMeta,yMeta+1,zMeta+1, getBeginnigOfOffsets()+innerWarpNumb] # tis queue offset
                # enable access to information is it bigger than 0 to all threads in block
                resShmem[threadIdxX()+1,innerWarpNumb+1,3] = offsetIter>0
            end #@ifY
        end#if
      @scanForDuplicates()                 
    end #outerWarpLoop    
end

singleVal = CUDA.zeros(14)

threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:17,2:18,4:10,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
iterThrougWarNumb = cld(blockDimY(),12)
resShmem = CuArray(falses(datBdim[1]+2, datBdim[2]+2, datBdim[3]+2 ))
xMeta = 1
yMeta=0
zMeta=0
resArray=CUDA.zeros(UInt32, 9*250*15,4)
#we set some offsets to make it simple for evaluation we will keeep it  each separated by 50 
offset = -49
for metaX in 1:3, metaY in 1:3, metaZ in 1:3
   for quueueNumb in 1:14
      offset+=350
      #set offset
      metaData[metaX,metaY,metaZ,getResOffsetsBeg()+quueueNumb]=offset
      #set counter
   end#for   
end #for   
#we are simulating some results in some of the result queues
offset = -49
   for quueueNumb in 1:14
         offset+=350
      for j in 1:quueueNumb
          for innerJ in 1:quueueNumb
        #the bigger the number the more repetitions and more non repeated elements 
           resList[offset+j+innerJ*quueueNumb,:]= [innerJ,innerJ+1,innerJ+2,1,1]
           metaData[1,1,1,getNewCountersBeg()+quueueNumb]+=1 
      end#inner j 
   end#for
end#for   

#should be first queue in block 3,3,3
resList[9*14*250] = [1,1,1,1,1]
resList[9*14*250+1] = [1,1,1,0,1]
resList[9*14*250+2] = [1,1,1,1,1]#repeat
resList[9*14*250+3] = [1,2,1,1,1]
resList[9*14*250+4] = [1,1,2,1,1]
resList[9*14*250+5] = [2,1,1,1,1]


metaData[3,3,3,getNewCountersBeg()+1]+=6 

####### check is test written well (testing the test)
@test metaData[3,3,3,getNewCountersBeg()+1] ==6


offset = -49
   for quueueNumb in 1:14
         offset+=350
        #the bigger the number the more repetitions and more non repeated elements 
         @test metaData[1,1,1,getNewCountersBeg()+quueueNumb]==quueueNumb*quueueNumb
   for j in 1:quueueNumb
         #we test first entries from innner loop
         @test resList[offset+j*quueueNumb,:]==[j,j+1,j+2,1,1]
        end#for innerj 
   end#for
end#for   



function loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,xMeta,yMeta,zMeta)
   MetadataAnalyzePass.@loadAndScanForDuplicates()  
    return
end

@cuda threads=threads blocks=blocks loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,xMeta,yMeta,zMeta)

@test metaData[3,3,3,getNewCountersBeg()+1] ==5


offset = -49
   for quueueNumb in 1:14
         offset+=350
        #the bigger the number the more repetitions and more non repeated elements 
         @test metaData[1,1,1,getNewCountersBeg()+quueueNumb]==quueueNumb#j instead of quueueNumb*quueueNumb
         tempList= resList[offset:offset+350,:]
          @test length(unique(tempList))==quueueNumb+1#+1 becouse of 0's entry
   end#for
end#for   



