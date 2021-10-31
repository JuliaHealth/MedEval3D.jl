
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Shuffle,Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates


threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
metaData =view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:4,1:4,1:4,:);
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
iterThrougWarNumb = cld(14,threads[2])
resShmem = CuArray(falses(datBdim[1]+2, datBdim[2]+2, datBdim[3]+2 ))
totalFp,totalFn = 100000,100000
resList,resListIndicies= allocateResultLists(totalFp,totalFn)


loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )
numbOfFp = 0
numbOfFn = 0
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



#setting number of fp ... so it should be validated as fit to be validated
offset = -49
   for quueueNumb in 1:14
         offset+=350
         #... so it should be validated as fit to be validated
           metaData[2,2,2,getBeginingOfFpFNcounts()+quueueNumb]+=4 # so all will remain active
           metaData[2,2,2,getNewCountersBeg()+quueueNumb]+=1 
         # so it should not be validated
         metaData[3,3,3,getBeginingOfFpFNcounts()+quueueNumb]+=1 # so all will be inactive
         metaData[3,3,3,getNewCountersBeg()+quueueNumb]+=1  

         if(isodd(quueueNumb)) 
            numbOfFp+=5
           else
            numbOfFn+=5
           end
end#for   



#we are simulating some results in some of the result queues
offset = -49
   for quueueNumb in 1:14
         offset+=350
      for j in 1:(quueueNumb*10)
            # if(quueueNumb ==2 )
            #    println("list $([innerJ+1,innerJ+2,innerJ+3,1,1,1]) ind $(offset+j+innerJ*quueueNumb)   ")
            # end  
        #the bigger the number the more repetitions and more non repeated elements 
           resList[offset+(j+1),:]= [mod(j,quueueNumb*5)+1,mod(j,quueueNumb*5)+2,mod(j,quueueNumb*5)+3,1,1,1]
           if(isodd(quueueNumb)) 
            numbOfFp+=1
           else
            numbOfFn+=1
           end 
           metaData[1,1,1,getNewCountersBeg()+quueueNumb]+=1 
           metaData[1,1,1,getBeginingOfFpFNcounts()+quueueNumb]+=2 # so all will remain active
   end#for
end#for   
# numbOfFp = 0
# numbOfFn = 0

#should be first queue in block 3,3,3
resList[127701+1,:] = [1,1,1,1,1,1]
resList[127701+2,:] = [1,1,1,0,1,1]
resList[127701+3,:] = [1,1,1,1,1,1]#repeat
resList[127701+4,:] = [1,2,1,1,1,1]
resList[127701+5,:] = [1,1,2,1,1,1]
resList[127701+6,:] = [2,1,1,1,1,1]

metaData[3,3,3,getNewCountersBeg()+1]+=6 
numbOfFp+=5
####### check is test written well (testing the test)
@test Int64(metaData[3,3,3,getNewCountersBeg()+1]) ==7

   
   #setting res list indicies
   for i in 1:(totalFp+totalFn)
      if(resList[i,1]!=0)
         resListIndicies[i] = UInt32(getResLinIndex(resList[i,1],resList[i,2],resList[i,3],resList[i,4],mainArrDims) )
      end   
   end  
   
 offsetForTwelve= Int64(metaData[1,1,1,getResOffsetsBeg()+12])
 notShuffl= resListIndicies[(offsetForTwelve+1):(offsetForTwelve+40)]= shuffle( resListIndicies[(offsetForTwelve+1):(offsetForTwelve+40)])

 offset = -49
for quueueNumb in 1:14
      offset+=350
     #the bigger the number the more repetitions and more non repeated elements 
      @test Int64.(metaData[1,1,1,getNewCountersBeg()+quueueNumb])==quueueNumb*10
      #we test  entries 
      tempList= resListIndicies[offset:offset+350]
      # println("**************************  $(quueueNumb)")
      # println(Int64.(filter(el->el>0,tempList)))
      # println(Int64.(unique(tempList)))

      @test length(filter(el->el>0,tempList))==quueueNumb*10        
      @test length(unique(tempList))==quueueNumb*5+1   

      
end#for quueueNumb
maxResListIndex= length(resListIndicies)

globalCurrentFnCount,globalCurrentFpCount= CUDA.zeros(UInt32,1),CUDA.zeros(UInt32,1)

function loadAndSanForDuplKernel(globalCurrentFnCount,globalCurrentFpCount,maxResListIndex,resListIndicies,metaData,iterThrougWarNumb,mainArrDims ,metaDataDims,loopXMeta,loopYZMeta,resList,datBdim)
   locArr= UInt32(0)
   localOffset= UInt32(0)
   offsetIter= UInt16(0)
   shmemSum =  @cuStaticSharedMem(UInt32,(36,14)) # we need this additional spots
   resShmem =  @cuDynamicSharedMem(Bool,(datBdim[1]+2,datBdim[2]+2,datBdim[3]+2)) # we need this additional 33th an 34th spots
   sourceShmem =  @cuDynamicSharedMem(Bool,(datBdim[1],datBdim[2],datBdim[3]))
   alreadyCoveredInQueues =@cuStaticSharedMem(UInt32,(15))

   for i in 1:30
     resShmem[(threadIdxX())+(i)*33]= false
   end
   for i in 1:15
      sourceShmem[(threadIdxX())+(i)*33]= false
   end
   for i in 1:14
      shmemSum[(threadIdxX()),i]= false
   end

   MetadataAnalyzePass.@metaDataWarpIter( metaDataDims,loopXMeta,loopYZMeta,
       begin

        # @exOnWarp 16 CUDA.@cuprint "idX $(threadIdxX())  xMeta $(xMeta) yMeta $(yMeta+1) zMeta $(zMeta+1)  \n "

        @loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter,localOffset)  
sync_threads()
for i in 1:30
   resShmem[(threadIdxX())+(i)*33]= false
 end
 for i in 1:15
    sourceShmem[(threadIdxX())+(i)*33]= false
 end
 for i in 1:14
    shmemSum[(threadIdxX()),i]= false
 end
 sync_threads()
       end)

       sync_threads()
       @ifXY 1 1 begin 
         if(blockIdxX()==1)
         summA=0
         suumB=0
         for i in 1:12
            summA+=alreadyCoveredInQueues[i]
            suumB+=i*5
         end   
         for i in 13:14
            summA+=alreadyCoveredInQueues[i]
            suumB+=i*10
         end  
         CUDA.@cuprint " alreadyCoveredInQueues[1] $(alreadyCoveredInQueues[1])   should be $(1*5)\n "
         CUDA.@cuprint " alreadyCoveredInQueues[2] $(alreadyCoveredInQueues[2]) should be $(2*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[3] $(alreadyCoveredInQueues[3]) should be $(3*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[4] $(alreadyCoveredInQueues[4]) should be $(4*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[5] $(alreadyCoveredInQueues[5]) should be $(5*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[6] $(alreadyCoveredInQueues[6]) should be $(6*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[7] $(alreadyCoveredInQueues[7]) should be $(7*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[8] $(alreadyCoveredInQueues[8]) should be $(8*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[9] $(alreadyCoveredInQueues[9]) should be $(9*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[10] $(alreadyCoveredInQueues[10]) should be $(10*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[11] $(alreadyCoveredInQueues[11]) should be $(11*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[12] $(alreadyCoveredInQueues[12]) should be $(12*5) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[13] $(alreadyCoveredInQueues[13]) should be $(13*10) \n "
         CUDA.@cuprint " alreadyCoveredInQueues[14] $(alreadyCoveredInQueues[14]) should be $(14*10) \n "
         CUDA.@cuprint "summA  $(summA) should be  $(suumB) \n "
   
      end 
         #   CUDA.@cuprint """  valuee fp $(alreadyCoveredInQueues[1]+ alreadyCoveredInQueues[3]+ alreadyCoveredInQueues[5]+ alreadyCoveredInQueues[7]+ alreadyCoveredInQueues[9]+ alreadyCoveredInQueues[11]+ alreadyCoveredInQueues[13]) 
            #   alreadyCoveredInQueues[1] $(alreadyCoveredInQueues[1]) alreadyCoveredInQueues[3] $(alreadyCoveredInQueues[3]) alreadyCoveredInQueues[5] $(alreadyCoveredInQueues[5]) alreadyCoveredInQueues[7] $(alreadyCoveredInQueues[7]) alreadyCoveredInQueues[9] $(alreadyCoveredInQueues[9]) alreadyCoveredInQueues[11] $(alreadyCoveredInQueues[11]) alreadyCoveredInQueues[13] $(alreadyCoveredInQueues[13]) 
              
            #   """
          atomicAdd(globalCurrentFpCount, alreadyCoveredInQueues[1]+ alreadyCoveredInQueues[3]+ alreadyCoveredInQueues[5]+ alreadyCoveredInQueues[7]+ alreadyCoveredInQueues[9]+ alreadyCoveredInQueues[11]+ alreadyCoveredInQueues[13]) 
       end
          @ifXY 2 1 atomicAdd(globalCurrentFnCount, alreadyCoveredInQueues[2]+ alreadyCoveredInQueues[4]+ alreadyCoveredInQueues[6]+ alreadyCoveredInQueues[8]+ alreadyCoveredInQueues[10]+ alreadyCoveredInQueues[12]+ alreadyCoveredInQueues[14]) 
       
       sync_threads()
       



   return
end

@cuda threads=threads blocks=blocks loadAndSanForDuplKernel(globalCurrentFnCount,globalCurrentFpCount,maxResListIndex,resListIndicies,metaData,iterThrougWarNumb,mainArrDims ,metaDataDims,loopXMeta,loopYZMeta,resList,datBdim)
@test Int64(metaData[3,3,3,getNewCountersBeg()+1]) ==6


@test Int64(globalCurrentFnCount[1]) ==numbOfFn
@test Int64(globalCurrentFpCount[1]) ==numbOfFp

Int64(globalCurrentFnCount[1]+globalCurrentFpCount[1])
offset = -49
   # for quueueNumb in 1:14
   for quueueNumb in 1:12
         offset+=350
        #the bigger the number the more repetitions and more non repeated elements 
         @test metaData[1,1,1,getNewCountersBeg()+quueueNumb]==quueueNumb*5#j instead of quueueNumb*quueueNumb
         tempList= Array(resListIndicies[offset:offset+350])
         @test length(filter(el->el>0,tempList))==quueueNumb*5#+1 becouse of 0's entry
  end#for

    quueueNumb =12
   offset = -49
   offset+=350*(quueueNumb)
   @test Int64(metaData[1,1,1,getNewCountersBeg()+quueueNumb])==quueueNumb*5#j instead of quueueNumb*quueueNumb
   tempList= Array(resListIndicies)[offset:offset+350];
   @test length(filter(el->el>0,tempList))==quueueNumb*5#+1 becouse of 0's entry
   


   quueueNumb =2
   offset = -49
   offset+=350*(quueueNumb)
   @test Int64(metaData[1,1,1,getNewCountersBeg()+quueueNumb])==quueueNumb*5#j instead of quueueNumb*quueueNumb
   tempList= Array(resListIndicies)[offset:offset+350];
   @test length(filter(el->el>0,tempList))==quueueNumb*5#+1 becouse of 0's entry
   Int64.(filter(el->el>0,tempList))
   Int64.(unique(filter(el->el>0,tempList)))

  ############### testing is to be analyzed
  #main block
  @test metaData[1,1,1,getIsToBeAnalyzedNumb()+15 ]==1
  @test metaData[1,1,1,getIsToBeAnalyzedNumb()+16 ]==1
  #right
  @test metaData[2,1,1,getIsToBeAnalyzedNumb()+3 ]==1
  @test metaData[2,1,1,getIsToBeAnalyzedNumb()+4 ]==1
  #bottom
  @test metaData[1,1,2,getIsToBeAnalyzedNumb()+11 ]==1
  @test metaData[1,1,2,getIsToBeAnalyzedNumb()+12 ]==1
  #anterior
  @test metaData[1,2,1,getIsToBeAnalyzedNumb()+7 ]==1
  @test metaData[1,2,1,getIsToBeAnalyzedNumb()+8 ]==1

  ### for block  2,2,2 all queaueas are active  and should activate surrounding blocks

  #main block
  metaX = 2
  metaY = 2
  metaZ = 2

  @test metaData[metaX,metaY,metaZ,getIsToBeAnalyzedNumb()+15 ]==1
  @test metaData[metaX,metaY,metaZ,getIsToBeAnalyzedNumb()+16 ]==1

  @test metaData[metaX-1,metaY,metaZ,getIsToBeAnalyzedNumb()+1 ]==1
  @test metaData[metaX-1,metaY,metaZ,getIsToBeAnalyzedNumb()+2 ]==1
  @test metaData[metaX+1,metaY,metaZ,getIsToBeAnalyzedNumb()+3 ]==1
  @test metaData[metaX+1,metaY,metaZ,getIsToBeAnalyzedNumb()+4 ]==1
  @test metaData[metaX,metaY-1,metaZ,getIsToBeAnalyzedNumb()+5 ]==1
  @test metaData[metaX,metaY-1,metaZ,getIsToBeAnalyzedNumb()+6 ]==1
  @test metaData[metaX,metaY+1,metaZ,getIsToBeAnalyzedNumb()+7 ]==1
  @test metaData[metaX,metaY+1,metaZ,getIsToBeAnalyzedNumb()+8 ]==1
  @test metaData[metaX,metaY,metaZ-1,getIsToBeAnalyzedNumb()+9 ]==1
  @test metaData[metaX,metaY,metaZ-1,getIsToBeAnalyzedNumb()+10 ]==1
  @test metaData[metaX,metaY,metaZ+1,getIsToBeAnalyzedNumb()+11 ]==1
  @test metaData[metaX,metaY,metaZ+1,getIsToBeAnalyzedNumb()+12 ]==1




  metaX = 3
  metaY = 3
  metaZ = 3

  @test metaData[metaX,metaY,metaZ,getIsToBeAnalyzedNumb()+15 ]==1
  @test metaData[metaX,metaY,metaZ,getIsToBeAnalyzedNumb()+16 ]==0

  @test metaData[metaX-1,metaY,metaZ,getIsToBeAnalyzedNumb()+1 ]==1
  @test metaData[metaX-1,metaY,metaZ,getIsToBeAnalyzedNumb()+2 ]==0
  @test metaData[metaX+1,metaY,metaZ,getIsToBeAnalyzedNumb()+3 ]==0
  @test metaData[metaX+1,metaY,metaZ,getIsToBeAnalyzedNumb()+4 ]==0
  @test metaData[metaX,metaY-1,metaZ,getIsToBeAnalyzedNumb()+5 ]==0
  @test metaData[metaX,metaY-1,metaZ,getIsToBeAnalyzedNumb()+6 ]==0
  @test metaData[metaX,metaY+1,metaZ,getIsToBeAnalyzedNumb()+7 ]==0
  @test metaData[metaX,metaY+1,metaZ,getIsToBeAnalyzedNumb()+8 ]==0
  @test metaData[metaX,metaY,metaZ-1,getIsToBeAnalyzedNumb()+9 ]==0
  @test metaData[metaX,metaY,metaZ-1,getIsToBeAnalyzedNumb()+10 ]==0
  @test metaData[metaX,metaY,metaZ+1,getIsToBeAnalyzedNumb()+11 ]==0
  @test metaData[metaX,metaY,metaZ+1,getIsToBeAnalyzedNumb()+12 ]==0

  Int64(resListIndicies[662])
  Int64(resListIndicies[307])
  Int64(resListIndicies[1372])
  Int64(resListIndicies[1727])
  Int64(resListIndicies[2437])



  
  ### for block 3,3,3 only queaue 1 should be active  and rest should not activate surrounding blocks






#mixed entries ...



#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  4  innerWarpNumb 1    idX 1  idY 1  xMeta 1 yMeta 0 zMeta 0 
#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  9  innerWarpNumb 2    idX 1  idY 2  xMeta 1 yMeta 0 zMeta 0
#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  14  innerWarpNumb 3    idX 1  idY 3  xMeta 1 yMeta 0 zMeta 0
#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  47  innerWarpNumb 4    idX 1  idY 4  xMeta 1 yMeta 0 zMeta 0
#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  177  innerWarpNumb 5    idX 1  idY 5  xMeta 1 yMeta 0 zMeta 0
#    zzzzzzzzzzz    shmemSum[36,innerWarpNumb]  1  innerWarpNumb 1    idX 3  idY 1  xMeta 3 yMeta 2 zMeta 2


#    offset = -49+(350*8)
#    Int64.(resList[offset:offset+10,:])



#    indd 1102501  xMeta 1 yMeta 1 zMeta 1 innerWarpNumb 2 tempCount 656
#    indd 1102501  xMeta 1 yMeta 1 zMeta 1 innerWarpNumb 2 tempCount 653

#    Int64.(unique(resListIndicies[offset:offset+350]))
#    Int64(resListIndicies[656 ]) #0
#    Int64(resListIndicies[653 ]) #223716...
#    Int64(resListIndicies[652 ]) #223716...
#    Int64(metaData[1102501]) #4


#    list [1, 2, 3, 1, 1, 1] ind 652   
#    list [2, 3, 4, 1, 1, 1] ind 654
#    list [1, 2, 3, 1, 1, 1] ind 653
#    list [2, 3, 4, 1, 1, 1] ind 655




   

# Int64(resListIndicies[127701] ) 
# Int64(resListIndicies[127701+1] ) 
# Int64(resListIndicies[127701+2] ) 
# Int64(resListIndicies[127701+3] ) 
# Int64(resListIndicies[127701+4] ) 
# Int64(resListIndicies[127701+5] ) 

# Int64(resListIndicies[127701] ) ==223181869
# Int64(resListIndicies[127701+1] ) ==223181869
# Int64(resListIndicies[127701+2] ) ==223181869
# Int64(resListIndicies[127701+3] ) ==223181869
# Int64(resListIndicies[127701+4] ) ==223181869
# Int64(resListIndicies[127701+5] ) ==223181869

# 223451221

# 223181869 -223181869


# for i in  0:0
#    println("111")
# end

# using BenchmarkTools
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
# BenchmarkTools.DEFAULT_PARAMETERS.seconds =60
# BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

# function toBench()
#    CUDA.@sync @cuda threads=threads blocks=blocks loadAndSanForDuplKernel(resListIndicies,metaData,iterThrougWarNumb,mainArrDims ,metaDataDims,loopXMeta,loopYZMeta,resList,datBdim)
# end
# bb2 = @benchmark toBench()

