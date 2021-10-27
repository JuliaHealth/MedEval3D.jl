
using Revise, Parameters, Logging, Test
using CUDA
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils
using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates


threads=(32,5)
blocks =8
mainArrDims= (516,523,826)
datBdim = (43,21,17)
metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:4,2:5,3:6,: );
#metaData = view(MetaDataUtils.allocateMetadata(mainArrDims,datBdim),1:9,2:3,4:6,: );
metaDataDims=size(metaData)
iterThrougWarNumb = cld(threads[2],12)
resShmem = CuArray(falses(datBdim[1]+2, datBdim[2]+2, datBdim[3]+2 ))

loopXMeta= fld(metaDataDims[1],threads[1])
loopYZMeta= fld(metaDataDims[2]*metaDataDims[3],blocks )

resList=allocateResultList()
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
          for innerJ in 0:(quueueNumb-1)
        #the bigger the number the more repetitions and more non repeated elements 
           resList[offset+j+innerJ*quueueNumb,:]= [innerJ+1,innerJ+2,innerJ+3,1,1]
           metaData[1,1,1,getNewCountersBeg()+quueueNumb]+=1 
      end#inner j 
   end#for
end#for   

#should be first queue in block 3,3,3
resList[9*14*250,:] = [1,1,1,1,1]
resList[9*14*250+1,:] = [1,1,1,0,1]
resList[9*14*250+2,:] = [1,1,1,1,1]#repeat
resList[9*14*250+3,:] = [1,2,1,1,1]
resList[9*14*250+4,:] = [1,1,2,1,1]
resList[9*14*250+5,:] = [2,1,1,1,1]


metaData[3,3,3,getNewCountersBeg()+1]+=6 

####### check is test written well (testing the test)
@test Int64(metaData[3,3,3,getNewCountersBeg()+1]) ==6

offset = -49
   for quueueNumb in 1:14
         offset+=350
        #the bigger the number the more repetitions and more non repeated elements 
         @test Int64.(metaData[1,1,1,getNewCountersBeg()+quueueNumb])==quueueNumb*quueueNumb
   for j in 1:quueueNumb
         #we test first entries from innner loop
         @test Int64.(resList[offset+j*quueueNumb,:])==[j,j+1,j+2,1,1]
        end#for j 
   end#for quueueNumb


function loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,loopXMeta,loopYZMeta,resList)
   locArr= UInt32(0)
   offsetIter= UInt16(0)
   shmemSum =  @cuStaticSharedMem(Float32,35,16)

   MetadataAnalyzePass.@metaDataWarpIter( metaDataDims,loopXMeta,loopYZMeta,
       begin
   @loadAndScanForDuplicates(iterThrougWarNumb,locArr,offsetIter)  
   for i in 1:15
      resShmem[(threadIdxX())+(i)*33]= false
   end
       end)
   

   return
end

@cuda threads=threads blocks=blocks loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,loopXMeta,loopYZMeta,resList)

@test Int64(metaData[3,3,3,getNewCountersBeg()+1]) ==5


offset = -49
   for quueueNumb in 1:14
         offset+=350
        #the bigger the number the more repetitions and more non repeated elements 
         @test metaData[1,1,1,getNewCountersBeg()+quueueNumb]==quueueNumb#j instead of quueueNumb*quueueNumb
         tempList= resList[offset:offset+350,:]
          @test length(unique(tempList))==quueueNumb+1#+1 becouse of 0's entry
   end#for



