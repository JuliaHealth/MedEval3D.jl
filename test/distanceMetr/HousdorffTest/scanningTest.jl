


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
resArray=allocateResArray()
#we set some offsets to make it simple for evaluation we will keeep it  each separated by 50 


#we are simulating some results
for i in 1:10
    
end#for    


function loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,xMeta,yMeta,zMeta)
   MetadataAnalyzePass.@loadAndScanForDuplicates()  
    return
end

@cuda threads=threads blocks=blocks loadAndSanForDuplKernel(metaData,iterThrougWarNumb ,resShmem,xMeta,yMeta,zMeta)
@test singleVal[1]==metaDataDims[1]*metaDataDims[2]*metaDataDims[3]



