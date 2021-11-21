
"""
at this point we already have res list filled with all fp and fn's in some cases it is enough in others we want futher actions to be done 
  - for example in case of lack of isometry in voxels we can now apply some corrections for it 
"""
module ResultProcess

"""
it will scan through result list - ignorea all that are zeroces in res indicies, add only the iteration numbers and in the end we will just dividy it by (fp+fn)
entriesPerBlock - amount of entries per block of threads 
totalLength - total length of result list
iterLoopResList - how many times given block need to iterate to aalyze all entries given to it 
globalSum - global variable accessed atomically holding sum of iter numbs of results 
"""
function getAverage(resList,resListIndicies,entriesPerBlock,totalLength,iterLoopResList ,globalSum )
 locSum = UInt32(0) 
 shmemSum = @cuStaticSharedMem(UInt32, (33,1))  
 offsetIter= UInt8(1)

  @iterateLinearlyForResList(iterLoopResList,entriesPerBlock,totalLength,begin 
      if(resListIndicies[i]>0)# we do not want 0 entries or duplicated ones  
        locSum+=resList[i,6]#adding iteration number
      end  
  end )
    
  #time for reduction 
  @redWitAct(offsetIter,shmemSum, locSum,+   )
  sync_threads()
  @addAtomic(shmemSum, globalSum)
end #getAverage


"""
iterate linearly result list 
entriesPerBlock - amount of entries per block of threads 
totalLength - total length of result list
"""
macro iterateLinearlyForResList(iterLoop,entriesPerBlock,totalLength, ex)
  return  esc(quote
  i = UInt32(0)
  @unroll for j in 0:($iterLoop)
    offset = ((blockIdxX()-1) *$entriesPerBlock)
    i= threadIdxX()+(threadIdxY()-1)*blockDimX()+ j* blockDimX()*blockDimY()
    if((i+offset)<=$totalLength && i<$entriesPerBlock)
      i+=offset
      $ex
    end
  end 
   end)
  end



"""
We will iterate over result list take all entries that are non zeo in res  indicies list now in a while loop we will lok for the source voxel that was responsible for the  result
so we can start fom naive approach where each lane will be responsible for finding source voxel that is the closest voxel from other mask
In order to achieve it we will go in the direction indicated by the dir value in the distance indicated by the dir in the result matrix  and the number of voxels indicated by the iteration number
It is probable that we will not find there he source voxel so we will need to iteratively increase number of voxels that go in orthogonal direction so for example
  when direction is down and iteration number 10 we will ook 10 voxels blow in straight line at first 
  then if wew ould not find we will look into voxel 9 down and in 1 left,right,anterior,posterior - so over orthogonal axes
  then we will  try going 8 pixels down and 1 pixel in 4 directions and futher pixel also in 4 directions  so 16 permutations 
  basically we will want to scan the area of the sphere with radius of iteration number and getting circle (or sth more like a square/rhombus as we analze 1 norm ...) crossections of it  starting from the point in the end of the axis in given
  direction and progress up circle crossesction by circle crossesction 
so the amount of voxels that we have to use at each iteration will increase in each step hence w have no free voxels to move at first step - then we have 4 possible position then I suppose 16...
so we can do it like that as we are fro example on step 3 and we are circling around with sth like circle around with radius 3 we can first analyze one axis for example left right so we go
3 voxls laeft and right we have a part of line now we get shorter line (by 1 on each end) anteriorly and posteriorly to the first axis then next row even shorter anteriorly and posteriorly ...
until we will reduce row length to 1 - this should lead to check the voxels at the end of anterior posterior ends of anterior posterior axis 

We can xperiment with getting corrections one by single line at the begining with some maximum number of ccles allowed - so if single lane wuld not be able to find result we would cease looking for it
and then scan one more time whole warps  for scanning those points that were not yet resolved 

additionally if we want to create intresting visualization we can add information about corrected distance to all points between source and target voxel..

  entriesPerBlock - amount of entries per block of threads 
  totalLength - total length of result list
  iterLoopResList - how many times given block need to iterate to aalyze all entries given to it 
  globalSum - global variable accessed atomically holding sum of iter numbs of results - so we will accumulate corrected values - to later return average 
  maxSingleThrIterNumb - maximum iteration number that should be analyzed using just a single lane; all points bigger than that will be analyzed used whole warps or maybe blocks
  """
function applyCorrection(resList,resListIndicies,entriesPerBlock,totalLength,iterLoopResList,globalSum,maxSingleThrIterNumb, referenceArrs )
  locSum = UInt32(0) 
  shmemSum = @cuStaticSharedMem(UInt32, (33,1))  
  offsetIter= Int16(1)
  locIterNumb=Int16(0)
  #indicates how coordinates should change relatively to the covered point from result list 
  xChange= Int16(0)
  yChange= Int16(0)
  zChange= Int16(0)

@iterateLinearlyForResList(iterLoopResList,entriesPerBlock,totalLength,begin 
  if(resListIndicies[i]>0)# we do not want to analyze 0 entries or duplicated ones  


  end  
end )
  


end

### looking for the true entry in given array  we iteratively scan area  where the original point from which dilatation started and led to this result that we are currently analyzing

"""
using single thread we iteratively scan area  where the original point from which dilatation started and led to this result that we are currently analyzing
referenceArrs= (reducedGoldB,reducedSegmB)
                top 6 
                bottom 5  
                left 2
                right 1 
                anterior 3
                posterior 4
"""
macro singleThreadScan(resList, i, referenceArrs )

  return  esc(quote

  offsetIter=$resList[$i,6] #storing iteration number
  locIterNumb=0 #will get incemented every time we move centrally in main axis 
  while offsetIter>0
    #first we need to  modify x,y or z depending on the direction given in dir variable of result list 
    dir = $resList[$i,5]
    xChange-= (dir==2)*offsetIter
    xChange+= (dir==1)*offsetIter

    yChange-= (dir==4)*offsetIter
    yChange+= (dir==3)*offsetIter

    zChange-= (dir==6)*offsetIter
    zChange+= (dir==5)*offsetIter

    #now we have established position over main axis - what is left is to scan by modifying orthogonal axes
    
    @unroll for orthoAxAmove in -locIterNumb:locIterNumb#so we need to iterate over one of the 
      @unroll for orthoAxBmove in [-locIterNumb+ CUDA.abs(orthoAxAmove),locIterNumb-CUDA.abs(orthoAxAmove)]
        #isXMain,isYMain - true if x or in second case y is main axis - we define main axis on the basis of dir variable
        isXMain = (dir==2 || dir==1)
        isYMain = (dir==3 || dir==4)

        #(dir!=2 && dir!=1) so x axis is not main axis
        xChange+= (!isXMain)*orthoAxAmove
        #(dir==2 || dir==1)*orthoAxBmove - so if x is main axis - it means that y is not  and it will be axis A
        yChange+=(!isXMain && !isYMain)*orthoAxBmove+ (isXMain)*orthoAxAmove
        #zChange can never affect axis A  will affect axis B if it is not main axis - so when either x or y is main axis
        zChange+=(!isXMain && !isYMain)*orthoAxBmove
        
        #here if we will find that there is true in this position we will assume  that those coordinates are coordinates of
        if(referenceArrs[2-$resList[$i,4]][ $resList[$i,1]+xChange,$resList[$i,2]+yChange,$resList[$i,3]+zChange])
            #now on the basis of new coordinates we need to calculate correction for non isometric voxels
            corrected =getCorrectedDistance(yDimSize,zDimSize, $resList[$i,1]+xChange,$resList[$i,2]+yChange,$resList[$i,3]+zChange)
            locSum+=corrected
            $resList[$i,6]=corrected
        end
    
        #resetting values
    xChange= Int16(0)
    yChange= Int16(0)
    zChange= Int16(0)                                
      end# 
    end#orthoAxAmove
    offsetIter-=1 
    locIterNumb+=1

      
  end#while
end)#quote
end #singleThreadScan


"""
function invoked in case image has non isometric voxels - this ussually is related to the 
  fact that voxel is smaller in z dimension than in other dimensions 
  IMPORTANT !!! the correction is not perfect as in dilatation step the lack of isometry is not taken into account source voxel may be diffrent than one we would find below, yet in most cases approximation should be good
  
  x dimension is generally set as a reference dimension and set to be 1 ; y dimension ussually is the same but in theory do not have to be
  z dimension will be given to this function as units where 1 unit is x dimension of voxel in mm 
    yDimSize,zDimSize - given in scale where 1 is x dimension of the voxel and presumed to be uniform in whole image
    xDisp,yDisp,zDisp - displacement in x,y,z axis from the covered voxel to source voxel
"""
function getCorrectedDistance(yDimSize,zDimSize, xDisp,yDisp,zDisp)
  return xDisp+ yDimSize*yDisp+ zDimSize*zDisp
end

end#ResultProcess
