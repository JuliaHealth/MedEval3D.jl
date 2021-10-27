"""
order of queues
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
13)   main block Fp  
14)   main block Fn  

15)   total block Fp  
16)   total block Fn  

In order to simplify the structure all will be represented as the UInt32  even values that are really booleans
in those cases false will be 0 and true 32 ...

"""
module MetaDataUtils
using CUDA
export getIsToBeActivatedInSegmNumb,getIsToBeActivatedInGoldNumb,getFullInSegmNumb,getFullInGoldNumb,setBlockasCurrentlyActiveInGold,setBlockasCurrentlyActiveInSegm,getActiveGoldNumb,getActiveSegmNumb,getResOffsetsBeg,getOldCountersBeg,getNewCountersBeg,getBeginingOfFpFNcounts,getBeginingOfXYZ,setBlockasCurrentlyActiveInGold,setBlockasCurrentlyActiveInSegm


"""
it will be 4 dimensional array - where fourth dimension will store actual data  in UInt32 format 
    -even Bool will be UInt32 for simplicity
1)activeInGold::Bool
2)activeInSegm::Bool
3)fullInGold::Bool
4)fullInSegm::Bool

5)isToBectivatedGold::Bool
6)isToBeActivatedSegm::Bool
7-9)x,y,z coordinates ::UInt32
10-24) isToBeAnalyzed::Bool
25-41) fpFNcounts::UInt32 last 2 will be totals

42-56) resOffsets::UInt32
57-59) totalOffsetBeginingAndEnd UInt32
60-74) oldCounters ::UInt32
75-89) newCounters::UInt32

arrDims - dimensions of the main data array
dataBDims - dimensions of the data block - part of the main array that is to be analyzed by single block
creates empty metadata on the basis of the main array dimensions and data block dimensions
"""
function allocateMetadata(arrDims,dataBDims)
    return CUDA.zeros(UInt32,cld(arrDims[1],dataBDims[1] ) 
            ,cld(arrDims[2],dataBDims[2] )
            ,cld(arrDims[3],dataBDims[3])
            ,90 )
end
"""
value pointing out where we start in 4th dimension counters
it is begining -1 as we will add thread idx to it ...
"""
function getBeginingOfFpFNcounts()::UInt32
    return UInt32(24)
end
"""
value pointing out where we start in 4th dimension x y z position (usfull when we have access only to linear index)
it is begining -1 as we will add thread idx to it ...
"""
function getBeginingOfXYZ()::UInt32
    return UInt32(6)
end


######## simple accessors - kept so we can easily can change metadata organisation if needed
function getActiveGoldNumb()::UInt32 return  1 end
function getActiveSegmNumb()::UInt32 return  2 end

function getFullInGoldNumb()::UInt32 return  3 end
function getFullInSegmNumb()::UInt32 return  4 end

function getIsToBeActivatedInGoldNumb()::UInt32 return  5 end
function getIsToBeActivatedInSegmNumb()::UInt32 return  6 end

function getResOffsetsBeg()::UInt32 return  42 end

function getOldCountersBeg()::UInt32 return  60 end
function getNewCountersBeg()::UInt32 return  75 end



"""
make is block currently active to true for gold standard pass
"""
function setBlockasCurrentlyActiveInGold(metaData, xMeta,yMeta,zMeta)
    metaData[xMeta,yMeta,zMeta,getActiveGoldNumb()]=1
end


"""
sets is active of given block to true for not gold pass
"""
function setBlockasCurrentlyActiveInSegm(metaData, xMeta,yMeta,zMeta)
    metaData[xMeta,yMeta,zMeta,getActiveSegmNumb()]=1
 end
 

"""
given linear index that is telling us about x,y,z location in one number (linIndex) and number that is the position in fourth dimesnsion (locFourthDim)
it will give us back this entryin meta data (metaData) 
for reducing the need of recalculations we will supply also the metaData3dimProd to speed up calculations
    metaDataDims dimensions of the meta data 
        """
function getMetaDataFieldFromLin(metaData,metaDataDims,linIndex,locFourthDim )
    return metaData[]
end


"""
given linear index that is telling us about x,y,z location in one number (linIndex) and number that is the position in fourth dimesnsion (locFourthDim)
it will set  this entry in meta data (metaData) to value (valuee)
"""
function setMetaDataFieldFromLin(metaData,metaDataDims,linIndex,locFourthDim,valuee)

end


    
"""
sets is active of given block to true for gold standard pass
"""
function setBlockToActiveInGold(metaData,linIndex)
    setMetaDataFieldFromLin(metaData,linIndex,1,UInt32(1))
end

    
       
    """
is block set to be activated for gold standard pass
"""
function isBlockToBeActivatedInGold(metaData, linIndex)
   return getMetaDataFieldFromLin(metaData,5,locFourthDim)
end
    
   
    
"""
true if block is full for gold standard pass
"""
function isBlockFullInGold(metaData, linIndex)
  return  getMetaDataFieldFromLin(metaData,linIndex,3)    
end

"""
is block currently Active for gold standard pass
"""
function isBlockCurrentlyActiveInGold(metaData, linIndex)
   getMetaDataFieldFromLin(metaData,1,locFourthDim)    
    end

            
"""
true if block is full for not gold pass
"""
function isBlockFullInSegm(metaData, linIndex)
   getMetaDataFieldFromLin(metaData,linIndex,4)
    
    end
"""
is block set to be activated for not gold pass
"""
function isBlockToBeActivatedInSegm(metaData, linIndex)

   getMetaDataFieldFromLin(metaData,6,locFourthDim)
    
    end
"""
is block currently Active for not gold pass
"""
function isBlockCurrentlyActiveInSegm(metaData, linIndex)
   getMetaDataFieldFromLin(metaData,2,locFourthDim)
    
    end
"""
make is block currently active to true for not gold pass
"""
function setBlockasCurrentlyActiveInSegm(metaData, linIndex)
    setMetaDataFieldFromLin(metaData,2,locFourthDim,UInt32(1))
    
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
checks is the total counter for the given result queue is smaller than the total amount of fp or fn for given list 
and sets appropriate variable isTpBeAnalyzed 
Important!!! we need to take into account corners so if we look from the top we only care of the top process  
    or we also needs tops of the sides and anterior posterior paddings? - this needs to be adressed
"""
macro setIstoBeAnalyzed(numb, mataData,linIndex,resShmem) 

krowa

   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
#in order to be able to skip some of the validations we will load now informations about this block and neighbouring blocks 
#like for example are there any futher results to be written in this block including border queues
#and is there sth in border queues of the neighbouring data blocks
basically give back the booleans that were calculated in setIstoBeAnalyzed
we will store the results in a corner of res shmem
"""
macro getIstoBeAnalyzed(resShmem,metaData,linIndex,isGold)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end


"""
get data is to be analyzed that was set in setIstoBeAnalyzed
and loaded into res shmem in getIstoBeAnalyzed
this function only return boolean from resshmem that tell us about wheather it makes sense to analyze the data inside
    the data block - so it cares about total data block fp or fn not yet covered
"""
function getIsTotalFPorFNnotYetCovered(resshmem ,isGold)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end


"""
We just access data from res shmem that was loaded into resshmem in getIstoBeAnalyzed
dim can be 1,2,3 and tell in what dimension is the plane of our analysis
            isStart may be either 1 or end of data block   - on this basis we can establish which padding is of our intrest ...
"""
function isPaddingToBeAnalyzed(resShmem,dim,isStart )

   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
sets values for amount of fp and fn in all of the ques specified above
"""
function setMetaFPandFN(numb, mataData,linIndex,valuee)
    
    setMetaDataFieldFromLin(metaData,linIndex,numb+24,valuee)
    
    end

#set the x,y,z coordinates - so we will able to query it efficiently also with linear index
#what is important later as we will use only part of meta data this indicies will need to be updated
function setMetaDataXYZ(metaData, xOuter,yOuter,zOuter  )
    setMetaDataFieldFromLin(metaData,linIndex,7,UInt32(xOuter))
    setMetaDataFieldFromLin(metaData,linIndex,8,UInt32(yOuter))
    setMetaDataFieldFromLin(metaData,linIndex,9,UInt32(zOuter))

    end
                
                
                
                
"""
reduce the values of selected metadata block by the supplied values
"""
function reduceMetaDataXYZ(metaData, minX,minY,minZ, linIndex  )
                 #   krowa ...
    setMetaDataFieldFromLin(metaData,linIndex,7,UInt32(xOuter))
    setMetaDataFieldFromLin(metaData,linIndex,8,UInt32(yOuter))
    setMetaDataFieldFromLin(metaData,linIndex,9,UInt32(zOuter))
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
13)   Data Main Fp  
14)   Data Main Fn  

accordin numb will invoke function that will give the number of fp and fn in given border or main part
    numb - will  tell what value as seen above we want
    mataData - metadata of data blocks
    linIndex- linear index of given block in metadata

    IMPORTANT if we will have numb above 14 we should just return 0 ///

"""
function getMetaResFPOrFNcount(numb, mataData,linIndex )
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  

accordin numb will invoke function that will give the result offset associated with values above
    numb - will  tell what value as seen above we want
    mataData - metadata of data blocks
    linIndex- linear index of given block in metadata

"""
function getMetaResOffsets(numb, mataData,linIndex )
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  


accordin numb will invoke function that will set  the result offset associated with values above
    numb - will  tell what value as seen above we want
    mataData - metadata of data blocks
    linIndex- linear index of given block in metadata
    value - value to et

"""
function setMetaResOffsets(numb, mataData,linIndex,value )
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  
get old result counters for the result lsts as seen above 
"""
function getOldCount(numb, mataData,linIndex)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  
get new counter value of result queues as specified above
"""
function getNewCount(numb, mataData,linIndex)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  

will return the diffrence between current counter value and old one for the result set lists associated with 
    quantities described just above
"""
function getCounterDiffrence(numb, mataData,linIndex)

   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
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
13)   Data Main Fp  
14)   Data Main Fn  

decrement value of a associated counter (look above ) by 1 
"""
function decrCounterByOne(numb, mataData,linIndex)

   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end
"""
on the basis of linear index we will find the  metadata x,y,Z
then on the basis of available maxX, maxY, maxZ of metadata we will establish is
in dimension supplied and in direction known from numb we will return boolean 
 that will e true if we have some block existing 
"""
function isNextBlockExists(metaData,dim, numb ,linIter, isPassGold, maxX,maxY,maxZ)::Bool
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end

"""
set block is to be activated property to be true 
"""
function  setAsToBeActivated(metaData,linIndex,isPassGold)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end 


"""
given metadata it pushes the result to the result list 
    metadata is needed mainly to push result into correct queue that is described by 
    appropriate offset and counter
"""
function appendResultPadding(metaData, linIndex, x,y,z,iterationnumber, dim,numb)

   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end


"""
we are adding the result to appropriate spot in the result list
on the basis of the main part offset from the  metadata
"""
function appendResultMainPart(metaData, linIndex, x,y,z,iterationnumber, direction)
   getMetaDataFieldFromLin(metaData,linIndex,locFourthDim)
    setMetaDataFieldFromLin(metaData,linIndex,locFourthDim,valuee)
    
    end

end#MetaDataUtils