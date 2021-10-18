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
13)   Total block Fp  
14)   Total block Fn  


16t - means 16 of such variables 
activeInGold::Bool
activeInSegm::Bool
fullInGold::Bool
fullInSegm::Bool

isToBectivatedGold::Bool
isToBeActivatedSegm::Bool
x,y,z coordinates ::UInt32
2t totalfpAndFnCount::UInt32
14T isToBeAnalyzed::Bool
14T fpFNcounts::UInt32

14T resOffsets::UInt32
2T totalOffsetBeginingAndEnd UInt32
16T oldCounters ::UInt32
16T new counters::UInt32

"""









"""
sets is active of given block to true for gold standard pass
"""
setBlockToActiveInGold(metaData,linIndex)

"""
true if block is full for gold standard pass
"""
isBlockFullInGold(metaData, linIndex)

"""
is block set to be activated for gold standard pass
"""
isBlockToBeActivatedInGold(metaData, linIndex)

"""
is block currently Active for gold standard pass
"""
isBlockCurrentlyActiveInGold(metaData, linIndex)

"""
make is block currently active to true for gold standard pass
"""
setBlockasCurrentlyActiveInGold(metaData, linIndex)



"""
sets is active of given block to true for not gold pass
"""
setBlockToActiveInSegm(metaData,linIndex)

"""
true if block is full for not gold pass
"""
isBlockFullInSegm(metaData, linIndex)

"""
is block set to be activated for not gold pass
"""
isBlockToBeActivatedInSegm(metaData, linIndex)

"""
is block currently Active for not gold pass
"""
isBlockCurrentlyActiveInSegm(metaData, linIndex)

"""
make is block currently active to true for not gold pass
"""
setBlockasCurrentlyActiveInSegm(metaData, linIndex)



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
macro setIstoBeAnalyzed() 




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

end


"""
get data is to be analyzed that was set in setIstoBeAnalyzed
and loaded into res shmem in getIstoBeAnalyzed
this function only return boolean from resshmem that tell us about wheather it makes sense to analyze the data inside
    the data block - so it cares about total data block fp or fn not yet covered
"""
function getIsTotalFPorFNnotYetCovered(resshmem )

end


"""
We just access data from res shmem that was loaded into resshmem in getIstoBeAnalyzed
dim can be 1,2,3 and tell in what dimension is the plane of our analysis
numb may be either 1 or 34  - on this basis we can establish which padding is of our intrest ...
"""
function isPaddingToBeAnalyzed(resShmem,dim,numb )


end

"""
sets given block is full property of either gold or other pass 
    as true
"""
function setBlockAsFull(metaData,linIndex, isGoldPass)

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
function setMetaFPandFN()
    
    
end




setMetaDataFnCount(metaData,locArr[1], xOuter,yOuter,zOuter) 



#set the x,y,z coordinates - so we will able to query it efficiently also with linear index
#what is important later as we will use only part of meta data this indicies will need to be updated
setMetaDataXYZ(metaData, xOuter,yOuter,zOuter  )
"""
reduce the values of selected metadata block by the supplied values
"""
reduceMetaDataXYZ(metaData, minX,minY,minZ, linIndex  )




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
getOldCount(numb, mataData,linIndex)

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
geNewCount(numb, mataData,linIndex)


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

end   

"""
on the basis of linear index we will find the  metadata x,y,Z
then on the basis of available maxX, maxY, maxZ of metadata we will establish is
in dimension supplied and in direction known from numb we will return boolean 
 that will e true if we have some block existing 
"""
function isNextBlockExists(metaData,dim, numb ,linIter, isPassGold, maxX,maxY,maxZ)::Bool
    
end

"""
set block is to be activated property to be true 
"""
function  setAsToBeActivated(metaData,linIndex,isPassGold)

end   


"""
given metadata it pushes the result to the result list 
    metadata is needed mainly to push result into correct queue that is described by 
    appropriate offset and counter
"""
function appendResultPadding(metaData, linIndex, x,y,z,iterationnumber, dim,numb)


end


"""
we are adding the result to appropriate spot in the result list
on the basis of the main part offset from the  metadata
"""
function appendResultMainPart(metaData, linIndex, x,y,z,iterationnumber, direction)

end #appendResultMainPart

