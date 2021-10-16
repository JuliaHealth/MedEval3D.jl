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





#first argument metadata second value from shared memory
setMetaLeftFP
setMetaLeftFN
setMetaRightFP
setMetaRightFN
setMetaPosteriorFP
setMetaPosteriorFN
setMetaAnteriorFP
setMetaAnteriorFN
setMetaTopFP
setMetaTopFN
setMetaBottomFP
setMetaBottomFN

#sets  count of fp, fn in main part

setMetaDataMainFpCount
setMetaDataMainFnCount


setMetaDataTotalFpCount
setMetaDataTotalFnCount

setMetaDataFnCount(metaData,locArr[1], xOuter,yOuter,zOuter) 



#set the x,y,z coordinates - so we will able to query it efficiently also with linear index
#what is important later as we will use only part of meta data this indicies will need to be updated
setMetaDataXYZ(metaData, xOuter,yOuter,zOuter  )
"""
reduce the values of selected metadata block by the supplied values
"""
reduceMetaDataXYZ(metaData, minX,minY,minZ, linIndex  )



#getters for metaData
getMetaLeftFP
getMetaLeftFN
getMetaRightFP
getMetaRightFN
getMetaPosteriorFP
getMetaPosteriorFN
getMetaAnteriorFP
getMetaAnteriorFN
getMetaTopFP
getMetaTopFN
getMetaBottomFP
getMetaBottomFN

#gets  count of fp, fn in main part

getMetaDataMainFpCount
getMetaDataMainFnCount

getMetaDataTotalFpCount
getMetaDataTotalFnCount


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


getMetaLeftFPOffset
getMetaLeftFNOffset
getMetaRightFPOffset
getMetaRightFNOffset
getMetaPosteriorFPOffset
getMetaPosteriorFNOffset
getMetaAnteriorFPOffset
getMetaAnteriorFNOffset
getMetaTopFPOffset
getMetaTopFNOffset
getMetaBottomFPOffset
getMetaBottomFNOffset
getMetaDataMainFpCountOffset
getMetaDataMainFnCountOffset




setMetaLeftFPOffset
setMetaLeftFNOffset
setMetaRightFPOffset
setMetaRightFNOffset
setMetaPosteriorFPOffset
setMetaPosteriorFNOffset
setMetaAnteriorFPOffset
setMetaAnteriorFNOffset
setMetaTopFPOffset
setMetaTopFNOffset
setMetaBottomFPOffset
setMetaBottomFNOffset
setMetaDataMainFpCountOffset
setMetaDataMainFnCountOffset