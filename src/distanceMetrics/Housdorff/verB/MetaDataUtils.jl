"""
sets is active of given block to true
"""
setBlockToActive(metaData,linIndex)


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