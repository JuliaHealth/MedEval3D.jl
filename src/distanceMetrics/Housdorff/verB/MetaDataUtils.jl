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