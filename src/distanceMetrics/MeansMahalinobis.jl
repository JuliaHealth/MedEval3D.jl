"""
new optimazation idea  - try to put all data in boolean arrays in shared memory  when getting means
next we would need only to read shared memory - yet first one need to check wheather there would be enough shmem on device

- we can also put the data abount slice wise count in shared memory - together with slicewise covariances

varianceZGlobal - should be reduced atomically in the end with all other variables - not slicewise like now
"""

module MeansMahalinobis
using ..CUDAGpuUtils, CUDA, ..IterationUtils, ..ReductionUtils, ..MemoryUtils
export meansMahalinobisKernel,prepareMahalinobisKernel

using KernelAbstractions

"""
prepares all equired arguments and gives back the  arguments and thread and block configuration 
for calculation of Mahalanobis distance - the latter is based on the occupancy API
"""
function prepareMahalinobisKernel()
   
    
    numberToLooFor=1
    #we will fill it after we work with launch configuration
    loopXdim = UInt32(1);loopYdim = UInt32(1) ;loopZdim = UInt32(1) ;
    sizz = (2,2,2);maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
    #gold
    totalXGold= CuArray([0.0]);
    totalYGold= CuArray([0.0]);
    totalZGold= CuArray([0.0]);
    totalCountGold= CuArray([0]);
    #segm
    totalXSegm= CuArray([0.0]);
    totalYSegm= CuArray([0.0]);
    totalZSegm= CuArray([0.0]);

    varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);
    varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm= CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]),CuArray([0.0]);

    totalCountSegm= CuArray([0]);
    totalCountGold= CuArray([0]);

    countPerZGold= CUDA.zeros(Float32,sizz[3]+1);
    countPerZSegm= CUDA.zeros(Float32,sizz[3]+1);

    # covariancesSliceWiseGold= CUDA.zeros(Float32,6,sizz[3]+1);
    # covariancesSliceWiseSegm= CUDA.zeros(Float32,6,sizz[3]+1);


    covarianceGlobal= CUDA.zeros(Float32,12,1);

    mahalanobisResGlobal= CUDA.zeros(1);


    args = (numberToLooFor
    ,loopYdim,loopXdim,loopZdim
    ,(maxX, maxY,maxZ)
    ,totalXGold,totalYGold,totalZGold,totalCountGold
    ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,
        countPerZGold, countPerZSegm,
        #,covariancesSliceWiseGold, covariancesSliceWiseSegm,
    varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
        ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
        ,mahalanobisResGlobal
        #, mahalanobisResSliceWise
    )
        
        
        get_shmem(threads) = (sizeof(UInt32)*3*4)
    
    threads ,blocks  = getThreadsAndBlocksNumbForKernel(get_shmem,meansMahalinobisKernel,(CUDA.zeros(2,2,2),CUDA.zeros(2,2,2),args...))

        loopXdim = UInt32(fld(maxX, threads[1]))
        loopYdim = UInt32(fld(maxY, threads[2])) 
        loopZdim = UInt32(fld(maxZ,blocks )) 
        

    args = (numberToLooFor
    ,loopYdim,loopXdim,loopZdim
    ,(maxX, maxY,maxZ)
    ,totalXGold,totalYGold,totalZGold,totalCountGold
    ,totalXSegm,totalYSegm,totalZSegm,totalCountSegm
        ,countPerZGold, countPerZSegm,
        #,covariancesSliceWiseGold, covariancesSliceWiseSegm,
    varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
        ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
        ,mahalanobisResGlobal
        #, mahalanobisResSliceWise
    
    )
        
    
  
  return(args,threads ,blocks)
end



"""
executed after prepareMahalinobisKernel - will execute on given arrays 
    and return the Mahalinobis distance between gold standard mask and the result of segmentation
"""
function calculateMalahlinobisDistance(goldGPU,segmGPU,args,threads ,blocks,numberToLooFor)

# setting main arays

##### setting to 0 all entries that require it    
for i in 6:28    
    CUDA.fill!(args[i],0)
end 

numbToLooFor,loopYdima,loopXdima,loopZdima,dims,totalXGold,totalYGold,totalZGold,totalCountGold,totalXSegm,totalYSegm,totalZSegm,totalCountSegm,countPerZGold, countPerZSegm,varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm,mahalanobisResGlobal=args

numberToLooFor
sizz = size(goldGPU)

maxX = UInt32(sizz[1]);maxY = UInt32(sizz[2]);maxZ = UInt32(sizz[3])
loopXdim = UInt32(fld(maxX, threads[1]))
loopYdim = UInt32(fld(maxY, threads[2]))
loopZdim = UInt32(fld(maxZ,blocks )) 
countPerZGold= CUDA.zeros(Float32,sizz[3]+1);
countPerZSegm= CUDA.zeros(Float32,sizz[3]+1);
# println( "loopXdim $(loopXdim) loopYdim $(loopYdim)  loopZdim $(loopZdim) maxX $(maxX) maxY $(maxY) maxZ $(maxZ) \n")
args = (numberToLooFor
,loopYdim,loopXdim,loopZdim
,(maxX, maxY,maxZ)
,totalXGold,totalYGold,totalZGold,totalCountGold
,totalXSegm,totalYSegm,totalZSegm,totalCountSegm
    ,countPerZGold, countPerZSegm,
    #,covariancesSliceWiseGold, covariancesSliceWiseSegm,
varianceXGlobalGold,covarianceXYGlobalGold,covarianceXZGlobalGold,varianceYGlobalGold,covarianceYZGlobalGold,varianceZGlobalGold
    ,varianceXGlobalSegm,covarianceXYGlobalSegm,covarianceXZGlobalSegm,varianceYGlobalSegm,covarianceYZGlobalSegm,varianceZGlobalSegm
    ,mahalanobisResGlobal
    #, mahalanobisResSliceWise

)

@cuda cooperative=true threads=threads blocks=blocks meansMahalinobisKernel(goldGPU,segmGPU,args...)

return  args[28][1]
end






"""
part of the above kernel that will be executed in order to get means
first of the gold standard array and then for other
    arrAnalyzed- gold standard or other array
    countPerZ- global memory array holding per slice counts of trues
    
"""
macro iterateForMeans(countPerZ,arrAnalyzed)
 return esc(quote

    @iter3dAdditionalzActs(arrDims,loopXdim,loopYdim,loopZdim,
    #inner expression
    if(  @inbounds($arrAnalyzed[x,y,z])  ==numberToLooFor)
        #updating variables needed to calculate means
        sumX+=Float32(x) ;  sumY+=Float32(y)  ; sumZ+=Float32(z)   ; count+=Float32(1)   
    end,
    #after z expression - we get slice wise true counts from it 
    begin
        sync_threads()
        #reducing count only
        if(z<=arrDims[3])
            countTemp = count
            @redWitAct(offsetIter,shmemSum, count,+)
            #saving to global memory count of this slice
            @ifXY 1 1 begin 
                 $countPerZ[z]=(shmemSum[1,1] - oldZVal[1] )
                oldZVal[1]=shmemSum[1,1]
            end
            #clear shared memory only first row was used and sync threads 
            clearSharedMemWarpLong(shmemSum, UInt8(1), Float32(0.0))
            count=countTemp#to preserve proper value for total count
        end#if ar dims
    end )#if bool in arr  

    #tell what variables are to be reduced and by what operation
    @redWitAct(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+)
     
end)#quote
end

"""
reset variables and arrays to be able to calculate the variances and Covariances
"""

macro resetForVarAndCov(totalX,totalY, totalZ, totalCount)
    return esc(quote
        #at this spot we should have all counts of x,y,z and counts - we can get means from it
        clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
      
        @ifY 4 @unroll for i in 1:5
            @ifX i @inbounds intermedieteRes[i]= Float32(0.0)
        end
    
        @ifXY 1 1  meanxyz[1]= $totalX[1]/$totalCount[1]
        @ifXY 1 2  meanxyz[2]= $totalY[1]/$totalCount[1]
        @ifXY 1 3  meanxyz[3]= $totalZ[1]/$totalCount[1]
        #from now one sumX is variance of x ,sumY is covariance xy ,sumZ is covariance xz 
        sumX,sumY,sumZ ,count= Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)

    end)#quote
end

"""
calculates variances and covariances needed for calculation of mahalanobis

"""
macro calculateVariancesAdCov(countPerZ,arrToAnalyze,varianceXGlobal,covarianceXYGlobal,covarianceXZGlobal,varianceYGlobal,covarianceYZGlobal,varianceZGlobal)
    return esc(quote
    @iter3dAdditionalxyzActsAndZcheck(arrDims,loopXdim,loopYdim,loopZdim
    #z check
    ,((z<= arrDims[3] && $countPerZ[z]>0.0)), 
    #inner expression
    if(  @inbounds($arrToAnalyze[x,y,z])  ==numberToLooFor)
        #getting variance x and x covariances
        sumX+=(Float32(x)-meanxyz[1])*(Float32(x)-meanxyz[1])
        sumY+=((Float32(y)-meanxyz[2])*(Float32(x)-meanxyz[1]))
        sumZ+=((Float32(z)-meanxyz[3])*(Float32(x)-meanxyz[1]))
        count+=Float32(1.0)   
         end,
    #x additional fun - currently nothing to implement here
    begin    end,
    #y additional fun - as we added all x variances and covariances we need now to update y variance and y covariance 
    #we need to add this the same amount of time as we added the x variables so we use count for it
    # of course if count is 0 we can ignore this step
    begin
   #we put reduced values into share memory 
    sync_threads()
  @redOnlyStepOne(offsetIter,shmemSum,  sumX,+,     sumY,+    ,sumZ,+   ,count,+);

   if(threadIdxX()==UInt32(1) && count>Float32(0.0))
       @inbounds shmemSum[threadIdxY(),1]+= sumX
       @inbounds shmemSum[threadIdxY(),2]+= sumY
       @inbounds shmemSum[threadIdxY(),3]+= sumZ
       #putting  variance y and covariance yz manually to shared memory multiply appropriate amount of time
       @inbounds shmemSum[threadIdxY(),4]+= count*(y-meanxyz[2])*(y-meanxyz[2])#variance y
       @inbounds shmemSum[threadIdxY(),5]+= count*((y-meanxyz[2])*(z-meanxyz[3]))#covariance yz
       @inbounds shmemSum[threadIdxY(),6]+= count

   end;
   #reset as values are already saved in shmemsum
   sumX,sumY,sumZ ,count= Float32(0.0),Float32(0.0),Float32(0.0),Float32(0.0)
   sync_warp()
  
    end,
    #z additional fun
    begin
        sync_threads()      
        #no point in analyzing if it is empty
        if(z<= arrDims[3] && $countPerZ[z]>0.0)
            #we do the last step of reductions to get all of the values into first spots of shared memory
            @redOnlyStepThree(offsetIter,shmemSum, +,+,+  ,+,+,+)
            sync_threads()

            #we will use it later to get slicewise results and in the end we will send those to global memory
            @ifY 1 @unroll for i in 1:5                
                @ifX i intermedieteRes[i]+=shmemSum[1,i]
            end 

            @ifXY 1 7 @inbounds intermedieteRes[6]+=(((z- meanxyz[3])*(z- meanxyz[3])  )*shmemSum[1,6])
            
            @ifXY 1 7 @inbounds CUDA.@atomic $varianceZGlobal[]+=((z- meanxyz[3])*(z- meanxyz[3])  )*shmemSum[1,6]
            ###########remove
            
            # @ifY 2 @unroll for i in 1:5
            #     @ifX i @inbounds $covariancesSliceWise[i,z]+=shmemSum[1,i]
                
            #     if(shmemSum[1,i]>0)
            #         CUDA.@cuprint " i $(i) val $(shmemSum[1,i]) "
            #     end    

            # end 
            #     @ifXY 2 6 @inbounds $covariancesSliceWise[6,z]+= ((z- meanxyz[3])*(z- meanxyz[3]))*shmemSum[1,6]
            sync_threads()
            # #clear shared memory
            clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))


        end#if covariance non empty
    end  )# iterations loop

    sync_threads()

    #at this point we should have all variances and covariances in intermedieteRes and we can send it to global results
    @ifXY 1 1  @inbounds CUDA.@atomic $varianceXGlobal[]+=intermedieteRes[1]  
    @ifXY 1 2 @inbounds CUDA.@atomic $covarianceXYGlobal[]+=intermedieteRes[2]  
    @ifXY 1 3  @inbounds CUDA.@atomic $covarianceXZGlobal[]+=intermedieteRes[3] 
    @ifXY 1 4  @inbounds CUDA.@atomic $varianceYGlobal[]+=intermedieteRes[4]   
    @ifXY 1 5 @inbounds CUDA.@atomic $covarianceYZGlobal[]+=intermedieteRes[5]  
    #@ifXY 1 7  @inboundsCUDA.@atomic $varianceZGlobal[]+=intermedieteRes[6]  

end)#quote 
end#calculateVariancesAdCov

"""
this will enable final calculations of the Mahalanobis distance

#original formula developed on basis of https://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation/147222#147222?newreg=a68aa51b2f8c45daaece49163105845c
#unrolled 3 by 3 cholesky decomposition 
# a = sqrt(varianceX)
# b = (covarianceXY)/a
# c = (covarianceXZ)/a
# e = sqrt(varianceY - b*b)
# d = (covarianceYZ -(c * b))/e
# #unrolled forward substitiution
# ya= x[1]/a 
# yb = (x[2]-b*ya)/e
# yc= (x[3]-yb*d-ya* c)/sqrt(varianceZ - c*c -d*d )
#taking square euclidean distance
# returnya*ya+yb*yb+yc*yc

"""
macro getFinalResults()
    return esc(quote
    #first getting total counts to all threads
    sumY= totalCountGold[1]#total count gold
    sumZ= totalCountSegm[1]#total count segm

    #values needed for calculating means 
    @ifXY 14 1 sumX = totalXGold[1]
    @ifXY 16 1 sumX = totalYGold[1]
    @ifXY 18 1 sumX = totalZGold[1]
    @ifXY 20 1 sumX = totalXSegm[1]
    @ifXY 22 1 sumX = totalYSegm[1]
    @ifXY 24 1 sumX = totalZSegm[1]
    sync_warp()
    @ifXY 14 1 shmemSum[7,1] = sumX/sumY #mean X gold
    @ifXY 16 1 shmemSum[8,1] = sumX/sumY #mean Y gold
    @ifXY 18 1 shmemSum[9,1] = sumX/sumY #mean Z gold
    @ifXY 20 1 shmemSum[10,1] = sumX/sumZ #mean X segm
    @ifXY 22 1 shmemSum[11,1] =  sumX/sumZ #mean Y segm
    @ifXY 24 1 shmemSum[12,1] = sumX/sumZ #mean Z segm
    sync_threads()# now all blocks should have available means

    if(blockIdxX()==1)# this one for global result should be executed only once
        @unroll for numb in 1:1
            #first we upload to registers variables needed to calculate Mahalinobis
            @ifXY 1 numb sumX = varianceXGlobalGold[1]
            @ifXY 2 numb shmemSum[13,1] = varianceXGlobalSegm[1]
            @ifXY 3 numb sumX = covarianceXYGlobalGold[1]
            @ifXY 4 numb shmemSum[14,1] = covarianceXYGlobalSegm[1]
            @ifXY 5 numb sumX = covarianceXZGlobalGold[1]
            @ifXY 6 numb shmemSum[15,1] = covarianceXZGlobalSegm[1]
            @ifXY 7 numb sumX = varianceYGlobalGold[1]
            @ifXY 8 numb shmemSum[16,1] = varianceYGlobalSegm[1]
            @ifXY 10 numb sumX = covarianceYZGlobalGold[1]
            @ifXY 11 numb shmemSum[17,1] = covarianceYZGlobalSegm[1]
            @ifXY 12 numb sumX = varianceZGlobalGold[1]
            @ifXY 13 numb shmemSum[18,1] = varianceZGlobalSegm[1]
                
            sync_warp()
            #sumX+=shfl_down_sync(active_mask(),sumX,UInt8(1))

            @ifXY 1 numb shmemSum[1,1]= (sumX+shmemSum[13,1])/(sumY+sumZ)  #common variance x
            @ifXY 3 numb shmemSum[2,1]= (sumX+shmemSum[14,1])/(sumY+sumZ) #common covariance xy
            @ifXY 5 numb shmemSum[3,1] = (sumX+shmemSum[15,1])/(sumY+sumZ)#common covariance xz
            @ifXY 7 numb shmemSum[4,1] = (sumX+shmemSum[16,1])/(sumY+sumZ) #common variance y
            @ifXY 10 numb shmemSum[5,1] = (sumX+shmemSum[17,1])/(sumY+sumZ) #common covariance yz
            @ifXY 12 numb shmemSum[6,1] = (sumX+shmemSum[18,1])/(sumY+sumZ) #common variance z
            sync_warp()
            @ifXY 1 1 begin
                # #unrolled 3 by 3 cholesky decomposition 
                a = CUDA.sqrt(shmemSum[1,1])  #getting varianceX
                b = (shmemSum[2,1])/a #getting covarianceXY
                c = (shmemSum[3,1])/a  #getting covarianceXZ
                e = CUDA.sqrt(shmemSum[4,1] - b*b)  #getting varianceY
                d = (shmemSum[5,1] -(c * b))/e #getting covarianceYZ
              #unrolled forward substitiution - we reuse the variables to reduce register preassure (hopefully)
                sumX= (shmemSum[7,1] -shmemSum[10,1]  )   /a 
                sumY = ( (shmemSum[8,1] -  shmemSum[11,1] )  -b*sumX)/e
                sumZ= ( (shmemSum[9,1] - shmemSum[12,1]  )  -sumY*d-sumX* c)/CUDA.sqrt(shmemSum[6,1] - c*c -d*d ) #getting  varianceZ
                #taking square euclidean distance
                mahalanobisResGlobal[1]= CUDA.sqrt(sumX*sumX+sumY*sumY+sumZ*sumZ)
            end # @ifXY  numb
          end#for numb  
    end# if block
#### now in case we want the slice wise results we need to loop through the 
#sync_threads()
#in order to maximize parallelization we will use 6 warps at a time (if needed )
# and each will use up diffrent row of shared memory so we will loop through slicewise
#and concurrently update the slicewise results list
# our data is stored in covariancesSliceWiseGold and  covariancesSliceWiseSegm   
#1) variance x 2) covariance xy 3) covariance xz 4) variance y 5) covariance yz 6) variance z this applies to both 





# @unroll for i in 0:cld(loopZdim,6) 
#     @unroll for j in 1:6
#         #we need to stay in the existing slices 
#         index = (i*6+j)*gridDim().x+blockIdxX()
#         if(index<arrDims[3])            
#         #first we load data from gold 
#         sumX,sumY,sumZ,count = Float32(0),Float32(0),Float32(0),Float32(0)
#         shmemSum[threadIdxX(),j]=Float32(0.0)
#             if(threadIdxX()<7 )
#                 sumx = covariancesSliceWiseGold[threadIdxX(),index]
#             #then from segm 
#             elseif(threadIdxX()<13 )
#                 shmemSum[12+threadIdxX() ,1] = covariancesSliceWiseSegm[threadIdxX()-6,index]
#             end #else if  
#             sync_warp()
#             #we set it on diffrent warps

#             @ifXY 1 j shmemSum[1,j] = (sumX+shmemSum[13,j])/(sumY+sumZ)  #common variance x
#             @ifXY 3 j shmemSum[2,j] = (sumX+shmemSum[14,j])/(sumY+sumZ) #common covariance xy
#             @ifXY 5 j shmemSum[3,j] = (sumX+shmemSum[15,j])/(sumY+sumZ)#common covariance xz
#             @ifXY 7 j shmemSum[4,j] = (sumX+shmemSum[16,j])/(sumY+sumZ) #common variance y
#             @ifXY 10 j shmemSum[5,j] = (sumX+shmemSum[17,j])/(sumY+sumZ) #common covariance yz
#             @ifXY 12 j shmemSum[6,j] = (sumX+shmemSum[18,j])/(sumY+sumZ) #common variance z
            
#            # CUDA.@cuprint " varx $(shmemSum[1,j]) cov xy  $(shmemSum[2,j])  cov xz $(shmemSum[3,j])  var y $(shmemSum[4,j]) covyz $(shmemSum[5,j]) var z  $(shmemSum[6,j]) \n"


#             sync_warp()


#             @ifXY 1 j begin
#                 # #unrolled 3 by 3 cholesky decomposition 
#                 a = CUDA.sqrt(shmemSum[1,j])  #getting varianceX

#                 b = (shmemSum[2,j])/a #getting covarianceXY
#                 c = (shmemSum[3,j])/a  #getting covarianceXZ
#                 e = CUDA.sqrt(shmemSum[4,j] - b*b)  #getting varianceY
#                 d = (shmemSum[5,j] -(c * b))/e #getting covarianceYZ
#               #unrolled forward substitiution - we reuse the variables to reduce register preassure (hopefully)
#                 sumX= (shmemSum[7,1] -shmemSum[10,1]  )   /a 
#                 sumY = ( (shmemSum[8,1] -  shmemSum[11,1] )  -b*sumX)/e
#                 sumZ= ( (shmemSum[9,1] - shmemSum[12,1]  )  -sumY*d-sumX* c)/CUDA.sqrt(shmemSum[6,j] - c*c -d*d ) #getting  varianceZ
#                 #taking square euclidean distance
#                 mahalanobisResSliceWise[index]= CUDA.sqrt(sumX*sumX+sumY*sumY+sumZ*sumZ)
#                 #CUDA.@cuprint "slice wise $(CUDA.sqrt(sumX*sumX+sumY*sumY+sumZ*sumZ))" 
#             end # @ifXY  numb
#         end#if
#     end#for
#     sync_warp()
# end#for 






    end)
end


"""
IMPORTANT x dim of threadblock needs to be always 32 
goldArr,segmArr  - arrays we analyze 
numberToLooFor - number we are intrested in in the array
loopYdim, loopXdim, loopZdim - number of times we nned to loop over those dimensions in order to cover all - important we start iteration from 0 hence we should use fld ...
maxX, maxY ,maxZ- maximum possible x and y - used for bound checking
totalX,totalY,totalZ - holding the results of summation of x,y and z's
totalCount - total number of non 0 entries
countPerZ - count of non empty entries per slice - particularly usefull in case it is 0
covariancesSliceWiseGold,covariancesSliceWiseSegm  - slice wise covariances - needed in case we want slice wise results - a matrix where each row entry is 
        I)gold standard mask values
        1)variance x    2)cov xy     3)cov xz     4)var y    5)cov yz      6)var z 
       
covarianceGlobal (just one column but entries exactly the same as above)
mahalanobisResGlobal - global result of Mahalinobis distance
mahalanobisResSliceWise - global result of Mahalinobis distance
"""
@kernel function meansMahalinobisKernel(
    goldArr, segmArr, numberToLooFor, loopYdim::UInt32, loopXdim::UInt32, loopZdim::UInt32, arrDims::Tuple{UInt32,UInt32,UInt32},
    totalXGold, totalYGold, totalZGold, totalCountGold, totalXSegm, totalYSegm, totalZSegm, totalCountSegm,
    countPerZGold, countPerZSegm,
    varianceXGlobalGold, covarianceXYGlobalGold, covarianceXZGlobalGold, varianceYGlobalGold, covarianceYZGlobalGold, varianceZGlobalGold,
    varianceXGlobalSegm, covarianceXYGlobalSegm, covarianceXZGlobalSegm, varianceYGlobalSegm, covarianceYZGlobalSegm, varianceZGlobalSegm,
    mahalanobisResGlobal
)
    # keeping counter of old z value - in order to be able to get slicewise z counter
    oldZVal = @localmem(Float32, 1)
    # summing coordinates of all voxels we are interested in
    sumX, sumY, sumZ, count = Float32(0), Float32(0), Float32(0), Float32(0)
    # count how many voxels of interest there are so we will get means
    # for storing results from warp reductions
    shmemSum = @localmem(Float32, (32, 6))
    clearSharedMemWarpLong(shmemSum, UInt8(6), Float32(0.0))
    if @index(Global) == 1
        oldZVal[1] = 0
    end
    # just needed for reductions
    offsetIter = UInt8(1)
    #### first we analyze gold standard array
    @synchronize

    @iterateForMeans(countPerZGold, goldArr)
end

# function executeMeansMahalanobisKernel(
#     goldArr, segmArr, numberToLooFor, loopYdim::UInt32, loopXdim::UInt32, loopZdim::UInt32, arrDims::Tuple{UInt32,UInt32,UInt32},
#     totalXGold, totalYGold, totalZGold, totalCountGold, totalXSegm, totalYSegm, totalZSegm, totalCountSegm,
#     countPerZGold, countPerZSegm,
#     varianceXGlobalGold, covarianceXYGlobalGold, covarianceXZGlobalGold, varianceYGlobalGold, covarianceYZGlobalGold, varianceZGlobalGold,
#     varianceXGlobalSegm, covarianceXYGlobalSegm, covarianceXZGlobalSegm, varianceYGlobalSegm, covarianceYZGlobalSegm, varianceZGlobalSegm,
#     mahalanobisResGlobal
# )
#     threads = (32, 32)
#     blocks = (cld(arrDims[1], threads[1]), cld(arrDims[2], threads[2]), cld(arrDims[3], threads[3]))

#     kernel = meansMahalanobisKernel(CPU(), threads, blocks)
#     kernel(
#         goldArr, segmArr, numberToLooFor, loopYdim, loopXdim, loopZdim, arrDims,
#         totalXGold, totalYGold, totalZGold, totalCountGold, totalXSegm, totalYSegm, totalZSegm, totalCountSegm,
#         countPerZGold, countPerZSegm,
#         varianceXGlobalGold, covarianceXYGlobalGold, covarianceXZGlobalGold, varianceYGlobalGold, covarianceYZGlobalGold, varianceZGlobalGold,
#         varianceXGlobalSegm, covarianceXYGlobalSegm, covarianceXZGlobalSegm, varianceYGlobalSegm, covarianceYZGlobalSegm, varianceZGlobalSegm,
#         mahalanobisResGlobal,
#         ndrange = blocks
#     )
# end

end#MeansMahalinobis
