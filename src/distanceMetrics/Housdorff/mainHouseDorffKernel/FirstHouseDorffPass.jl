"""
we already converted source array to booleans now we know the number of false positives; false negatives and 
min,max x,y,z of part of data where we have any intreasting maxresultPoints

Now we need to: 
    -populate metadata
        -mark also whether the block is active, foll or neither - from both perspectives of both masks
        -we can set the min,max dims of data block
    -as we already loaded data from global memory - what is supposedly most expensive part - we can do the first dilatation step
    -we prepare rudamentary work plan for  next step    


Toexperiment
    reducePaddingPlane   #to be experimented is it better to reduce to varyiong y or x 



"""
module FirstHouseDorffPass
using KernelAbstractions, ..CUDAGpuUtils, Logging,StaticArrays

"""
prepare first pass of Housedorff kernel run
datablockDim - the edge length of cube describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32x32
"""
"""


main kernel of muodule it will get 4 data arrays -   reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB
    we work on all simultaneously

    initializes work queue (invoked only in first pass)
    basically we need to add to the one dimensional queue all of the  active blocks we find in first pass- crerating basic queue that will be processed by normals passes
        -we will make one block to be responsible for one slice  - and we will run this without cooperative threads limitation ... 
        -hence we will have double loop of all data blocks metadata  iterating over x and y dimension with z dimension constant
        -now every time when block is empty or full we will update its metadata
        -every time the block is active we will add it to the end of mainWorkQueeue - we will use mainQuesCounter for it
        -in the end of first pass or at the bigining of second we need to set  mainActiveCounterNow= mainQuesCounter
    
    reducedArrays =  [reducedSegmA,reducedGoldB,reducedGoldA,reducedSegmB] 
        - list of 3 dimensional boolean arrays with main data that we will work on

    metaData - basic data about data blocks - stored in 3 dimensional array
    metadataDims - dimensions of metaData array
    resArrays = [resArrayA, resArrayB]- 3 dimensional array of size of reduced source array where entry will be in place where we had covered the voxel, and value will mark in which itwration
        - we have two result arrays separate for each pass    
        - when we will cover some point  we need to report it in the resArray

    datablockDim - the edge length  describing the block of data we are working on  - defoult is 32 - which would produce blocks of size 32x32xzDim
        - we will have  the plane of threads of size 32x32 and in order to preserve memory coalescence we should have a warp have the  set of next x positions
    
    mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
    mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 
"""
firstHouseDorffPassKernel = @kernel function firstHouseDorffPassKernel(reducedArrays, metaData, metadataDims::Tuple{UInt8,UInt8,UInt8}, resArrays, datablockDim, mainQuesCounter, mainWorkQueue)
    resShmem = @StaticSharedMem(Bool, (34, 34, 34))
    isMaskFull = Ref(false)
    isMaskEmpty = Ref(false)
    locArr = Ref(Int32(0))

    for xdim = UInt8(1):metadataDims[1], ydim = UInt8(1):metadataDims[2], ispassGoldd in [true, false]
        singleDataBlockPass(reducedArrays[ispassGoldd * 2 + 1], reducedArrays[ispassGoldd * 2 + 2], UInt16(1), ((xdim - 1) * 32) + 1, ((ydim - 1) * 32) + 1, ((blockIdx().x - 1) * 32) + 1, isMaskFull, isMaskEmpty, resShmem, locArr, metaData, metadataDims, ispassGoldd, xdim, ydim, blockIdx().x, mainQuesCounter, mainWorkQueue, resArrays[ispassGoldd + 1])
    end
end
#for
# in this spot we should have already filled work queue, set information which blocks are full or empty, and got throught first dilatation step

function executefirstHouseDorffPassKernel(datablockDim, metaData)
    threadsNum = (datablockDim, datablockDim)
    blocksNum = size(metaData)

    kernel = firstHouseDorffPassKernel(CPU(), threadsNum, blocksNum)
    kernel(reducedArrays, metaData, metadataDims, resArrays, datablockDim, mainQuesCounter, mainWorkQueue, ndrange = blocksNum)
end

end #FirstHouseDorffPassKernel

"""
collects multiple functions that will be invoked over a single data blocks
analyzedArr - array that we dilatate
refAray - array we are comparing with
iterationNumber - in first pass it is 1 
blockBeginingX,blockBeginingY,blockBeginingZ - coordinates where our block is begining - will be used as offset by our threads
isMaskFull,isMaskEmpty - needed for establishing state  of a current block after processing 
resShmem- shared memory
locArr - thread local array with values loaded from global memory
metaData - 3 dim array with metadata of data blocks
metadataDims - dimensions of metadataDims
currMatadataBlockX,currMatadataBlockY, currMatadataBlockZ - cartesian coordinates of current block in metadaa!!!
mainQuesCounter - counter that we will update atomically and will be usefull to populate the work queue
mainWorkQueue - the list of the indicies of  data blocks in metadata with additional information is it referencing the goldpass or second one 
"""
# function singleDataBlockPass(analyzedArr
#                 ,refAray
#                 ,iterationNumber
#                 ,blockBeginingX
#                 ,blockBeginingY
#                 ,blockBeginingZ
#                 ,isMaskFull
#                 ,isMaskEmpty
#                 ,resShmem
#                 ,locArr
#                 ,metaData
#                 ,metadataDims::Tuple{UInt8,UInt8,UInt8}
#                 ,isPassGold::Bool
#                 ,currMatadataBlockX::UInt8
#                 ,currMatadataBlockY::UInt8
#                 ,currMatadataBlockZ::UInt8
#                 ,mainQuesCounter
#                 ,mainWorkQueue
#                 ,resArray )
#             ############### execution
#             executeDataIterFirstPass(analyzedArr, refAray,iterationNumber,blockBeginingX,blockBeginingY,blockBeginingZ,isMaskFull,isMaskEmpty,resShmem,locArr,resArray,mainQuesCounter)
#             #for futher processing we need to have space in main shmem
#             clearMainShmem(shmem)
#             #now we need to deal with padding in shmem res
#             processAllPaddingPlanes(x,y,z,shmem,currBlockX,currBlockY,currBlockZ,sourceArray,referenceArray,resArray,metaData,metadataDims,isPassGold,locArr)
#             # now let's check weather block is eligible for futher processing - for this we need sums ...
#             isActiveForFirstPass(isMaskFull, isMaskEmpty,resShmem,currMatadataBlockX,currMatadataBlockY,currMatadataBlockZ,isPassGold,metaData,mainQuesCounter,mainWorkQueue)
# end#singleDataBlockPass
#FirstHouseDorffPassKernel