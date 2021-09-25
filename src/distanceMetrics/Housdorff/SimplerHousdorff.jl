"""
Work will have 3 parts - 3 kernels 
    1) will prepare data - will return source array in boolean format and will give benchmark
    max and min x,y,z of either mask - to describe smallest possible cube holding all necessary data 
    2)first pass through the data  that will mark which data blocks are acteve or full ...  - it will also prepare  the rudamentary worplan for first iteration in phase 3
    3)the actual working part - there will be shceduling block that will pass what work should be done by other blocks - copying da, deciding to terminate etc.
        - rest of the block will iteratively analyze data blocks scheduled by scheduling block


So we will need the block of data representing each wall of the bit cube bit cuve will have edge length of 32 - apart from corner cases

Simplification
Get reduced array of bits from Tpfp..fn modified kernel
Assign blocks of size 32x32x32 for each we will have data structure holding data as pointed below all will be in UInt16
    Is activeOrFullForFirsPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForFirsPasssegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassSegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
   activeIterNumbMaskA - The number of iteration when block was last active - needed to coordinate which block will be working now
    activeIterNumbMaskB - The number of iteration when block was last active - needed to coordinate which block will be working now
    Data will be stored in UInt16 4 dimensional array in such a way that data blocks block id x,y,z will point to metadata position

ResArray storing all coordinates that were covered and in which iteration it occured  and in pass from which mask it occured

work scheduling structures

workSchedule- matrix that will have  number of rows that is equal to number of thread blocks that will work on a task
    and in each column we will put metadata needed to localize on what data block we are working currently 
    - so x,y,z indexes and 1 if we do it from perspective of gold sandard mask and 2 if from second mask
LocalRes -  matrix that will have  number of rows that is equal to number of thread blocks that will work on a task - so blocks will write results to this array
        and then scheduling block will collect those results from all matrix in the ResArray
LocalResLastEntryList - list with length equal to the number of thread blocks - will store the last number of occuppied  spot in Local Res (so block will know where to put next results)        
isWorking - bollean list of length equal the number of thread blocks that will point out wheather given thread block is working or not 
            - data needed for scheduling block 
fp, fn - number of false positive and false negatives - we will get it from algorithm responsible for preparing data 
    - and it will be needed  to know how much data needs to be allocated andfor the scheduling block to be able to decide whether we finished


Algo
Check blocks to assign weather they become inactive or full
    We do it by storing in register UInt32 (UInt numb) and then when streaming (each lane is responsible for 32 fields) will modify the bit of stored numb associated with step in a loop we are ; at first pass we do it for both masks at the same time
    Next we can use voting mechanism  and then shared memory reduction to check 2 times weather all numbs are first all 0’s - so numb will be 0 - in this case we will make it inactive second all 1’s so numbs will be max value of UINt32  - in this case we will make it full - we apply the procedure to both masks if neither we will make it active
    We scan in blocks for next active block available and mark using atomics by setting the activeIterNumb of a block to the current iteration number, so no other block can modify it further  !!! we need to check what does the atomic return …
    When working on a block we have 3 jobs 
    set to the 1’s all values around each one we find - and write it to global memory immediately or first to shared memory - to be experimented upon 
    Add coordinates of covered values to ResArray 
    Activate neighbouring blocks if needed
    Activation of neighbour blocks
    When we found the 1 (before dilatation step) in position of max/ min x/y/z we need to save information to the shared memory that this block needs to be activated according to the data present - we will choose the block depending on which dimension was min/max if we have corners or edges we will activate multiple blocks
    Not in all cases the block will get activated it will not get activated if
    Is full
    Neighbours in given direction are all inactive in second mask
    We finish when there are no more active blocks

Consideration for fact that Housdorff needs to be scanned from 2 perspectives 
we need to have a copy of original data for both masks so modification of one pass of housdorff will not affect the other - in this case also order is not so important
Each block needs to be aware for which pass it is currently working - hence to know what to dilatate and where to write the results
To maximize occupancy the blocks will concurrently work on both passes at once - it will reduce number of grid sync call at each iterations - and increase occupancy at the end of algorithm

"""
module SimplerHousdorff
using CUDA, Main.CUDAGpuUtils, Logging
"""
Metadata
divide array into blocks and assigns the metadata for each 
    Is activeOrFullForFirsPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForFirsPasssegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassSegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    activeIterNumbMaskA - The number of iteration when block was last active - needed to coordinate which block will be working now
    activeIterNumbMaskB - The number of iteration when block was last active - needed to coordinate which block will be working now
    Data will be stored in n times 34 UInt16 matrix in global memory

So in order to achieve it we need first to get the dimensions of the array we got - as it is reduced array it do not have to have the dimensions that would be simmilar
to the medical image dimensions - yet by construction all dimensions need to be multiple by 32!!! 
As we start from place where x,y,z is 0 and we will proceed (concurrently from here )
        we know by construction that min x, min y, and min z will be always some multiply of 32 away from begining 
        now we can lounch multiple small blocks where each will calculate the the required metadata - amount of those blocks will be just equal to the number of blocks tha we want to describe 
        and will be calculated by ceildiv of given dimension so sth like sizz = size[arrGold]-> will be mapped
        to blocks as sizz-> (celdiv(sizz[1]) ...) by all dimensions ; blocks will be elongated in one dimension and constructed in a way that single warp will be responsible to set metadata for single block

 arrGold - boollean 3 dim Cu array with gold standard (reduced to the smallest block with all true entries of both masks)
 arrSegm - segmentation 3 dim boolean Cu array we want to compare  
 
"""
function getBlockMetaData(arrGold, arrSegm,maxThrPerB::Int64=1024)
    arrGoldDims= size(arrGold)
    metadata = allocateMomory(arrGoldDims)
    metaDataDims = size(metadata)
    blocks,threadNumPerBlock = getBlockNumb(metaDataDims)# for metadata kernel 
    @info "blocks " blocks
    @info "threadNumPerBlock " threadNumPerBlock

    @cuda threads=1024 blocks=metaDataDims[1]*metaDataDims[2]*metaDataDims[3] housedorffMetadataKernel(metadata,metaDataDims,arrGoldDims ) 
    return metadata
end#getBlockMetaData


   
"""
kernel that output metadata for Housedorff
first all blocks are active - we intend to have all blocks o be active at first pass - and then prograssively  work only on this data that we are intrested in 
matadata - 4 dimensional data with block metadata

dataArrs - array of 2 arrays where first is arrGold and second arrSegm
    arrGold - boollean 3 dim Cu array with gold standard (reduced to the smallest block with all true entries of both masks)
    arrSegm - segmentation 3 dim boolean Cu array we want to compare

iterationNumber - variable that we will set in order to mark on what iteration we are curently

arrGoldDims - dimensions of the main array
dimOfThreadBlock - how big is the edge of a cube describing data block - for example for 1024 threads - we will have 32x32x32 size hence dimOfThreadBlock= 32
we will have in basic type 32x32 threads and work on 32x32x32 data block 
"""
function housedorffMetadataKernel(matadata
                                ,arrGold
                                ,arrSegm
                                ,arrGoldDims::Tuple{Int64, Int64, Int64}
                                ,iterationNumber::UInt16
                                ,debugginArr
                                ,dimOfThreadBlock::UInt16 )


    #initializing shared array (remember this is autmatically initated to semi random numbers)
    shemm= CuDynamicSharedArray(Bool,(dimOfThreadBlock,dimOfThreadBlock,dimOfThreadBlock) )
    #as we constructed data dimensions o be always multiple of 32 we do not need to do bound checks
    @unroll for k in 1:32
        shemm[blockIdx().x,blockIdx().y,k]=  [ (blockIdx().x-1)*dimOfThreadBlock+threadIdxX(),(blockIdx().y-1)*dimOfThreadBlock+threadIdxY(),(blockIdx().z-1)*dimOfThreadBlock+k]
    end#for

                                # each lane will be responsible for one meta data  
    # @unroll for k in 0:metadataDims[1]
    #     matadata[threadIdxX(),k+1, blockIdx().x,1]=(threadIdxX()-1)*32   #min x
    #     matadata[threadIdxX(),k+1, blockIdx().x,2]=min(threadIdxX()*32,arrGoldDims[1])   #max x
    #     matadata[threadIdxX(),k+1, blockIdx().x,3]=k*32   #min y
    #     matadata[threadIdxX(),k+1, blockIdx().x,4]=min((k+1)*32,arrGoldDims[2] )   #max y
    #     matadata[threadIdxX(),k+1, blockIdx().x,5]=(blockIdx().x-1)*32   #min z
    #     matadata[threadIdxX(),k+1, blockIdx().x,6]=min((blockIdx().x)*32,arrGoldDims[3])   #max z
    # end#for
end #housedorffMetadataKernel


"""
so given metadata and some helper constants we need to choose on what data our block of threads will work on
at the begining - first iteration it is not very important as all blocks are active - hence we can schedule work by just evenly dividing all data blocks to thread blocks
Then- some of the blocks will be marked as either inactive or full hence  we need to redistribute the work - so each thread block will have the same (+-1) number of data blocks to work on  
    - in order to achieve this -  single block will be scheduled to redistribute work - it will run at the very end of iteration cycle  to redistribute work  for next one 
    ; and at the begining - to redistribute blocks activated  in last round of previous iteration 
"""
function chooseBlockToWorkOn()

end#chooseBlockToWorkOn

"""
controls allocation of GPU memory - instantiating Cu arrays
 first we alocate the place for  metadata -  

   Metadata -  Assign blocks of size 32x32x32 for each we will have data structure holding data as pointed below all will be in UInt16
      1) isActiveOrFullForPasssegm - true if other  mask is acive for modifications  
      2) isActiveOrFullForPassgold -  true if gold standard mask is acive for modifications 
      3) isFullSegm - true if other mask is full (only ones)
      4) isFullGold - true if gold standard mask is full (only ones)

      Data will be stored in Bool 4 dimensional array in such a way that data blocks block id x,y,z will point to metadata position  


    resArray storing all coordinates that were covered and in which iteration it occured  and in pass from which mask it occured

    work scheduling structures
    

    mainWorkQueue - basically we need to add to the one dimensional queue all of the  active blocks we find in first pass- crerating basic queue that will be processed by normals passes
    of course during those normal passes some blocks will be added to the queue and some will be removed in order to mage 
        it will store 3 indicies (UInt8) of the place of the block in metadata plus  wheater we are referencing main pass or second one (UInt8 wchich will be 1 or 0)
        so entry1 - x position in metadata 2)y pos 3) z pos 4) boolean marking is it gold pass or not
        it we would need additional helper structures
    so we will have: 
    mainQuesCounter - telling us how many entries we have in work queue - we will divide this by number of thread blocks + some constant 
        - to get some amount of the data blocks to be processed + tail queue that will be accessed in atomic way - but it can be accessed by all blocks
        so if some block will finish work before others it will start processing this tail queue
    mainActiveCounterNow,mainActiveCounterNext - at first main queue will have only active blocks but progressively it will have more and more empty spots
        so we need to get the second counter that will keep track on the ramaining active blocks in curent iteration  
            -mainActiveCounterNow - will be reduced every time thread block finish processing block - we will sync grid and start next iteration when it will reach 0 
            -mainActiveCounterNext - will be increased every time we activate some block - will become the  mainActiveCounterNow in next iteration if it will reach 0 we will 
                call it the end and finish kernel       

    fp, fn - number of false positive and false negatives - we will get it from algorithm responsible for preparing data 
        - and it will be needed  to know how much data needs to be allocated andfor the scheduling block to be able to decide whether we finished

   arguments     
    fpNumb,fnNumb - number of false positives and false negatives
    blocksNum - number of thread blocks that are able to run on the same grid
    arrGoldDims - tuple with dimensions of main data array
    dataBlocksNum- number of dataBlocks
    dimOfThreadBlock - how big is the edge of a cube describing data block - for example for 1024 threads - we will have 32x32x32 size hence dimOfThreadBlock= 32
"""
function allocateMomory(arrGoldDims::Tuple{Int64, Int64, Int64}
                        ,fpNumb::Int64
                        ,fnNumb::Int64
                        ,blocksNum::Int64
                        ,dataBlocksNum::Int64
                        ,dimOfThreadBlock::UInt16  )
    maxresultPoints = fpNumb+fnNumb+1
    #number of data blocks in given dimension
    x=cld(arrGoldDims[1],dimOfThreadBlock)
    y=cld(arrGoldDims[2],dimOfThreadBlock)
    z=cld(arrGoldDims[3],dimOfThreadBlock)


    #we need just some blocks to set in the schedule for the begining - the scheduling block will refine it 
    # workNumbPerBlock = cld(x*y*z,dataBlocksNum)
    # workScheduleCPU = zeros(UInt16,blocksNum, workNumbPerBlock  ,4)
    # Threads.@threads for i in 1:blocksNum
    #                         for j in 1:workNumbPerBlock
    #                         workScheduleCPU
    #                         end    
    #                     end


    workSchedule= CUDA.zeros(UInt16,blocksNum, cld(x*y*z,dataBlocksNum)  ,4) #in each entry 1) x;2)y;3)z;4)1 if gold standard pass and 2 if segm pass  ; length is set so we will have approximately equal number of blocks to work on


    metaData= CUDA.zeros(UInt16,x,y,z,12)
    resArray=CUDA.zeros(UInt16, blocksNum,maxresultPoints, 4)#in each entry 1) x;2)y;3)z;4)1 if gold standard pass and 2 if segm pass 
    localRes= CUDA.zeros(UInt16,maxresultPoints/2,4  ) #maxresultPoints/2 very conservative  we may experiment with far smaller number to  decrese memory usage
    localResLastEntryList= CUDA.zeros(UInt16, blocksNum) # 1 entry per thread block
    innerLoopStep= CUDA.zeros(UInt16, blocksNum) # 1 entry per thread block
    worksScheduleLastStep= CUDA.zeros(UInt16, blocksNum) # 1 entry per thread block
    isWorking = CUDA.zeros(Bool, blocksNum) # 1 entry per thread block
    # for boolean arrays that will store all boolean data about analyzed array#we have two copies as Housdorff is composed of 2 passes 
    reducedGoldA= CUDA.zeros(Bool,arrGoldDims)
    reducedSegmA =CUDA.zeros(Bool,arrGoldDims)
    
    reducedGoldB= CUDA.zeros(Bool,arrGoldDims)
    reducedSegmB =CUDA.zeros(Bool,arrGoldDims)

return (metaData)
end#allocateMomory


end#SimplerHousdorff





    # #first we choose biggest dimension and this will be dimension of elongated block
    # arrGoldDims= size(arrGold)
    # maxDim = argmax(size(arrGold))
    # localArr = [0,0,0]
    # for i in 1:3
    #     if(i!=maxDim)
    #         localArr[i]= cld(arrGoldDims[i],32)
    #     else
    #         #we are now in max dimension
    #         localArr[i]= cld(arrGoldDims[i],maxThrPerB)
    #     end#if    
    # end#for



# """
# getting dimensions  needed to run number of blocks that will cover all of data to prepare metadata  
# metasataDims - dimensions of metasata array - 4 dims tuple
# maxThrPerB - maximum number of threads per block defoult is 1024
#     return the information to the kernel how to create its blocks of threads (how many); and the number of threads in a block 
# """
# function getBlockNumb(metasataDims::Tuple{Int64, Int64, Int64, Int64},maxThrPerB::Int64=1024)  

#     return ( metasataDims[3] ,maxThrPerB)
# end#getBlockNumb


    
# """
# kernel that output metadata for Housedorff
#     matadata - Int16 4 dimensional array where x,y,z dims identical to data blocks  and 4 dim stores 6 numbers 
#         Is activeOrFullForFirsPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
#         Is activeOrFullForFirsPasssegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
#         Is activeOrFullForSecondPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
#         Is activeOrFullForSecondPassSegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
#         activeIterNumbMaskA - The number of iteration when block was last active - needed to coordinate which block will be working now
#         activeIterNumbMaskB - The number of iteration when block was last active - needed to coordinate which block will be working now
#         metadataDims - dimensions of metadata - 4 dims tuple
#         arrGoldDims - dimensions of the main array
#     IMPORTANT - althought this is 1 dimensional kernel we will output 3 dim data
#         so blockIdx().x - gives info about z - dimension 
#         threadIdxX() - tell about position in x dimension
#         in y dimension we will iterate in a loop the number that is equal to the number of the size of the  y dimension (remember we are talking about metadata array)


# """
# function housedorffMetadataKernel(matadata
#                                 ,metadataDims::Tuple{Int64, Int64, Int64, Int64}
#                                 ,arrGoldDims::Tuple{Int64, Int64, Int64} )
#     # each lane will be responsible for one meta data  
#     @unroll for k in 0:metadataDims[1]
#         matadata[threadIdxX(),k+1, blockIdx().x,1]=(threadIdxX()-1)*32   #min x
#         matadata[threadIdxX(),k+1, blockIdx().x,2]=min(threadIdxX()*32,arrGoldDims[1])   #max x
#         matadata[threadIdxX(),k+1, blockIdx().x,3]=k*32   #min y
#         matadata[threadIdxX(),k+1, blockIdx().x,4]=min((k+1)*32,arrGoldDims[2] )   #max y
#         matadata[threadIdxX(),k+1, blockIdx().x,5]=(blockIdx().x-1)*32   #min z
#         matadata[threadIdxX(),k+1, blockIdx().x,6]=min((blockIdx().x)*32,arrGoldDims[3])   #max z
#     end#for
# end #housedorffMetadataKernel
