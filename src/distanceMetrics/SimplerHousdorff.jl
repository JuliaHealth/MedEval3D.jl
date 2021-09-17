"""
So we will need the block of data representing each wall of the bit cube bit cuve will have edge length of 32 - apart from corner cases

Simplification
Get reduced array of bits from Tpfp..fn modified kernel
Assign blocks of size 32x32x32 for each we will have data structure holding data as pointed below all will be in UInt16
    Min x max x; miny max y; min z max z - first 6 numbers
    All adjacent blocks indices next 26 numbers
    Is activeOrFullForFirsPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForFirsPasssegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassSegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
   activeIterNumbMaskA - The number of iteration when block was last active - needed to coordinate which block will be working now
    activeIterNumbMaskB - The number of iteration when block was last active - needed to coordinate which block will be working now
    In total 6+26+1+1=34 numbers in array for a block
    Data will be stored in n times 34 UInt16 matrix in global memory
ResArray storing all coordinates that were covered and in which iteration it occured

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

"""
divide array into blocks and assigns the metadata for each 
    Min x max x; miny max y; min z max z - first 6 numbers
    All adjacent blocks indices next 26 numbers
    Is activeOrFullForFirsPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForFirsPasssegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassgold - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    Is activeOrFullForSecondPassSegm - 2 if neither in first mask , 1 if full in first mask; 0 if active in first mask 
    activeIterNumbMaskA - The number of iteration when block was last active - needed to coordinate which block will be working now
    activeIterNumbMaskB - The number of iteration when block was last active - needed to coordinate which block will be working now
    Data will be stored in n times 34 UInt16 matrix in global memory

So in order to achieve it we need first to get the dimensions of the array we got - as it is reduced array it do not have to have the dimensions that would be simmilar
to the medical image dimensions 
As we start from place where x,y,z is 0 and we will proceed (concurrently from here )
        we know by construction that min x, min y, and min z will be always some multiply of 32 away from begining 
        now we can lounch multiple small blocks where each will calculate the the required metadata - amount of those blocks will be just equal to the number of blocks tha we want to describe 
        and will be calculated by ceildiv of given dimension so sth like sizz = size[arrGold]-> will be mapped
        to blocks as sizz-> (celdiv(sizz[1]) ...) by all dimensions - let's make those blocks the size of the warp so 32

        Now (blockIdx-1) *32 will be min x of a block and blockIdx*32 max x ; the same for other dimensions 
        we calculate linear index to have position of the given block; we do the same  calculation for blocks that are -1 and +1 in each dimension ... so 26 blocks to analyze 
        futher numbers are already correctly initialized to 0 


 arrGold - boollean 3 dim Cu array with gold standard
 arrSegm - segmentation 3 dim boolean Cu array we want to compare   
"""
function getBlockMetaData(arrGold, arrSegm)


end#getBlockMetaData

end#SimplerHousdorff