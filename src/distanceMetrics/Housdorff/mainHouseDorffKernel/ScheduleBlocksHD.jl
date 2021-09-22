
"""
We want to analyze only those data blocks that are active - so neither full nor empty

mainWorkQueue - basically we need to add to the one dimensional queue all of the  active blocks we find in first pass- crerating basic queue that will be processed by normals passes
of course during those normal passes some blocks will be added to the queue and some will be removed in order to mage it we would need additional helper structures
so we will have: 
mainQuesCounter - telling us how many entries we have in work queue - we will divide this by number of thread blocks + some constant 
    - to get some amount of the data blocks to be processed + tail queue that will be accessed in atomic way - but it can be accessed by all blocks
    so if some block will finish work before others it will start processing this tail queue
mainActiveCounterNow,mainActiveCounterNext - at first main queue will have only active blocks but progressively it will have more and more empty spots
    so we need to get the second counter that will keep track on the ramaining active blocks in curent iteration  
        -mainActiveCounterNow - will be reduced every time thread block finish processing block - we will sync grid and start next iteration when it will reach 0 
        -mainActiveCounterNext - will be increased every time we activate some block - will become the  mainActiveCounterNow in next iteration if it will reach 0 we will 
            call it the end and finish kernel       
"""
module ScheduleBlocksHD
using CUDA, Main.GPUutils, Logging,StaticArrays




"""
run periodically in normal pass - each thread block is reponsible to update workplan for itself 
    - and add some additional to the general work queue so all thread blocks will get work for next iteration if possible
    we will get data about what blocks are active from analyzing metadata chunk that this block is responsible for 
 """
function updateWorkScheduleNormalPass(metadata
                                     ,mataDataDims)

end#updateWorkScheduleNormalPass    


"""
we have 3 dimensional data and in order to distribute work evenly and using consecutive memory chunks
    based on https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
"""
function getLinearIndexFrom3d()::UInt32
    blockId = blockIdx().x + blockIdx().y * gridDim.x
    + gridDim().x * gridDim().y * blockIdx().z

    return blockId * (blockDim().x * blockDim().y * blockDim().z)
    + threadIdx().z * blockDim().x * blockDim().y
    + threadIdx().y * blockDim().x + threadIdx().x
end



end#ScheduleBlocksHD