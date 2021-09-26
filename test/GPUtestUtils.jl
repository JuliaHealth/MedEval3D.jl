

function fillGlobalFromShmem(testArrInn,resShmem)
    for z in 1:34   
        testArrInn[threadIdxX()+1,threadIdxY()+1,z ]=resShmem[threadIdxX()+1,threadIdxY()+1,z ]
    end    
    
        testArrInn[1,threadIdxX()+1,threadIdxY()+1]=  resShmem[1,threadIdxX()+1,threadIdxY()+1]
        testArrInn[34,threadIdxX()+1,threadIdxY()+1]=  resShmem[34,threadIdxX()+1,threadIdxY()+1]
        testArrInn[threadIdxX()+1,1,threadIdxY()+1]=  resShmem[threadIdxX()+1,1,threadIdxY()+1]
        testArrInn[threadIdxX()+1,34,threadIdxY()+1]=  resShmem[threadIdxX()+1,34,threadIdxY()+1]
 

end


function getIndiciesWithTrue(arr)
    indicies = CartesianIndices(arr)
    return filter(ind-> arr[ind] ,indicies)


end


# function fillGlobalFromShmem(testArrInn,resShmem)
#     for z in 1:34   
#         testArrInn[threadIdxX()+1,threadIdxY()+1,z ]=resShmem[threadIdxX()+1,threadIdxY()+1,z ]
#         sync_threads()
#     end    
    
#     for z in 1:32
#         testArrInn[threadIdxX()+1,threadIdxY()+2,z+1 ]=  resShmem[threadIdxX()+1,threadIdxY()+2,z+1 ]
#         sync_threads()
#         testArrInn[threadIdxX()+1,threadIdxY(),z+1 ]=  resShmem[threadIdxX()+1,threadIdxY()+2,z+1 ]
#         sync_threads()
#         testArrInn[threadIdxX()+2,threadIdxY()+1,z+1 ]=  resShmem[threadIdxX()+2,threadIdxY()+1,z+1 ]
#         sync_threads()
#         testArrInn[threadIdxX(),threadIdxY()+1,z+1 ]=  resShmem[threadIdxX(),threadIdxY()+1,z+1 ]
#         sync_threads()
#     end

# end