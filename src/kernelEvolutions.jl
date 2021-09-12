"""
Holding developed kernels  in progression when new optimazation is povided
in order to be able to benchmark each on given data 
"""
module BasicPreds
using CUDA
export getBigTestBools,getSmallTestBools


function getSmallTestBools()

    nx=512
    ny=512
    nz=317

    #first we initialize the metrics on CPU so we will modify them easier
    goldBool= zeros(Float32,nx,ny,nz); #mimicks gold standard mask
    segmBool= zeros(Float32,nx,ny,nz); #mimicks mask     
# so we  have 2 cubes that are overlapped in their two thirds
    cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
    cartTrueSegm =  CartesianIndices(zeros(3,3,3) ).+CartesianIndex(4,5,5); 
    goldBool[cartTrueGold].=1.0
    segmBool[cartTrueSegm].=1.0



    cartTrue =  CartesianIndices(zeros(9,9,9) ).+CartesianIndex(80,80,80);
    cartTrueB =  CartesianIndices(zeros(9,9,9) ).+CartesianIndex(200,200,200);

    goldBool[cartTrue].=1.0
    segmBool[cartTrue].=1.0


    #for storing output total first tp than TN than Fp and Fn
    tp= CuArray([0]);
    tn= CuArray([0]);
    fp= CuArray([0]);
    fn = CuArray([0]);
    
    #for storing metrics for slice
    tpArr = CuArray(zeros(Int16,nz));
    tnArr = CuArray(zeros(Int16,nz));
    fpArr = CuArray(zeros(Int16,nz));
    fnArr = CuArray(zeros(Int16,nz));
### calculating correct results (unoptimazied way) just for unit testing

# FlattG = vec(goldBool);
# FlattSeg = vec(segmBool);
ff = zeros(Float32,1000)
FlattG = vcat(vec(goldBool),ff)
FlattSeg = vcat(vec(segmBool),ff)


FlattGoldGPU= CuArray( FlattG)
FlattSegGPU= CuArray( FlattSeg )
# total for all slices
# tpTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length
# tnTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==false ,collect(enumerate(FlattG)))|>length
# fpTotalTrue = filter(pair->!pair[2] && FlattB[pair[1]] ,collect(enumerate(FlattG)))|>length
# fnTotalTrue = filter(pair->pair[2] && !FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length

# correct result per slice 

# toIterSlices =  collect(enumerate(collect(eachslice(goldBool, dims = 3)))) 
# tpPerSliceTrue =   map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==true         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# tnPerSliceTrue =  map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==false         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# fpPerSliceTrue =    map(slicePair->   filter( pair->  !pair[2] && vec(segmBool[:,:,slicePair[1]])[pair[1]]       ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# fnPerSliceTrue =   map(slicePair->   filter( pair->  pair[2] && !vec(segmBool[:,:,slicePair[1]])[pair[1]]          ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
# #sum for all dummy image
# tpTotalTrue=  tpPerSliceTrue|>sum 
# fpTotalTrue= fpPerSliceTrue|>sum 
# fnTotalTrue= fnPerSliceTrue|>sum 


toIterSlices = []#collect(enumerate(collect(eachslice(goldBool, dims = 3)))) 
tpPerSliceTrue = []# map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==true         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
tnPerSliceTrue = []#map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==false         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fpPerSliceTrue = []#  map(slicePair->   filter( pair->  !pair[2] && vec(segmBool[:,:,slicePair[1]])[pair[1]]       ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fnPerSliceTrue = []# map(slicePair->   filter( pair->  pair[2] && !vec(segmBool[:,:,slicePair[1]])[pair[1]]          ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
#sum for all dummy image
tpTotalTrue= 747# tpPerSliceTrue|>sum 
fpTotalTrue= 9#fpPerSliceTrue|>sum 
fnTotalTrue= 27#fnPerSliceTrue|>sum 


tnTotalTrue= (nx*ny*nz)-tpTotalTrue-fpTotalTrue-fnTotalTrue

goldBoolGPU= CuArray( goldBool)
segmBoolGPU= CuArray( segmBool )




blockNum = Int64(round(length(FlattG)/1024))

# array needs to hold 3 values tp, fp and false negatives from each block

intermediateResTp = CUDA.zeros(Int32, blockNum+2)
intermediateResFp = CUDA.zeros(Int32, blockNum+2)
intermediateResFn = CUDA.zeros(Int32, blockNum+2)

#intermediateResults = CUDA.zeros(Int32, Int64(((nx*ny*nz)/32)+100)  , 3)

    ## so there should be 9  true positives, 

# returning bunch of values so writing all will be simpler
    return (goldBoolGPU,segmBoolGPU,tp,tn,fp,fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,FlattG, FlattSeg ,FlattGoldGPU,FlattSegGPU,intermediateResTp,intermediateResFp,intermediateResFn)
    
    end
    





"""
most primitive working example
"""
function primitiveAtomicKernel(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp, fn,nx,ny,nz,xthreads, ythreads,zthreads)
        # getting all required indexes
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        #if(i< (nx*ny*nz)/ythreads ) #i<nx*ny && j<ny && z<nz
       # CUDA.@cuprint "goldBoolGPU[i,j,z] $(goldBoolGPU[i,j,z]) segmBoolGPU[i,j,z] $(segmBoolGPU[i,j,z]) i $(i) j $(j) z $(z) "
            if(goldBoolGPU[i] & segmBoolGPU[i] )
                @atomic tp[]+=1
            elseif (!goldBoolGPU[i] & !segmBoolGPU[i] )
                @atomic tn[]+=1
            elseif (!goldBoolGPU[i] & segmBoolGPU[i] )
                @atomic fp[]+=1    
            elseif (goldBoolGPU[i] & !segmBoolGPU[i] )
                @atomic fn[]+=1    
            end
        #else
         #   CUDA.@cuprint "i $(i) j $(j) z $(z)  \n"
    
    
        return  
    
        end



 end   


 



 """
 adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
 starting to using warp primitives 
 
 """
#  function kernelFunction(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp,fn,warpsInBlock::Int64)
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#    wid, lane = fldmod1(threadIdx().x, warpsize())

#    grid_handle = this_grid() # handle forsynchronizing cooperative groups 
#    shmemA = @cuStaticSharedMem(Int32, (26))           
#    #shmemB = @cuStaticSharedMem(Float32, (8, 8,8,2))           

#    #shared memory probably do not make sense as we are not intrested in the values more than once
#    @inbounds goldb::Bool =goldBoolGPU[i]
#    @inbounds segmb::Bool =segmBoolGPU[i] 
#    #using native function we calculate how many threads pass our criteria 
#    maskTp = vote_ballot_sync(FULL_MASK,goldb & segmb)  
#    maskFp = vote_ballot_sync(FULL_MASK,~goldb & segmb)  
#    maskFn = vote_ballot_sync(FULL_MASK,goldb & ~segmb)  
   
#    # generally values for  maskTp, maskFp, maskFn are constant across the warp  so in order to prevent adding the same number couple times we need modulo operator
#    #modul = threadIdx().x % 32
   
#    if(lane==1)

# #   CUDA.@cuprint "maskTp  $(maskTp)  val  $(CUDA.popc(maskTp)[1]) \n"  
# #   end  
#     @atomic tp[]+= CUDA.popc(maskTp)[1]*1
#     @atomic fp[]+= CUDA.popc(maskFp)[1]*1
#     @atomic fn[]+= CUDA.popc(maskFn)[1]*1

# end#if  

#    return  

#     end



"""
using warp reduce in a block

"""

# function kernelFunction(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp,fn,intermediateResults::CuDeviceMatrix{Int32, 1} )
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     blockId = blockIdx().x
#    wid, lane = fldmod1(threadIdx().x, warpsize())

#    #shared memory for storing results from warp reductions
#    shmemTp = @cuStaticSharedMem(Int32, (33))
#    shmemFp = @cuStaticSharedMem(Int32, (33))
#    shmemFn = @cuStaticSharedMem(Int32, (33))

   
#    @inbounds goldb::Bool =goldBoolGPU[i]
#    @inbounds segmb::Bool =segmBoolGPU[i] 
#    #using native function we calculate how many threads pass our criteria 
#    maskTp = vote_ballot_sync(FULL_MASK,goldb & segmb)  
#    maskFp = vote_ballot_sync(FULL_MASK,~goldb & segmb)  
#    maskFn = vote_ballot_sync(FULL_MASK,goldb & ~segmb)  
   
#    #we are adding on separate threads results from warps to shared memory
#     if(lane==1)
#         @inbounds  shmemTp[wid]= CUDA.popc(maskTp)[1]*1
#     elseif(lane==2) 
#         @inbounds shmemFp[wid]+= CUDA.popc(maskFp)[1]*1
#     elseif(lane==3)
#         if(CUDA.popc(maskFn)[1]>0)        
#         end    

#         @inbounds shmemFn[wid]+= CUDA.popc(maskFn)[1]*1
#     end#if  

# #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
# sync_threads()
#     # in case we have only 32 warps as we set we will not go out of bounds
#       if(wid==1 )
#         vallTp = reduce_warp(shmemTp[lane])
#         #probably we do not need to sync warp as shfl dow do it for us        
#         if(lane==1)
#             @atomic tp[]+=vallTp
#             end    
#        elseif(wid==2 )   
#         vallFp = reduce_warp(shmemFp[lane])
#         if(lane==1)
#             @atomic fp[]+=vallFp
#         end    
#        elseif(wid==3)  
#         vallFn = reduce_warp(shmemFn[lane])
#         if(lane==1)
#             @atomic fn[]+=vallFn
#         end  
#         end


#    return  
"""
using warp and get output to intermediate array
"""
# function getBlockTpFpFn(goldBoolGPU::CuDeviceArray{Bool,1, 1}, segmBoolGPU::CuDeviceArray{Bool, 1, 1},tp,tn,fp,fn,intermediateResTp::CuDeviceArray{Int32, 1, 1},intermediateResFp::CuDeviceArray{Int32, 1, 1},intermediateResFn::CuDeviceArray{Int32, 1, 1} )
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     blockId = blockIdx().x
#    wid, lane = fldmod1(threadIdx().x, warpsize())

#    #shared memory for storing results from warp reductions
#    shmemTp = @cuStaticSharedMem(Int32, (33))
#    shmemFp = @cuStaticSharedMem(Int32, (33))
#    shmemFn = @cuStaticSharedMem(Int32, (33))

   
#    @inbounds goldb::Bool =goldBoolGPU[i]
#    @inbounds segmb::Bool =segmBoolGPU[i] 
#    #using native function we calculate how many threads pass our criteria 
#    maskTp = vote_ballot_sync(FULL_MASK,goldb & segmb)  
#    maskFp = vote_ballot_sync(FULL_MASK,~goldb & segmb)  
#    maskFn = vote_ballot_sync(FULL_MASK,goldb & ~segmb)  
   
#    #we are adding on separate threads results from warps to shared memory
#     if(lane==1)
#         @inbounds  shmemTp[wid]= CUDA.popc(maskTp)[1]*1
#     elseif(lane==2) 
#         @inbounds shmemFp[wid]+= CUDA.popc(maskFp)[1]*1
#     elseif(lane==3)
#          @inbounds shmemFn[wid]+= CUDA.popc(maskFn)[1]*1
#     end#if  

# #now all data about of intrest should be in  shared memory so we will get all rsults from warp reduction in the shared memory
# sync_threads()
#     # in case we have only 32 warps as we set we will not go out of bounds
#       if(wid==1 )
#         vallTp = reduce_warp(shmemTp[lane])
#         #probably we do not need to sync warp as shfl dow do it for us        
#         if(lane==1)
#             @inbounds intermediateResTp[blockId]=vallTp
#         end    
#        elseif(wid==3 )   
#         vallFp = reduce_warp(shmemFp[lane])
#         if(lane==1)
#             @inbounds intermediateResFp[blockId]=vallFp
#         end    
#        elseif(wid==5)  
#         vallFn = reduce_warp(shmemFn[lane])
#         if(lane==1)
#             @inbounds intermediateResFn[blockId]=vallFn
#         end  
#         end
#    return  
#    end









# using CUDA

# A=rand(Bool,2,2,2)
# CuArray(view(A,: ))


# function getBigTestBools()
# goldBool= falses(128,128,10);#mimicks gold standard mask
# segmBool= falses(128,128,10);# mimicks mask 
# goldBool= falses(128,128,10);#mimicks gold standard mask
# segmBool= falses(128,128,10);# mimicks mask 
# cartTrueGold = [CartesianIndex(100,100,5), CartesianIndex(100,101,6), CartesianIndex(100,102,7)];
# cartTrueSegm = [CartesianIndex(99,100,5), CartesianIndex(100,101,6), CartesianIndex(100,102,7) ];
# CartesianIndex(100,100,5); # false negative;
# CartesianIndex(99,100,5); # false positive
# (CartesianIndex(100,101,6), CartesianIndex(100,102,7)); # true positive 
# goldBool[cartTrueGold].=true;
# segmBool[cartTrueSegm].=true;
# goldBoolGPU= CuArray(goldBool);
# segmBoolGPU= CuArray(segmBool);



# tp = CuArray([0])
# tn = CuArray([0])
# fp = CuArray([0])
# fn = CuArray([0])

# n = 128*128*10

# blockNum = Int64(ceil((n)/(8*8*8)))



# return (goldBoolGPU,segmBoolGPU,tp,tn,fp, fn,blockNum, n   )
# end


