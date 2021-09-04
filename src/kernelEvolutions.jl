"""
Holding developed kernels  in progression when new optimazation is povided
in order to be able to benchmark each on given data 
"""
module BasicPreds
using CUDA
export getBigTestBools,getSmallTestBools


function getSmallTestBools()

    nx=32
    ny=32
    nz=16
    #sets nuber of threads in thread blocks
    xthreads = 16
    ythreads = 16
    zthreads = 1
    #first we initialize the metrics on CPU so we will modify them easier
    goldBool= falses(nx,ny,nz); #mimicks gold standard mask
    segmBool= falses(nx,ny,nz); #mimicks mask     
# so we  have 2 cubes that are overlapped in their two thirds
    cartTrueGold =  CartesianIndices(zeros(3,3,5) ).+CartesianIndex(5,5,5);
    cartTrueSegm =  CartesianIndices(zeros(3,3,3) ).+CartesianIndex(4,5,5); 
    goldBool[cartTrueGold].=true
    segmBool[cartTrueSegm].=true



    #for storing output total
    tp = CuArray([0]);
    tn = CuArray([0]);
    fp = CuArray([0]);
    fn = CuArray([0]);
    #for storing metrics for slice
    tpArr = CuArray(ones(Int16,nz));
    tnArr = CuArray(ones(Int16,nz));
    fpArr = CuArray(ones(Int16,nz));
    fnArr = CuArray(ones(Int16,nz));
### calculating correct results (unoptimazied way) just for unit testing

FlattG = vec(goldBool);
FlattSeg = vec(segmBool);

FlattGoldGPU= CuArray( FlattG)
FlattSegGPU= CuArray( FlattSeg )
FlattGoldGPU,FlattSegGPU
# total for all slices
# tpTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length
# tnTotalTrue = filter(pair->pair[2]== FlattB[pair[1]] ==false ,collect(enumerate(FlattG)))|>length
# fpTotalTrue = filter(pair->!pair[2] && FlattB[pair[1]] ,collect(enumerate(FlattG)))|>length
# fnTotalTrue = filter(pair->pair[2] && !FlattB[pair[1]] ==true ,collect(enumerate(FlattG)))|>length

# correct result per slice 
toIterSlices = collect(enumerate(collect(eachslice(goldBool, dims = 3)))) 
tpPerSliceTrue =  map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==true         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
tnPerSliceTrue = map(slicePair->   filter( pair->  pair[2]== vec(segmBool[:,:,slicePair[1]])[pair[1]] ==false         ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fpPerSliceTrue =  map(slicePair->   filter( pair->  !pair[2] && vec(segmBool[:,:,slicePair[1]])[pair[1]]       ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
fnPerSliceTrue =  map(slicePair->   filter( pair->  pair[2] && !vec(segmBool[:,:,slicePair[1]])[pair[1]]          ,  collect(enumerate(vec(collect(slicePair[2])))) )|>length  , toIterSlices   )
#sum for all dummy image
tpTotalTrue= tpPerSliceTrue|>sum 
tnTotalTrue= tnPerSliceTrue|>sum 
fpTotalTrue= fpPerSliceTrue|>sum 
fnTotalTrue= fnPerSliceTrue|>sum 

goldBoolGPU= CuArray( goldBool)
segmBoolGPU= CuArray( segmBool )

blockNum =Int64(ceil((nx*ny*nz)/(xthreads*ythreads*zthreads)));

    ## so there should be 9  true positives, 

# returning bunch of values so writing all will be simpler
    return (goldBoolGPU,segmBoolGPU,tp,tn,fp, fn, tpArr,tnArr,fpArr, fnArr, blockNum , nx,ny,nz,xthreads, ythreads,zthreads ,tpTotalTrue,tnTotalTrue,fpTotalTrue, fnTotalTrue ,tpPerSliceTrue,  tnPerSliceTrue,fpPerSliceTrue,fnPerSliceTrue ,FlattG, FlattSeg ,FlattGoldGPU,FlattSegGPU)
    
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


