 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\kernelEvolutions.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")
using Main.BasicPreds, Main.GPUutils,CUDA,Cthulhu,BenchmarkTools,Revise 

goldBoolGPU,segmBoolGPU,tp,tn,fp, fn,blockNum, nx,ny,nz,xthreads, ythreads,zthreads  = getSmallTestBools()

function kernelFunction(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},tp,tn,fp, fn,nx,ny,nz,xthreads, ythreads,zthreads)
    # getting all required indexes
    i= (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z  
    #if(i< (nx*ny*nz)/ythreads ) #i<nx*ny && j<ny && z<nz
   # CUDA.@cuprint "goldBoolGPU[i,j,z] $(goldBoolGPU[i,j,z]) segmBoolGPU[i,j,z] $(segmBoolGPU[i,j,z]) i $(i) j $(j) z $(z) "
        if(goldBoolGPU[i,j,z] & segmBoolGPU[i,j,z] )
            @atomic tp[]+=1
        elseif (!goldBoolGPU[i,j,z] & !segmBoolGPU[i,j,z] )
            @atomic tn[]+=1
        elseif (!goldBoolGPU[i,j,z] & segmBoolGPU[i,j,z] )
            @atomic fp[]+=1    
        elseif (goldBoolGPU[i,j,z] & !segmBoolGPU[i,j,z] )
            @atomic fn[]+=1    
        end
    #else
     #   CUDA.@cuprint "i $(i) j $(j) z $(z)  \n"


    return  

    end

# @benchmark CUDA.@sync  blockNum
@cuda threads=(xthreads, ythreads,zthreads) blocks=9 kernelFunction(goldBoolGPU,segmBoolGPU,tp,tn,fp, fn, nx,ny,nz,xthreads, ythreads,zthreads) #shmem= 10*10*10*2  threads=(8,8,8)   blocks=ceil(Int,n/8*8*8)

tn[1]


2100/256

2046-1790

tp[1]==2 && tn[1]==(nx*ny*nz)-6 && fp[1]==1 && fn[1]==3

(nx*ny*nz)/(xthreads*ythreads*zthreads)
#@cuda threads=(8,8,8) blocks=blockNum  kernelFunction(goldBoolGPU,segmBoolGPU,tp,tn,fp, fn) #shmem= 10*10*10*2  threads=(8,8,8)   blocks=ceil(Int,n/8*8*8)




using CUDA
goldBoolGPU= CuArray(falses(16,16,2));
segmBoolGPU= CuArray(falses(16,16,2));
fn = CuArray([0])
function kernelFunct(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},fn)
    i= (blockIdx().x) * blockDim().x + threadIdx().x
    j = (blockIdx().y) * blockDim().y + threadIdx().y
    z = (blockIdx().z) * blockDim().z + threadIdx().z 

    if (goldBoolGPU[i,j,z] & !segmBoolGPU[i,j,z] )
        @atomic fn[]+=1    
        end
    return  
    end
@cuda threads=(4, 4,1) blocks=32  kernelFunct(goldBoolGPU,segmBoolGPU,fn) 
#I get error ERROR: Out-of-bounds array access.


fn

(16*16*2)/(4*4)


128*128*2/16

    # kernel = @cuda launch=false kernelFunction(arr1, arr2, res)
    #config = launch_configuration(kernel.fun, shmem=threads-> 2 * sum(threads) * sizeof(Float32))
    




#z= view(arr1,:,:,1) # in such configuration data is contiguous  Base.iscontiguous(x)
#arr1=  CUDA.ones(10,128,128) ;  # 3 dim array of ones



    # CUDA.@cuprint "i $(i) ; (blockIdx().x-1) $(blockIdx().x-1) ; blockDim().x $(blockDim().x) ; threadIdx().x $(threadIdx().x)               \n "
    # CUDA.@cuprint "j $(j) ; blockIdx().y $(blockIdx().y-1) ; blockDim().y $(blockDim().y) ; threadIdx().y $(threadIdx().y)               \n"
    # CUDA.@cuprint "z $(z) ; blockIdx().z $(blockIdx().z-1) ; blockDim().z $(blockDim().z) ; threadIdx().z $(threadIdx().z)               \n"
    #CUDA.@cuprint "goldBoolGPU[i,j,z] $(goldBoolGPU[i,j,z]) segmBoolGPU[i,j,z] $(segmBoolGPU[i,j,z]) i $(i) j $(j) z $(z)  \n"



    # function kernelFunction(goldBoolGPU::CuDeviceArray{Bool, 3, 1}, segmBoolGPU::CuDeviceArray{Bool, 3, 1},tp,tn,fp, fn)
    #     # getting all required indexes
    #     i,j,z = defineIndicies()
    #         if(goldBoolGPU[i,j,z] && segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint "  gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) tp + \n"
    #         @atomic tp[]+=1
    #         elseif (!goldBoolGPU[i,j,z] && !segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint "  gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) tn +\n"
    #         @atomic tn[]+=1
    #         elseif (!goldBoolGPU[i,j,z] && segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint " gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z])  fp +\n"      
    #         @atomic fp[]+=1    
    #         elseif (goldBoolGPU[i,j,z] && !segmBoolGPU[i,j,z] )
    #         CUDA.@cuprint " gold $(goldBoolGPU[i,j,z])  segm  $(segmBoolGPU[i,j,z]) fn +\n"       
    #         @atomic fn[]+=1    
    #         end
    #       return  
    #     end