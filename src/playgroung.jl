using CUDA,Cthulhu
arr1=  CUDA.ones(30,30,30) ;  # 3 dim array of ones
arr2=  CUDA.ones(30,30,30).*2; # 3 dim array of two’s
res = CUDA.zeros(30,30,30); # preallocation of memory
function kernelFunction(arr1::CuDeviceArray{Float32, 3, 1}, arr2::CuDeviceArray{Float32, 3, 1}, res::CuDeviceArray{Float32, 3, 1})
    # getting all required indexes

    i= (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z  
    #shmem = @cuDynamicSharedMem(Float32, (blockDim().x, blockDim().y,blockDim().z,2))
       shmem = @cuStaticSharedMem(Float32, (8, 8,8,2))           
       @inbounds shmem[threadIdx().x ,threadIdx().y ,threadIdx().z,1] = arr1[i,j,z]
       @inbounds shmem[threadIdx().x ,threadIdx().y ,threadIdx().z,2] = arr2[i,j,z]
       sync_threads() 

      @inbounds res[i, j,z] = shmem[threadIdx().x ,threadIdx().y ,threadIdx().z,1] + shmem[threadIdx().x ,threadIdx().y ,threadIdx().z,2]
    return
    end
    n=30*30*30

    # kernel = @cuda launch=false kernelFunction(arr1, arr2, res)
    # config = launch_configuration(kernel.fun, shmem=threads-> 60 * sum(threads) * sizeof(Float32))
    @cuda threads=(8,8,8) blocks=ceil(Int,n/8*8*8)  kernelFunction(arr1, arr2, res) #shmem= 10*10*10*2 

sum(res)

sum(arr2)+ sum(arr1) == sum(res)

(8*8*8)/32

cbrt(512)







# using CUDA,Cthulhu
# arr1=  CUDA.ones(30,30,30) ;  # 3 dim array of ones
# arr2=  CUDA.ones(30,30,30).*2; # 3 dim array of two’s
# res = CUDA.zeros(30,30,30); # preallocation of memory
# function kernelFunction(arr1::CuDeviceArray{Float32, 3, 1}, arr2::CuDeviceArray{Float32, 3, 1}, res::CuDeviceArray{Float32, 3, 1})
#     # getting all required indexes

#     i= (blockIdx().x-1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y-1) * blockDim().y + threadIdx().y
#     z = (blockIdx().z-1) * blockDim().z + threadIdx().z  
#     # CUDA.@cuprint "i $(i) ; (blockIdx().x-1) $(blockIdx().x-1) ; blockDim().x $(blockDim().x) ; threadIdx().x $(threadIdx().x)               \n "
#     # CUDA.@cuprint "j $(j) ; blockIdx().y $(blockIdx().y-1) ; blockDim().y $(blockDim().y) ; threadIdx().y $(threadIdx().y)               \n"
#     # CUDA.@cuprint "z $(z) ; blockIdx().z $(blockIdx().z-1) ; blockDim().z $(blockDim().z) ; threadIdx().z $(threadIdx().z)               \n"



#        shmem = @cuStaticSharedMem(Float32, (10, 10,10))           
#        @inbounds shmem[threadIdx().x ,threadIdx().y ,threadIdx().z] = arr1[i,j,z]+arr2[i,j,z]
#        sync_threads() 

#       @inbounds res[i, j,z] = shmem[threadIdx().x ,threadIdx().y ,threadIdx().z]
#     return
#     end
#     n=30*30*30
#     @cuda threads=(3,3,3) blocks=ceil(Int,n/10*10*10) kernelFunction(arr1, arr2, res)

# sum(res)

# sum(arr2)+ sum(arr1) == sum(res)
