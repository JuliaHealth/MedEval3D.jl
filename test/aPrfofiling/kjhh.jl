using CUDA
ff = CUDA.zeros(3)
function gggg(ff)

  shmemblockDataa = @cuDynamicSharedMem(Float32,(1,2))
    return
end
@cuda threads=(32,2) blocks=2 shmem = 3 gggg(ff)
ff[1]