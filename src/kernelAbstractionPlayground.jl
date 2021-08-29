const BACKEND = :CUDA
using Pkg,KernelAbstractions, Adapt, OffsetArrays,CUDAKernels
Pkg.activate(string(BACKEND, "Env"))
using CUDA, CUDAKernels


@kernel function saxpy!(z, α, x, y)
    I = @index(Global)
    @inbounds z[I] = α * x[I] + y[I]
end

"""
The workgroupsize is a local block of threads that are co-executed. The ndrange specifies the global index space
. This index space will be subdivided by the workgroupsize and each group will be executed in parallel.
"""
ndrange = (128,)
workgroupsize = (16,)
blocks, workgroupsize, dynamic = KernelAbstractions.NDIteration.partition(ndrange, workgroupsize)

"""
Initialization
"""
kernel = saxpy!(KernelAbstractions.Device())
x = adapt(ArrayT, rand(64, 32))
y = adapt(ArrayT, rand(64, 32))
z = similar(x)

