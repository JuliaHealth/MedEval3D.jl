
"""
calculates Mahalanobis distance between two segmentations
1)so easist is to think about those segmentations as sparse matricies where we store in 3 columns x,y,z coordinates of all entries where we have true
	-we need mean from those entries only we ignore zeros
	- hence first pass can be iterating over both arrays - do the reduction understood in two ways
		- we need 3 sums and count of the x , y znad z components - in order to achieve this we can apply classic reduction techniques
		all of blocks will accumulate sum of elements  and count, it will also add point coordinates (x,y,z) to shared memory queue - if shared memory queue will
		get filled to its capacity we will update the main global sparse matrix representation using atomics to update the counter
		- summation will be done as ussual using first accumulation in private variables than summation across 
2) so we should have 3 sums (of x,y and z coordinates) and their count and a 3 column table with all x,y,z coordinates of the non empty entries		
	- we will have this data for both  arrays - representing gold standard and other segmentation
	now we need to calculate covariance matriscies of each point using global means and accumulate (add) all of those matricies
	We will use WMMA to speed things up - and in order to utilize WMMA constraints we will write calculation of covariance matrix in 
	matrix multiplication notation look into https://docs.google.com/document/d/1G5FqjRj7WrDs4LWtBH667_orA8HzGa_JtTQ5Yb8vu8E/edit
	- in order to get those matricies we just need to load values prepared in step 1 - probably best way would be to keep means in shared memory 

https://github.com/JuliaGPU/GemmKernels.jl/blob/40c1dacb2ff2d24d7d392fc784c389e5e7fd8307/src/kernel.jl

example of WMMA https://github.com/JuliaGPU/CUDA.jl/blob/800b7b89c4c19b9b98a7d150a813a0e3d0e18be5/examples/wmma/high-level.jl
look into https://math.stackexchange.com/questions/4240707/mahalanobis-distance-between-two-3-dimensional-boolean-arrays


taken from https://github.com/JuliaGPU/GemmKernels.jl/blob/40c1dacb2ff2d24d7d392fc784c389e5e7fd8307/src/kernel.jl
- probably how to load element by element to fragment

"""
module Mahalanobis
export WMMAkernel
using CUDA, Main.CUDAGpuUtils
"""
First we upload means meanx, mean y and mean z to shared memory 
We defined shared memory for fragmentA , fragmentB and C - A and B we will define on every iteration 
	fragment C will accumulate all
boolMatr- We define boolean matrix of size identical to thread block - so each block will have one boolean to write to
	when it will heat true of predicate it will write to this shared memory matrix	
so we do on iteration each warp will mark i  
	some of threads in a warp my have a valid x in their local memory some will not we will know 
it by analyzing boolMatr
	first we sync warp - we do it after bounds check so we will not have  kernel blocked indefinitely
	so we do fast loop through 	indicies 1 to 32 from shared memory 
	when we will have true in this shared memory we
		sync warp
		using shfl operator we get access to x rest of needed variables is available from all threads
		now sadly we probably need to do big block of ifs to fill the matrix as designed in  https://docs.google.com/document/d/1G5FqjRj7WrDs4LWtBH667_orA8HzGa_JtTQ5Yb8vu8E/edit
		next we execute 
		we set all booleans in boolean matrix to 0s
after we got through the z iteration  if we are intrested in slice wise results 
	we check weather our accumulated matrix  has sth else than 0s  if yes we can calculate
		Mahalinobis for this slice 
after all iterations in a block we add covariance matrix to global memory so it will be accumulated 

dataShmem -  x,y,z and their global means
ones - 4 by 4 ones array - usefull for intialization of fragmentC
"""
function WMMAkernel(dataShmem,ones,d_out,fragA,fragB)
	#  fragA = @cuStaticSharedMem(Float16, (4,4))
	#  fragB = @cuStaticSharedMem(Float16, (4,4))
	# fragC = @cuStaticSharedMem(Float16, (4,4))

	 if(threadIdxX()<17)
		a =  ((threadIdxX()-1) & (3))+1
		b = ~((threadIdxX()+3)>>2 & 1)+3
		c = ((threadIdxX()-1)>>2 )+1
		fragA[a,c] = dataShmem[b,a]*( ((threadIdxX()>4 && threadIdxX()<13)*-2)+1 )
	else
		a = ((threadIdxX()-1) & (3))+1
		c = (((threadIdxX()-16)-1)>>2 )+1
		d = ((threadIdxX()-17) >>3)+1
		fragB[ c,a] = dataShmem[ d,a ] 	
	end	
	 conf = WMMA.Config{16, 16, 16, Float16}
	 a_frag = WMMA.load_a(pointer(fragA), 16, WMMA.ColMajor, conf)
	 b_frag = WMMA.load_b(pointer(fragB), 16, WMMA.ColMajor, conf)
	 c_frag = WMMA.load_c(pointer(ones), 16, WMMA.ColMajor, conf)
	# #  c_frag = 0.5f0 .* c_frag
	#d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
	d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
	d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
	WMMA.store_d(pointer(d_out), d_frag, 16, WMMA.ColMajor, conf)
	 
		 return
	end
	 
		

end#module








