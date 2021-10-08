"""
not finished experiments on using WMMA tensor cores
"""
# """
# calculates Mahalanobis distance between two segmentations
# 1)so easist is to think about those segmentations as sparse matricies where we store in 3 columns x,y,z coordinates of all entries where we have true
# 	-we need mean from those entries only we ignore zeros
# 	- hence first pass can be iterating over both arrays - do the reduction understood in two ways
# 		- we need 3 sums and count of the x , y znad z components - in order to achieve this we can apply classic reduction techniques
# 		all of blocks will accumulate sum of elements  and count, it will also add point coordinates (x,y,z) to shared memory queue - if shared memory queue will
# 		get filled to its capacity we will update the main global sparse matrix representation using atomics to update the counter
# 		- summation will be done as ussual using first accumulation in private variables than summation across 
# 2) so we should have 3 sums (of x,y and z coordinates) and their count and a 3 column table with all x,y,z coordinates of the non empty entries		
# 	- we will have this data for both  arrays - representing gold standard and other segmentation
# 	now we need to calculate covariance matriscies of each point using global means and accumulate (add) all of those matricies
# 	We will use WMMA to speed things up - and in order to utilize WMMA constraints we will write calculation of covariance matrix in 
# 	matrix multiplication notation look into https://docs.google.com/document/d/1G5FqjRj7WrDs4LWtBH667_orA8HzGa_JtTQ5Yb8vu8E/edit
# 	- in order to get those matricies we just need to load values prepared in step 1 - probably best way would be to keep means in shared memory 

# https://github.com/JuliaGPU/GemmKernels.jl/blob/40c1dacb2ff2d24d7d392fc784c389e5e7fd8307/src/kernel.jl

# example of WMMA https://github.com/JuliaGPU/CUDA.jl/blob/800b7b89c4c19b9b98a7d150a813a0e3d0e18be5/examples/wmma/high-level.jl
# look into https://math.stackexchange.com/questions/4240707/mahalanobis-distance-between-two-3-dimensional-boolean-arrays


# taken from https://github.com/JuliaGPU/GemmKernels.jl/blob/40c1dacb2ff2d24d7d392fc784c389e5e7fd8307/src/kernel.jl
# - probably how to load element by element to fragment

# """
# module Mahalanobis
# export WMMAkernel
# using CUDA,Main.IterationUtils, Main.ReductionUtils, Main.MemoryUtils
# """

# First we upload means meanx, mean y and mean z to shared memory 
# We defined shared memory for fragmentA , fragmentB and C - A and B we will define on every iteration 
# 	fragment C will accumulate all
# Before we start we need to set all shared memory to 0

# We want to iterate over our arrays - first the gold than other one
# we will check is in given spot of the array the predicate of equality is met - if yes we will write the output into shared memory fragments and update local counter
# counters and fragments will be warp owned and managed 
# every time the  true will be found the counter will be updated and on this basis value (old value from counter) will be put into the fragA and fragB
# x loop 
# 	- every time we will finish  the iteration of x loop we will check the warp counter and on the basis of it we will either continue  or not
# 	in the simplest version if warp private counter will be greater  than (224-33) we will update fragmentA and B  execute WMMA 
# 	- prepareing the frag A and B will be mainly related to fill with as many (xses - meanx) as we have first  14 rows for frag A and first 14 columns in B 
# 		last columns/rows need to be filled by current (y - mean),(z-mean)	
# 	-when we will fill the matrix before pushing it into WMMA we need to add some modificatiions 
# 		- in the main diagonal we have variances of x entries (apart from last 2 in basic algorithm) - here we do not need to achange anything 	
# 		- in last two entries of main diagonal we will have 16 y variances and 16 z variances - this number needs to adjusted to be equal to local counter
# 			so we divide by 16 and multiply by local counter value
# 		- in entries [15,16] and [16,15] we will have yz covariance 	
# 		- covariances of xses with x and y will be in last two columns apart from right bottom corner	
# 	-after each WMMA we need to set all entries of fragments and counter to 0 
# y loop 
#    - we check wheather result counter evaluated to more than 0 if yes we fire procedure described above if not we get to next y we are intrested in
# z loop 
# 	we check weather our results are not empty by checking the value of [1,1] of frag A if it is grater than 0	
#     - in case we are intreseted in slicewise results we save the covariance data into  global memory slice wise covariance data
# 	- in this case we reset	
# after we got through all iterations we add data about covariances to global memory (atomically)	


# goldArr,segmArr  - arrays we analyze 
# numberToLooFor - number we are intrested in in the array
# arrDims - dimensions of main array
# loopXdim, loopYdim,  loopZdim - indicates how many times we need to loop over dimensions to cover all
# """
# function WMMAkernel(goldArr,segmArr
#     ,numberToLooFor
# 	,arrDims, loopXdim, loopYdim,  loopZdim
# 	 )

# 	warpNum = 1 
# 	#store matricies we will multiply
# 	warpFragA = @cuStaticSharedMem(Float16, (16,16))
# 	warpFragB = @cuStaticSharedMem(Float16, (16,16))
# 	#matrix with accumulated results
# 	warpFragC = @cuStaticSharedMem(Float16, (16,16))
# 	#keeps track how many x true entries we have 
# 	warpXCounter = @cuStaticSharedMem(UInt16, (1))
# 	#resetting all
# 	@ifXY warpNum 1 warpXCounter=UInt16(0)
# 	#resetting using multiple warps
# 	@ifY warpNum begin clearFrag(warpFragA) ;  clearFrag(warpFragB)  ;clearFrag(warpFragC)  end

# 	#iterating over data looking for places that meet our predicate
# 	@iter3dAdditionalxyzActs(arrDims,loopXdim,loopYdim,loopZdim
# 	#expression in very center
# 	,begin    end
# 	#additionalActionAfterX
# 	,begin    end
# 	#additionalActionAfterY
# 	,begin    end
# 	#additionalActionAfterZ
# 	,begin    end
# 	) 

# 	 conf = WMMA.Config{16, 16, 16, Float16}
# 	 a_frag = WMMA.load_a(pointer(fragA), 16, WMMA.ColMajor, conf)
# 	 b_frag = WMMA.load_b(pointer(fragB), 16, WMMA.ColMajor, conf)
# 	 c_frag = WMMA.load_c(pointer(ones), 16, WMMA.ColMajor, conf)
# 	d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
# 	WMMA.store_d(pointer(d_out), d_frag, 16, WMMA.ColMajor, conf)
	 
# 		 return
# 	end 
		

# end#module


# """
# we have a 16 by 16 fragment and we need to fill it with 0s
# """
# function clearFrag(frag)
# 	@unroll for i in 1:8
# 		#indexing so for example on first iteration so i=1 when threadIdx<=16 we will cover in that wayy all first row ; and if it is bigger than 16 we will covering second row 
# 		frag[((i-1) & (2^4 - 1))+1,(((threadIdxX()-1)>>4 )+1)*i]= Float16(0)
# 	end#for
# end#clearFrag



# # function WMMAkernel(dataShmem,ones,d_out,fragA,fragB)
# # 	#  fragA = @cuStaticSharedMem(Float16, (4,4))
# # 	#  fragB = @cuStaticSharedMem(Float16, (4,4))
# # 	# fragC = @cuStaticSharedMem(Float16, (4,4))

# # 	 if(threadIdxX()<17)
# # 		a =  ((threadIdxX()-1) & (3))+1
# # 		b = ~((threadIdxX()+3)>>2 & 1)+3
# # 		c = ((threadIdxX()-1)>>2 )+1
# # 		fragA[a,c] = dataShmem[b,a]*( ((threadIdxX()>4 && threadIdxX()<13)*-2)+1 )
# # 	else
# # 		a = ((threadIdxX()-1) & (3))+1
# # 		c = (((threadIdxX()-16)-1)>>2 )+1
# # 		d = ((threadIdxX()-17) >>3)+1
# # 		fragB[ c,a] = dataShmem[ d,a ] 	
# # 	end	
# # 	 conf = WMMA.Config{16, 16, 16, Float16}
# # 	 a_frag = WMMA.load_a(pointer(fragA), 16, WMMA.ColMajor, conf)
# # 	 b_frag = WMMA.load_b(pointer(fragB), 16, WMMA.ColMajor, conf)
# # 	 c_frag = WMMA.load_c(pointer(ones), 16, WMMA.ColMajor, conf)
# # 	d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
# # 	WMMA.store_d(pointer(d_out), d_frag, 16, WMMA.ColMajor, conf)
	 
# # 		 return
# # 	end





