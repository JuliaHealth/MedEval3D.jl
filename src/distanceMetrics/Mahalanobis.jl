
"""
calculates Mahalanobis distance between two segmentations
1)so easist is to think about those segmentations as sparse matricies where we store in 3 columns x,y,z coordinates of all entries where we have true
	-we need mean from those entries only we ignore zeros
	- hence first pass can be iterating over both arrays - do the reduction understood in two ways
		- we need 3 sums and count of the x , y znad z components - in order to achieve this we can apply classic reduction techniques
		all of blocks will accumulate sum of elements  and count, it will also add point coordinates (x,y,z) to shared memory queue - if shared memory queue will
		get filled to its capacity we will update the main global sparse matrix representation using atomics to update the counter
		- summation will be done as ussual using first accumulation in private variables than summation across 
2) sow we should have 3 sums (of x,y and z coordinates) and their count and a 3 column table with all x,y,z coordinates of the non empty entries		
	- we will have this data for both 

example of WMMA https://github.com/JuliaGPU/CUDA.jl/blob/800b7b89c4c19b9b98a7d150a813a0e3d0e18be5/examples/wmma/high-level.jl
"""
module MahalanobisDist

look into https://math.stackexchange.com/questions/4240707/mahalanobis-distance-between-two-3-dimensional-boolean-arrays

"""
calculating means 
  - for 2 dimensional case we need just 2 numbers
  - for 3 dimensional case we need 3
"""
function calcMean()
  
mat #static vector with 3 entries 
	while ( xxx ){
			double val =it.Get(); # given value of a voxel
			if(val>thd){ # accepting voxel as "true" only if it bigger than treshold (applied for fuzzy case)
				ImageType::IndexType index = it.GetIndex(); # getting index 
				mat[0] += index[0];
				mat[1] += index[1];
				mat[2] += index[2];
				count++; # to know how many had been above treshold !!
			}
			++it;
		}
		mat = mat/count;
  
  
  
  
end#function


		
		
		
		
### START
using Test

using CUDA

a     = rand(Float16, (16, 16))
b     = rand(Float16, (16, 16))
c     = rand(Float32, (16, 16))

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(c)
d_dev = similar(c_dev)

function kernel(a_dev, b_dev, c_dev, d_dev)
    conf = WMMA.Config{16, 16, 16, Float32}

    a_frag = WMMA.load_a(pointer(a_dev), 16, WMMA.ColMajor, conf)
    b_frag = WMMA.load_b(pointer(b_dev), 16, WMMA.ColMajor, conf)
    c_frag = WMMA.load_c(pointer(c_dev), 16, WMMA.ColMajor, conf)

    c_frag = 0.5f0 .* c_frag

    d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)

    WMMA.store_d(pointer(d_dev), d_frag, 16, WMMA.ColMajor, conf)

    return
end

@cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
d = Array(d_dev)

@test all(isapprox.(a * b + 0.5 * c, d; rtol=0.01))
### END
		
		
		
		
		
		
		

end#module








