"""
Holds collection of functions - mainly macros that when invoked from inside of the kernel will lead to iteration
Generally by convention we will use UInt32 as this is a format of threadid.x and other basic CUDA variables
arr - 3 dimensional data we analyze
maxX maxY maxZ- we will need also to supply data for variables called maxX maxY maxZ (if we have 3 dimensional loop) - those will create boundary checks  - for 2 dimensional case we obviously need only max X and max Y
minX, minY and minZ - this are assumed to be 1 if not we can supply them in the macro
loopX, loopY,loopZ - how many iterations should be completed while iterating over given dimension

we need also to supply functions of iterating the 3 dimensional data but with chosen dimension fixed - 
  so for example if we want to analyze top padding from shared memory we will set the dimension 3 at 1
"""
module IterationUtils
export @iter3d

"""
full 3d iteration 
arrDims - we will get maxX maxY maxZ from tuple with dimensions of the array analyzed
loopXdim, loopYdim,loopZdim - how many iterations should be completed while iterating over given dimension
ex - expression we want to invoke it will have x,y z
"""
macro iter3d(arrDims, loopXdim, loopYdim,loopZdim,ex  )
  return esc(quote
  @unroll for zdim::UInt32 in 0:$loopZdim
    z::UInt32 = (zdim*gridDim().x) + blockIdxX()#multiply by blocks to avoid situation that a single block of threads would have no work to do
    if( (z<= $arrDims[3]) )    
        @unroll for ydim::UInt32 in  0:$loopYdim
            y::UInt32 = ((ydim* blockDimY()) +threadIdxY())
              if( (y<=$arrDims[2])  )
                @unroll for xdim::UInt32 in 0:$loopXdim
                    x::UInt32=(xdim* blockDimX()) +threadIdxX()
                    if((x <=$arrDims[1]))
                      $ex
                    end#if x
               end#for x dim 
          end#if y
        end#for  yLoops 
    end#if z   
end#for z dim
end 
)

end#iter3d



# mainArr- array from which we want to get values
# minX::UInt32 =UInt32(0), minY::UInt32 =UInt32(0), minZ::UInt32 =UInt32(0)
# minX, minY and minZ - this are assumed to be 1 if not we can supply them in the macro



"""
iteration through 3 dimensional data but  with one dimension fixed - ie we will analyze plane or part of the plane where value of given dimension is as we supply it 
this can be used to analyze for example padding in the stencil - of course it may lead to non coalesced memory access 
chosenDim (1,2 or 3) - the dimension of choice through which we are NOT iterating
dimValue - value of this dimension that we will keep fixed
        DimA, DimB - those dimensions will be chosen to iterate over (macro will analyze those on its own - no need to supply them explicitely)
              - in case chosenDim = 1  DimA, DimB will be 2,3 
              - in case chosenDim = 2  DimA, DimB will be 1,3 
              - in case chosenDim = 3  DimA, DimB will be 1,2
maxDimA, maxDimB - maximum values available in source array
loopDimA, loopDimB - how many times we should iterate in those dimensions

"""
macro iterDimFixed(chosenDim, dimValue,maxDimA, maxDimB,loopDimA, loopDimB  )

end#iterDimFixed


end#module
