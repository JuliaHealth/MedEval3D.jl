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
using CUDA,Logging
export @iter3d, @iter3dAdditionalxyzActsAndZcheck, @iter3dAdditionalxyzActs, @iter3dAdditionalzActs

"""
full 3d iteration 
arrDims - we will get maxX maxY maxZ from tuple with dimensions of the array analyzed
loopXdim, loopYdim,loopZdim - how many iterations should be completed while iterating over given dimension
ex - expression we want to invoke it will have x,y z
"""
macro iter3d(arrDims, loopXdim, loopYdim,loopZdim,ex)
  zOffset= :((zdim*gridDim().x))
  zAdd = :(blockIdxX())
  yOffset= :(ydim* blockDimY())
  yAdd= :(threadIdxY())
  xOffset= :(xdim* blockDimX())
  xAdd = :(threadIdxX())
  xCheck=:(x <=$arrDims[1])
  yCheck =:(y<=$arrDims[2])
  zCheck=:(z<= $arrDims[3])
  additionalActionAfterZ= :()
  additionalActionAfterY= :()
  additionalActionAfterX= :()
  is3d=true
  mainExp = generalizedItermultiDim(  loopXdim, loopYdim,loopZdim,zOffset,zAdd ,yOffset,yAdd,xOffset,xAdd,xCheck,yCheck,zCheck,additionalActionAfterZ,additionalActionAfterY,additionalActionAfterX,is3d,ex) ;
  
  return esc(:( $mainExp))
  
end#iter3d

"""
modification of iter3d loop  where wa allow additional action after z check
"""
macro iter3dAdditionalxyzActsAndZcheck(arrDims, loopXdim, loopYdim,loopZdim
  ,zCheck
  ,ex,additionalActionAfterX,additionalActionAfterY,additionalActionAfterZ)
  zOffset= :((zdim*gridDim().x))
  zAdd = :(blockIdxX())
  yOffset= :(ydim* blockDimY())
  yAdd= :(threadIdxY())
  xOffset= :(xdim* blockDimX())
  xAdd = :(threadIdxX())
  xCheck=:(x <=$arrDims[1])
  yCheck =:(y<=$arrDims[2])
  is3d=true
  mainExp = generalizedItermultiDim(  loopXdim, loopYdim,loopZdim,zOffset,zAdd ,yOffset,yAdd,xOffset,xAdd,xCheck,yCheck,zCheck,additionalActionAfterZ,additionalActionAfterY,additionalActionAfterX,is3d,ex) ;
  
  return esc(:( $mainExp))
end#iter3dAdditionalxyzActs


"""
modification of iter3d loop  where wa allow additional actions to be performed after each loop check
"""
macro iter3dAdditionalzActs(arrDims, loopXdim, loopYdim,loopZdim,ex,additionalActionAfterZ)
  zOffset= :((zdim*gridDim().x))
  zAdd = :(blockIdxX())
  yOffset= :(ydim* blockDimY())
  yAdd= :(threadIdxY())
  xOffset= :(xdim* blockDimX())
  xAdd = :(threadIdxX())
  xCheck=:(x <=$arrDims[1])
  yCheck =:(y<=$arrDims[2])
  zCheck=:(z<= $arrDims[3])
  additionalActionAfterY= :()
  additionalActionAfterX= :()
  is3d=true
  mainExp = generalizedItermultiDim(  loopXdim, loopYdim,loopZdim,zOffset,zAdd ,yOffset,yAdd,xOffset,xAdd,xCheck,yCheck,zCheck,additionalActionAfterZ,additionalActionAfterY,additionalActionAfterX,is3d,ex) ;
  
  return esc(:( $mainExp))
end#iter3dAdditionalxyzActs




"""
generalized version of iter3d we will specilize it in the macro on the basis of multiple dispatch
  loopXdim, loopYdim,loopZdim - information how many times we need to loop over given dimension
  zOffset,zAdd - calculate offset and thread/block dependent add 
  Offset,xOffset,xAdd, yAdd - offsets for x and y  and what to add to them
  xCheck,yCheck,zCheck - checks performed just after for - and determining wheather to continue
  additionalActionAfterZ,additionalActionAfterY,additionalActionAfterX  - gives possibility of invoking more actions  
      - invoked after the checks (checks are avoided to prevent warp stall if warp sync will be invoked)
  is3d - if true we use 3 dimensional loop if not we iterate only over x and y     
  ex - main expression around which we build loop    
"""
function generalizedItermultiDim(
   loopXdim, loopYdim,loopZdim
  ,zOffset,zAdd
  ,yOffset,yAdd
  ,xOffset,xAdd
  ,xCheck,yCheck,zCheck
  ,additionalActionAfterZ,additionalActionAfterY,additionalActionAfterX ,
  is3d ,ex  )
#we will define expressions from deepest to most superficial
exp1 = quote
      @unroll for xdim::UInt32 in 0:$loopXdim
          x::UInt32= $xOffset +threadIdxX()
          if( $xCheck)
            $ex
          end#if x
        $additionalActionAfterX  
        end#for x dim
        end#quote

exp2= quote
        @unroll for ydim::UInt32 in  0:$loopYdim
          y::UInt32 = $yOffset +threadIdxY()
            if($yCheck)
              $exp1
        end#if y
        $additionalActionAfterY
        end#for  yLoops 
      end

 exp3= quote
    @unroll for zdim::UInt32 in 0:$loopZdim
      z::UInt32 = $zOffset + $zAdd#multiply by blocks to avoid situation that a single block of threads would have no work to do
      if($zCheck)    
        $exp2
      end#if z 
      $additionalActionAfterZ  
  end#for z dim
  end 
#if it is 3d we will return x,y,z iterations if not only x and y 
if(is3d)
  return exp3
else
  return exp2

end


end#generalizedIter3d




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


# function generalizedIter3d( arrDims, loopXdim, loopYdim,loopZdim,zOffset,zAdd ,ex  )
#   quote
#     @unroll for zdim::UInt32 in 0:$loopZdim
#       z::UInt32 = (zdim*gridDim().x) + blockIdxX()#multiply by blocks to avoid situation that a single block of threads would have no work to do
#       if( (z<= $arrDims[3]) )    
#           @unroll for ydim::UInt32 in  0:$loopYdim
#               y::UInt32 = ((ydim* blockDimY()) +threadIdxY())
#                 if( (y<=$arrDims[2])  )
#                   @unroll for xdim::UInt32 in 0:$loopXdim
#                       x::UInt32=(xdim* blockDimX()) +threadIdxX()
#                       if((x <=$arrDims[1]))
#                         $ex
#                       end#if x
#                  end#for x dim 
#             end#if y
#           end#for  yLoops 
#       end#if z   
#   end#for z dim
#   end 



# function generalizedIter3dFullChecks( arrDims, loopXdim, loopYdim,loopZdim,zOffset,zAdd,yOffset,xOffset,ex  )
#   quote
#     @unroll for zdim::UInt32 in 0:$loopZdim
#       z::UInt32 = $zOffset + $zAdd#multiply by blocks to avoid situation that a single block of threads would have no work to do
#       if( (z<= $arrDims[3]) )    
#           @unroll for ydim::UInt32 in  0:$loopYdim
#               y::UInt32 = $yOffset +threadIdxY()
#                 if( (y<=$arrDims[2])  )
#                   @unroll for xdim::UInt32 in 0:$loopXdim
#                       x::UInt32= $xOffset +threadIdxX()
#                       if((x <=$arrDims[1]))
#                         $ex
#                       end#if x
#                  end#for x dim 
#             end#if y
#           end#for  yLoops 
#       end#if z   
#   end#for z dim
#   end 
# end#generalizedIter3d
