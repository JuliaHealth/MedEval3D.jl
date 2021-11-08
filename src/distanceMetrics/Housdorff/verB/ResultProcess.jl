
"""
at this point we already have res list filled with all fp and fn's in some cases it is enough in others we want futher actions to be done 
  - for example in case of lack of isometry in voxels we can now apply some corrections for it 
"""
module ResultProcess

"""
it will scan through result list - ignorea all that are zeroces in res indicies, add only the iteration numbers and in the end we will just dividy it by (fp+fn)
"""
function getAverage()
  
end  

"""
We will iterate over result list take all entries that are non zeo in res  indicies list now in a while loop we will lok for the source voxel that was responsible for the  result
so we can start fom naive approach where each lane will be responsible for finding source voxel that is the closest voxel from other mask
In order to achieve it we will go in the direction indicated by the dir value in the distance indicated by the dir in the result matrix  and the number of voxels indicated by the iteration number
It is probable that we will not find there he source voxel so we will need to iteratively increase number of voxels that go in orthogonal direction so for example
  when direction is down and iteration number 10 we will ook 10 voxels blow in straight line at first 
  then if wew ould not find we will look into voxel 9 down and in 1 left,right,anterior,posterior - so over orthogonal axes
  then we will  try going 8 pixels down and 1 pixel in 4 directions and futher pixel also in 4 directions  so 16 permutations 
  basically we will want to scan the area of the sphere with radius of iteration number and getting circle crossections of it  starting from the point in the end of the axis in given
  direction and progress up circle crossesction by circle crossesction 

We can xperiment with getting corrections one by single line at the begining with some maximum number of ccles allowed - so if single lane wuld not be able to find result we would cease looking for it
and then scan one more time whole warps  for scanning those points that were not yet resolved 
"""
function applyCorrection()
  
  
end




end#ResultProcess
