
"""
calculates Mahalanobis distance between two segmentations
General plan to implement algorithm
1) calculate means for all row, column and "depth vectors" in both arrays
2)in case of each voxel we subtract values related to the mean of 3 axes that are passing through this point
  then we add both covariance matrices from gold standard and segmentation output  and take its inverse 
4)Then we apply classical definition of Mahalanobis from 2 dimensional case and we iterate over 3 possible "slicing" 
  of covariance 3 dimensional array (using planes passing in all 3 axes) sum results and take square root of it
More Detailed plan
1)We need to filter out all of the  areas that are not important becouse of our data sparsity 
  - so we will use voteany mechanism to find the smallest possible cubic space enclosing all non zero entries in both segmentations
2)No we should consider a scheme that will make it possible to calculate this distance for both slice wise and volume wise cases
3)What will be required In all cases is volume wide 3 dimensional array 
I) slice wise
  a)

"""
module MahalanobisDist


function calcMean()
  
  
end#function



end#module








