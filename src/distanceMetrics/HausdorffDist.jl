

"""
calculatin Hausdorff distance between 2 segmentations
 let's assume we are already working with minimum cube that encompass all non 0 entries  for example thanks to function __match_all_sync()


1)In a kernel we need to first load to shared memory the data that is associated with given block of threads 
  (it will probably be a vcube of 8x8x8 dimensionality) plus padding of 1 around and syncThreads()

2)using ballot scync( we can look for only false positives  false negatives and true positives  )
  - we will also need to extablish the cartesian coordinate of masks in +x; -x; +y,-y, +z and - z directions that is the only couse of need of true positives
  - experimentation will say weather we will do it separately to step 3 or in the same time

3)We can use ballot sync  to get a mask for a warp that will include only entries that has more than one and less then 8 true entries in pixels around
in case of 2D and more than for example 2 and less than for exxample 20 in case of 3d (we will experiment with this numbers
  - data to this point will be taken from shared memory
  - basic idea if it has neighbours all around it is not a border pixel  if it is solo pixel without neighbours it is probably just a noise
  - we will calculate only mask for the  gold standard
  - we will experiment with chacking immidiately all 9 neighbours in another mask to see weathere local housdorff is not just 1...  

4) we will apply mask from 3 to get cartesian indicies or thread ids of those indexes that passed all conditions so in this step we should have 
  - border pixels of gold standard (thanks to padding from shared memory this should be real border s not only birders of a block)
  - false negative and false positives that will dictate Housdorf distance
  - each warp should also give  maximum cartesian coordinate of masks in +x; -x; +y,-y, +z and - z  so we will than able to reduce those and obtain borders 
5) we will reduce the   maximum and minimums of each axes to get  border points in each block, apart from this each block should return also the  set of coordinates of all border points of gold standard and  FP and fn

6) we will reduce borders between the blocks probably cooperative groups will be achievable as bitmasks are compact enough to get into GPU memory and accumulate rest of the block outputs

7) next as we are still in the same cooperative group and after all block had synchronized we will need to
  - on the basis of the maximal points in each axis establish the approximate center of the gold standard shape 
      -we will need to solve system of linear equations  describing lines connecting mentioned points to get their place of crossing
8) we would like to use spherical coordinate system centered at the point calculated at point 7 this will enable us to increse locality of search 
  - the set of axes will be created which number will equal total number of device threads/ 32 - so one warp will manage one axis
  - this step weather makes sense will be established by exprimentation initially axes will be evenly spaced and  points from step 6 will be assigned to the axis closest 
   ie with most similar angles of spherical coordinate system in case we would have a lot of empty exes we can redistribute warps to the are of high density of points
   - we can also ignore  all of the axes that do not inclode any false positive or negative in them or vicinity (this last will require reduction in a block)
   - blocks should contain nearby axes and point data should be loaded into shared memory
   - from each warp set of for example 2 closest and 2 most distance points to origin should be returned (number to be specified via experimentation) - those groups should be kept separate
   - in case in this axis and surrounding ones  there would be no  false positive or false negatives we should ignore them 
   - axes and points that remained should be grouped into blocks with overlapping (border axes of given block should be included in neighbouring block 
        - how have the overlapping should be - to be established
        - how big blocks should be - to be established - yet with increased size complexity will increse combinatorically ...
 - finally in each block we will analyze distance between each point from group close to the center of gold standard mask and group distant 
 - depending on needs and experimentation we will increase only maximal distance - most probably to reduce dependence on outliers all of them 
    in case we would like to visualize  the results we would also need to return cartesian coordinates related to the closest and most distant pair of points  and distance between them ...

"""
module HausdorffDist




end#HausdorffDist

















