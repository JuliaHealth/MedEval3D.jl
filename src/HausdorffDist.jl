

"""
calculatin Hausdorff distance between 2 segmentations
 let's assume we are already working with minimum cube that encompass all non 0 entries  for example thanks to function __match_all_sync()

1)In a kernel we need to first load to shared memory the data that is associated with given block of threads 
  (it will probably be a vcube of 8x8x8 dimensionality) plus padding of 1 around and syncThreads()
2)using ballot scync( we can look for only false positives and false negatives ) - experimentation will say weather we will do it separately to step 3 or in the same time
3)We can use ballot sync  to get a mask for a warp that will include only entries that has more than one and less then 8 true entries in pixels around
in case of 2D and more than for example 2 and less than for exxample 20 in case of 3d (we will experiment with this numbers
  - basic idea if it has neighbours all around it is not a border pixel  if it is solo pixel without neighbours it is probably just a noise
  - data to this point will be taken from shared memory
  - we will calculate only mask for the  gold standard
  - we will experiment with chacking immidiately all 9 neighbours in another mask to see weathere local housdorff is not just 1...  

4) we will apply mask from 3 to get cartesian indicies or thread ids
"""
module HausdorffDist




end#HausdorffDist

















