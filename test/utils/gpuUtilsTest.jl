using DrWatson
@quickactivate "Medical segmentation evaluation"
using Test

module CUDAGpuUtilsTest




s = SubArray( a, (1:3,))


A = [1 2 3; 
     3 4 5 ;
     6 7 8 ];

B = ones(3,3,3)
B[1,:,:] = A
B[2,:,:] = A.*10
B[3,:,:] = A.*100
B




V = view(B, :, :,1)
Base.iscontiguous(V)


