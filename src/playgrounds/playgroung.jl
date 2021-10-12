
x = [4.0,5.0,9.0]
A = [ 6.0 15.0 55.0;
    15.0 55.0 225.0;
    55.0 225.0 979.0  ]
y = [0.0,0.0,0.0]
L2 = zeros(Float32,3,3)

varianceX= A[1,1]
covarianceXY= A[2,1]
covarianceXZ = A[3,1]
varianceY = A[2,2]
covarianceYZ = A[3,2]
varianceZ = A[3,3]






a = CUDA.zeros(1)
function shuffleTestKernel(a)
    x = 1.0
    @ifX 1 x = 5.0 
    x= @getFromLane(x,1)
    CUDA.@cuprint "x $(x) \n" 
    a[1]= x
    return
end
@cuda threads=(32) blocks=(1) shuffleTestKernel(a)
a[1]