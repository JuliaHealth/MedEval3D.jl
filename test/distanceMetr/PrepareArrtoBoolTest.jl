using Test,Revise
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\gpuUtils.jl")


includet("C:/GitHub/GitHub/NuclearMedEval/src/kernels/PrepareArrtoBool.jl")
using Main.PrepareArrtoBool, Main.GPUutils
using CUDA

@testset "getBoolCubeKernel" begin 
    sliceW = 512
    sliceH = 512
    sliceNumb = 91
    arrGold = zeros(Int64,sliceW,sliceH,sliceNumb);
    arrSegm = zeros(Int64,sliceW,sliceH,sliceNumb);

arrDims = size(arrSegm)
    # so we should get all mins as 2 and all maxes as 5
 
    arrGold[9,2,5]= 2
    arrGold[3,6,4]= 2
    arrGold[3,3,4]= 2
    
    arrGold[4,4,4]= 2
    arrSegm[4,4,4]= 2

    arrSegm[3,18,4]= 2
    arrSegm[5,2,11]= 2

     fn=CuArray([UInt32(0)]);     fp=CuArray([UInt32(0)]);     minxRes=CuArray([UInt32(100000)]);     maxxRes=CuArray([UInt32(0)]);
     minyRes=CuArray([UInt32(10000)]);     maxyRes=CuArray([UInt32(0)]);     minZres=CuArray([UInt32(100000)]);     maxZres=CuArray([UInt32(0)]);

     reducedGoldA=CuArray(falses(arrDims));
     reducedSegmA=CuArray(falses(arrDims));
     reducedGoldB=CuArray(falses(arrDims));
     reducedSegmB=CuArray(falses(arrDims));


    Main.PrepareArrtoBool.getBoolCube!(
     CuArray(arrGold)
    ,CuArray(arrSegm)
    , sliceNumb
    ,fn    ,fp    ,minxRes
    ,maxxRes    ,minyRes
    ,maxyRes    ,minZres
    ,maxZres    ,2
    , CuArray([UInt32(0)])
    , reducedGoldA
    , reducedSegmA
    , reducedGoldB
    ,reducedSegmB
      );

      Int64( maxxRes[])

      @test Int64(fn[])==3
      @test Int64(fp[])==2
      @test Int64(minxRes[])==3
      @test Int64( maxxRes[])==9
      @test Int64(minyRes[])==2
      @test  Int64(maxyRes[])==18
      @test  Int64( minZres[])==4
      @test  Int64( maxZres[])==11

     vv =  collect(view(reducedGoldA,Int64(minxRes[]) : Int64( maxxRes[]) , Int64(minyRes[]) : Int64(maxyRes[]),Int64( minZres[]) :Int64( maxZres[]) ))
typeof(vv)
size(vv)


      @test  reducedGoldA[9,2,5]==true
      @test   reducedGoldA[3,6,4]==true
      @test   reducedGoldA[3,3,4]==true
      @test   reducedGoldA[4,4,4]==true
      @test   reducedGoldA[5,5,5]==false
      @test   reducedGoldA[5,2,5]==false
      @test   reducedGoldA[5,2,90]==false

      @test  reducedGoldB[9,2,5]==true
      @test   reducedGoldB[3,6,4]==true
      @test   reducedGoldB[3,3,4]==true
      @test   reducedGoldB[4,4,4]==true
      @test   reducedGoldB[5,5,5]==false
      @test   reducedGoldB[5,2,5]==false
      @test   reducedGoldB[5,2,90]==false

      @test  reducedSegmA[3,18,4]==true
      @test   reducedSegmA[5,2,11]==true
      @test   reducedSegmA[4,4,4]==true
      @test   reducedSegmA[5,5,5]==false
      @test   reducedSegmA[5,2,5]==false
      @test   reducedSegmA[5,2,90]==false

      @test  reducedSegmB[3,18,4]==true
      @test   reducedSegmB[5,2,11]==true
      @test   reducedSegmB[4,4,4]==true
      @test   reducedSegmB[5,5,5]==false
      @test   reducedSegmB[5,2,5]==false
      @test   reducedSegmB[5,2,90]==false


end#testset

t = true
f = false
 0-t