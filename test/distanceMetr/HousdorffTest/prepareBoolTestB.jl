
    using Revise, Parameters, Logging, Test
    using CUDA
    includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\includeAllUseFullForTest.jl")
    using Main.CUDAGpuUtils ,Main.IterationUtils,Main.ReductionUtils , Main.MemoryUtils,Main.CUDAAtomicUtils, Main.PrepareArrtoBool, Main.BitWiseUtils
    using Main.ResultListUtils, Main.MetadataAnalyzePass,Main.MetaDataUtils,Main.WorkQueueUtils,Main.ProcessMainDataVerB,Main.HFUtils, Main.ScanForDuplicates, Main.Housdorff
    


    mainArrDims= (512,512,500)

    arrGold = zeros(Int32,mainArrDims);
    arrSegm = zeros(Int32,mainArrDims);

        # so we should get all mins as 2 and all maxes as 5
    
        arrGold[9,2,5]= 2
        arrGold[3,6,4]= 2
        arrGold[3,3,4]= 2
        
        arrGold[4,4,4]= 2
        arrSegm[4,4,4]= 2

        arrSegm[3,18,4]= 2
        arrSegm[5,2,11]= 2   



        gold3d = CuArray(arrGold)
        segm3d= CuArray(arrSegm)
        numberToLooFor =Int32(2)
        robustnessPercent = 0.9

        boolKernelArgs, mainKernelArgs,threadsBoolKern,blocksBoolKern ,threadsMainKern,blocksMainKern ,shmemSizeBool,shmemSizeMain=    preparehousedorfKernel(gold3d,segm3d,robustnessPercent,numberToLooFor)
        mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength=boolKernelArgs
   

        arrGold[dataBdim[1]*3+1,dataBdim[2]*4+1,dataBdim[3]*5+1]= 2
        arrSegm[dataBdim[1]*3+1,dataBdim[2]*4+1,dataBdim[3]*5+1]= 2

        golddd = CuArray(arrGold);
        segmmm= CuArray(arrSegm);

        # using CUDA
        # ff = CUDA.zeros(1)
        # function gggg(ff)
        # #   shmemblockData = @cuDynamicSharedMem(Float32,(dataBdim[1], dataBdim[2]))
        # #   shmemblockData = @cuDynamicSharedMem(Float32,(1,2))
        #    shmemblockData = CuDynamicSharedArray(Int, 6)
        #     return
        # end
        # @cuda threads=(32,2) blocks=2 gggg(ff)
        # ff[1]
   
        metaDataDims
        metaDataLength
        metaDataLength/blocksBoolKern
        loopMeta
        dataBdim
        Int64(blocksBoolKern)
        function locForKernel(goldGPU,segmGPU,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)

          @getBoolCubeKernel()

        return
    end
    minxRes[1]


        @cuda shmem=shmemSizeBool threads=threadsBoolKern blocks=blocksBoolKern locForKernel(golddd,segmmm,mainArrDims,dataBdim,metaData,metaDataDims,reducedGoldA,reducedSegmA,loopXinPlane,loopYinPlane,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp ,numberToLooFor,shmemblockDataLoop,shmemblockDataLenght,loopAXFixed,loopBXfixed,loopAYFixed,loopBYfixed,loopAZFixed,loopBZfixed,loopdataDimMainX,loopdataDimMainY,loopdataDimMainZ,inBlockLoopX,inBlockLoopY,inBlockLoopZ,metaDataLength,loopMeta,loopWarpMeta,clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength,sourceShmemTotalLength)
          Int64( maxzRes[1])

        @test Int64(fn[])==3
        @test Int64(fp[])==2
        
       Int64(minxRes[1])
         Int64( maxxRes[])
         Int64(minyRes[])
          Int64(maxyRes[])
          Int64( minzRes[])
          Int64( maxzRes[])


          @test  Int64(minxRes[])==1
          @test  Int64( maxxRes[])==4
          @test  Int64(minyRes[])==1
          @test   Int64(maxyRes[])==5
          @test  Int64( minzRes[])==1
          @test  Int64( maxzRes[])==6

        #   mainArrDims
        # @test Int64(minxRes[])==2
        # @test Int64( maxxRes[])==5
        # @test Int64(minyRes[])==1
        # @test  Int64(maxyRes[])==9
        # @test  Int64( minzRes[])==2
        # @test  Int64( maxzRes[])==6
        # arrGold[32*3+1,32*4+1,32*5+1]
        # arrSegm[32*3+1,32*4+1,32*5+1]

Int64(sum(reducedGoldA))

Int64(reducedGoldA[4,4,1])
Int64(reducedSegmA[4,4,1])

Int64(reducedGoldA[3,18,1])

        @test  isBit1AtPos(reducedGoldA[9,2,1],5)==true

        @test  isBit1AtPos(reducedGoldA[3,3,1],4)==true
        @test  isBit1AtPos(reducedGoldA[3,3,1],4)==true
        @test  isBit1AtPos(reducedGoldA[4,4,1],4)==true
        @test  isBit1AtPos(reducedGoldA[5,5,1],5)==false
        @test  isBit1AtPos(reducedGoldA[9,2,1],5)==true
        @test  isBit1AtPos(reducedSegmA[3,18,1],4)==true
        @test  isBit1AtPos(reducedGoldA[3,18,1],4)==false


        @test  isBit1AtPos(reducedSegmA[dataBdim[1]*3+1,dataBdim[2]*4+1,6],1)==true

       