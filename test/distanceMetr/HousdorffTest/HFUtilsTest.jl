#module HFUtilsTest


using  Test, Revise 
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\utils\\CUDAGpuUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\src\\distanceMetrics\\Housdorff\\mainHouseDorffKernel\\HFUtils.jl")
includet("C:\\GitHub\\GitHub\\NuclearMedEval\\test\\GPUtestUtils.jl")
using Main.HFUtils
using Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA, StaticArrays

using Main.HFUtils
using Main.CUDAGpuUtils,Cthulhu,BenchmarkTools , CUDA, StaticArrays


@testset "clearMainShmem" begin 

    testArr = CUDA.zeros(Bool,34,34,34);
    function testKernForClearShmem(testArrInn)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 
        resShmem[threadIdxX()+1,threadIdxY()+1,1]= 1
        testArrInn[threadIdxX()+1,threadIdxY()+1,1]= 1
        resShmem[threadIdxX()+1,threadIdxY()+1,2]= 1
        testArrInn[threadIdxX()+1,threadIdxY()+1,2]= 1

         HFUtils.clearMainShmem( resShmem)
         HFUtils.clearMainShmem( testArrInn)       
        return
    end    
    @cuda threads=(32,32) blocks=1 testKernForClearShmem(testArr) 
    @test  sum(testArr)==1024 
    CUDA.reclaim()# just to destroy from gpu our dummy data

end # clearMainShmem



@testset "clearLocArr" begin 

    testArr = CUDA.zeros(Bool,32);
    function testKernForClearArrA(testArrInn)
        if(threadIdxX()==UInt32(1) &&threadIdxY()==UInt32(1) )
        locArr= zeros(MVector{32,Bool})
        locArr[1]=1
        locArr[2]=1
        testArrInn[1] = locArr[1]
        testArrInn[2] = locArr[2]
        end
        return
    end    

    @cuda threads=(32,32) blocks=1 testKernForClearArrA(testArr) 
    @test  sum(testArr)==2 

    testArr = CUDA.zeros(Bool,32);

    function testKernForClearArrB(testArrInn)
        @ifXY 1 1 begin 
        locArr= zeros(MVector{32,Bool})
        locArr[1]=1
        locArr[2]=1
        clearLocArr(locArr)
        testArrInn[1] = locArr[1]
        testArrInn[2] = locArr[2]
        end
        return
    end    

    @cuda threads=(32,32) blocks=1 testKernForClearArrB(testArr) 
    @test  sum(testArr)==0 


    CUDA.reclaim()# just to destroy from gpu our dummy data

end # clearLocArr




@testset "clearPadding" begin 
  

    testArr = CUDA.zeros(Bool,34,34,34);

    function testclearPadding(testArrInn)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 


        for z in 1:34
            resShmem[threadIdxX(),threadIdxY(),z ]=1
            resShmem[threadIdxX()+2,threadIdxY()+2,z ]=1
            resShmem[threadIdxX()+2,threadIdxY(),z ]=1
            resShmem[threadIdxX(),threadIdxY()+2,z ]=1
        end 
        fillGlobalFromShmem(testArrInn,resShmem)

        # clearPadding(resShmem)

        # for z in 1:34   
        #     testArrInn[threadIdxX(),threadIdxY(),z ]=resShmem[threadIdxX(),threadIdxY(),z ]
        #     testArrInn[threadIdxX()+2,threadIdxY()+2,z ]=  resShmem[threadIdxX()+2,threadIdxY()+2,z ]
        #     testArrInn[threadIdxX(),threadIdxY()+2,z ]=  resShmem[threadIdxX(),threadIdxY()+2,z ]
        #     testArrInn[threadIdxX()+2,threadIdxY(),z ]=  resShmem[threadIdxX()+2,threadIdxY(),z ]
        # end    
                
        return
    end    

    @cuda threads=(32,32) blocks=1 testclearPadding(testArr) 
    @test (length(testArr)-(sum(testArr)+(4*34)+(8*32)  )) ==0



    testArrB = CUDA.zeros(Bool,34,34,34);

    function testclearPadding(testArrInn)
        resShmem =  @cuStaticSharedMem(Bool,(34,34,34))#+2 in order to get the one padding 


        for z in 1:34
            resShmem[threadIdxX(),threadIdxY(),z ]=1
            resShmem[threadIdxX()+2,threadIdxY()+2,z ]=1
            resShmem[threadIdxX()+2,threadIdxY(),z ]=1
            resShmem[threadIdxX(),threadIdxY()+2,z ]=1
        end 
sync_threads()
        clearPadding(resShmem)
        clearMainShmem(resShmem)
        sync_threads()

        fillGlobalFromShmem(testArrInn,resShmem)

        return
    end    
    @cuda threads=(32,32) blocks=1 testclearPadding(testArrB) 
    @test sum(testArrB) ==0


end#clearPadding

##############







# macro addArguments(x, ex)
#     return esc(:(if threadIdxX()==$x
#         $ex
#     end))

# end


# macro times3(ex)
#     return _times3(ex)
# end



# function _times3(ex)
#     # if ex.head == :call && ex.args[1] == :+
#     #     ex.args[1] = :*
#     # end   
#     push!(ex.args,3)

#     return ex
# end


# function addd(aa,bb)
#     return aa+bb
# end    

# a = 2; b = 3

# @times3 addd(a)


# function outer(bb)
#     aa=5
#     addd(bb)
# end

# using MacroTools

# ex = quote
#     struct someStr
#       x::Int
#       y
#     end
#   end

#   prettify(ex)


#   @capture(ex, struct T_ fields__ end)
#   T, fields


#   exB = quote
#     xi::Int = 2
#     xiB::Int = 2
#     xiC::Int = 2
#     xiD::Int = 2
#   end

#   MacroTools.prewalk(x -> @show(x) isa Symbol ? x : x, exB);

#   @capture(exB, Symbol__ )