
using Distances, Test, Revise 
includet("src/kernelEvolutions.jl")
using Main.BasicPreds


x = rand(Bool,3,3,3)
y = rand(Bool,3,3,3)

r = evaluate(Jaccard(), goldBool, segmBool)


CartesianIndices(zeros(3,3,3) ).+CartesianIndex(5,5,5)