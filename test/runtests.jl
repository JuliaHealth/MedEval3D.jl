using Test

# Recursively include all .jl files in the test directory (except this file)
function include_all_tests(dir)
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, ".jl") && file != "runtests.jl"
                include(joinpath(root, file))
            end
        end
    end
end

include_all_tests(@__DIR__)