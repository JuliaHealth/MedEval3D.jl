
module TestUtils
using Conda
using PyCall
using Pkg
Pkg.build("HDF5")
using HDF5
Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "h5py")

sitk = pyimport("SimpleITK")
np= pyimport("numpy")


end#TestUtils