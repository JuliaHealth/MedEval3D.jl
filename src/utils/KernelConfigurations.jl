"""
all kernels need some gpu data structures that we can divide into couple types 
    - global results - sibgle Float32 that will give given global metrics
    - slice wise results - array where each entry tells about given metric 
    - global single helper structures - single variables in global memory 
    - complex helper structures - arrays or tuples holding intermediate data - will take far more memory than single helper structures

We will store all of the symbols and types of all above entries as list of tuples where we will have symbol of the variable and the type
on the basis of this we can create struct for each kernel 
    we can also create a clearing function of multiple types - like for esxample only resetting values - by setting all to 0's or freeing memory in case we are memory contrained

Whats more we should also supply the function calculating the amount of required shared memory 

for given kernel compilation we need also to supply data type and dimensions of the data that will be passed - on all of those data we will calculate best thread block configuration
    using the occupancy calculator - this data can be potentially cached in permanent disk cache - as it shopuld be constant given the same GPU and the same array size
    apart from number of threads and thread blocks we need also to calculate iteration constants - how many time we need to loop trough x,y,z dimensions
    - in some cases here we will also calculate some additional data like size, dimensions of metadata etc...    

"""