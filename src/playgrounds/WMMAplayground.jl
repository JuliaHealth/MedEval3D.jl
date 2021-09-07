TCSQ =  256 
WARPSIZE = 32
BSIZE 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

kernel_singlepass<<<grid, block>>>(Adh, outd, n, bs);
Adh - pointer to some half
outd - some float
n- probably length of Array
int bs = BSIZE >> 5;
R- ???????


"""
kernel_singlepass(half *a, float *out, int N,int bs){
	int offset = blockIdx.x * (bs * TCSQ * R); 
	if(offset < N){
		REAL sumf = block_reduce_tc(N, a, offset);
        //if((threadIdx.x & 31) == 0){
        //    //printf("offset %i \n",offset);
        //    atomicAdd(out, sumf);
        //}
        if(threadIdx.x == 0){
            //printf("offset %i \n",offset);
            atomicAdd(out, sumf);
        }
	}
}
"""


"""
__inline__ __device__ REAL block_reduce_tc(int N, half *a, int offset){
	__shared__ REAL shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	int wid = tid >> 5;
	REAL val = reduction_tc_warp(N, a, offset + wid*TCSQ*R, lane, wid << 8);

    if(lane == 0){
		shared[wid] = val;
    }
	__syncthreads();
    //printf("thread %i val %f\n", threadIdx.x, val);
	val = (tid < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
	if(wid == 0){
        val = warp_shuffle_reduction_real(val);
    }
	return val;
}



"""


#WMMA Part 
#below we see that 1) first we fefine WMMA fragments
#2)   we fill fragment a with ones and d with zeros
"""
// kernel
__inline__ __device__ REAL reduction_tc_warp(int N, half *A, int offset, int lane, int warpoff){
    // definicion de offset y fragmentos
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> r_frag;
    
    // (1) cargar datos de memoria global a A, B y C frags
    wmma::fill_fragment(a_frag, 1.0f);
    //wmma::fill_fragment(b_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    // (2) mejora MMA multiples encadenados
    //const int bigoffset = gridDim.x * 32 * TCSQ;
    //if(offset >= N){ return 0.0f; }
    #pragma loop unroll
    for(int i=0; i<R; ++i){
        //if(threadIdx.x == 0) printf("offset %i \n",offset + TCSQ*32*(i+1));
        wmma::load_matrix_sync(b_frag, A + offset + (i<<8), TCSIZE);
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
    }

    // (3) preparando datos para segundo MMA
    wmma::fill_fragment(b_frag, 1.0f);
    // [OPCION 1] copia manual
    //#pragma loop unroll
    //for(int i=0; i < 8 ; ++i){
    //    a_frag.x[i] = d_frag.x[i];
    //    a_frag.x[i+8] = d_frag.x[i];
    //}
   
    //int offwid = (threadIdx.x/32)*256;
    // [OPCION 2] copia a shared mem
    __shared__ half As[DIFF];
    wmma::store_matrix_sync(As+warpoff, d_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+warpoff, TCSIZE);
    wmma::fill_fragment(d_frag, 0.0f);




    //// (4) MMA
    wmma::mma_sync(r_frag, a_frag, b_frag, d_frag);

    // (5) Almacenar resultado
    if(lane == 0){
        //printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
        //printf("%f\n",(float)d_frag.x[0]);
        return r_frag.x[0];
        //return 1.0f;
    }
    else return 0.0f;
}

"""