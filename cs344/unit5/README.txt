1. Ways of optimization:
    1. Choose better alg
    2. Common patterns (e.g. memory coalescing)
    3. Architecture based details (not much gain)
    4. More detailed (less gain)


2. Speedup upperbound (Amdahl’s Law):  1/(1-P) > 1/((1-P)+xP), where xP is the time used for portion P after speedup

3. optimization: 1. computation 2. memory access (usual bottleneck in gpu)

In gpu computing, bottleneck is usually memory. always start by computing memory bandwidth and see where it’s achieving ~60% of peak memory bandwidth. (data moving / time spent)
1st thing to look at for memory: bad coalescing?

--------Memory Optimization----------------
4. A brief explanation of what a "warp" is:
http://stackoverflow.com/questions/3606636/cuda-model-what-is-warp-size

Example:
CL_DEVICE_MAX_COMPUTE_UNITS: 30, # of SMs. Maximal of 8 blocks can run on a same SM at the same time

CL_DEVICE_MAX_WORK_ITEM_SIZES: 512 / 512 / 64, Limit of (gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z)

CL_DEVICE_MAX_WORK_GROUP_SIZE: 512, Maximal # threads in a block

CL_NV_DEVICE_WARP_SIZE: 32, warp size, granularity of memory transfer (coalescing) and instruction dispatch.
                                  | 1.X   | 2.X   | 2.1   |
memory transfer unit / warp       | 0.5   | 1.0   | 1.0   |
instruction dispatch unit / warp  | 0.5   | 0.5   | 1.0   |
2 important points: 
    1. your memory access should be "groupable" in sets of 16 or 32. So keep the X dimension of your blocks a multiple of 32.
    2. (most important) to get the most from a specific gpu, you need to maximize occupancy.

5. Little's law for memory system:
Number of bytes delivered = Average latency of each transaction (time between adjacent transactions) x (usefule) Bandwidth
(For uncoalesced memory access, not all the bandwidth are useful. some accessed memory are not used)

6. Occupancy (#threads / max #threads)
  1. To affect occupacy: control amount of shared mem, control # bloacks / # threads
  2. Increase occupancy help to a point:
      + exposed more parallelism, transactions in flight
      - may force gpu to run less efficiently

7. Shared memory bank conflicts: padding shared memory


--------Computation Optimization----------------
8. Minimize time waiting at barriers
use smaller block size

9. Minimize thread diverging
(Concentrate on warp)
warp: set of threads executing the same instructions at a time
3d blocks are aligned first x-based and then y-based and finally z-based on warps (thread(x, y, z) and thread(x+1, y, z) are adjacent, thread(x, y, z) and thread(x, y+1, z) are separated by blockDim.x threads.
block of threads is devided into warps
SIMD(CPU): single instruction, multiple data
SIMT(GPU)(per warp): single instruction, multiple threads (threads when branches diverge) (in one warp, at one time, execute threads on one branch only and deactivate other threads)



Summary
1. APOD: Analyse, Parallelize, Optimize, Deploy (Only p/o where and when it's needed)
2. measure & improve memory bandwidth
  1. Assure sufficient occupancy
  2. Coalesce global memory access
  3. Little's law: minimize latency between accesses
3. minimize thread diverge
  1. within warp
  2. avoid branchy code
  3. avoid thread workload imbalance
  4. don't freak out
4. consider fast math
  1. intrinsics: __sin, __cos, __exp
  2. use double precision on perpose: 3.14 != 3.14f
5. use streams
  1. overlap computation and cpu-gpu memory transfers
