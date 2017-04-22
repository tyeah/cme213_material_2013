Big picture

SpMV
1. Keeping your threads busy is important
2. Managing communication is important
Usually a tradeoff: per element threads are more balanced (less idle threads, most threads are busy) but suffer more from communitcation (usually based on shared memory)

BFS
We want:
  1. Lots of parallelism
  2. Coalescing memory access
  3. Minimum execution divergence
  4. Easy to implement
