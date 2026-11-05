# TriTune report: agent on matmul:1024x1024x1024:fp16 (+12)

> **SIMULATED RESULTS: no GPU was used. Latencies and profiler counters come from TriTune's analytic model; numerical checks were emulated on the CPU. Do not quote these numbers as hardware performance.**

- experiment: `20261104-183444-0988` (agent, status done)
- backend: simulated (no GPU; analytic model, numerics emulated on CPU)
- geomean speedup vs untuned Triton baseline: **1.62x** over 13/13 problems (correctness-verified candidates only)
- candidates evaluated: 100, compile failures: 0, numerical failures: 0, unverified: 0
- wall time: 1.83 s
- cache: 42 hits / 299 misses this run
- params: seed=0 reps=50 warmup=10 budget=8

## Results

| problem | baseline p50 | best p50 | speedup | best config | evals | to-best | failed |
|---|---:|---:|---:|---|---:|---:|---:|
| `matmul:1024x1024x1024:fp16` | 46.9 us | 38.5 us | 1.22x | `grouped BLOCK_K=32 BLOCK_M=64 BLOCK_N=64 GROUP_M=8 w4 s4` | 9 | 7 | 0 |
| `matmul:2048x2048x2048:fp16` | 0.289 ms | 0.185 ms | 1.57x | `grouped BLOCK_K=32 BLOCK_M=128 BLOCK_N=128 GROUP_M=8 w8 s3` | 9 | 4 | 0 |
| `matmul:4096x4096x1024:fp16` | 0.460 ms | 0.261 ms | 1.76x | `grouped BLOCK_K=32 BLOCK_M=128 BLOCK_N=128 GROUP_M=8 w8 s3` | 9 | 4 | 0 |
| `matmul:1000x1000x1000:fp16` | 45.7 us | 32.7 us | 1.40x | `grouped BLOCK_K=32 BLOCK_M=32 BLOCK_N=64 GROUP_M=8 w4 s4` | 9 | 9 | 0 |
| `matmul:2048x512x2048:fp32` | 0.314 ms | 0.118 ms | 2.67x | `grouped BLOCK_K=32 BLOCK_M=64 BLOCK_N=64 GROUP_M=4 w4 s3` | 9 | 9 | 0 |
| `softmax:4096x1024:fp16` | 15.7 us | 15.7 us | 1.00x | `single_pass BLOCK_N=1024 w4 s2` | 5 | 1 | 0 |
| `softmax:8192x512:fp16` | 15.6 us | 15.6 us | 1.00x | `single_pass BLOCK_N=512 w4 s2` | 5 | 1 | 0 |
| `softmax:2048x4096:fp32` | 78.4 us | 51.9 us | 1.51x | `single_pass BLOCK_N=4096 w16 s2` | 6 | 4 | 0 |
| `softmax:1823x781:fp16` | 7.4 us | 7.4 us | 1.00x | `single_pass BLOCK_N=1024 w4 s2` | 5 | 1 | 0 |
| `softmax:1024x16384:fp32` | 0.925 ms | 0.112 ms | 8.24x | `single_pass BLOCK_N=16384 w32 s2` | 7 | 4 | 0 |
| `reduction:1048576:fp32` | 8.0 us | 6.4 us | 1.24x | `atomic BLOCK_SIZE=2048 TILES=2 w4 s2` | 9 | 7 | 0 |
| `reduction:16777216:fp32` | 83.3 us | 51.1 us | 1.63x | `atomic BLOCK_SIZE=2048 TILES=16 w4 s2` | 9 | 6 | 0 |
| `reduction:4194311:fp16` | 17.1 us | 9.9 us | 1.72x | `atomic BLOCK_SIZE=1024 TILES=8 w4 s2` | 9 | 4 | 0 |

## Latency distributions

| problem | which | n | p50 | p95 | min | max | histogram |
|---|---|---:|---:|---:|---:|---:|---|
| `matmul:1024x1024x1024:fp16` | baseline | 50 | 46.9 us | 48.4 us | 44.7 us | 57.3 us | `▁▃█▆▃▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:1024x1024x1024:fp16` | best | 50 | 38.5 us | 39.2 us | 36.7 us | 49.6 us | `▂▄█▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:2048x2048x2048:fp16` | baseline | 50 | 0.289 ms | 0.299 ms | 0.279 ms | 0.406 ms | `▃█▂▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:2048x2048x2048:fp16` | best | 50 | 0.185 ms | 0.189 ms | 0.178 ms | 0.192 ms | `▂▁▂▂▂▂▃█▂▃▃▂▃▂▁▁` |
| `matmul:4096x4096x1024:fp16` | baseline | 50 | 0.460 ms | 0.474 ms | 0.447 ms | 0.586 ms | `▅█▃▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:4096x4096x1024:fp16` | best | 50 | 0.261 ms | 0.268 ms | 0.253 ms | 0.269 ms | `▁▂▁▃▁▁█▃▅▃▃▅▂▁▁▃` |
| `matmul:1000x1000x1000:fp16` | baseline | 50 | 45.7 us | 47.2 us | 43.9 us | 63.1 us | `▃█▃▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:1000x1000x1000:fp16` | best | 50 | 32.7 us | 33.4 us | 31.8 us | 39.7 us | `▂█▆▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:2048x512x2048:fp32` | baseline | 50 | 0.314 ms | 0.390 ms | 0.304 ms | 0.439 ms | `▇█▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `matmul:2048x512x2048:fp32` | best | 50 | 0.118 ms | 0.122 ms | 0.114 ms | 0.140 ms | `▂▆█▄▂▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:4096x1024:fp16` | baseline | 50 | 15.7 us | 16.2 us | 15.0 us | 19.6 us | `▁▃█▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:4096x1024:fp16` | best | 50 | 15.7 us | 16.2 us | 15.0 us | 19.6 us | `▁▃█▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:8192x512:fp16` | baseline | 50 | 15.6 us | 16.1 us | 15.1 us | 21.0 us | `▃█▂▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:8192x512:fp16` | best | 50 | 15.6 us | 16.1 us | 15.1 us | 21.0 us | `▃█▂▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:2048x4096:fp32` | baseline | 50 | 78.4 us | 80.1 us | 75.9 us | 90.1 us | `▂▃█▆▄▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:2048x4096:fp32` | best | 50 | 51.9 us | 53.3 us | 50.3 us | 63.6 us | `▃█▆▃▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:1823x781:fp16` | baseline | 50 | 7.4 us | 7.6 us | 7.2 us | 7.7 us | `▃▂▁▂▅█▃▆▃▄▃▄▁▁▁▁` |
| `softmax:1823x781:fp16` | best | 50 | 7.4 us | 7.6 us | 7.2 us | 7.7 us | `▃▂▁▂▅█▃▆▃▄▃▄▁▁▁▁` |
| `softmax:1024x16384:fp32` | baseline | 50 | 0.925 ms | 0.951 ms | 0.896 ms | 1.130 ms | `▅▅█▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| `softmax:1024x16384:fp32` | best | 50 | 0.112 ms | 0.131 ms | 0.109 ms | 0.154 ms | `▆█▃▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `reduction:1048576:fp32` | baseline | 50 | 8.0 us | 8.2 us | 7.7 us | 8.2 us | `▁▁▁▂▃▂▄█▂▃▆▄▄▃▁▃` |
| `reduction:1048576:fp32` | best | 50 | 6.4 us | 6.6 us | 6.2 us | 8.2 us | `▅█▆▂▁▁▁▁▁▁▁▁▁▁▁▁` |
| `reduction:16777216:fp32` | baseline | 50 | 83.3 us | 84.9 us | 80.2 us | 0.116 ms | `▄█▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `reduction:16777216:fp32` | best | 50 | 51.1 us | 52.4 us | 49.5 us | 70.9 us | `▃█▁▁▁▁▁▁▁▁▁▁▁▁▁▁` |
| `reduction:4194311:fp16` | baseline | 50 | 17.1 us | 17.5 us | 16.6 us | 18.6 us | `▂▅▆▅█▃▃▂▁▁▁▁▁▁▁▁` |
| `reduction:4194311:fp16` | best | 50 | 9.9 us | 11.5 us | 9.5 us | 12.9 us | `▂█▇▂▁▁▁▁▁▁▁▁▁▁▁▁` |

## Resource usage (best candidate)

| problem | regs/thread | spills | shared mem | occupancy | DRAM % of peak |
|---|---:|---:|---:|---:|---:|
| `matmul:1024x1024x1024:fp16` | 88 | 0 | 32768 | 31% (profiler) | 24% |
| `matmul:2048x2048x2048:fp16` | 120 | 0 | 49152 | 25% (profiler) | 52% |
| `matmul:4096x4096x1024:fp16` | 120 | 0 | 49152 | 25% (profiler) | 59% |
| `matmul:1000x1000x1000:fp16` | 68 | 0 | 24576 | 38% (profiler) | 30% |
| `matmul:2048x512x2048:fp32` | 104 | 0 | 49152 | 19% (profiler) | 48% |
| `softmax:4096x1024:fp16` | 44 | 0 | 32 | 62% (profiler) | 69% |
| `softmax:8192x512:fp16` | 32 | 0 | 32 | 100% (profiler) | 69% |
| `softmax:2048x4096:fp32` | 44 | 0 | 128 | 50% (profiler) | 83% |
| `softmax:1823x781:fp16` | 44 | 0 | 32 | 62% (profiler) | 49% |
| `softmax:1024x16384:fp32` | 68 | 0 | 256 | 0% (estimate) | 77% |
| `reduction:1048576:fp32` | 50 | 0 | 32 | 56% (profiler) | 42% |
| `reduction:16777216:fp32` | 50 | 0 | 32 | 56% (profiler) | 84% |
| `reduction:4194311:fp16` | 34 | 0 | 32 | 75% (profiler) | 54% |

## Compilation failures

None.

## Numerical validation failures

None.

## Optimization decisions

1. **start** [`cdcdd847bb76`]: agent run with planner=heuristic
2. **thought**: Start by looking at the baseline kernel and its launch config.
4. **thought**: Compile the baseline to get register / shared memory numbers.
6. **thought**: Check the baseline is numerically correct before trusting any timing.
8. **thought**: Measure the baseline; every later candidate is compared against this.
9. **accept** [`cdcdd847bb76`]: first measured candidate, this is the incumbent to beat
11. **thought**: Profile the baseline to see what it's bound by.
13. **thought**: Ask the advisor what the current numbers point to.
15. **thought**: 64x64 tile reads each A/B element too many times. Hypothesis: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed [small_tiles]
16. **hypothesis** [`cc5e7b5653d0`]: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed
18. **thought**: Compile the new candidate.
20. **thought**: It compiles. Validate before timing it.
22. **thought**: Numerics check out. Benchmark it against the incumbent.
23. **reject** [`cc5e7b5653d0`]: not kept: regression: 15.4% slower
25. **thought**: Ask the advisor what the current numbers point to.
27. **thought**: row-major order with N=1024. Hypothesis: Grouped tile order keeps a band of B resident in L2 across consecutive programs [row_major_order]
28. **hypothesis** [`c53430a34e0d`]: Grouped tile order keeps a band of B resident in L2 across consecutive programs
30. **thought**: Compile the new candidate.
32. **thought**: It compiles. Validate before timing it.
34. **thought**: Numerics check out. Benchmark it against the incumbent.
35. **reject** [`c53430a34e0d`]: not kept: within noise (+1.4%, threshold 1.6%)
37. **thought**: Ask the advisor what the current numbers point to.
39. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
40. **hypothesis** [`c5897e7b9895`]: One more stage of cp.async prefetch to hide global load latency
42. **thought**: Compile the new candidate.
44. **thought**: It compiles. Validate before timing it.
46. **thought**: Numerics check out. Benchmark it against the incumbent.
47. **accept** [`c5897e7b9895`]: kept: 12.7% faster (> 1.6% noise threshold)
49. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
51. **thought**: Ask the advisor what the current numbers point to.
53. **thought**: row-major order with N=1024. Hypothesis: Grouped tile order keeps a band of B resident in L2 across consecutive programs [row_major_order]
54. **hypothesis** [`c737e8ae7bc0`]: Grouped tile order keeps a band of B resident in L2 across consecutive programs
56. **thought**: Compile the new candidate.
58. **thought**: It compiles. Validate before timing it.
60. **thought**: Numerics check out. Benchmark it against the incumbent.
61. **accept** [`c737e8ae7bc0`]: kept: 1.8% faster (> 1.3% noise threshold)
63. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
65. **thought**: Ask the advisor what the current numbers point to.
67. **thought**: small output tile. Hypothesis: Grow BLOCK_M only, a smaller step in arithmetic intensity [small_tiles]
68. **hypothesis** [`c773babc5451`]: Grow BLOCK_M only, a smaller step in arithmetic intensity
70. **thought**: Compile the new candidate.
72. **thought**: It compiles. Validate before timing it.
74. **thought**: Numerics check out. Benchmark it against the incumbent.
75. **reject** [`c773babc5451`]: not kept: regression: 37.4% slower
77. **thought**: Ask the advisor what the current numbers point to.
79. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
80. **hypothesis** [`c84084d87e6a`]: Local refinement: num_stages 3 -> 4
82. **thought**: Compile the new candidate.
84. **thought**: It compiles. Validate before timing it.
86. **thought**: Numerics check out. Benchmark it against the incumbent.
87. **accept** [`c84084d87e6a`]: kept: 4.4% faster (> 1.1% noise threshold)
89. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
91. **thought**: Ask the advisor what the current numbers point to.
93. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: GROUP_M 8 -> 4 [local search]
94. **hypothesis** [`ceb09f5a0891`]: Local refinement: GROUP_M 8 -> 4
96. **thought**: Compile the new candidate.
98. **thought**: It compiles. Validate before timing it.
100. **thought**: Numerics check out. Benchmark it against the incumbent.
101. **reject** [`ceb09f5a0891`]: not kept: within noise (+0.5%, threshold 1.2%)
103. **thought**: Ask the advisor what the current numbers point to.
105. **thought**: structural alternative not explored yet. Hypothesis: Try the tiled variant with similar parameters [local search]
106. **hypothesis** [`c1d566d0dd0c`]: Try the tiled variant with similar parameters
108. **thought**: Compile the new candidate.
110. **thought**: It compiles. Validate before timing it.
112. **thought**: Numerics check out. Benchmark it against the incumbent.
113. **reject** [`c1d566d0dd0c`]: not kept: within noise (-1.8%, threshold 7.4%)
115. **thought**: Ask the advisor what the current numbers point to.
117. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 64 -> 32 [local search]
119. **thought**: Candidate budget is used up. Wrapping up.
120. **final** [`c84084d87e6a`]: Candidate budget is used up. Kept the fastest verified candidate.
122. **start** [`cb7738d006dc`]: agent run with planner=heuristic
123. **thought**: Start by looking at the baseline kernel and its launch config.
125. **thought**: Compile the baseline to get register / shared memory numbers.
127. **thought**: Check the baseline is numerically correct before trusting any timing.
129. **thought**: Measure the baseline; every later candidate is compared against this.
130. **accept** [`cb7738d006dc`]: first measured candidate, this is the incumbent to beat
132. **thought**: Profile the baseline to see what it's bound by.
134. **thought**: Ask the advisor what the current numbers point to.
136. **thought**: 64x64 tile reads each A/B element too many times. Hypothesis: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed [small_tiles]
137. **hypothesis** [`cbcdf2926995`]: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed
139. **thought**: Compile the new candidate.
141. **thought**: It compiles. Validate before timing it.
143. **thought**: Numerics check out. Benchmark it against the incumbent.
144. **accept** [`cbcdf2926995`]: kept: 24.7% faster (> 1.6% noise threshold)
146. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
148. **thought**: Ask the advisor what the current numbers point to.
150. **thought**: row-major order with N=2048. Hypothesis: Grouped tile order keeps a band of B resident in L2 across consecutive programs [row_major_order]
151. **hypothesis** [`c9b45c9ff9d1`]: Grouped tile order keeps a band of B resident in L2 across consecutive programs
153. **thought**: Compile the new candidate.
155. **thought**: It compiles. Validate before timing it.
157. **thought**: Numerics check out. Benchmark it against the incumbent.
158. **accept** [`c9b45c9ff9d1`]: kept: 2.1% faster (> 1.5% noise threshold)
160. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
162. **thought**: Ask the advisor what the current numbers point to.
164. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
165. **hypothesis** [`c10daa0fb578`]: One more stage of cp.async prefetch to hide global load latency
167. **thought**: Compile the new candidate.
169. **thought**: It compiles. Validate before timing it.
171. **thought**: Numerics check out. Benchmark it against the incumbent.
172. **accept** [`c10daa0fb578`]: kept: 13.5% faster (> 1.5% noise threshold)
174. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
176. **thought**: Ask the advisor what the current numbers point to.
178. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 128 -> 256 [local search]
179. **hypothesis** [`c6088fbeb2f8`]: Local refinement: BLOCK_M 128 -> 256
181. **thought**: Compile the new candidate.
183. **thought**: It compiles. Validate before timing it.
185. **thought**: Numerics check out. Benchmark it against the incumbent.
186. **reject** [`c6088fbeb2f8`]: not kept: regression: 22.7% slower
188. **thought**: Ask the advisor what the current numbers point to.
190. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 128 -> 64 [local search]
191. **hypothesis** [`ce069ecb5234`]: Local refinement: BLOCK_M 128 -> 64
193. **thought**: Compile the new candidate.
195. **thought**: It compiles. Validate before timing it.
197. **thought**: Numerics check out. Benchmark it against the incumbent.
198. **reject** [`ce069ecb5234`]: not kept: regression: 9.9% slower
200. **thought**: Ask the advisor what the current numbers point to.
202. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_N 128 -> 256 [local search]
203. **hypothesis** [`c72a7c9bc215`]: Local refinement: BLOCK_N 128 -> 256
205. **thought**: Compile the new candidate.
207. **thought**: It compiles. Validate before timing it.
209. **thought**: Numerics check out. Benchmark it against the incumbent.
210. **reject** [`c72a7c9bc215`]: not kept: regression: 21.3% slower
212. **thought**: Ask the advisor what the current numbers point to.
214. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_N 128 -> 64 [local search]
215. **hypothesis** [`cc12c18206b3`]: Local refinement: BLOCK_N 128 -> 64
217. **thought**: Compile the new candidate.
219. **thought**: It compiles. Validate before timing it.
221. **thought**: Numerics check out. Benchmark it against the incumbent.
222. **reject** [`cc12c18206b3`]: not kept: regression: 16.4% slower
224. **thought**: Ask the advisor what the current numbers point to.
226. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 8 -> 4 [local search]
227. **hypothesis** [`cd40ab53943c`]: Local refinement: num_warps 8 -> 4
229. **thought**: Compile the new candidate.
231. **thought**: It compiles. Validate before timing it.
233. **thought**: Numerics check out. Benchmark it against the incumbent.
234. **reject** [`cd40ab53943c`]: not kept: regression: 30.6% slower
236. **thought**: Ask the advisor what the current numbers point to.
238. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
240. **thought**: Candidate budget is used up. Wrapping up.
241. **final** [`c10daa0fb578`]: Candidate budget is used up. Kept the fastest verified candidate.
243. **start** [`cef45563c2ea`]: agent run with planner=heuristic
244. **thought**: Start by looking at the baseline kernel and its launch config.
246. **thought**: Compile the baseline to get register / shared memory numbers.
248. **thought**: Check the baseline is numerically correct before trusting any timing.
250. **thought**: Measure the baseline; every later candidate is compared against this.
251. **accept** [`cef45563c2ea`]: first measured candidate, this is the incumbent to beat
253. **thought**: Profile the baseline to see what it's bound by.
255. **thought**: Ask the advisor what the current numbers point to.
257. **thought**: 64x64 tile reads each A/B element too many times. Hypothesis: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed [small_tiles]
258. **hypothesis** [`c66b494cc0e1`]: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed
260. **thought**: Compile the new candidate.
262. **thought**: It compiles. Validate before timing it.
264. **thought**: Numerics check out. Benchmark it against the incumbent.
265. **accept** [`c66b494cc0e1`]: kept: 33.1% faster (> 1.6% noise threshold)
267. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
269. **thought**: Ask the advisor what the current numbers point to.
271. **thought**: row-major order with N=4096. Hypothesis: Grouped tile order keeps a band of B resident in L2 across consecutive programs [row_major_order]
272. **hypothesis** [`c3785456f3a0`]: Grouped tile order keeps a band of B resident in L2 across consecutive programs
274. **thought**: Compile the new candidate.
276. **thought**: It compiles. Validate before timing it.
278. **thought**: Numerics check out. Benchmark it against the incumbent.
279. **accept** [`c3785456f3a0`]: kept: 5.8% faster (> 1.6% noise threshold)
281. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
283. **thought**: Ask the advisor what the current numbers point to.
285. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
286. **hypothesis** [`cdbade94c1cb`]: One more stage of cp.async prefetch to hide global load latency
288. **thought**: Compile the new candidate.
290. **thought**: It compiles. Validate before timing it.
292. **thought**: Numerics check out. Benchmark it against the incumbent.
293. **accept** [`cdbade94c1cb`]: kept: 9.9% faster (> 1.6% noise threshold)
295. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
297. **thought**: Ask the advisor what the current numbers point to.
299. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 128 -> 256 [local search]
300. **hypothesis** [`ce529cd4633d`]: Local refinement: BLOCK_M 128 -> 256
302. **thought**: Compile the new candidate.
304. **thought**: It compiles. Validate before timing it.
306. **thought**: Numerics check out. Benchmark it against the incumbent.
307. **reject** [`ce529cd4633d`]: not kept: regression: 8.1% slower
309. **thought**: Ask the advisor what the current numbers point to.
311. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 128 -> 64 [local search]
312. **hypothesis** [`c9a0d2dc1e19`]: Local refinement: BLOCK_M 128 -> 64
314. **thought**: Compile the new candidate.
316. **thought**: It compiles. Validate before timing it.
318. **thought**: Numerics check out. Benchmark it against the incumbent.
319. **reject** [`c9a0d2dc1e19`]: not kept: regression: 27.0% slower
321. **thought**: Ask the advisor what the current numbers point to.
323. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_N 128 -> 256 [local search]
324. **hypothesis** [`c92d39f115ce`]: Local refinement: BLOCK_N 128 -> 256
326. **thought**: Compile the new candidate.
328. **thought**: It compiles. Validate before timing it.
330. **thought**: Numerics check out. Benchmark it against the incumbent.
331. **reject** [`c92d39f115ce`]: not kept: regression: 13.6% slower
333. **thought**: Ask the advisor what the current numbers point to.
335. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_N 128 -> 64 [local search]
336. **hypothesis** [`ce805c086fea`]: Local refinement: BLOCK_N 128 -> 64
338. **thought**: Compile the new candidate.
340. **thought**: It compiles. Validate before timing it.
342. **thought**: Numerics check out. Benchmark it against the incumbent.
343. **reject** [`ce805c086fea`]: not kept: regression: 8.1% slower
345. **thought**: Ask the advisor what the current numbers point to.
347. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 8 -> 4 [local search]
348. **hypothesis** [`ce2149090f25`]: Local refinement: num_warps 8 -> 4
350. **thought**: Compile the new candidate.
352. **thought**: It compiles. Validate before timing it.
354. **thought**: Numerics check out. Benchmark it against the incumbent.
355. **reject** [`ce2149090f25`]: not kept: regression: 42.4% slower
357. **thought**: Ask the advisor what the current numbers point to.
359. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
361. **thought**: Candidate budget is used up. Wrapping up.
362. **final** [`cdbade94c1cb`]: Candidate budget is used up. Kept the fastest verified candidate.
364. **start** [`c4ac3d79e3dc`]: agent run with planner=heuristic
365. **thought**: Start by looking at the baseline kernel and its launch config.
367. **thought**: Compile the baseline to get register / shared memory numbers.
369. **thought**: Check the baseline is numerically correct before trusting any timing.
371. **thought**: Measure the baseline; every later candidate is compared against this.
372. **accept** [`c4ac3d79e3dc`]: first measured candidate, this is the incumbent to beat
374. **thought**: Profile the baseline to see what it's bound by.
376. **thought**: Ask the advisor what the current numbers point to.
378. **thought**: 64x64 tile reads each A/B element too many times. Hypothesis: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed [small_tiles]
379. **hypothesis** [`c8f0f17584bc`]: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed
381. **thought**: Compile the new candidate.
383. **thought**: It compiles. Validate before timing it.
385. **thought**: Numerics check out. Benchmark it against the incumbent.
386. **reject** [`c8f0f17584bc`]: not kept: regression: 15.6% slower
388. **thought**: Ask the advisor what the current numbers point to.
390. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
391. **hypothesis** [`c027af7fcf39`]: One more stage of cp.async prefetch to hide global load latency
393. **thought**: Compile the new candidate.
395. **thought**: It compiles. Validate before timing it.
397. **thought**: Numerics check out. Benchmark it against the incumbent.
398. **accept** [`c027af7fcf39`]: kept: 12.9% faster (> 1.6% noise threshold)
400. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
402. **thought**: Ask the advisor what the current numbers point to.
404. **thought**: small output tile. Hypothesis: Grow BLOCK_M only, a smaller step in arithmetic intensity [small_tiles]
405. **hypothesis** [`c2ee0e6e4a64`]: Grow BLOCK_M only, a smaller step in arithmetic intensity
407. **thought**: Compile the new candidate.
409. **thought**: It compiles. Validate before timing it.
411. **thought**: Numerics check out. Benchmark it against the incumbent.
412. **reject** [`c2ee0e6e4a64`]: not kept: regression: 38.3% slower
414. **thought**: Ask the advisor what the current numbers point to.
416. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
417. **hypothesis** [`cb964a7aa998`]: Local refinement: num_stages 3 -> 4
419. **thought**: Compile the new candidate.
421. **thought**: It compiles. Validate before timing it.
423. **thought**: Numerics check out. Benchmark it against the incumbent.
424. **reject** [`cb964a7aa998`]: not kept: within noise (+3.7%, threshold 5.3%)
426. **thought**: Ask the advisor what the current numbers point to.
428. **thought**: structural alternative not explored yet. Hypothesis: Try the grouped variant with similar parameters [local search]
429. **hypothesis** [`c378e4ad133f`]: Try the grouped variant with similar parameters
431. **thought**: Compile the new candidate.
433. **thought**: It compiles. Validate before timing it.
435. **thought**: Numerics check out. Benchmark it against the incumbent.
436. **accept** [`c378e4ad133f`]: kept: 1.5% faster (> 1.4% noise threshold)
438. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
440. **thought**: Ask the advisor what the current numbers point to.
442. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
443. **hypothesis** [`c0600c122d68`]: Local refinement: num_stages 3 -> 4
445. **thought**: Compile the new candidate.
447. **thought**: It compiles. Validate before timing it.
449. **thought**: Numerics check out. Benchmark it against the incumbent.
450. **accept** [`c0600c122d68`]: kept: 4.2% faster (> 1.3% noise threshold)
452. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
454. **thought**: Ask the advisor what the current numbers point to.
456. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: GROUP_M 8 -> 4 [local search]
457. **hypothesis** [`c00d0cb892c8`]: Local refinement: GROUP_M 8 -> 4
459. **thought**: Compile the new candidate.
461. **thought**: It compiles. Validate before timing it.
463. **thought**: Numerics check out. Benchmark it against the incumbent.
464. **reject** [`c00d0cb892c8`]: not kept: within noise (+0.3%, threshold 7.0%)
466. **thought**: Ask the advisor what the current numbers point to.
468. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_M 64 -> 32 [local search]
469. **hypothesis** [`c1d2e609e45b`]: Local refinement: BLOCK_M 64 -> 32
471. **thought**: Compile the new candidate.
473. **thought**: It compiles. Validate before timing it.
475. **thought**: Numerics check out. Benchmark it against the incumbent.
476. **accept** [`c1d2e609e45b`]: kept: 12.8% faster (> 1.3% noise threshold)
478. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
480. **thought**: Ask the advisor what the current numbers point to.
482. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 4 -> 3 [local search]
484. **thought**: Candidate budget is used up. Wrapping up.
485. **final** [`c1d2e609e45b`]: Candidate budget is used up. Kept the fastest verified candidate.
487. **start** [`c2d5f1608525`]: agent run with planner=heuristic
488. **thought**: Start by looking at the baseline kernel and its launch config.
490. **thought**: Compile the baseline to get register / shared memory numbers.
492. **thought**: Check the baseline is numerically correct before trusting any timing.
494. **thought**: Measure the baseline; every later candidate is compared against this.
495. **accept** [`c2d5f1608525`]: first measured candidate, this is the incumbent to beat
497. **thought**: Profile the baseline to see what it's bound by.
499. **thought**: Ask the advisor what the current numbers point to.
501. **thought**: 64x64 tile reads each A/B element too many times. Hypothesis: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed [small_tiles]
502. **hypothesis** [`c7ab7966ab17`]: 128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed
504. **thought**: Compile the new candidate.
506. **thought**: It compiles. Validate before timing it.
508. **thought**: Numerics check out. Benchmark it against the incumbent.
509. **reject** [`c7ab7966ab17`]: not kept: regression: 21.6% slower
511. **thought**: Ask the advisor what the current numbers point to.
513. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
514. **hypothesis** [`c31f120a202a`]: One more stage of cp.async prefetch to hide global load latency
516. **thought**: Compile the new candidate.
518. **thought**: It compiles. Validate before timing it.
520. **thought**: Numerics check out. Benchmark it against the incumbent.
521. **reject** [`c31f120a202a`]: not kept: within noise (-4.9%, threshold 12.0%)
523. **thought**: Ask the advisor what the current numbers point to.
525. **thought**: small output tile. Hypothesis: Grow BLOCK_M only, a smaller step in arithmetic intensity [small_tiles]
526. **hypothesis** [`c9a8dbe6a8e6`]: Grow BLOCK_M only, a smaller step in arithmetic intensity
528. **thought**: Compile the new candidate.
530. **thought**: It compiles. Validate before timing it.
532. **thought**: Numerics check out. Benchmark it against the incumbent.
533. **reject** [`c9a8dbe6a8e6`]: not kept: within noise (+8.5%, threshold 12.0%)
535. **thought**: Ask the advisor what the current numbers point to.
537. **thought**: structural alternative not explored yet. Hypothesis: Try the grouped variant with similar parameters [local search]
538. **hypothesis** [`c40d2be2e936`]: Try the grouped variant with similar parameters
540. **thought**: Compile the new candidate.
542. **thought**: It compiles. Validate before timing it.
544. **thought**: Numerics check out. Benchmark it against the incumbent.
545. **accept** [`c40d2be2e936`]: kept: 48.6% faster (> 12.0% noise threshold)
547. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
549. **thought**: Ask the advisor what the current numbers point to.
551. **thought**: num_stages=2. Hypothesis: One more stage of cp.async prefetch to hide global load latency [shallow_pipeline]
552. **hypothesis** [`c13b3b8a9961`]: One more stage of cp.async prefetch to hide global load latency
554. **thought**: Compile the new candidate.
556. **thought**: It compiles. Validate before timing it.
558. **thought**: Numerics check out. Benchmark it against the incumbent.
559. **accept** [`c13b3b8a9961`]: kept: 13.5% faster (> 4.2% noise threshold)
561. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
563. **thought**: Ask the advisor what the current numbers point to.
565. **thought**: small output tile. Hypothesis: Grow BLOCK_M only, a smaller step in arithmetic intensity [small_tiles]
566. **hypothesis** [`cf3f17a2be41`]: Grow BLOCK_M only, a smaller step in arithmetic intensity
568. **thought**: Compile the new candidate.
570. **thought**: It compiles. Validate before timing it.
572. **thought**: Numerics check out. Benchmark it against the incumbent.
573. **reject** [`cf3f17a2be41`]: not kept: regression: 29.1% slower
575. **thought**: Ask the advisor what the current numbers point to.
577. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 3 -> 4 [local search]
578. **hypothesis** [`ca65ab7971ac`]: Local refinement: num_stages 3 -> 4
580. **thought**: Compile the new candidate.
582. **thought**: It compiles. Validate before timing it.
584. **thought**: Numerics check out. Benchmark it against the incumbent.
585. **reject** [`ca65ab7971ac`]: not kept: regression: 22.8% slower
587. **thought**: Ask the advisor what the current numbers point to.
589. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: GROUP_M 8 -> 4 [local search]
590. **hypothesis** [`c26332107055`]: Local refinement: GROUP_M 8 -> 4
592. **thought**: Compile the new candidate.
594. **thought**: It compiles. Validate before timing it.
596. **thought**: Numerics check out. Benchmark it against the incumbent.
597. **accept** [`c26332107055`]: kept: 15.7% faster (> 4.2% noise threshold)
599. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
601. **thought**: Ask the advisor what the current numbers point to.
603. **thought**: occupancy 18.8% limited by shared memory. Hypothesis: Fewer stages frees shared memory so more blocks fit per SM [low_occupancy]
605. **thought**: Candidate budget is used up. Wrapping up.
606. **final** [`c26332107055`]: Candidate budget is used up. Kept the fastest verified candidate.
608. **start** [`c8ade2395d67`]: agent run with planner=heuristic
609. **thought**: Start by looking at the baseline kernel and its launch config.
611. **thought**: Compile the baseline to get register / shared memory numbers.
613. **thought**: Check the baseline is numerically correct before trusting any timing.
615. **thought**: Measure the baseline; every later candidate is compared against this.
616. **accept** [`c8ade2395d67`]: first measured candidate, this is the incumbent to beat
618. **thought**: Profile the baseline to see what it's bound by.
620. **thought**: Ask the advisor what the current numbers point to.
622. **thought**: structural alternative not explored yet. Hypothesis: Try the online variant with similar parameters [local search]
623. **hypothesis** [`cde30ceb82da`]: Try the online variant with similar parameters
625. **thought**: Compile the new candidate.
627. **thought**: It compiles. Validate before timing it.
629. **thought**: Numerics check out. Benchmark it against the incumbent.
630. **reject** [`cde30ceb82da`]: not kept: regression: 26.7% slower
632. **thought**: Ask the advisor what the current numbers point to.
634. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 8 [local search]
635. **hypothesis** [`c525ea36699c`]: Local refinement: num_warps 4 -> 8
637. **thought**: Compile the new candidate.
639. **thought**: It compiles. Validate before timing it.
641. **thought**: Numerics check out. Benchmark it against the incumbent.
642. **reject** [`c525ea36699c`]: not kept: within noise (+0.4%, threshold 1.7%)
644. **thought**: Ask the advisor what the current numbers point to.
646. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 2 [local search]
647. **hypothesis** [`c5eab6ac1d16`]: Local refinement: num_warps 4 -> 2
649. **thought**: Compile the new candidate.
651. **thought**: It compiles. Validate before timing it.
653. **thought**: Numerics check out. Benchmark it against the incumbent.
654. **reject** [`c5eab6ac1d16`]: not kept: within noise (-5.6%, threshold 7.6%)
656. **thought**: Ask the advisor what the current numbers point to.
658. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
659. **hypothesis** [`cfe32cd961c0`]: Local refinement: num_stages 2 -> 1
661. **thought**: Compile the new candidate.
663. **thought**: It compiles. Validate before timing it.
665. **thought**: Numerics check out. Benchmark it against the incumbent.
666. **reject** [`cfe32cd961c0`]: not kept: within noise (+0.1%, threshold 1.7%)
668. **thought**: Ask the advisor what the current numbers point to.
670. **thought**: No untried changes left that the observations justify.
671. **final** [`c8ade2395d67`]: Stopped when the advisor ran out of justified changes; kept the fastest verified candidate.
673. **start** [`c8568dfa90cf`]: agent run with planner=heuristic
674. **thought**: Start by looking at the baseline kernel and its launch config.
676. **thought**: Compile the baseline to get register / shared memory numbers.
678. **thought**: Check the baseline is numerically correct before trusting any timing.
680. **thought**: Measure the baseline; every later candidate is compared against this.
681. **accept** [`c8568dfa90cf`]: first measured candidate, this is the incumbent to beat
683. **thought**: Profile the baseline to see what it's bound by.
685. **thought**: Ask the advisor what the current numbers point to.
687. **thought**: structural alternative not explored yet. Hypothesis: Try the online variant with similar parameters [local search]
688. **hypothesis** [`ca68fcc987c9`]: Try the online variant with similar parameters
690. **thought**: Compile the new candidate.
692. **thought**: It compiles. Validate before timing it.
694. **thought**: Numerics check out. Benchmark it against the incumbent.
695. **reject** [`ca68fcc987c9`]: not kept: regression: 14.3% slower
697. **thought**: Ask the advisor what the current numbers point to.
699. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 8 [local search]
700. **hypothesis** [`c01616836bd7`]: Local refinement: num_warps 4 -> 8
702. **thought**: Compile the new candidate.
704. **thought**: It compiles. Validate before timing it.
706. **thought**: Numerics check out. Benchmark it against the incumbent.
707. **reject** [`c01616836bd7`]: not kept: within noise (+0.1%, threshold 1.7%)
709. **thought**: Ask the advisor what the current numbers point to.
711. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 2 [local search]
712. **hypothesis** [`cdca37f56a51`]: Local refinement: num_warps 4 -> 2
714. **thought**: Compile the new candidate.
716. **thought**: It compiles. Validate before timing it.
718. **thought**: Numerics check out. Benchmark it against the incumbent.
719. **reject** [`cdca37f56a51`]: not kept: within noise (-0.8%, threshold 1.7%)
721. **thought**: Ask the advisor what the current numbers point to.
723. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
724. **hypothesis** [`cc99bab0eea8`]: Local refinement: num_stages 2 -> 1
726. **thought**: Compile the new candidate.
728. **thought**: It compiles. Validate before timing it.
730. **thought**: Numerics check out. Benchmark it against the incumbent.
731. **reject** [`cc99bab0eea8`]: not kept: within noise (-0.1%, threshold 1.7%)
733. **thought**: Ask the advisor what the current numbers point to.
735. **thought**: No untried changes left that the observations justify.
736. **final** [`c8568dfa90cf`]: Stopped when the advisor ran out of justified changes; kept the fastest verified candidate.
738. **start** [`cf50c1f74a38`]: agent run with planner=heuristic
739. **thought**: Start by looking at the baseline kernel and its launch config.
741. **thought**: Compile the baseline to get register / shared memory numbers.
743. **thought**: Check the baseline is numerically correct before trusting any timing.
745. **thought**: Measure the baseline; every later candidate is compared against this.
746. **accept** [`cf50c1f74a38`]: first measured candidate, this is the incumbent to beat
748. **thought**: Profile the baseline to see what it's bound by.
750. **thought**: Ask the advisor what the current numbers point to.
752. **thought**: structural alternative not explored yet. Hypothesis: Try the online variant with similar parameters [local search]
753. **hypothesis** [`c1609d2b3835`]: Try the online variant with similar parameters
755. **thought**: Compile the new candidate.
757. **thought**: It compiles. Validate before timing it.
759. **thought**: Numerics check out. Benchmark it against the incumbent.
760. **reject** [`c1609d2b3835`]: not kept: regression: 28.5% slower
762. **thought**: Ask the advisor what the current numbers point to.
764. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 8 [local search]
765. **hypothesis** [`c6961f1f64d4`]: Local refinement: num_warps 4 -> 8
767. **thought**: Compile the new candidate.
769. **thought**: It compiles. Validate before timing it.
771. **thought**: Numerics check out. Benchmark it against the incumbent.
772. **accept** [`c6961f1f64d4`]: kept: 22.6% faster (> 1.6% noise threshold)
774. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
776. **thought**: Ask the advisor what the current numbers point to.
778. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 8 -> 16 [local search]
779. **hypothesis** [`c788d85e3dc1`]: Local refinement: num_warps 8 -> 16
781. **thought**: Compile the new candidate.
783. **thought**: It compiles. Validate before timing it.
785. **thought**: Numerics check out. Benchmark it against the incumbent.
786. **accept** [`c788d85e3dc1`]: kept: 14.4% faster (> 1.6% noise threshold)
788. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
790. **thought**: Ask the advisor what the current numbers point to.
792. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 16 -> 32 [local search]
793. **hypothesis** [`cba6d7c1392c`]: Local refinement: num_warps 16 -> 32
795. **thought**: Compile the new candidate.
797. **thought**: It compiles. Validate before timing it.
799. **thought**: Numerics check out. Benchmark it against the incumbent.
800. **reject** [`cba6d7c1392c`]: not kept: within noise (+0.9%, threshold 1.3%)
802. **thought**: Ask the advisor what the current numbers point to.
804. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
805. **hypothesis** [`cbd086cdd08a`]: Local refinement: num_stages 2 -> 1
807. **thought**: Compile the new candidate.
809. **thought**: It compiles. Validate before timing it.
811. **thought**: Numerics check out. Benchmark it against the incumbent.
812. **reject** [`cbd086cdd08a`]: not kept: within noise (-0.7%, threshold 1.3%)
814. **thought**: Ask the advisor what the current numbers point to.
816. **thought**: No untried changes left that the observations justify.
817. **final** [`c788d85e3dc1`]: Stopped when the advisor ran out of justified changes; kept the fastest verified candidate.
819. **start** [`cebf1cea5c64`]: agent run with planner=heuristic
820. **thought**: Start by looking at the baseline kernel and its launch config.
822. **thought**: Compile the baseline to get register / shared memory numbers.
824. **thought**: Check the baseline is numerically correct before trusting any timing.
826. **thought**: Measure the baseline; every later candidate is compared against this.
827. **accept** [`cebf1cea5c64`]: first measured candidate, this is the incumbent to beat
829. **thought**: Profile the baseline to see what it's bound by.
831. **thought**: Ask the advisor what the current numbers point to.
833. **thought**: DRAM 49%, SM 48%. Hypothesis: More warps in flight to cover DRAM latency [latency_bound]
834. **hypothesis** [`ca9600e2e18a`]: More warps in flight to cover DRAM latency
836. **thought**: Compile the new candidate.
838. **thought**: It compiles. Validate before timing it.
840. **thought**: Numerics check out. Benchmark it against the incumbent.
841. **reject** [`ca9600e2e18a`]: not kept: within noise (+0.4%, threshold 1.2%)
843. **thought**: Ask the advisor what the current numbers point to.
845. **thought**: structural alternative not explored yet. Hypothesis: Try the online variant with similar parameters [local search]
846. **hypothesis** [`c1b268eda3a1`]: Try the online variant with similar parameters
848. **thought**: Compile the new candidate.
850. **thought**: It compiles. Validate before timing it.
852. **thought**: Numerics check out. Benchmark it against the incumbent.
853. **reject** [`c1b268eda3a1`]: not kept: regression: 39.4% slower
855. **thought**: Ask the advisor what the current numbers point to.
857. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 2 [local search]
858. **hypothesis** [`c921f6576fe3`]: Local refinement: num_warps 4 -> 2
860. **thought**: Compile the new candidate.
862. **thought**: It compiles. Validate before timing it.
864. **thought**: Numerics check out. Benchmark it against the incumbent.
865. **reject** [`c921f6576fe3`]: not kept: regression: 7.0% slower
867. **thought**: Ask the advisor what the current numbers point to.
869. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
870. **hypothesis** [`c62e7a2990b9`]: Local refinement: num_stages 2 -> 1
872. **thought**: Compile the new candidate.
874. **thought**: It compiles. Validate before timing it.
876. **thought**: Numerics check out. Benchmark it against the incumbent.
877. **reject** [`c62e7a2990b9`]: not kept: within noise (-0.5%, threshold 1.2%)
879. **thought**: Ask the advisor what the current numbers point to.
881. **thought**: No untried changes left that the observations justify.
882. **final** [`cebf1cea5c64`]: Stopped when the advisor ran out of justified changes; kept the fastest verified candidate.
884. **start** [`ce5ba5f5a813`]: agent run with planner=heuristic
885. **thought**: Start by looking at the baseline kernel and its launch config.
887. **thought**: Compile the baseline to get register / shared memory numbers.
889. **thought**: Check the baseline is numerically correct before trusting any timing.
891. **thought**: Measure the baseline; every later candidate is compared against this.
892. **accept** [`ce5ba5f5a813`]: first measured candidate, this is the incumbent to beat
894. **thought**: Profile the baseline to see what it's bound by.
896. **thought**: Ask the advisor what the current numbers point to.
898. **thought**: register pressure (n_regs=255, n_spills=149). Hypothesis: Doubling num_warps halves the elements each thread holds, which should remove the spills [spills]
899. **hypothesis** [`c5ef19b599b9`]: Doubling num_warps halves the elements each thread holds, which should remove the spills
901. **thought**: Compile the new candidate.
903. **thought**: It compiles. Validate before timing it.
905. **thought**: Numerics check out. Benchmark it against the incumbent.
906. **accept** [`c5ef19b599b9`]: kept: 76.2% faster (> 1.4% noise threshold)
908. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
910. **thought**: Ask the advisor what the current numbers point to.
912. **thought**: register pressure (n_regs=212). Hypothesis: Doubling num_warps halves the elements each thread holds, which should remove the spills [register_pressure]
913. **hypothesis** [`c54aafbcceda`]: Doubling num_warps halves the elements each thread holds, which should remove the spills
915. **thought**: Compile the new candidate.
917. **thought**: It compiles. Validate before timing it.
919. **thought**: Numerics check out. Benchmark it against the incumbent.
920. **accept** [`c54aafbcceda`]: kept: 30.2% faster (> 1.6% noise threshold)
922. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
924. **thought**: Ask the advisor what the current numbers point to.
926. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 16 -> 32 [local search]
927. **hypothesis** [`c9bcf3336361`]: Local refinement: num_warps 16 -> 32
929. **thought**: Compile the new candidate.
931. **thought**: It compiles. Validate before timing it.
933. **thought**: Numerics check out. Benchmark it against the incumbent.
934. **accept** [`c9bcf3336361`]: kept: 26.9% faster (> 8.2% noise threshold)
936. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
938. **thought**: Ask the advisor what the current numbers point to.
940. **thought**: whole 16384-wide row lives in registers (n_regs=None). Hypothesis: Stream the row in chunks with online softmax instead of holding it all in registers [low_occupancy]
941. **hypothesis** [`ca00acc91aef`]: Stream the row in chunks with online softmax instead of holding it all in registers
943. **thought**: Compile the new candidate.
945. **thought**: It compiles. Validate before timing it.
947. **thought**: Numerics check out. Benchmark it against the incumbent.
948. **reject** [`ca00acc91aef`]: not kept: within noise (+1.3%, threshold 8.2%)
950. **thought**: Ask the advisor what the current numbers point to.
952. **thought**: structural alternative not explored yet. Hypothesis: Try the online variant with similar parameters [local search]
953. **hypothesis** [`cd2e11bac066`]: Try the online variant with similar parameters
955. **thought**: Compile the new candidate.
957. **thought**: It compiles. Validate before timing it.
959. **thought**: Numerics check out. Benchmark it against the incumbent.
960. **reject** [`cd2e11bac066`]: not kept: within noise (+2.9%, threshold 8.2%)
962. **thought**: Ask the advisor what the current numbers point to.
964. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
965. **hypothesis** [`cffe3e882ba0`]: Local refinement: num_stages 2 -> 1
967. **thought**: Compile the new candidate.
969. **thought**: It compiles. Validate before timing it.
971. **thought**: Numerics check out. Benchmark it against the incumbent.
972. **reject** [`cffe3e882ba0`]: not kept: within noise (-0.5%, threshold 8.2%)
974. **thought**: Ask the advisor what the current numbers point to.
976. **thought**: No untried changes left that the observations justify.
977. **final** [`c9bcf3336361`]: Stopped when the advisor ran out of justified changes; kept the fastest verified candidate.
979. **start** [`c74310836232`]: agent run with planner=heuristic
980. **thought**: Start by looking at the baseline kernel and its launch config.
982. **thought**: Compile the baseline to get register / shared memory numbers.
984. **thought**: Check the baseline is numerically correct before trusting any timing.
986. **thought**: Measure the baseline; every later candidate is compared against this.
987. **accept** [`c74310836232`]: first measured candidate, this is the incumbent to beat
989. **thought**: Profile the baseline to see what it's bound by.
991. **thought**: Ask the advisor what the current numbers point to.
993. **thought**: DRAM 34%, SM 1%. Hypothesis: More warps in flight to cover DRAM latency [latency_bound]
994. **hypothesis** [`c72d3c5b1cf6`]: More warps in flight to cover DRAM latency
996. **thought**: Compile the new candidate.
998. **thought**: It compiles. Validate before timing it.
1000. **thought**: Numerics check out. Benchmark it against the incumbent.
1001. **reject** [`c72d3c5b1cf6`]: not kept: within noise (+0.1%, threshold 1.3%)
1003. **thought**: Ask the advisor what the current numbers point to.
1005. **thought**: structural alternative not explored yet. Hypothesis: Try the two_pass variant with similar parameters [local search]
1006. **hypothesis** [`ce64afcedb70`]: Try the two_pass variant with similar parameters
1008. **thought**: Compile the new candidate.
1010. **thought**: It compiles. Validate before timing it.
1012. **thought**: Numerics check out. Benchmark it against the incumbent.
1013. **reject** [`ce64afcedb70`]: not kept: regression: 12.6% slower
1015. **thought**: Ask the advisor what the current numbers point to.
1017. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 1024 -> 2048 [local search]
1018. **hypothesis** [`c1557156c81a`]: Local refinement: BLOCK_SIZE 1024 -> 2048
1020. **thought**: Compile the new candidate.
1022. **thought**: It compiles. Validate before timing it.
1024. **thought**: Numerics check out. Benchmark it against the incumbent.
1025. **accept** [`c1557156c81a`]: kept: 12.7% faster (> 4.6% noise threshold)
1027. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1029. **thought**: Ask the advisor what the current numbers point to.
1031. **thought**: DRAM 39%, SM 2%. Hypothesis: More warps in flight to cover DRAM latency [latency_bound]
1032. **hypothesis** [`c72a9737c89b`]: More warps in flight to cover DRAM latency
1034. **thought**: Compile the new candidate.
1036. **thought**: It compiles. Validate before timing it.
1038. **thought**: Numerics check out. Benchmark it against the incumbent.
1039. **reject** [`c72a9737c89b`]: not kept: within noise (+0.0%, threshold 4.6%)
1041. **thought**: Ask the advisor what the current numbers point to.
1043. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 2048 -> 4096 [local search]
1044. **hypothesis** [`cc551e72693d`]: Local refinement: BLOCK_SIZE 2048 -> 4096
1046. **thought**: Compile the new candidate.
1048. **thought**: It compiles. Validate before timing it.
1050. **thought**: Numerics check out. Benchmark it against the incumbent.
1051. **reject** [`cc551e72693d`]: not kept: regression: 7.3% slower
1053. **thought**: Ask the advisor what the current numbers point to.
1055. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: TILES 1 -> 2 [local search]
1056. **hypothesis** [`cdc9ad9a6d6e`]: Local refinement: TILES 1 -> 2
1058. **thought**: Compile the new candidate.
1060. **thought**: It compiles. Validate before timing it.
1062. **thought**: Numerics check out. Benchmark it against the incumbent.
1063. **accept** [`cdc9ad9a6d6e`]: kept: 7.6% faster (> 4.6% noise threshold)
1065. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1067. **thought**: Ask the advisor what the current numbers point to.
1069. **thought**: DRAM 42%, SM 2%. Hypothesis: More warps in flight to cover DRAM latency [latency_bound]
1070. **hypothesis** [`ce3218c591ac`]: More warps in flight to cover DRAM latency
1072. **thought**: Compile the new candidate.
1074. **thought**: It compiles. Validate before timing it.
1076. **thought**: Numerics check out. Benchmark it against the incumbent.
1077. **reject** [`ce3218c591ac`]: not kept: within noise (-0.3%, threshold 1.6%)
1079. **thought**: Ask the advisor what the current numbers point to.
1081. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 2048 -> 1024 [local search]
1082. **hypothesis** [`c414cb1edba6`]: Local refinement: BLOCK_SIZE 2048 -> 1024
1084. **thought**: Compile the new candidate.
1086. **thought**: It compiles. Validate before timing it.
1088. **thought**: Numerics check out. Benchmark it against the incumbent.
1089. **reject** [`c414cb1edba6`]: not kept: regression: 8.1% slower
1091. **thought**: Ask the advisor what the current numbers point to.
1093. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: TILES 2 -> 4 [local search]
1095. **thought**: Candidate budget is used up. Wrapping up.
1096. **final** [`cdc9ad9a6d6e`]: Candidate budget is used up. Kept the fastest verified candidate.
1098. **start** [`c17cccb49a24`]: agent run with planner=heuristic
1099. **thought**: Start by looking at the baseline kernel and its launch config.
1101. **thought**: Compile the baseline to get register / shared memory numbers.
1103. **thought**: Check the baseline is numerically correct before trusting any timing.
1105. **thought**: Measure the baseline; every later candidate is compared against this.
1106. **accept** [`c17cccb49a24`]: first measured candidate, this is the incumbent to beat
1108. **thought**: Profile the baseline to see what it's bound by.
1110. **thought**: Ask the advisor what the current numbers point to.
1112. **thought**: 16384 atomics on one address per call. Hypothesis: Fold more tiles per program so there are 4x fewer same-address atomics [atomic_contention]
1113. **hypothesis** [`c517f56a4676`]: Fold more tiles per program so there are 4x fewer same-address atomics
1115. **thought**: Compile the new candidate.
1117. **thought**: It compiles. Validate before timing it.
1119. **thought**: Numerics check out. Benchmark it against the incumbent.
1120. **accept** [`c517f56a4676`]: kept: 30.0% faster (> 5.1% noise threshold)
1122. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1124. **thought**: Ask the advisor what the current numbers point to.
1126. **thought**: 4096 atomics on one address per call. Hypothesis: Fold more tiles per program so there are 4x fewer same-address atomics [atomic_contention]
1127. **hypothesis** [`cf9b640ebe86`]: Fold more tiles per program so there are 4x fewer same-address atomics
1129. **thought**: Compile the new candidate.
1131. **thought**: It compiles. Validate before timing it.
1133. **thought**: Numerics check out. Benchmark it against the incumbent.
1134. **accept** [`cf9b640ebe86`]: kept: 10.5% faster (> 5.1% noise threshold)
1136. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1138. **thought**: Ask the advisor what the current numbers point to.
1140. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: TILES 16 -> 8 [local search]
1141. **hypothesis** [`c818ca4bf1f6`]: Local refinement: TILES 16 -> 8
1143. **thought**: Compile the new candidate.
1145. **thought**: It compiles. Validate before timing it.
1147. **thought**: Numerics check out. Benchmark it against the incumbent.
1148. **reject** [`c818ca4bf1f6`]: not kept: regression: 3.2% slower
1150. **thought**: Ask the advisor what the current numbers point to.
1152. **thought**: structural alternative not explored yet. Hypothesis: Try the two_pass variant with similar parameters [local search]
1153. **hypothesis** [`cd8beece79ec`]: Try the two_pass variant with similar parameters
1155. **thought**: Compile the new candidate.
1157. **thought**: It compiles. Validate before timing it.
1159. **thought**: Numerics check out. Benchmark it against the incumbent.
1160. **reject** [`cd8beece79ec`]: not kept: regression: 2.3% slower
1162. **thought**: Ask the advisor what the current numbers point to.
1164. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 1024 -> 2048 [local search]
1165. **hypothesis** [`ccadec6c0d5e`]: Local refinement: BLOCK_SIZE 1024 -> 2048
1167. **thought**: Compile the new candidate.
1169. **thought**: It compiles. Validate before timing it.
1171. **thought**: Numerics check out. Benchmark it against the incumbent.
1172. **accept** [`ccadec6c0d5e`]: kept: 2.1% faster (> 1.2% noise threshold)
1174. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1176. **thought**: Ask the advisor what the current numbers point to.
1178. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 2048 -> 4096 [local search]
1179. **hypothesis** [`c954a9347c02`]: Local refinement: BLOCK_SIZE 2048 -> 4096
1181. **thought**: Compile the new candidate.
1183. **thought**: It compiles. Validate before timing it.
1185. **thought**: Numerics check out. Benchmark it against the incumbent.
1186. **reject** [`c954a9347c02`]: not kept: regression: 30.2% slower
1188. **thought**: Ask the advisor what the current numbers point to.
1190. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 8 [local search]
1191. **hypothesis** [`c0bd10f431a0`]: Local refinement: num_warps 4 -> 8
1193. **thought**: Compile the new candidate.
1195. **thought**: It compiles. Validate before timing it.
1197. **thought**: Numerics check out. Benchmark it against the incumbent.
1198. **reject** [`c0bd10f431a0`]: not kept: within noise (+0.2%, threshold 1.5%)
1200. **thought**: Ask the advisor what the current numbers point to.
1202. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 2 [local search]
1203. **hypothesis** [`c9c254aaf5b5`]: Local refinement: num_warps 4 -> 2
1205. **thought**: Compile the new candidate.
1207. **thought**: It compiles. Validate before timing it.
1209. **thought**: Numerics check out. Benchmark it against the incumbent.
1210. **reject** [`c9c254aaf5b5`]: not kept: regression: 22.6% slower
1212. **thought**: Ask the advisor what the current numbers point to.
1214. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_stages 2 -> 1 [local search]
1216. **thought**: Candidate budget is used up. Wrapping up.
1217. **final** [`ccadec6c0d5e`]: Candidate budget is used up. Kept the fastest verified candidate.
1219. **start** [`c5e2cf5e0b93`]: agent run with planner=heuristic
1220. **thought**: Start by looking at the baseline kernel and its launch config.
1222. **thought**: Compile the baseline to get register / shared memory numbers.
1224. **thought**: Check the baseline is numerically correct before trusting any timing.
1226. **thought**: Measure the baseline; every later candidate is compared against this.
1227. **accept** [`c5e2cf5e0b93`]: first measured candidate, this is the incumbent to beat
1229. **thought**: Profile the baseline to see what it's bound by.
1231. **thought**: Ask the advisor what the current numbers point to.
1233. **thought**: 4097 atomics on one address per call. Hypothesis: Fold more tiles per program so there are 4x fewer same-address atomics [atomic_contention]
1234. **hypothesis** [`c123af1ef0a4`]: Fold more tiles per program so there are 4x fewer same-address atomics
1236. **thought**: Compile the new candidate.
1238. **thought**: It compiles. Validate before timing it.
1240. **thought**: Numerics check out. Benchmark it against the incumbent.
1241. **accept** [`c123af1ef0a4`]: kept: 35.8% faster (> 1.4% noise threshold)
1243. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1245. **thought**: Ask the advisor what the current numbers point to.
1247. **thought**: DRAM 49%, SM 4%. Hypothesis: More warps in flight to cover DRAM latency [latency_bound]
1248. **hypothesis** [`cd0e597ed861`]: More warps in flight to cover DRAM latency
1250. **thought**: Compile the new candidate.
1252. **thought**: It compiles. Validate before timing it.
1254. **thought**: Numerics check out. Benchmark it against the incumbent.
1255. **reject** [`cd0e597ed861`]: not kept: within noise (+0.2%, threshold 5.8%)
1257. **thought**: Ask the advisor what the current numbers point to.
1259. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: TILES 4 -> 8 [local search]
1260. **hypothesis** [`c474438bf798`]: Local refinement: TILES 4 -> 8
1262. **thought**: Compile the new candidate.
1264. **thought**: It compiles. Validate before timing it.
1266. **thought**: Numerics check out. Benchmark it against the incumbent.
1267. **accept** [`c474438bf798`]: kept: 9.7% faster (> 7.8% noise threshold)
1269. **thought**: New incumbent. Re-profile it so the next hypothesis uses fresh numbers.
1271. **thought**: Ask the advisor what the current numbers point to.
1273. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: TILES 8 -> 16 [local search]
1274. **hypothesis** [`c32d417c3062`]: Local refinement: TILES 8 -> 16
1276. **thought**: Compile the new candidate.
1278. **thought**: It compiles. Validate before timing it.
1280. **thought**: Numerics check out. Benchmark it against the incumbent.
1281. **reject** [`c32d417c3062`]: not kept: within noise (+5.2%, threshold 7.8%)
1283. **thought**: Ask the advisor what the current numbers point to.
1285. **thought**: structural alternative not explored yet. Hypothesis: Try the two_pass variant with similar parameters [local search]
1286. **hypothesis** [`c6ccb481f49e`]: Try the two_pass variant with similar parameters
1288. **thought**: Compile the new candidate.
1290. **thought**: It compiles. Validate before timing it.
1292. **thought**: Numerics check out. Benchmark it against the incumbent.
1293. **reject** [`c6ccb481f49e`]: not kept: regression: 19.9% slower
1295. **thought**: Ask the advisor what the current numbers point to.
1297. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 1024 -> 2048 [local search]
1298. **hypothesis** [`cfb69022249a`]: Local refinement: BLOCK_SIZE 1024 -> 2048
1300. **thought**: Compile the new candidate.
1302. **thought**: It compiles. Validate before timing it.
1304. **thought**: Numerics check out. Benchmark it against the incumbent.
1305. **reject** [`cfb69022249a`]: not kept: within noise (+5.4%, threshold 7.8%)
1307. **thought**: Ask the advisor what the current numbers point to.
1309. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: BLOCK_SIZE 1024 -> 512 [local search]
1310. **hypothesis** [`ca7bf53a2e11`]: Local refinement: BLOCK_SIZE 1024 -> 512
1312. **thought**: Compile the new candidate.
1314. **thought**: It compiles. Validate before timing it.
1316. **thought**: Numerics check out. Benchmark it against the incumbent.
1317. **reject** [`ca7bf53a2e11`]: not kept: regression: 10.7% slower
1319. **thought**: Ask the advisor what the current numbers point to.
1321. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 8 [local search]
1322. **hypothesis** [`c8a0aa6b30e8`]: Local refinement: num_warps 4 -> 8
1324. **thought**: Compile the new candidate.
1326. **thought**: It compiles. Validate before timing it.
1328. **thought**: Numerics check out. Benchmark it against the incumbent.
1329. **reject** [`c8a0aa6b30e8`]: not kept: within noise (-0.3%, threshold 7.8%)
1331. **thought**: Ask the advisor what the current numbers point to.
1333. **thought**: no specific bottleneck left to target; probing a neighbouring config. Hypothesis: Local refinement: num_warps 4 -> 2 [local search]
1335. **thought**: Candidate budget is used up. Wrapping up.
1336. **final** [`c474438bf798`]: Candidate budget is used up. Kept the fastest verified candidate.

## Reproduce

```bash
TRITUNE_SEED=0 TRITUNE_REPS=50 TRITUNE_WARMUP=10 tritune ...  # see experiment params above
```

Environment fingerprint: `{"backend": "sim", "gpu": "SIM-A100-40GB (simulated)", "cc": null, "driver": null, "cuda": null, "torch": null, "triton": null, "tritune": "0.1.0", "model": "sim-roofline-1"}`
