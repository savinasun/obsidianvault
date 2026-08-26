### [[dl_moc]]
# **GPU Architecture**

**Tags**:: #llm
**Links**:: [[LLMs]] · [[Optimization Algorithms]] · [[Attention]] · [[Transformer]]


This note follows the Cornell Virtual Workshop roadmap [*Understanding GPU Architecture*](https://cvw.cac.cornell.edu/gpu-architecture) (Steve Lantz & Zilu Wang, Cornell Center for Advanced Computing; revised through 4/2026). The roadmap's thesis: GPU hardware is optimized for **highly parallel processing**, but the design differs from a CPU's more in *proportions and terminology* than in kind — a CUDA core is closer to a *vector lane* than to a CPU core, a warp is a *vector*, and an SM is the real analog of a CPU core. Three concrete devices are dissected: the **Blackwell B200** (TACC Horizon), the **Quadro RTX 5000** (TACC Frontera), and the **Tesla V100** (TACC's former Longhorn). Sections marked *(beyond the tutorial)* collect material added from other sources.

**How to use this note.** The [[#Glossary]] defines every term used below; each entry ends with `→` links to the section(s) where it is explained. In the body, a term's first mention in a section links back to its glossary entry. [[#7. NVIDIA Architecture Timeline]] collects the per-generation numbers.

## Glossary
---
### Execution model
- **GPGPU** — *General-Purpose computing on GPUs*: using the graphics processor's floating-point engine for non-graphics workloads with huge flop counts and simple data-access patterns. → [[#Heterogeneous Applications (GPGPU)]] ^gpgpu
- **CUDA** — NVIDIA's API/language for GPGPU (C++ with `__global__` kernels and `<<<M, N>>>` launch syntax). Directive-based alternatives: **OpenACC**, **OpenMP offloading**. → [[#Heterogeneous Applications (GPGPU)]], [[#Kernels and SMs]] ^cuda
- **Heterogeneous computing** — programming model in which a CPU *host* runs control logic and offloads massively parallel work to GPU *devices*. → [[#Heterogeneous Applications (GPGPU)]] ^heterogeneous
- **SIMD** — *Single Instruction, Multiple Data*: one instruction acts identically on every element of a vector (CPU vector units: 8–16 lanes, e.g. AVX-512 = 16 floats). → [[#SIMT and Warps]] ^simd
- **SIMT** — *Single Instruction, Multiple Threads*: NVIDIA's relaxed SIMD in which individual threads can be *masked* active/inactive, so branching is possible (but inefficient). Every thread executes the kernel code as written. → [[#SIMT and Warps]] ^simt
- **Thread** — the stream of instructions + data assigned to one CUDA core; has a unique ID used for addressing and control. No exact CPU equivalent. → [[#Threads and Cores Redefined]] ^thread
- **Warp** — group of **32 threads** with consecutive indices executing the same instruction together on different data; processed by 32 CUDA cores. CPU analog: a *vector*. (Named after weaving.) → [[#SIMT and Warps]] ^warp
- **Branch divergence** — threads within one warp taking different branches; the branches execute serially with inactive threads idling. Volta+ *independent thread scheduling* allows finer interleaving/sync. → [[#SIMT and Warps]] ^divergence
- **Thread block** — the group of N threads launched together by `<<<M, N>>>`; a block runs on exactly one SM (never split) and its threads can share data through shared memory. → [[#Kernels and SMs]] ^block
- **Kernel** — a function (`__global__` in CUDA) executed on the device N times in parallel by N threads, organized as M blocks. → [[#Kernels and SMs]] ^kernel
- **CUDA core** — the unit that processes one data item at a time for its share of a SIMT instruction; grouped in sets of 32. CPU analog: a *vector lane*, not a core. Typed by datatype (FP32, INT32, FP64). → [[#Threads and Cores Redefined]] ^cuda-core
- **SM (Streaming Multiprocessor)** — the unit that executes thread blocks; home of the CUDA cores, tensor cores, schedulers, register file, L1/shared memory. CPU analog: a *core*. Subdivided into **4 processing blocks**, each with its own warp scheduler. → [[#Kernels and SMs]], [[#Inside a Blackwell B200 SM]], [[#Inside a Volta SM]] ^sm
- **Warp scheduler** — per-processing-block unit that, every cycle, assigns a ready warp to a set of 32 CUDA cores (4 per SM since Volta). → [[#Kernels and SMs]] ^scheduler
- **TPC / GPC** — *Texture Processing Cluster* (pair of SMs) / *GPU Processing Cluster* (group of TPCs); a graphics-pipeline hierarchy irrelevant to GPGPU. → [[#Blackwell Chip Diagram]], [[#Turing Block Diagram]] ^tpc-gpc
- **SFU (Special Function Unit)** — per-SM unit for transcendental functions (sin, exp, rsqrt). → [[#Inside a Volta SM]] ^sfu
- **LSU (Load/Store Unit)** — per-processing-block units that move data between the memory hierarchy and registers (V100/B200: 32 per SM; Turing: 16). → [[#Inside a Blackwell B200 SM]] ^lsu
- **Tensor core** — matrix-multiply-accumulate unit computing $D = A\times B + C$ on small tiles per clock in reduced precision; introduced with Volta (gen 1), FP4 on Blackwell (gen 5). → [[#Tensor Cores, Fifth Generation]], [[#Tensor Cores (1st gen)]] ^tensor-core
- **TMA (Tensor Memory Accelerator)** — Hopper+ engine that performs bulk *asynchronous* tile loads for tensor cores, independent of the CUDA cores. → [[#Inside a Blackwell B200 SM]] ^tma
- **TMEM (Tensor Memory)** — Blackwell's dedicated 64 KB-per-processing-block (256 KB/SM) storage for tensor-core accumulators. → [[#Inside a Blackwell B200 SM]] ^tmem
- **RT core / ROP / PolyMorph / Raster engine** — graphics-only units (ray tracing, render output, geometry, rasterization) found on Turing-class cards; explain *why* the SM hierarchy exists. → [[#Turing Block Diagram]] ^graphics-units
- **Compute capability (CC)** — version number encoding an SM's feature set; major = architecture (7 Volta, 7.5 Turing, 8 Ampere, 9 Hopper, 10 Blackwell), minor = revision. → [[#Compute Capability]] ^cc
- **FMA (Fused Multiply-Add)** — $a\cdot b + c$ in one instruction = **2 flop**; the factor of 2 in every peak-rate formula. → [[#Context and Peak Rates]] ^fma
- **Precision formats** — **FP64** (double, 64-bit), **FP32** (single), **TF32** (19-bit tensor-core format, Ampere+), **FP16 / BF16** (half), **FP8**, **FP4** (Blackwell), **INT8 / INT4**. Tensor-core throughput roughly doubles with each halving of precision. → [[#Performance: GPU vs. CPU]], [[#Tensor Cores, Fifth Generation]] ^precision
- **Latency hiding** — the GPU's main defense against high memory latency: launch far more warps than cores so a ready warp always exists while others stall on data. → [[#Comparison to CPU Memory]] ^latency-hiding
- **CUDA streams** — independent queues of transfers/kernels; used to overlap host↔device copies with computation. → [[#Comparison to CPU Memory]], [[#B200 Memory & NVLink 5.0]] ^streams

### Memory hierarchy
- **Register file** — fastest storage, feeds the CUDA cores directly; 65,536 × 4-byte registers = **256 KB per SM**, organized in **32 banks** (one per warp lane). → [[#Memory Types]] ^registers
- **Register spilling** — when live data exceed the register file, the excess spills to L1, then L2, then *local memory*, with a steeper penalty at each level. → [[#Memory Types]] ^spilling
- **L1 cache** — per-SM cache for recently used global data and spilled registers; physically the same SRAM as shared memory (**unified data cache**, 96–256 KB depending on generation). → [[#Memory Types]] ^l1
- **Shared memory** — the application-declared, block-visible partition of the L1 SRAM; the mechanism for threads in a block to communicate. Split vs L1 is configurable. → [[#Memory Types]] ^shared-memory
- **Constant cache** — per-SM cache (64 KB) for read-only constants; ideal for **broadcasting** one value to a whole warp. → [[#Memory Types]] ^constant-cache
- **L2 cache** — the single on-chip cache shared by all SMs (6 MB V100 → ~126 MB B200); *all* traffic — SMs, memory controllers, host link — passes through it. "A proxy for device memory." → [[#Memory Levels (B200 numbers)]] ^l2
- **Global memory** — the device's main DRAM (HBM or GDDR), equivalent to CPU RAM; largest and slowest on-device level. → [[#Memory Types]] ^global-memory
- **Local memory** — per-SM-private region of global memory used for spilled registers; cached write-back in L2. → [[#Memory Types]] ^local-memory
- **Texture / constant memory** — read-only regions of global memory cached in L1 / the constant caches respectively. → [[#Memory Types]] ^texture-constant
- **Cache line / sector** — GPU cache lines are **128 B = 32 lanes × 4 B** (one warp's operands), split into **four 32 B sectors** that can be fetched independently. CPUs move full 64 B lines. → [[#Memory Levels (B200 numbers)]], [[#Appendix: Finer Memory Slices (per CUDA core vs per vector lane)]] ^cache-line
- **Write-through / write-back** — GPU L1 is write-through for global data (coherence handled entirely in L2, no L1-coherence fabric needed) and write-back for SM-private local memory. → [[#Memory Types]] ^coherence
- **SRAM** — *Static RAM*: fast, small, transistor-based on-chip memory used for registers, L1/shared memory and L2. In DL performance discussions "SRAM" usually means the per-SM shared memory that FlashAttention tiles into. → [[#Memory Levels (B200 numbers)]], [[#Flash Attention]] ^sram
- **DRAM** — *Dynamic RAM*: dense, capacitor-based off-chip memory technology behind HBM, GDDR, and CPU DDR; high capacity, high latency. → [[#Memory Types]] ^dram
- **VRAM** — colloquial term for a GPU's total on-board memory (its global memory), whatever the technology (GDDR6, HBM3e). "80 GB of VRAM" = 80 GB of global memory. → [[#Memory Types]] ^vram
- **HBM (High Bandwidth Memory)** — 3D-stacked DRAM on the GPU package connected via a very wide bus (4096-bit on V100/B200); HBM2 (V100, 900 GB/s) → HBM2e (A100) → HBM3 (H100) → HBM3e (B200, 8 TB/s). → [[#B200 Memory & NVLink 5.0]], [[#V100 Memory & NVLink 2.0]] ^hbm
- **GDDR** — *Graphics DDR* SDRAM: conventional discrete memory chips on graphics cards (GDDR6 on RTX 5000, 448 GB/s); cheaper, lower bandwidth than HBM. → [[#RTX 5000 Memory & PCIe 3.0]] ^gddr
- **Memory controller** — on-chip interface to DRAM; V100 and B200 each have **8 × 512-bit** controllers (4096 bits total). → [[#Blackwell Chip Diagram]] ^memory-controller
- **ECC (Error-Correcting Code) memory** — memory with parity bits to detect/correct bit flips; standard on HPC CPUs and on data-center GPUs (V100, RTX 5000, B200). → [[#B200 Memory & NVLink 5.0]] ^ecc
- **Unified Memory** — single virtual address space across host and device (`cudaMallocManaged()`); CUDA migrates pages on demand. → [[#B200 Memory & NVLink 5.0]] ^unified-memory
- **Host memory** — the CPU's RAM; effectively the slowest level of the GPU's hierarchy because every byte must cross PCIe or NVLink. → [[#Comparison to CPU Memory]] ^host-memory

### Interconnects
- **PCIe (PCI Express)** — general-purpose host↔device bus; bandwidth scales with lanes and generation (Gen3 x16 = 32 GB/s bidirectional; Gen5 x16 = 128 GB/s). The classic host↔GPU bottleneck. → [[#RTX 5000 Memory & PCIe 3.0]], [[#B200 Memory & NVLink 5.0]] ^pcie
- **NVLink** — NVIDIA's high-bandwidth GPU↔GPU (and, on POWER9/Grace hosts, CPU↔GPU) interconnect: 2.0 = 300 GB/s (V100), 3.0 = 600 GB/s (A100), 4.0 = 900 GB/s (H100), 5.0 = 1.8 TB/s (B200). → [[#V100 Memory & NVLink 2.0]], [[#B200 Memory & NVLink 5.0]] ^nvlink
- **NVLink-C2C** — chip-to-chip NVLink providing a coherent 900 GB/s CPU↔GPU link inside Grace Hopper / Grace Blackwell superchips, replacing PCIe. → [[#B200 Memory & NVLink 5.0]] ^nvlink-c2c
- **NVSwitch** — switch fabric that ties many NVLinks together for **all-to-all** GPU communication within an NVLink domain (8-GPU HGX board, 72-GPU NVL72 rack). → [[#B200 Memory & NVLink 5.0]], [[#Multi-GPU Systems]] ^nvswitch
- **Die-to-die link** — Blackwell's 10 TB/s connection joining two dies into one logical GPU. → [[#Blackwell Chip Diagram]] ^d2d
- **InfiniBand** — low-latency, high-throughput network for **scale-out** across nodes; NVIDIA Quantum adds in-network computing (**SHARP** reductions). → [[#Multi-GPU Systems]] ^infiniband
- **RoCE / Spectrum-X** — RDMA over Converged Ethernet; NVIDIA's Ethernet-based alternative fabric for AI clusters. → [[#Multi-GPU Systems]] ^roce
- **GPUDirect RDMA / Storage** — direct DMA between GPU memory and a NIC or storage device, bypassing CPU bounce buffers. → [[#Multi-GPU Systems]] ^gpudirect

### Systems & software
- **Grace / GB200 / NVL4 / NVL72** — Grace = NVIDIA's 72-core Arm CPU; GB200 Superchip = 1 Grace + 2 B200 via NVLink-C2C; NVL4 = 2 superchips per server (Horizon); NVL72 = 72-GPU rack with an NVSwitch fabric. → [[#Context and Peak Rates]] ^grace
- **HGX** — NVIDIA's **baseboard/reference design** (8 GPUs + NVSwitch) sold to OEMs who add CPUs, storage, networking, chassis. → [[#Multi-GPU Systems]] ^hgx
- **DGX** — NVIDIA's **turnkey server** built around an HGX board, pre-configured with CPUs, NVMe, networking, OS and software stack. → [[#Multi-GPU Systems]] ^dgx
- **Fabric Manager** — system software managing NVSwitch-based HGX/DGX systems. → [[#Multi-GPU Systems]] ^fabric-manager
- **NCCL** — NVIDIA Collective Communications Library; implements **all-reduce**, **all-gather**, **reduce-scatter**, broadcast over NVLink/PCIe/InfiniBand; the layer under PyTorch DDP/FSDP, Megatron, NeMo. → [[#Multi-GPU Systems]] ^nccl
- **Rank** — one participating process/device in an NCCL collective; all ranks must call the collective with matching shapes/dtypes. → [[#Multi-GPU Systems]] ^rank
- **NVLS / NVLink SHARP, Multimem, GIN** — advanced device-initiated communication: NVLink-SHARP multicast/reduction in NVSwitch, multimem addressing, GPU-Initiated Networking. → [[#Multi-GPU Systems]] ^nvls
- **Flash Attention** — IO-aware exact attention that tiles $Q,K,V$ into SRAM and never materializes the $T\times T$ matrix in HBM. → [[#Flash Attention]] ^flash-attention
- **Top500** — twice-yearly ranking of the fastest supercomputers (LINPACK); NVIDIA accelerators in 87 % of entries as of Nov 2025. → [[#Context and Peak Rates]] ^top500

## GPU Characteristics
---
### Design: GPU vs. CPU
GPUs were built to **render graphics** — shading, texturing, and rasterizing thousands of independent polygons — while CPUs were built to **control the logical flow** of arbitrary programs. The roles dictate the silicon budget:

| | CPU | GPU |
| --- | --- | --- |
| Transistor budget | Sophisticated **instruction processing** (branch prediction, out-of-order, deep pipelines), large **caches** | Many **arithmetic units** (ALUs/FPUs), high **aggregate memory bandwidth** |
| Clock speed | Faster | Slower |
| Control per compute | One control unit per core | One control unit per *row* of many compute units |
| Individual ALU/FPU | Fewer, individually more capable | Many, individually simpler |

The canonical figure (CUDA C++ Programming Guide, v11.2) draws green compute boxes, gold control, purple [[#^l1|L1]], blue higher-level cache, orange [[#^dram|DRAM]]. Its one-gold-box-per-row-of-green-boxes layout encodes the key point: **GPUs are designed for workloads that can be parallelized to a significant degree**.

### Performance: GPU vs. CPU
CPUs target *multitasking and fast serial processing*; GPUs target *high computational throughput* via massive parallelism. The historical Gflop/s chart (CUDA C Programming Guide v9.1, 2009–2016) shows GPUs pulling far ahead in both FP32 and FP64 — which raises the question the rest of the roadmap answers: *can this floating-point engine be used for something other than graphics?*

> **[[#^precision|Precision]] terms.** *Single precision* = 32-bit floats (FP32); *double precision* = 64-bit (FP64). FP32 is often processed **2×** faster than FP64 (and on consumer/graphics parts, far more — see RTX 5000 below).

### Heterogeneous Applications (GPGPU)
Almost any application dominated by **huge amounts of floating-point operations with simple data-access patterns** gains from GPUs — [[#^gpgpu|GPGPU]]. [[#^cuda|CUDA]] was the enabling API (matrix manipulation, FFTs, decryption, …); directive-based **OpenACC** and **OpenMP offloading** came later. Fields cited: deep learning, CFD, structural mechanics, seismic processing, bioinformatics, materials science, molecular dynamics, quantum chemistry, computational physics.

GPUs are always **hosted** by CPU systems, so the natural programming model is [[#^heterogeneous|heterogeneous]]: offload the massively parallel, numerically intensive parts to one or more GPUs; leave the less-parallel control logic on the CPU. Two device classes serve this role — standard graphics cards (**Quadro**, now RTX) and HPC accelerators (**Tesla**, now data-center A/H/B-series) with extra FP64, special memory ([[#^hbm|HBM]], [[#^ecc|ECC]]), and [[#^nvlink|NVLink]].

Raw FP32 peaks quoted for TACC hardware:

| Device | Peak FP32 |
| --- | --- |
| Intel Xeon Platinum 8280 "Cascade Lake" (28 cores) | 4.3 Tflop/s |
| NVIDIA Quadro RTX 5000 | 11.2 Tflop/s |
| NVIDIA Tesla V100 | 15.7 Tflop/s |

…*assuming the application can make full use of the hardware's parallelism*.

### Threads and Cores Redefined
The graphics pipeline is fast because (i) a scene is many **independent objects** (e.g. a mesh of triangles) and (ii) **the same sequence of steps** is applied to each — so all objects can be processed in parallel. CPUs have also become parallel machines, and the vocabulary overlaps *confusingly*:

> A single [[#^cuda-core|CUDA core]] is much more like a single **vector lane** in a CPU's vector unit than like a CPU core. CUDA cores work in teams of **32** to execute one instruction on many data ([[#^simd|SIMD]]-style); CPU vector units do the same over 8 or 16 elements (AVX-512: 16 floats).

| GPU term | Definition | CPU equivalent |
| --- | --- | --- |
| [[#^thread|**thread**]] | Stream of instructions + data assigned to one CUDA core; a Single Instruction applies to Multiple Threads ([[#^simt|SIMT]]) | — |
| [[#^cuda-core|**CUDA core**]] | Processes one data item after another, executing its share of a SIMT instruction stream | **vector lane** |
| [[#^warp|**warp**]] | Group of **32 threads** executing the same instruction stream together on different data | **vector** |
| [[#^kernel|**kernel**]] | Function that runs on the device; subdivided into [[#^block|**thread blocks**]] | thread(s) |
| [[#^sm|**SM**]] (streaming multiprocessor) | Unit that executes a thread block; several SMs cooperate on one kernel | **core** |

### SIMT and Warps
**[[#^simt|SIMT]] vs [[#^simd|SIMD]].** In pure SIMD one instruction acts identically on all data. SIMT loosens this: individual threads can be **activated/deactivated**, so an instruction is applied only on the *active* threads while inactive threads' data are left unchanged. Consequently SIMT supports **branching — but inefficiently** ([[#^divergence|branch divergence]]): for `if (cond) {A; B} else {X; Y}`, threads with `cond==true` run the if-clause while the others idle, then roles swap. Results are correct; utilization is not.

- **Pre-Volta**: the whole if-clause (A, B) must complete, then the whole else-clause (X, Y), then all threads reconverge before Z.
- **Volta and later** (*independent thread scheduling*): finer interleaving, and synchronization of shared data is permitted at intermediate points (e.g. after A and X).
- CPUs have a form of SIMT too: **masked** x86-64 vector instructions, where a mask operand "hides" selected lanes — vectorized branching, to a limited extent.
- In CUDA, SIMT parallelism is *not* written as vectorized loops: **every GPU thread executes the kernel code as written**, which justifies the "thread" nomenclature. (OpenMP/OpenACC directive code ends up looking like vectorized CPU code instead.)

**[[#^warp|Warps]].** At runtime each [[#^block|thread block]] is divided into **warps of 32 threads with consecutive indices**, and a warp is processed together by a set of 32 CUDA cores — exactly as a vectorized CPU loop is chunked into fixed-size vectors for the lanes. The number 32 is simply how NVIDIA groups CUDA cores in hardware; chunking a block into warps simplifies the SM's scheduling. (The word comes from weaving — the set of vertical threads the shuttle passes through; Lindholm et al. 2008: *"the first parallel-thread technology"*.)

**Programmer implication.** Warps are a hardware detail, but **branch divergence within a warp** serializes execution — so avoid data-dependent branching within warp-sized sets of loop iterations (e.g. keep `threadIdx`-dependent conditions warp-aligned).

### Kernels and SMs
**[[#^kernel|Kernel]] (software).** A function executed in parallel on the GPU, marked `__global__` in [[#^cuda|CUDA]] (callable from host *or* device, executes on device). It runs **N times in parallel by N threads**, each with a unique ID used to compute addresses and make control decisions. The **execution configuration** `fun<<<M, N>>>(x, y, z)` launches **M [[#^block|blocks]] of N threads**.

**[[#^sm|SM]] (hardware).** A kernel call is executed by one or more SMs — the hardware homes of the CUDA cores, always arranged in sets of 32 to execute full warps. SM count depends on architecture and model (V100: 80; RTX 5000: 48; B200: 160).

Mapping rules:
- At most **M blocks** can be spread across SMs; **a block is never split across SMs**; if M > #SMs, several blocks share an SM. Independent blocks thus run in parallel on different SMs.
- Each SM splits its block's N threads into warps; **every cycle, each SM's [[#^scheduler|schedulers]] (4 per SM since Volta) assign full warps to available sets of 32 CUDA cores**. A leftover partial warp still occupies a full set of 32 cores.
- Each SM owns private memory usable only by its own CUDA cores: [[#^registers|registers]], [[#^l1|L1 cache]], [[#^constant-cache|constant caches]], [[#^shared-memory|shared memory]] (§2).

### Compute Capability
A version number ([[#^cc|CC]]) summarizing the SM's technical properties / supported feature set; discoverable at runtime. **Major number = core architecture**, minor = incremental revision:

| CC | Architecture | Example |
| --- | --- | --- |
| 7.0 | Volta | Tesla V100 |
| 7.5 | Turing (incremental update of Volta) | Quadro RTX 5000, T4 |
| 8.0 / 8.6 | Ampere | A100 / RTX 30-series |
| 9.0 | Hopper | H100 |
| 10.0 | Blackwell | B200 |

Full list of generations in [[#7. NVIDIA Architecture Timeline]]. NVIDIA's CUDA-enabled GPU list gives each card's CC, which indexes the full feature/spec tables in the CUDA Programming Guide (e.g. instruction throughput per SM per cycle, Table 3).

## GPU Memory
---
![[gpu_memory_b200.png|600]]
The GPU, like a CPU, relies on a **hierarchy from RAM through cache levels** to keep its processing engines fed; an SM, like a CPU core, can only compute on data that reach its [[#^registers|registers]].

### Memory Levels (B200 numbers)
Per SM: **256 KB [[#^registers|register file]]** (65,536 × 4-byte registers), **256 KB unified [[#^l1|L1 data cache]] / [[#^shared-memory|shared memory]]**, **64 KB [[#^constant-cache|constant caches]]**. Shared by all SMs: **~126 MB [[#^l2|L2]]** (the tutorial's diagram rounds to 128 MB) and **192 GB HBM3e [[#^global-memory|global memory]]**.

Key structural facts:
- **Registers**: an SM has *far more* registers than a CPU core, and larger total register capacity — even though each GPU register holds one 4-byte float while an AVX-512 register holds 16.
- **Caches**: the CPU wins on total cache. Every Intel core has private L1 + L2 plus a share of L3; the GPU has nothing beyond a single shared **L2**.
- **[[#^cache-line|Cache line]] = 128 bytes = 32 × 4 bytes** — exactly one warp's worth of 4-byte operands moving together between cache, registers and CUDA cores.
- **Everything flows through L2.** [[#^memory-controller|Memory controllers]], the host link ([[#^pcie|PCIe]]/[[#^nvlink|NVLink]]) and all SMs connect to the L2; incoming host data pass through L2 into global memory. *"The L2 cache is a proxy for device memory"* — evictions write the oldest lines back to DRAM.
- All of these on-chip levels are [[#^sram|SRAM]]; global memory is [[#^dram|DRAM]] ([[#^hbm|HBM]] or [[#^gddr|GDDR]]).

### Memory Types
**On-chip, per SM (closest to the CUDA cores):**
- **[[#^registers|Register File]]** — feeds the CUDA cores directly. Organized as a big matrix of 4-byte elements with **32 banks** (columns) matching the 32 threads of a warp: a warp operates on a full row, each thread on its own column.
- **[[#^l1|L1 Cache]]** — normal recently-used-data cache for RAM traffic, *and* the overflow region for [[#^spilling|register spilling]] when active data exceed the register file. Banked like the register file.
- **[[#^shared-memory|Shared Memory]]** — physically the same SRAM as L1, but **explicitly declared** by the application and **accessible by any thread in the thread block** → the mechanism for intra-block communication. The application sets the L1/shared split.
- **[[#^constant-cache|Constant Caches]]** — for variables declared read-only constants in global memory; best use is **broadcasting one constant to all threads of a warp**.

**Off-SM, shared by all SMs (larger, more distant):**
- **[[#^l2|L2 Cache]]** — one on-chip cache shared by all SMs; also sits in the path of data moving on/off the device via PCIe or NVLink.
- **[[#^global-memory|Global Memory]]** — bulk device RAM (what a spec sheet calls [[#^vram|VRAM]]). **[[#^hbm|HBM]]** (high-bandwidth, 3D-stacked) on B200 / V100; **[[#^gddr|GDDR6]]** on RTX 5000.
- **[[#^texture-constant|Texture & Constant Memory]]** — read-only regions of global memory; when fetched to an SM, readable by any thread in the kernel (an expanded form of shared memory). Texture memory is cached in L1, constant memory in the constant caches.
- **[[#^local-memory|Local Memory]]** — per-SM mapped regions of global memory: when register spilling overflows L1, the excess goes to L2, then local memory. Each extra level traversed on reload steepens the penalty.

**[[#^coherence|Coherence policy]].** L2 holds both globally visible data (global/texture/constant) and SM-private data (local memory):
- *Global data*: **write-through** — an SM write immediately evicts the L1 line to L2. Consistency is maintained entirely in L2, so **the GPU needs no complex L1-coherence interconnect** like a CPU's.
- *Local memory*: **write-back** — L1→L2 eviction may be delayed since the L2 region is not shared.

### Comparison to CPU Memory
GPUs are **register-heavy**, CPUs **cache-heavy with wider data paths**. Per computational unit (bold = larger):

| Memory type | Tesla V100 (per SM) | Intel Cascade Lake / Skylake SP (per core) |
| --- | --- | --- |
| Register file | **256 KB** | 10.5 KB |
| L1 cache | **128 KB** (max) | 32 KB |
| Constant caches | **64 KB** | — |
| L2 cache | 0.075 MB (share of 6 MB) | **1 MB** (private) |
| L3 cache | — | **1.375 MB** (share) |
| Global / RAM | 0.4 GB (share of 32 GB) | **> 3.4 GB** |

The GPU concentrates capacity at the *closest* levels; the CPU wins further out. **Speed reverses the picture** — the CPU has far better cache *latency*, the GPU wins on *bandwidth* at the distant levels:

| Memory type | V100 latency | V100 bandwidth (per SM) | Xeon latency | Xeon bandwidth (per core) |
| --- | --- | --- | --- | --- |
| L1 | 28 cycles | 128 B/cycle | 4 cycles | 192 B/cycle |
| Private L2 | — | — | 14 cycles | 64 B/cycle |
| Shared L2 / L3 | 193 cycles | 17.6 B/cycle | 50–70 cycles | 14.3 B/cycle |
| Global / RAM | 220–350 cycles | 7.4 B/cycle | 190–220 cycles | 1.9–2.5 B/cycle |

*(V100 data from Jia et al., [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826); device-wide bandwidths converted to B/cycle/SM by dividing by 1.53 GHz × 80 SMs, e.g. 2155 GB/s L2. Registers have no latency/bandwidth figure — one speaks of instruction throughput, ≥ 1/cycle if operand pipelines stay full.)*

**Consequences for GPU programming:**
- Individual global-memory reads/writes have **high latency** despite huge bandwidth. Shared memory/L1 help to an extent, but the primary remedy is [[#^latency-hiding|latency hiding]]: launch *very* many threads so that whenever one warp stalls on data, another warp can issue its next instruction.
- The [[#^host-memory|host's memory]] is effectively one more (slowest) level. Host↔device transfer — especially over [[#^pcie|PCIe]] — can dominate; once data are on the device, keep them there.
- Send data in **big batches**, and use [[#^streams|CUDA streams]] so computation on some batches overlaps transfer of others.

### Appendix: Finer Memory Slices (per CUDA core vs per vector lane)
- **Registers**: V100 SM register file = $65536/(2\times32) = 1024$ floats per CUDA core. Cascade Lake/Skylake physical register file = 168 vector registers × 64 B = 10.5 KB → $(16\times168)/(2\times16) = 84$ floats per lane — **an order of magnitude less**. (Factors of 2: a CPU core issues 2 × 16-float vectors/cycle; a GPU SM issues 2 × 32-float warps/cycle.)
- **Caches**: V100 SM: 128 KB L1 + 64 KB constant + 6/80 = 0.075 MB L2 → 0.26 MB per SM → **0.0041 MB per CUDA core**. Xeon core: 32 KB L1 + 1 MB L2 + 1.375 MB L3 share → 2.4 MB → **0.75 MB per vector lane** — two orders of magnitude more.
- **[[#^cache-line|Cache lines]]**: GPU 128-byte lines are **four 32-byte sectors**; on a miss only the needed sectors are filled, so 4 sectors from 4 different lines cost the same as 1 full line (the 32 B sector persists even though V100's HBM2 has a 64 B interface). Intel moves full 64-byte lines all the way from RAM to registers.

## Horizon GPUs: Blackwell B200
---
### Context and Peak Rates
NVIDIA met scientific computing's **FP64** need in 2007 with the Tesla line; as of Nov 2025 NVIDIA accelerators are in **87 % of [[#^top500|Top500]] systems** and account for 55 % of total performance. TACC's **Horizon** contributes 4,000 GPUs in [[#^grace|Grace Blackwell]] nodes: a GB200 Superchip = **2 × B200 GPUs + 1 × 72-core Grace CPU**; two superchips per **GB200 NVL4** server, configured as 2 nodes each with 2 B200s (185 GiB HBM3 per GPU) + 1 Grace (120 GiB LPDDR5X). Horizon uses the **B200** rather than the Blackwell Ultra **B300** because the B300 trades FP64 for reduced-precision AI throughput.

Peak FP64/FP32 ([[#^fma|FMA]] = 2 flop/core/cycle):
$$
10{,}240\ \text{FP64 cores}\times 2\ \tfrac{\text{flop}}{\text{core}\cdot\text{cycle}}\times 1.965\ \text{GHz}\approx 40\ \text{Tflop/s}
$$
$$
20{,}480\ \text{FP32 cores}\times 2\times 1.965\ \text{GHz}\approx 80\ \text{Tflop/s}
$$
vs. the companion Grace CPU at max turbo, 4 × 128-bit VPUs per core:
$$
72\ \text{cores}\times 4\ \tfrac{\text{VPU}}{\text{core}}\times 2\ \tfrac{\text{FP64 lanes}}{\text{VPU}}\times 2\times 3.44\ \text{GHz}\approx 4.0\ \text{Tflop/s}
$$
A 10× FP64 advantage — but only if the application exposes **many thousands of [[#^simt|SIMT]] threads** of fine-grained parallelism.

### Blackwell Chip Diagram
- The B200 is unique in being **two dies joined by a 10 TB/s [[#^d2d|die-to-die link]]**, functioning as one chip.
- Hierarchy: SMs paired into [[#^tpc-gpc|**TPCs**]] (Texture Processing Clusters), grouped into **GPCs** (GPU Processing Clusters). **8 GPCs × 10 TPCs × 2 = 160 SMs.** (The tutorial's die shot is actually the B300; the B200 layout is nearly identical.) The TPC/GPC hierarchy matters only for graphics.
- Edges: [[#^nvlink-c2c|**NVLink-C2C**]] to the Grace host (**900 GB/s**); [[#^nvlink|**NVLink 5.0**]] bridges to other GPUs (**1.8 TB/s**); **eight 512-bit [[#^memory-controller|memory controllers]] (4096 bits)** to **192 GB [[#^hbm|HBM3e]]**; central **L2**.

### Inside a Blackwell B200 SM
Per [[#^sm|SM]]: **128 FP32 + 128 INT32 + 64 FP64 [[#^cuda-core|CUDA cores]]**, **4 fifth-gen [[#^tensor-core|Tensor Cores]]**, 4 [[#^sfu|Special Function Units]], 4 Texture Units. Divided into **4 processing blocks** so that per clock either **4 warps of FP32/INT32** or **2 warps of FP64** (or a mix) can be issued.

Data path: **32 [[#^lsu|Load/Store units]]** per block. Loads target the **64 KB register file** or the **64 KB [[#^tmem|Tensor Memory (TMEM)]]** of each block → **256 KB registers + 256 KB TMEM per SM**. Stores go to the **256 KB L1 data cache** (shrinkable to carve out shared memory), then L2, then global memory. The [[#^tma|**Tensor Memory Accelerator (TMA)**]] performs bulk *asynchronous* loads into the tensor cores independently of the CUDA cores.

### Tensor Cores, Fifth Generation
Matrix multiplication is the heart of neural-network training and inference; since the V100 (1st-gen), [[#^tensor-core|tensor cores]] accelerate it whenever reduced [[#^precision|precision]] (FP16/FP8/FP4) suffices (CUDA ≥ 9.1 uses them automatically for convolutions).

- **Basic op** (1st gen): $D = A\times B + C$ on **4×4** matrices, $A,B$ FP16, $C,D$ FP16 or FP32. The 64 element-wise products "light up" at once and are summed down columns — conceptually a multi-stage FMA pipeline with successive $C$ matrices accumulating partial sums.
- **Generational pattern**: each generation *doubles matrix size and halves precision* — Ada (4th gen): (4×8)×(8×4) in FP8; Blackwell (5th gen): **(4×16)×(16×4) in FP4** — *and* per-SM dense FP16 rate doubled in each HPC family (Volta → Ampere → Hopper → Blackwell = gen 1 → 3 → 4 → 5; gen 2/Turing equaled gen 1 at FP16 but added INT8/INT4).
- **B200 count**: **640 tensor cores = 4/SM × 160 SMs**. One 5th-gen tensor core = **1024 FP16 FMA/clock**; one SM = 4096 FMA = **8192 flop/clock**.
$$
8192\ \tfrac{\text{flop}}{\text{SM}\cdot\text{cycle}}\ (\text{FP16})\times 160\ \text{SMs}\times 4\ \tfrac{\text{FP4}}{\text{FP16}}\times 1.965\ \text{GHz}\approx 10\ \text{Pflop/s (FP4, dense)}
$$
Compare V100: 640 tensor cores (8/SM × 80), 64 FMA/clock each → 1024 flop/SM/clock; 8× less per SM than B200 at FP16.

### B200 Memory & NVLink 5.0
- **[[#^hbm|HBM3e]]** stacked on-package: 4 chips per stack, 2 stacks per die pair (8 stacks total) → **192 GB** at **8 TB/s** peak. [[#^ecc|ECC]] supported (as on V100).
- **[[#^unified-memory|Unified Memory]]**: `cudaMallocManaged()` replaces `malloc`/`new`; CUDA migrates pages to whichever processor touches them. Especially attractive with the coherent **NVLink-C2C** between Grace and each B200.
- **Interconnects**: [[#^pcie|PCIe]] 5.0 x16 = 4 GB/s/lane/direction → **128 GB/s bidirectional**. Grace Blackwell instead uses [[#^nvlink-c2c|**NVLink-C2C**]] at 900 GB/s (CPU↔GPU, coherent) and [[#^nvlink|**NVLink 5.0**]] at 1.8 TB/s (GPU↔GPU, also across nodes via [[#^nvswitch|**NVSwitch**]]). NVLink 5.0 rivals memory bandwidth itself: B200 HBM3e 8 TB/s; Grace LPDDR5X 1 TB/s.
- Overlap several [[#^streams|transfer streams]] with kernels so some compute while others wait for data.

> **Bottom line.** High flop/s only matters if host↔device and memory data rates keep up; NVLink-C2C + NVLink 5.0 + HBM3e are what let data-heavy applications use the B200's full speed.

## Frontera GPUs: Quadro RTX 5000
---
### Device and Peak Rate
Frontera's remaining GPU subsystem: **360 Quadro RTX 5000** cards (Turing **TU104**, [[#^cc|CC]] 7.5) in Dell/Intel Broadwell servers, 4 per node. A Turing [[#^sm|SM]] has the *same* 64 FP32 + 64 INT32 [[#^cuda-core|CUDA cores]] as Volta, but **no FP64 CUDA cores**, and the chip has only **48 SMs** (vs 80):
$$
3072\ \text{FP32 cores}\times 2\times 1.815\ \text{GHz}\approx 11.2\ \text{Tflop/s}
$$
FP64 [[#^fma|FMAs]] are possible but at a token **2 results/cycle/SM** (CUDA Programming Guide, Table 3 footnote for CC 7.5). Not "inferior" — a graphics card judged on flop/s/W or cost/(flop/s) can still be attractive for HPC.

### Turing Block Diagram
48 SMs → 24 [[#^tpc-gpc|TPCs → 6 GPCs]], superficially like GV100, but each level hosts a [[#^graphics-units|graphics-pipeline unit]] absent in GV100:

| Block | Special unit | Count in TU104 |
| --- | --- | --- |
| SM | Ray Tracing (RT) Core | 48 |
| TPC | PolyMorph Engine | 24 |
| GPC | Raster Engine | 6 |

Plus **8 ROPs** (Render Output units) per memory controller × 8 = 64 ROPs. Irrelevant for number-crunching, but they explain *why* the SM hierarchy exists.

### Inside a Turing SM
64 FP32 + 64 INT32 CUDA cores, **8 Turing [[#^tensor-core|Tensor Cores]]** (2nd gen; FP32/FP16/**INT8/INT4** multi-precision for inference), 16 [[#^sfu|SFUs]], 4 Texture Units, **1 RT core**; 4 processing blocks issuing up to 2 warps of FP32/INT32 per cycle; **16 [[#^lsu|Load/Store units]]** (vs 32 in Volta); no FP64.

### Frontera's GPU Subsystems
Original Frontera had two: **Longhorn** (400+ V100s in 100+ IBM POWER9 AC922 servers, 4 GPUs each; decommissioned 2022) for double precision, and the RTX 5000 subsystem — together 11 Pflop/s FP32 for AI/ML and molecular dynamics. Nearly everything is water-cooled; the Quadros are **fully immersed in liquid coolant** (GRC).

### RTX 5000 Memory & PCIe 3.0
**16 GB [[#^gddr|GDDR6]]** (no 3D-stacked memory), 8 [[#^memory-controller|memory controllers]] → **448 GB/s**; **4 MB [[#^l2|L2]]**; **96 KB** unified [[#^l1|L1]]/[[#^shared-memory|shared]] per SM. Half the V100's memory size/bandwidth, but with 60 % of the SMs — so *per SM* the two are roughly on a par except for the 25 % smaller L1/shared.

| | Quadro RTX 5000 | Tesla V100 |
| --- | --- | --- |
| Main memory | GDDR6 SDRAM | HBM2 |
| ECC | yes | yes |
| Size | 16 GB | 32 GB |
| Bandwidth | 448 GB/s | 900 GB/s |
| L2 | 4 MB | 6 MB |
| Shared + L1 per SM | 96 KB | 128 KB |
| SMs | 48 | 80 |
| FP32/INT32/FP64 cores per SM | 64/64/0 | 64/64/32 |
| Tensor cores per SM | 8 | 8 |
| Boost clock | 1.815 GHz | 1.53 GHz |

**Interconnect.** Each Quadro has its own [[#^pcie|**PCIe 3.0 x16**]] to the host (**32 GB/s** bidirectional) and a **single [[#^nvlink|NVLink 2.0]] link (50 GB/s)** used to pair Quadros within a node. The V100 has **6 NVLink 2.0 links × 50 GB/s = 300 GB/s**; in the AC922 it is split 150 GB/s to host + 150 GB/s to a neighbor GPU.

| | Quadro RTX 5000 | Tesla V100 (IBM host) |
| --- | --- | --- |
| Host↔device | 1 × PCIe 3 x16 — 32 GB/s | 3 × NVLink 2.0 — 150 GB/s |
| Peer↔peer (per pair) | 1 × NVLink 2.0 — 50 GB/s | 3 × NVLink 2.0 — 150 GB/s |

Measured on Frontera, the Quadro's NVLink peer link often runs *slower* in bandwidth than its PCIe host link (though with better latency). Mitigations: **maximize on-device memory residency** and **overlap transfers with compute** ([[#^streams|streams]]) — stream data in while other tasks run (hence "*streaming* multiprocessor").

## GPU Example: Tesla V100
---
### Device and Peak Rates
Volta **GV100**, [[#^cc|CC]] 7.0; by 2021 Volta devices were in 20 % of [[#^top500|Top500]] systems. **2560 FP64 [[#^cuda-core|CUDA cores]]**, each doing an [[#^fma|FMA]] per cycle:
$$
2560\times 2\times 1.53\ \text{GHz}\approx 7.8\ \text{Tflop/s (FP64)},\qquad 5120\times 2\times 1.53\ \text{GHz}\approx 15.7\ \text{Tflop/s (FP32)}
$$
vs. Frontera's Xeon Platinum 8280 at all-core turbo, 2 AVX-512 VPUs/core:
$$
56\ \text{VPUs}\times 8\ \tfrac{\text{FP64 lanes}}{\text{VPU}}\times 2\times 2.4\ \text{GHz}\approx 2.15\ \text{Tflop/s}
$$

### Volta Block Diagram
Full GV100: **6 [[#^tpc-gpc|GPCs × 7 TPCs]] × 2 = 84 SMs**; the V100 product enables **80** (so two GPCs have 6 TPCs). Edges: [[#^pcie|**PCIe 3.0**]] to host; [[#^nvlink|**NVLink 2.0**]] bridges to other GPUs and to POWER9 hosts; **8 × 512-bit [[#^memory-controller|memory controllers]]** to [[#^hbm|**HBM2**]]; central **6 MB [[#^l2|L2]]**.

### Inside a Volta SM
**64 FP32 + 64 INT32 + 32 FP64** CUDA cores, **8 [[#^tensor-core|Tensor Cores]]** (1st gen; 2 per processing block), 16 [[#^sfu|SFUs]], 4 Texture Units; **4 processing blocks** → up to **2 warps of FP32/INT32 (or 1 each) or 1 warp of FP64** per cycle; **32 [[#^lsu|Load/Store units]]**.

### Tensor Cores (1st gen)
**640 total (8 × 80 SMs)**; each does **64 FP16 FMA/clock** → 512 FMA = **1024 flop/SM/clock**. Operation $D = A\times B + C$ on 4×4 with FP16 inputs and FP16/FP32 accumulate (details in [[#Tensor Cores, Fifth Generation]]).

### V100 Memory & NVLink 2.0
- **[[#^hbm|HBM2]]**: 8 chips per stack × 4 stacks → **32 GB** at **900 GB/s**; [[#^ecc|ECC]].
- [[#^unified-memory|Unified Memory]] via `cudaMallocManaged()`; multiple [[#^streams|streams]] to overlap transfer and compute.
- [[#^pcie|**PCIe 3.0 x16**]]: 1 GB/s/lane/direction → **32 GB/s** bidirectional. In Longhorn's IBM AC922, **POWER9** CPUs spoke [[#^nvlink|NVLink 2.0]] directly: each of the two CPUs connects to two V100s at **150 GB/s** each, and those two V100s share a 150 GB/s link (a lone GPU could get the full 300 GB/s).


## NVIDIA Architecture Timeline *(beyond the tutorial)*
---
Tutorial architectures in bold. CC = [[#^cc|compute capability]]; tensor-core generation per NVIDIA's numbering; "flagship" = the data-center part (consumer example in parentheses).

| Architecture | Year | CC | Process | Flagship (consumer) | Tensor cores | Memory / bandwidth | GPU↔GPU link | Key introductions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tesla (G80/GT200) | 2006–08 | 1.0–1.3 | 90/65 nm | Tesla C870, C1060 (GeForce 8800) | — | GDDR3 | — | Unified shaders, **CUDA** (2007); FP64 arrives with GT200 |
| Fermi | 2010 | 2.0–2.1 | 40 nm | Tesla C2050/M2090 (GTX 480) | — | GDDR5, ~180 GB/s | — | First true L1/L2 cache hierarchy, **ECC**, 512 cores, unified address space |
| Kepler | 2012 | 3.0–3.7 | 28 nm | Tesla K20/K40/K80 (GTX 680/780) | — | GDDR5, 288 GB/s (K40) | — | Dynamic parallelism, Hyper-Q, GPU Boost, 192-core SMX |
| Maxwell | 2014 | 5.0–5.3 | 28 nm | Tesla M40/M60 (GTX 980) | — | GDDR5 | — | Power efficiency, 128-core SMM; weak FP64 (graphics-focused) |
| Pascal | 2016 | 6.0–6.2 | 16 nm | Tesla P100 (GTX 1080) | — | **HBM2** 16 GB, 732 GB/s | **NVLink 1.0**, 160 GB/s | First HBM2 + NVLink, FP16 arithmetic, Unified Memory page faulting, 64-core SM |
| **Volta** | 2017 | 7.0 | 12 nm | **Tesla V100** (Titan V) | **1st gen** (8/SM, FP16) | HBM2 32 GB, 900 GB/s | NVLink 2.0, 300 GB/s | Tensor cores, independent thread scheduling, unified L1/shared (128 KB) → [[#5. GPU Example: Tesla V100]] |
| **Turing** | 2018 | 7.5 | 12 nm | T4, **Quadro RTX 5000** (RTX 2080) | 2nd gen (INT8/INT4) | GDDR6, 448 GB/s (RTX 5000) | NVLink 2.0, 50–100 GB/s | RT cores, no FP64 units, 16 LSU/SM → [[#4. Frontera GPUs: Quadro RTX 5000]] |
| Ampere | 2020 | 8.0 / 8.6 | 7 nm (GA100) / 8 nm (GA10x) | A100 (RTX 3090) | 3rd gen (TF32, BF16, 2:4 sparsity) | HBM2e 40/80 GB, 1.6–2.0 TB/s | NVLink 3.0, 600 GB/s | TF32, structured sparsity, **MIG** partitioning, 192 KB L1/shared, 40 MB L2, async copy |
| Ada Lovelace | 2022 | 8.9 | TSMC 4N | L40 / L40S (RTX 4090) | 4th gen (FP8) | GDDR6/6X, ~1 TB/s | — (no NVLink) | Consumer/inference line; FP8, 3rd-gen RT cores, large L2 (72–96 MB) |
| Hopper | 2022 | 9.0 | TSMC 4N | H100, H200 (—) | 4th gen (FP8, Transformer Engine) | HBM3 80 GB, 3.35 TB/s (H200: HBM3e 141 GB, 4.8 TB/s) | NVLink 4.0, 900 GB/s | **TMA**, thread-block clusters + distributed shared memory, FP8 Transformer Engine, 50 MB L2; Grace Hopper GH200 (NVLink-C2C) |
| **Blackwell** | 2024 | 10.0 (consumer 12.0) | TSMC 4NP | **B200**, GB200 (RTX 5090) | **5th gen** (FP4/FP6, microscaling) | HBM3e 192 GB, 8 TB/s | NVLink 5.0, 1.8 TB/s | Dual-die 10 TB/s link, **TMEM**, 2nd-gen Transformer Engine, decompression engine, NVL72 rack → [[#3. Horizon GPUs: Blackwell B200]] |
| Blackwell Ultra | 2025 | 10.3 | TSMC 4NP | B300, GB300 | 5th gen (FP4 dense 15 PF) | HBM3e 288 GB, 8 TB/s | NVLink 5.0, 1.8 TB/s | 1.5× FP4 vs B200 at the cost of FP64 throughput (why Horizon chose B200) |
| Rubin (announced) | 2026 | — | TSMC 3 nm | R100 / Vera Rubin NVL144 | 6th gen | **HBM4** | NVLink 6.0 | Announced at GTC 2025; Vera Arm CPU replaces Grace; figures preliminary |

Cross-generation invariants worth remembering: **warp = 32**, **cache line = 128 B = 4 × 32 B sectors**, **256 KB registers/SM** (since Fermi), **4 processing blocks/SM** (since Volta), **everything goes through L2**.

### Data-center flagship specs
Tutorial devices in bold; P100/A100/H100 from NVIDIA whitepapers.

| | P100 (Pascal) | **V100** (Volta) | **RTX 5000** (Turing) | A100 (Ampere) | H100 SXM (Hopper) | **B200** (Blackwell) |
| --- | --- | --- | --- | --- | --- | --- |
| Compute capability | 6.0 | 7.0 | 7.5 | 8.0 | 9.0 | 10.0 |
| SMs | 56 | 80 | 48 | 108 | 132 | 160 |
| FP32 / FP64 cores per SM | 64 / 32 | 64 / 32 | 64 / 0 | 64 / 32 | 128 / 64 | 128 / 64 |
| Tensor cores per SM (gen) | — | 8 (1st) | 8 (2nd) | 4 (3rd) | 4 (4th) | 4 (5th) |
| Registers per SM | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB | 256 KB (+256 KB TMEM) |
| L1 / shared per SM | 24 KB L1 + 64 KB shared | 128 KB | 96 KB | 192 KB | 256 KB | 256 KB |
| L2 | 4 MB | 6 MB | 4 MB | 40 MB | 50 MB | ~126 MB |
| Global memory | 16 GB HBM2 | 32 GB HBM2 | 16 GB GDDR6 | 40/80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| Memory bandwidth | 0.73 TB/s | 0.9 TB/s | 0.45 TB/s | 1.6–2.0 TB/s | 3.35 TB/s | 8 TB/s |
| GPU↔GPU link | NVLink 1.0, 160 GB/s | NVLink 2.0, 300 GB/s | NVLink 2.0, 50 GB/s | NVLink 3.0, 600 GB/s | NVLink 4.0, 900 GB/s | NVLink 5.0, 1.8 TB/s |
| Peak FP64 / FP32 | 5.3 / 10.6 T | 7.8 / 15.7 T | ~0.35 / 11.2 T | 9.7 / 19.5 T | 34 / 67 T | 40 / 80 T |


## Flash Attention
---
Standard [[Attention]] computes `softmax(Q @ K.T / scale) @ V`, materializing the full $T\times T$ attention matrix in [[#^hbm|HBM]] — $O(T^2)$ memory and, more importantly, $O(T^2)$ HBM reads/writes. [[#^flash-attention|**Flash Attention**]] ([Dao et al. 2022](https://arxiv.org/abs/2205.14135)) computes the *exact same result* with a **tiling algorithm** that:
- processes $Q, K, V$ in blocks sized to fit in [[#^sram|SRAM]] / [[#^shared-memory|shared memory]] (the per-SM L1 region),
- uses the **online softmax** trick to accumulate the normalizer incrementally so the full $T\times T$ matrix is never written to HBM,
- reduces memory from $O(T^2)$ to $O(T)$ and — the real win — is **IO-aware**: it minimizes traffic across the slow HBM↔SRAM boundary, which is the bottleneck for memory-bound attention (recompute in the backward pass rather than store).
This is the textbook instance of the tutorial's lesson that global-memory *latency and bandwidth*, not flop/s, bound many kernels; FlashAttention-3 additionally exploits Hopper's [[#^tma|TMA]] and asynchronous tensor-core execution.

### Multi-GPU Systems
[[#^dgx|**DGX**]] and [[#^hgx|**HGX**]] are NVIDIA's platforms for multi-GPU servers:
- **HGX** — the **baseboard / reference design**: 8 GPUs + [[#^nvswitch|NVSwitch]] interconnects, sold to OEMs (Dell, HPE, Supermicro, …) who add CPUs, storage, networking, chassis, cooling. *The engine sold to car manufacturers.*
- **DGX** — NVIDIA's **turnkey server** built around the HGX board: CPUs, RAM, NVMe, [[#^infiniband|InfiniBand]]/Ethernet, OS, software stack; pre-configured and validated. *NVIDIA's own car with that engine.*
- **DGX GB200 / SuperPOD / cluster**: multiple DGX/HGX systems networked for scale-out training.

| System | GPUs | Notes |
| --- | --- | --- |
| HGX A100 / DGX A100 | 8× A100 | Baseboard / complete server |
| HGX H100 / DGX H100 | 8× H100 | Baseboard / complete server |
| HGX B200 | 8× B200 | Typical OEM training node |
| DGX GB200 / NVL72 | 72 GPUs | Rack-scale system with an NVLink network (NVSwitch) |
| GB200 NVL4 (Horizon) | 4 GPUs + 2 Grace | HPC building block described in [[#3. Horizon GPUs: Blackwell B200]] |

**Communication stack, inside-out:**
- *Intra-node*: [[#^nvlink|NVLink]] ↔ [[#^nvswitch|NVSwitch]] (all-to-all within the NVLink domain), managed by the [[#^fabric-manager|Fabric Manager]]. [[#^pcie|PCIe]] still carries GPU–NIC and GPU–storage traffic; keep GPU↔GPU traffic on NVLink.
- *Inter-node*: [[#^infiniband|InfiniBand]] (Quantum, with SHARP in-network reductions) or [[#^roce|Spectrum-X / RoCE Ethernet]]; [[#^gpudirect|GPUDirect RDMA]] lets the NIC read/write GPU memory directly (GPUDirect Storage: the same for NVMe).
- *Library layer*: [[#^nccl|NCCL]] implements the collectives (all-reduce for data parallelism, reduce-scatter / all-gather for FSDP/ZeRO, all-to-all for MoE) across [[#^rank|ranks]]; mismatched shapes/dtypes across ranks → hangs or corruption. Advanced modes ([[#^nvls|NVLS, Multimem, GIN]]) offload reductions into the NVSwitch fabric or let GPUs initiate network traffic — useful later, not day one.
