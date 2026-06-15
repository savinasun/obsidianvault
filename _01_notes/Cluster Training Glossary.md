### [[dl_moc]] · [[sequence_llms_moc]]  
# **Cluster Training**

**Tags**:: #llm
**Links**:: [[LLMs]] · [[Optimization Algorithms]]

https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm
## Glossary
---
**DGX** and **HGX** are NVIDIA's platforms for multi-GPU servers:
- **HGX**: NVIDIA’s scale-up server platform for multi-GPU systems. Think of it as the GPU baseboard/platform level.
	**HGX** — the **baseboard/reference design**
	- A board containing 8 GPUs + NVSwitch interconnects
	- Sold to OEMs (Dell, HPE, Supermicro, etc.) who build their own servers around it
	- Just the GPU subsystem — the server vendor adds CPUs, storage, networking, chassis, cooling
- **DGX**: NVIDIA’s integrated AI server built on top of those technologies.
	**DGX** — NVIDIA's **turnkey server**
	- A complete server built by NVIDIA around the HGX baseboard
	- Includes CPUs, RAM, NVMe storage, networking (InfiniBand/Ethernet), OS, software stack
	- Comes pre-configured and validated — plug in and run
	- Essentially NVIDIA's "reference server"
- **DGX GB200 / SuperPOD / cluster**: multiple DGX/HGX systems networked together for scale-out training.
Think of it like:
- **HGX** = engine sold to car manufacturers
- **DGX** = NVIDIA's own car with that engine
**Product examples:**

|System|GPUs|Notes|
|---|---|---|
|HGX A100|8x A100|Baseboard, used by OEMs|
|DGX A100|8x A100|Complete NVIDIA server|
|HGX H100|8x H100|Baseboard|
|DGX H100|8x H100|Complete server|
|HGX B200|8x B200|Your node is likely this or similar|
|DGX GB200 / NVL72|72 GPUs|Rack-scale system with NVLink network|

- **NVLink**: the direct **GPU-to-GPU interconnect**.
- **NVSwitch**: the switch fabric that ties many NVLinks together to provide **all-to-all GPU communication** inside an NVLink domain.
- **Fabric Manager**: the system software that manages **NVSwitch-based HGX/DGX** systems.
- **PCIe**: the general-purpose host/device interconnect. It matters for GPU-NIC and GPU-storage paths, but for fast intra-node training you generally want communication to stay on **NVLink/NVSwitch** whenever possible.
- **InfiniBand**: the low-latency, high-throughput network typically used for **scale-out** training across nodes. NVIDIA’s Quantum InfiniBand platform emphasizes offload and in-network computing such as **SHARP**.
- **Spectrum-X / RoCE Ethernet**: the Ethernet-based alternative for AI clusters. NVIDIA positions Spectrum-X as an Ethernet fabric for large GPU deployments, with **RoCE** connectivity between GPU servers.
- **GPUDirect RDMA**: direct data exchange between **GPU memory and a NIC/peer device**, avoiding CPU bounce buffers. This is a core idea for efficient inter-node GPU communication. **GPUDirect Storage** does the analogous thing for storage-to-GPU paths.
- **NCCL**: NVIDIA’s collective communication library. This is the workhorse under most PyTorch/Megatron/NeMo distributed training runs. It provides collectives like **all-reduce**, **all-gather**, and **reduce-scatter** across multi-GPU and multi-node jobs.
- **Rank**: in NCCL terms, each participating device/process in a collective. NCCL’s docs emphasize that collectives must be called on each rank with matching shapes/types, or you get hangs or corruption.
- **NVLink SHARP / NVLS**, **Multimem**, **GIN**: advanced terms. NCCL now documents device-initiated communication modes including NVLink SHARP-backed multicast/reduction and GPU-initiated networking over the network fabric. These are useful later, not day one.
- **SRAM** (Static Random-Access Memory): a type of fast, on-chip memory in a GPU.


## GPU memory hierarchy (fast → slow)
---
https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm

| Level | Name                     | Size               | Speed     |
| ----- | ------------------------ | ------------------ | --------- |
| 1     | **Registers**            | ~256 KB per SM     | Fastest   |
| 2     | **SRAM / Shared Memory** | ~128-228 KB per SM | Very fast |
| 3     | **L2 Cache**             | ~few MB            | Fast      |
| 4     | **HBM (Global Memory)**  | 24-80 GB           | Slowest   |
### What Flash Attention does

Standard attention computes `softmax(Q @ K.T / scale) @ V`, which requires materializing the full `T x T` attention matrix in GPU memory — **O(T²)** memory.

**Flash Attention** computes the exact same result but uses a **tiling algorithm** that:
- Processes the attention matrix in small blocks that fit in **SRAM (fast on-chip memory)**
- Never materializes the full T×T matrix in **HBM (slow global memory)**
- Reduces memory from **O(T²) to O(T)**