**DGX** and **HGX** are NVIDIA's platforms for multi-GPU servers:

**HGX** — the **baseboard/reference design**

- A board containing 8 GPUs + NVSwitch interconnects
- Sold to OEMs (Dell, HPE, Supermicro, etc.) who build their own servers around it
- Just the GPU subsystem — the server vendor adds CPUs, storage, networking, chassis, cooling

**DGX** — NVIDIA's **turnkey server**

- A complete server built by NVIDIA around the HGX baseboard
- Includes CPUs, RAM, NVMe storage, networking (InfiniBand/Ethernet), OS, software stack
- Comes pre-configured and validated — plug in and run
- Essentially NVIDIA's "reference server"

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