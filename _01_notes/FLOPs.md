### [[dl_moc]]  
# **Floating-Point Operations**

**Tags**::  
**Links**::

## Definition
---
FLOPs: floating-point operations (measure of computation done)
FLOP/s (FLOPS ): floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware

Examples:
+ Training GPT-3 took 3.14e23 FLOPs.
+ Training GPT-4 is speculated to take 2e25 FLOPs.
+ A100 has a peak performance of 312 teraFLOP/s
+ H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without



+ How long does it take to train a 70B parameter model on 15T tokens on 1024 H100?
+ Answer: total flops = $6\times 70e9 \times 15e12$
	+ Forward Pass: 2 * (number of data points) * (number of parameters) FLOPs
	+ Backward Pass: 4 * (number of data points) * (number of parameters) FLOPs
	+ Total: 6 * (number of data points) * (number of parameters) FLOPs

+ Matrix multiplication dominate: 2 (m n p) FLOPs
+ FLOP/s depends on hardware (H100 >> A100) and datatype (bfloat16 >> float32)
## Model FLOPs Utilization (MFU)

Definition: the actual number of FLOP/s divided by the promised number of FLOP/s
$$\textrm{mfu} = \textrm{actual\_flop\_per\_sec} / \textrm{promised\_flop\_per\_sec}$$
Usually MFU>=0.5 is quite good.

