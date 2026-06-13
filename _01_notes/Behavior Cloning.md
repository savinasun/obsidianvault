### [[dl_moc]]  
# **Behavior Cloning**

**Tags**::  
**Links**::

## Learning from Demonstration
---
At each time step $t$, our goal is to learn a policy $\pi$ that uses the given observation $o_t$ to predict an optimal action distribution for the task: $a_t \sim \pi(\cdot|o_t)$, where actions $a_t$ are commands sent to the robot controller and the observations $o_t$ consist of information of current image and robot joint information $o_t=[i_t, j_t]$.

The policy $\pi$ must be learned given a set of expert demonstrations $\mathcal{D}=\{\tau_1, \dots, \tau_n\}$ where each demonstration $\tau_i=[(o_0, a_0), \dots, (o_T, a_T)]$ is a trajectory with optimal observation-action tuples (i.e. collected by a proficient agent).

