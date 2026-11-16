### [[dl_moc]]  
# **Robot Kinematics**

**Tags**:: #math #rl
**Links**:: [[Linear Algebra]] · [[SVD]] · [[VLA]] · [[Behavior Cloning]] · [[Flow Matching]] · [[ODEs, SDEs, and Probability Paths]] · [[World Models]] · [[Camera Calibration]] · [[Hand-Eye Calibration]]


**Robot kinematics** is the geometry of motion for articulated bodies, with forces and masses deliberately ignored: it maps <span style="color:rgb(255, 255, 0)">joint variables</span> to the pose of every <span style="color:rgb(255, 255, 0)">link</span> (**forward kinematics**, FK), <span style="color:rgb(255, 255, 0)">joint velocities</span> to <span style="color:rgb(255, 255, 0)">link velocities</span> (**velocity kinematics**, via the **Jacobian**), and desired poses back to joint variables (**inverse kinematics**, IK). It is the interface between what a learned policy emits (end-effector targets, joint targets) and what the hardware executes, and it fixes the geometry — $SO(3)$, $SE(3)$ — in which poses, actions, and proprioception live. 

Classical foundations: the Denavit–Hartenberg link parameters ([Denavit & Hartenberg, 1955](https://doi.org/10.1115/1.4011045)) and the screw-theoretic product-of-exponentials formulation of Brockett (1984). The note follows *Modern Robotics* ([Lynch & Park, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 3–7; free PDF) and Craig's *Introduction to Robotics* for the DH material.

**Notation.** Joint vector $\theta \in \mathbb{R}^n$ (the $q_t$ of the [[VLA]] problem setup — same quantity, *Modern Robotics* symbol); pose $T = (R, p) \in SE(3)$; twist $\mathcal{V} = (\omega, v) \in \mathbb{R}^6$ (angular first); $[\omega]$ the skew-symmetric matrix with $[\omega]x = \omega \times x$; $[\mathrm{Ad}_T]$ the adjoint map. Frames: $\{s\}$ space (base), $\{b\}$ body (end-effector), $\{d\}$ desired. Clash with the learning notes: here $\theta$ is a **joint angle**, not network parameters, and $T$ is a rigid transform, not a time horizon or temperature.

## Configuration Space and Degrees of Freedom
---
**Configuration space** $\mathcal{C}$ is the set of all configurations of the robot; its dimension is the number of **degrees of freedom** (dof). For a mechanism with $N$ links (ground included), $J$ joints, $m = 3$ (planar) or $6$ (spatial), and joint $i$ having $f_i$ dof, **Grübler's formula** gives

$$
\text{dof} = m\,(N - 1 - J) + \sum_{i=1}^{J} f_i .
$$
Check: a planar four-bar ($N{=}4$, $J{=}4$ revolute, $f_i{=}1$) has $3(4-1-4)+4 = 1$ dof. For an open chain of $n$ one-dof joints this is just $n$. Topologically, $\mathcal{C}$ is generally *not* $\mathbb{R}^n$: a revolute joint is a circle $S^1$, so a 6R arm lives on a torus $T^6$ (before joint limits) — the same "data lives on a manifold" point as [[Flow Matching]] on tori and $SE(3)$.

![[kinematic_chains.jpg|400]]

*A 6R serial manipulator: rigid links connect the base (the ground link) to the end-effector through six revolute joints. Counting the ground, $N = 7$ and $J = 6$, so Grübler gives $6(7-1-6) + 6 = 6$ dof.*

The **task space** is the space of end-effector poses, $SE(3)$ (dim 6) for a rigid end-effector. A robot is **redundant** when $n > \dim(\text{task})$ (e.g. a 7-dof arm for a 6-dof pose), which buys a continuous family of IK solutions (a self-motion manifold).

## Rigid-Body Pose: $SO(3)$ and $SE(3)$
---
**Rotations and poses as matrix groups.**

$$
SO(3) = \{R \in \mathbb{R}^{3\times3} : R^\top R = I,\ \det R = 1\}, \qquad
SE(3) = \left\{T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}\right\}, \quad T_1 T_2 = \begin{bmatrix} R_1R_2 & R_1p_2 + p_1 \\ 0 & 1 \end{bmatrix},\ T^{-1} = \begin{bmatrix} R^\top & -R^\top p \\ 0 & 1\end{bmatrix}.
$$

They are 3- and 6-dimensional Lie groups: $R^\top R = I$ imposes 6 constraints on 9 entries. The homogeneous form makes composition a matrix product, and a change of frame $T_{ab}T_{bc} = T_{ac}$ reads like index contraction. Note $SO(3)$ is **non-commutative** — rotation order matters.

**Parameterizing rotations.** No global, singularity-free, minimal parameterization of $SO(3)$ exists (the group is not homeomorphic to $\mathbb{R}^3$), so each choice pays somewhere:

| Representation                              | Dim | Catch                                                                | Notes                                                                                                                                    |                                                                                     |
| ------------------------------------------- | --- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Rotation matrix                             | 9   | redundant; drifts off $SO(3)$ under noise                            | project back via [[SVD]]: $R = U\,\mathrm{diag}(1,1,\det(UV^\top))V^\top$                                                                |                                                                                     |
| Euler angles (e.g. ZYX)                     | 3   | **gimbal lock** at pitch $\pm 90^\circ$ loses a dof; order-dependent | human-readable; poor for learning                                                                                                        |                                                                                     |
| Axis-angle / exp. coords $\hat\omega\theta$ | 3   | non-unique at $\theta = 0$ and $\theta = \pi$                        | the natural Lie-algebra coordinates                                                                                                      |                                                                                     |
| Unit quaternion                             | 4   | **double cover**: $q$ and $-q$ are the same rotation                 | Separate actor $\pi_\theta$ and critic $V_\phi$ networks![[6dof.jpeg]]                               $R$ by Gram–Schmidt + cross product | continuous map from $SO(3)$ ([Zhou et al., 2019](https://arxiv.org/abs/1812.07035)) |

The last row matters for learning: Zhou et al. prove that any representation of 3D rotations in $\le 4$ real dimensions is *discontinuous* (so Euler angles and quaternions both have regions a network cannot regress smoothly), while 5D and 6D representations are continuous. This is the case for outputting 6D rotations from learned pose regressors.

**Roll, pitch, yaw.** The most common Euler-angle convention (aerospace and ROS REP-103: $x$ forward, $y$ left, $z$ up) names the rotations about the three body axes — **roll** $\phi$ about $x$, **pitch** $\vartheta$ about $y$, **yaw** $\psi$ about $z$ — and composes them as

$$
R = R_z(\psi)\,R_y(\vartheta)\,R_x(\phi),
$$

read either as *intrinsic* z–y–x (yaw, then pitch about the new $y$, then roll about the new $x$) or as *extrinsic* rotations about fixed axes in the order $x$, $y$, $z$. Other fields order the axes differently, so always state the convention. Pitch is written $\vartheta$ here to avoid the clash with the joint angle $\theta$.

![[roll_pitch_yaw.svg|500]]

*Roll, pitch, yaw in a right-handed frame; each arc is the positive (right-hand-rule) rotation about its axis.*

**Gimbal lock** is visible in this composition: at $\vartheta = \pm 90^\circ$ the roll and yaw axes coincide, and $R$ depends only on $\psi - \phi$ (at $+90^\circ$) or $\psi + \phi$ (at $-90^\circ$). One degree of freedom is lost, and nearby orientations require large, discontinuous changes in $(\phi, \psi)$.

## Screws and the Exponential Map
---
**Exponential coordinates.** Every rotation is a rotation by $\theta$ about a unit axis $\hat\omega$; the matrix exponential gives **Rodrigues' formula**

$$
e^{[\hat\omega]\theta} = I + \sin\theta\,[\hat\omega] + (1-\cos\theta)\,[\hat\omega]^2 \in SO(3).
$$

**Screw axis.** *Chasles' theorem*: every rigid-body displacement is a rotation about an axis combined with a translation along it — a screw motion. (Hand–eye calibration rests on this: the two motions in $AX = XB$ are one screw seen in two frames, with equal angle and slide — see [[Hand-Eye Calibration]].) A **screw axis** is $\mathcal{S} = (\omega, v)$ with either $\|\omega\| = 1$ and $v = -\omega \times q + h\omega$ (axis through point $q$, **pitch** $h$) or $\omega = 0,\ \|v\| = 1$ (pure translation). Its matrix form $[\mathcal{S}] = \begin{bmatrix}[\omega] & v \\ 0 & 0\end{bmatrix} \in se(3)$ exponentiates to a pose (for $\|\omega\|=1$):

$$
e^{[\mathcal{S}]\theta} = \begin{bmatrix} e^{[\omega]\theta} & \big(I\theta + (1-\cos\theta)[\omega] + (\theta - \sin\theta)[\omega]^2\big)v \\ 0 & 1 \end{bmatrix}.
$$

**Exponential map as a flow.** $T(\theta)=e^{[\mathcal{S}]\theta}T(0)$ solves the linear ODE $\frac{dT}{d\theta} = [\mathcal{S}]\,T$: a constant-twist vector field on $SE(3)$ integrated for "time" $\theta$ — exactly the flow-of-a-vector-field picture of [[ODEs, SDEs, and Probability Paths]], on a group instead of $\mathbb{R}^d$. The inverse, $\log: SE(3) \to se(3)$, turns a pose error into a 6-vector twist; the IK iteration below uses it.

**Adjoint.** A twist expressed in one frame converts to another via
$$
[\mathrm{Ad}_T] = \begin{bmatrix} R & 0 \\ [p]R & R \end{bmatrix}, \qquad \mathcal{V}_a = [\mathrm{Ad}_{T_{ab}}]\,\mathcal{V}_b .
$$

## Forward Kinematics
---
**Forward kinematics** computes the end-effector pose $T(\theta) \in SE(3)$ from joint values. Two standard formulations:

**Denavit–Hartenberg (DH).** Attach a frame to each link; four parameters per joint $(\alpha_{i-1}, a_{i-1}, d_i, \theta_i)$ determine the transform between consecutive frames. In Craig's (modified) convention

$$
T_{i-1,i} = \mathrm{Rot}_x(\alpha_{i-1})\,\mathrm{Trans}_x(a_{i-1})\,\mathrm{Rot}_z(\theta_i)\,\mathrm{Trans}_z(d_i), \qquad T(\theta) = T_{01}(\theta_1)\,T_{12}(\theta_2)\cdots T_{n-1,n}(\theta_n).
$$

DH is minimal (4 numbers per joint) but needs a frame-placement convention (standard vs. modified DH differ in index placement), and the common normal is not unique when adjacent axes are parallel. URDF/MJCF files sidestep this by giving each joint an arbitrary origin transform plus axis.

**Product of exponentials (PoE).** Fix the zero pose $M = T(0)$ and express each joint as a screw axis $\mathcal{S}_i$ in the space frame; then

$$
T(\theta) = e^{[\mathcal{S}_1]\theta_1}\,e^{[\mathcal{S}_2]\theta_2}\cdots e^{[\mathcal{S}_n]\theta_n}\,M \quad \text{(space form)}, \qquad T(\theta) = M\,e^{[\mathcal{B}_1]\theta_1}\cdots e^{[\mathcal{B}_n]\theta_n} \quad \text{(body form)},
$$

with $\mathcal{B}_i = [\mathrm{Ad}_{M^{-1}}]\,\mathcal{S}_i$. PoE needs only one fixed frame plus the joint screws — no per-link frames — and each factor has a clear geometric meaning ("rotate about this line").

**Worked example (planar 2R, link lengths $L_1, L_2$).**

$$
x = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2), \qquad y = L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2).
$$

## Velocity Kinematics and the Jacobian
---
**Jacobian.** Differentiating FK gives the linear map from joint rates to end-effector twist, $\mathcal{V}_s = J_s(\theta)\,\dot\theta$. In PoE form the columns are the joint screws, each transported by the joints *before* it:

$$
J_{s,i}(\theta) = \big[\mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}\big]\,\mathcal{S}_i, \qquad J_b(\theta) = [\mathrm{Ad}_{T_{bs}}]\,J_s(\theta),
$$

so $J_s$ gives the twist in the base frame and $J_b$ in the end-effector frame. (This "geometric" Jacobian is *not* the derivative of an Euler-angle parameterization; the two differ by the rotation-rate-to-Euler-rate map, which is itself singular at gimbal lock.)

**Statics duality.** Under the principle of virtual work, the joint torques that balance an end-effector wrench $F$ are

$$
\tau = J_b^\top(\theta)\,F_b .
$$

Velocity maps through $J$, force through $J^\top$ — the same matrix, transposed.

**Singularities and manipulability.** A configuration is **singular** when $J$ loses rank: the end-effector cannot move in some task direction no matter the joint rates, and joint rates blow up for motion near that direction. For the planar 2R arm,

$$
J = \begin{bmatrix} -L_1\sin\theta_1 - L_2\sin(\theta_1{+}\theta_2) & -L_2\sin(\theta_1{+}\theta_2) \\ L_1\cos\theta_1 + L_2\cos(\theta_1{+}\theta_2) & L_2\cos(\theta_1{+}\theta_2)\end{bmatrix}, \qquad \det J = L_1 L_2 \sin\theta_2,
$$

singular at $\theta_2 \in \{0, \pi\}$ — arm fully stretched or folded, losing the radial direction. The **manipulability ellipsoid** $\{J\dot\theta : \|\dot\theta\| = 1\}$ has semi-axes equal to the singular values of $J$ ([[SVD]]); Yoshikawa's measure $w = \sqrt{\det(JJ^\top)} = \prod_i \sigma_i$ and the condition number $\sigma_{\max}/\sigma_{\min}$ quantify closeness to singularity. (Mixing angular and linear rows has unit-dependent meaning; treat the two blocks separately.)

## Inverse Kinematics
---
**Inverse kinematics** solves $T(\theta) = T_d$ for $\theta$. Unlike FK it is generally **non-unique, sometimes non-existent, and nonlinear**: no solution outside the **workspace**; multiple solutions inside (a general 6R arm has up to 16); a continuum for redundant arms.

**Closed form.** For the planar 2R arm, the law of cosines gives two solutions (elbow up / elbow down):

$$
\cos\theta_2 = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2L_1L_2}, \qquad \theta_1 = \mathrm{atan2}(y,x) - \mathrm{atan2}\big(L_2\sin\theta_2,\ L_1 + L_2\cos\theta_2\big).
$$

For spatial 6R arms, **Pieper's criterion** (1968): if three consecutive joint axes intersect at a point (a spherical wrist) or are parallel, the position and orientation problems decouple and IK has an analytic solution — the reason most industrial arms have spherical wrists. Closed-form solvers are fast and enumerate all branches, but are robot-specific.

**Numerical (Newton–Raphson).** Linearize FK about the current guess $\theta^k$. With the body-frame pose error expressed as a twist $\mathcal{V}_b = \log\!\big(T_{sb}^{-1}(\theta^k)\,T_{sd}\big)^\vee$, iterate

$$
\theta^{k+1} = \theta^k + J_b^{\dagger}(\theta^k)\,\mathcal{V}_b, \qquad J^\dagger = J^\top (JJ^\top)^{-1}\ \text{(full row rank)},
$$

until $\|\mathcal{V}_b\|$ is small. It converges quadratically near a solution, finds only one branch (the one nearest the initial guess), and diverges near singularities where $J^\dagger$ explodes.

**Damped least squares (DLS).** Regularize: $\min_{\delta\theta} \|J\delta\theta - \mathcal{V}\|^2 + \lambda^2\|\delta\theta\|^2$, giving

$$
\delta\theta = J^\top (JJ^\top + \lambda^2 I)^{-1}\mathcal{V}.
$$

In SVD terms each mode's gain $1/\sigma_i$ becomes $\sigma_i/(\sigma_i^2 + \lambda^2)$, so it is bounded by $1/(2\lambda)$ instead of blowing up as $\sigma_i \to 0$. This is the Levenberg–Marquardt damping idea from nonlinear least squares applied to the pose residual — trading tracking accuracy near singularities for bounded joint velocity. The same Levenberg–Marquardt solver refines camera intrinsics and poses against reprojection error in [[Camera Calibration]].

**Redundancy resolution.** For $n > 6$, the general solution of $J\dot\theta = \mathcal{V}$ is

$$
\dot\theta = J^\dagger\mathcal{V} + (I - J^\dagger J)\,\dot\theta_0,
$$

where $(I - J^\dagger J)$ projects onto the **null space** of $J$ (self-motions that do not move the end-effector). Choosing $\dot\theta_0 = -\nabla\phi(\theta)$ optimizes a secondary objective $\phi$ (joint limits, manipulability, obstacle clearance, posture) without disturbing the primary task. Humanoid whole-body control and motion retargeting stack such priorities.

```python
import numpy as np
from scipy.linalg import expm, logm

def hat3(w):  return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
def hat6(S):  return np.block([[hat3(S[:3]), S[3:, None]], [np.zeros((1, 4))]])   # (ω,v) -> [S]
def vee6(X):  return np.r_[X[2, 1], X[0, 2], X[1, 0], X[:3, 3]]                   # [S] -> (ω,v)
def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [hat3(p) @ R, R]])

def fk(M, Slist, th):                    # T(θ) = e^{[S1]θ1} … e^{[Sn]θn} M
    T = np.eye(4)
    for S, t in zip(Slist, th): T = T @ expm(hat6(S) * t)
    return T @ M

def jacobian_space(Slist, th):           # column i = Ad_{e^{[S1]θ1}…e^{[S_{i-1}]θ_{i-1}}} S_i
    J, T = np.zeros((6, len(th))), np.eye(4)
    for i, (S, t) in enumerate(zip(Slist, th)):
        J[:, i] = adjoint(T) @ S
        T = T @ expm(hat6(S) * t)
    return J

def ik(M, Slist, T_d, th, lam=1e-2, iters=100, tol=1e-8):
    for _ in range(iters):
        T = fk(M, Slist, th)
        V = vee6(logm(np.linalg.inv(T) @ T_d).real)                  # body-frame error twist
        if np.linalg.norm(V) < tol: break
        Jb = adjoint(np.linalg.inv(T)) @ jacobian_space(Slist, th)
        th = th + Jb.T @ np.linalg.solve(Jb @ Jb.T + lam**2 * np.eye(6), V)   # DLS step
    return th
```

## Kinematics in Learning-Based Robotics
---
**Action space is a kinematic choice.** A policy's output is either **joint-space** (targets $\theta^\star$ or $\Delta\theta$) or **task-space** (end-effector pose deltas $\Delta T$, plus gripper), and the choice trades off:

| | Joint-space actions | End-effector (task-space) actions |
|---|---|---|
| Embodiment transfer | tied to one arm's joint layout and dof | arm-agnostic given a controller that tracks $T_d$ |
| Needs IK / Jacobian | no | yes — inherits singularities, branch choice, joint-limit handling |
| Redundancy | policy resolves it implicitly | IK/null-space controller resolves it, hidden from the policy |
| Typical use | ACT-style bimanual, humanoids | many cross-embodiment generalist policies on Open X-Embodiment-style data |

This is the kinematic content of the "action representation" axis in [[VLA]] and of the action-chunk definition ($A_t$ = joint targets or end-effector deltas) there. Proprioception $\theta_t$ (and its FK image $T(\theta_t)$) is the other half of the observation $o_t$ in [[Behavior Cloning]].

**Rotation outputs.** Regressing end-effector orientation with a quaternion has the $q \equiv -q$ ambiguity and (per Zhou et al.) cannot be continuous in $\le 4$ dims; Euler angles wrap and gimbal-lock; hence 6D rotations (or small axis-angle *deltas*, which stay far from the $\theta=\pi$ ambiguity) are natural policy-output choices. Orientation losses should respect the group geometry (e.g. geodesic angle $\arccos\frac{\mathrm{tr}(R_1^\top R_2) - 1}{2}$), not Euclidean distance on a chart.

**Generative models on $SE(3)$.** Grasp poses and end-effector trajectories live on $SE(3)$, not $\mathbb{R}^d$; Riemannian Flow Matching ([Chen & Lipman, 2024](https://arxiv.org/abs/2302.03660)) handles such manifolds with velocities in tangent spaces, and the exponential map above is what keeps samples on the group. See [[Flow Matching]].

**Differentiable FK.** $T(\theta)$ is a composition of matrix exponentials, so gradients flow through it: pose-space or keypoint-space losses can supervise joint-space networks, and IK can be unrolled as a differentiable layer.

**What kinematics leaves out.** It assumes the commanded motion is achieved. Forces, inertia, friction, contact, and actuator limits belong to *dynamics* ($M(\theta)\ddot\theta + c(\theta,\dot\theta) + g(\theta) = \tau + J^\top F$); kinematics supplies the $J$ in that equation.

## Relationship to Other Notes
---
- [[VLA]] — its action representation (joint targets vs. end-effector deltas, gripper commands) and proprioceptive state $q_t$ are exactly the joint-space / task-space distinction here.
- [[Behavior Cloning]] — demonstrations are kinematic trajectories; compounding error is amplified by IK failures near singularities, and the action space sets what "mimic the expert" means.
- [[SVD]] — the Jacobian's singular values give the manipulability ellipsoid; pseudoinverse and DLS are SVD filter functions; rotation re-orthogonalization is an SVD projection.
- [[Linear Algebra]] — rank, null space, pseudoinverse and least squares underlie velocity kinematics, redundancy resolution, and IK.
- [[ODEs, SDEs, and Probability Paths]] — $e^{[\mathcal{S}]\theta}$ is the flow of a constant-twist linear ODE on $SE(3)$; the vector-field/flow vocabulary carries over to Lie groups.
- [[Flow Matching]] — Riemannian FM on $SE(3)$ and tori for pose and trajectory generation; the manifold viewpoint of $SO(3)$/$SE(3)$ is shared.
- [[World Models]] — action-conditioned predictors take kinematic actions as input; the choice of action space affects what must be learned vs. what a controller handles.
- [[Camera Calibration]] — camera extrinsics and poses are $SE(3)$ transforms in the same $T_{ab}$ algebra.
- [[Hand-Eye Calibration]] — solves $AX = XB$ with $A$ built from forward kinematics; the screw invariants of $A, B$ and Park–Martin's log-map solution use the $SE(3)$ machinery of this note, and only relative kinematic accuracy matters.

## References
---
- [Lynch & Park — *Modern Robotics: Mechanics, Planning, and Control*, Cambridge Univ. Press, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Ch. 3 Rigid-Body Motions, 4 Forward Kinematics, 5 Velocity Kinematics and Statics, 6 Inverse Kinematics, 7 Kinematics of Closed Chains
- [Denavit & Hartenberg, 1955 — A Kinematic Notation for Lower-Pair Mechanisms Based on Matrices, *J. Applied Mechanics* 22(2)](https://doi.org/10.1115/1.4011045)
- Craig — *Introduction to Robotics: Mechanics and Control* (DH convention, Pieper's solution)
- Brockett, 1984 — Robotic manipulators and the product of exponentials formula
- [Zhou et al., 2019 — On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)
- [Chen & Lipman, 2024 — Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660)
