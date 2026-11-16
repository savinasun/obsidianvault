### [[dl_moc]]  
# **Hand–Eye Calibration**

**Tags**:: #rl #math #vision
**Links**:: [[Camera Calibration]] · [[Robot Kinematics]] · [[SVD]] · [[Linear Algebra]] · [[VLA]] · [[Behavior Cloning]]


**Hand–eye calibration** estimates the fixed rigid transform between a robot and a camera. For a camera riding on the wrist (*eye-in-hand*), that is the camera's pose in the gripper frame; for a camera mounted in the workcell (*eye-to-hand*), it is the camera's pose in the base frame. It is what lets a robot (i) map camera measurements into its own workspace and (ii) move the camera precisely ([Horaud & Dornaika, 1995](https://doi.org/10.1177/027836499501400301)), so every camera-guided grasp, every point-cloud policy and every multi-camera robot dataset depends on it. Because a calibration target is hard to place at a precisely known pose in robot coordinates, the problem is posed on *relative* motions as the matrix equation $AX = XB$ ([Shiu & Ahmad, 1989](https://doi.org/10.1109/70.88014); [Tsai & Lenz, 1989](https://doi.org/10.1109/70.34770)). The classical solvers are Tsai–Lenz, Park–Martin ([1994](https://doi.org/10.1109/70.326576)), Horaud–Dornaika's quaternion method, Andreff et al.'s Kronecker formulation ([2001](https://doi.org/10.1177/02783640122067372)) and Daniilidis's dual quaternions ([1999](https://doi.org/10.1177/02783649922066213)).

The note follows two course sources. Georgia Tech **ME 6406 Machine Vision** (K.-M. Lee) assigns the Tsai–Lenz paper as its hand–eye reading ([course copy](http://kmlee.gatech.edu/me6406/handeye.pdf)); it supplies the station construction, the closed form and the accuracy observations below. The GdR Robotics Winter School **Robotics Principia** ([Chaumette, *Requirements for Sensor-based Control*, slides 9–14](https://project.inria.fr/roboticsprincipia/files/2019/01/Requirements_SBC.pdf)) supplies the eye-in-hand / eye-to-hand derivation and the rank argument. *Modern Robotics* ([Lynch & Park, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 3) provides the $SE(3)$ machinery, as in [[Robot Kinematics]]. Horaud & Dornaika ([arXiv:2311.12655](https://arxiv.org/abs/2311.12655)) and Andreff et al. ([arXiv:2311.11808](https://arxiv.org/abs/2311.11808)) are the comparative references, and the OpenCV `calib3d` docs and source fix the implementation conventions.

**Notation.** Frames and transforms as in [[Robot Kinematics]] and [[Camera Calibration]]; the last column translates Tsai–Lenz, OpenCV and Andreff et al.

| Symbol | Meaning | Source equivalents · clashes |
|---|---|---|
| $\{b\}, \{g\}, \{c\}, \{t\}$ | robot base, gripper (flange), camera, calibration target | Tsai–Lenz: RW, $G_i$, $C_i$, CW |
| $T_{ab}$ | pose of frame $b$ in frame $a$: $\mathbf x_a = T_{ab}\mathbf x_b$, $T_{ab}T_{bc} = T_{ac}$ | OpenCV: ${}^{a}T_{b}$, named `b2a` (e.g. `gripper2base` $= T_{bg}$) |
| $T_{bg}^{(i)}$ | gripper pose at station $i$, from forward kinematics | Tsai–Lenz: $H_{gi}$; OpenCV `R_gripper2base`, `t_gripper2base` |
| $T_{ct}^{(i)}$ | target pose in the camera at station $i$, from PnP | Tsai–Lenz: $H_{ci}$; OpenCV `R_target2cam`, `t_target2cam` |
| $X$ | unknown hand–eye transform: $T_{gc}$ (eye-in-hand) or $T_{bc}$ (eye-to-hand) | Tsai–Lenz: $H_{cg}$ ("from $C_i$ to $G_i$"); OpenCV `cam2gripper` |
| $A, B$ | relative gripper / camera motion between stations $i, j$ | Tsai–Lenz: $H_{gij}$, $H_{cij}$; Andreff et al. swap the roles ($A$ = camera motion) |
| $R_\bullet, t_\bullet$ | rotation / translation block of a transform, e.g. $R_A$, $t_X$ | Tsai–Lenz: $T$ for translation |
| $\theta, n$ | rotation angle and unit axis of a motion | not a joint angle (cf. [[Robot Kinematics]]) |
| $P = 2\sin(\theta/2)\,n$ | Tsai–Lenz's modified Rodrigues vector | Tsai–Lenz: $P_r$; OpenCV: `2*rot2quatMinimal` |
| $\alpha, \beta$ | exponential coordinates $\theta n$ of $R_A$, $R_B$ | Park–Martin; unrelated to CS231A's focal lengths in [[Camera Calibration]] |
| $d = n\cdot t$ | slide of a motion along its screw axis | — |
| $q$; $\check q = q + \epsilon q'$ | unit quaternion; unit dual quaternion | Horaud–Dornaika; Daniilidis |
| $\mathrm{vec}(\cdot)$, $\otimes$ | row-major vectorization; Kronecker product | Andreff et al.; OpenCV reshapes row-major |
| $Y$ | second unknown in $AX = YB$ (here $T_{bt}$) | OpenCV writes $AX = ZB$ with $A = T_{ct}$, $X = T_{tb}$, $Z = T_{cg}$, $B = T_{gb}$ |
| $N$ | number of stations (robot poses at which an image is taken) | Tsai–Lenz: $N$ |

## Problem Setup
---
<span style="color:rgb(255, 255, 0)"><b>Two configurations.</b></span> In **eye-in-hand**, the camera rides on the gripper and views a target fixed in the workcell; the unknown is $X = T_{gc}$. In **eye-to-hand**, a static camera views a target carried by the gripper; the unknown is $X = T_{bc}$. In both, the robot visits $N$ stations, and at each one we record two observables: the gripper pose $T_{bg}^{(i)}$ from forward kinematics ([[Robot Kinematics]]) and the target pose $T_{ct}^{(i)}$ from PnP with known intrinsics ([[Camera Calibration]]).

![[hand_eye_calibration.svg|650]]

*Eye-in-hand. The loop base → gripper → camera → target closes on the fixed but unknown $T_{bt}$ at every station; eliminating it between two stations gives $AX = XB$.*

![[hand_eye_eye_to_hand.svg|650]]

*Eye-to-hand. Now the target rides on the gripper ($T_{gt}$ fixed but unknown) and the static camera's pose in the base, $X = T_{bc}$, is sought; the same elimination gives $AX = XB$ with a different $A$.*

**Why not measure the target pose directly?** If the target's pose in robot coordinates were known, a single station would give $X$ (eye-in-hand: $X = (T_{bg})^{-1}T_{bt}(T_{ct})^{-1}$). Tsai & Lenz point out that it is very difficult to fix a calibration block in a precisely known relationship to the robot. So $T_{bt}$ (or $T_{gt}$) is treated as an unknown nuisance transform and eliminated by comparing stations.

**Only relative robot motion matters.** The solution depends on $T_{bg}^{(i)}$ only through relative motions between stations. Tsai & Lenz: "even if $H_{gi}$ may be bad, so long as $H_{gij}$ is good, there is no problem." Absolute kinematic errors shared by nearby poses cancel; errors in the motion *between* stations do not.

## From Two Stations to AX = XB
---
<span style="color:rgb(255, 255, 0)"><b>Eye-in-hand.</b></span> The target does not move, so $T_{bg}^{(i)}\,X\,T_{ct}^{(i)} = T_{bt}$ at every station (Chaumette, slide 10). Equating stations $i$ and $j$ and rearranging:

$$
\underbrace{\big(T_{bg}^{(j)}\big)^{-1}T_{bg}^{(i)}}_{A\ \text{(gripper motion, from FK)}}\; X \;=\; X\; \underbrace{T_{ct}^{(j)}\big(T_{ct}^{(i)}\big)^{-1}}_{B\ \text{(camera motion, from PnP)}} .
$$

<span style="color:rgb(255, 255, 0)"><b>Eye-to-hand.</b></span> Here the target is fixed on the gripper, so $X\,T_{ct}^{(i)} = T_{bg}^{(i)}\,T_{gt}$ at every station (slide 13). The same elimination gives $A = T_{bg}^{(j)}\big(T_{bg}^{(i)}\big)^{-1}$ with the same $B$. Equivalently, feed the inverted kinematic poses $T_{gb}^{(i)}$ to an eye-in-hand solver, which is what the OpenCV documentation prescribes.

**Block form.** Writing out the rotation and translation blocks (slide 11; Tsai & Lenz Eq. 15):

$$
R_A R_X = R_X R_B, \qquad (R_A - I)\,t_X = R_X t_B - t_A .
$$

The rotation equation does not involve translation, so $R_X$ can be solved first and $t_X$ second (the *separable* solvers below).

![[hand_eye_axxb_square.svg|700]]

*$AX = XB$ as a commutative square: the gripper–camera rig is rigid, so going from $g_j$ to $c_i$ via the gripper motion and then $X$ equals going via $X$ and then the camera motion. The right panel previews the consequences derived next.*

## Geometry: Screw Invariants and Observability
---
*Proposition (screw invariants).* If $AX = XB$, then $A$ and $B$ have the same rotation angle, $\theta_A = \theta_B$; their axes are related by $n_A = R_X n_B$; and they have the same slide along the screw axis, $n_A\cdot t_A = n_B\cdot t_B$.

*Proof.* $R_A = R_X R_B R_X^\top$ is a similarity, so $R_A$ and $R_B$ share eigenvalues and hence angles. Moreover $R_A (R_X n_B) = R_X R_B n_B = R_X n_B$, so $R_X n_B$ is the axis of $R_A$ (Horaud & Dornaika, Eq. 18). For the slide, multiply $t_A = R_X t_B - (R_A - I)\,t_X$ on the left by $n_A^\top$: since $n_A^\top(R_A - I) = 0$ and $n_A^\top R_X = (R_X^\top n_A)^\top = n_B^\top$, we get $n_A^\top t_A = n_B^\top t_B$. ∎

In the language of [[Robot Kinematics]] (Chasles' theorem), $A$ and $B$ are *the same screw motion expressed in two frames*, and Chen (1991) based the uniqueness analysis of hand–eye geometry on this screw representation. <span style="color:rgb(0, 176, 240)">The invariants cost nothing to check: for every station pair, compare angle and slide of $A$ and $B$. A mismatch beyond noise flags a timestamp offset between robot and camera, a wrong pose, or a swapped image — before any solver runs.</span>

**One motion is not enough.** For a single pair, $\mathrm{Skew}(P_A + P_B)$ in the Tsai–Lenz rotation system and $(R_A - I)$ in the translation system both have rank 2. Rotation about $n_A$ and translation along $n_A$ stay free, a 2-dof family of solutions. Tsai & Lenz (Lemma VII) conclude that $R_X$ cannot be computed from two stations; Chaumette (slide 11) notes that a third orientation is necessary.

**Two motions with non-parallel rotation axes suffice** for a unique $X$ (shown algebraically by Tsai & Lenz and geometrically by Chen, as summarized by Andreff et al.). That means at least three stations; OpenCV requires three and recommends many more.

**Partial calibration.** Andreff et al. (2001, Table 1, Propositions 1–2, Lemmas 1–2) work out what degenerate motion sets still determine:

| Motions available | Determined | Undetermined |
|---|---|---|
| one general motion | — | rotation about $n_A$ and translation along $n_A$ (2 dof) |
| ≥ 2 linearly independent pure translations | $R_X$, from $t_A = R_X t_B$ | $t_X$ entirely |
| planar motions: all rotation axes parallel, plus translations | $R_X$ and the part of $t_X$ perpendicular to the axis | $t_X$ along the common axis (the "height") |
| ≥ 2 motions with non-parallel rotation axes | $R_X$, $t_X$ | — |

<span style="color:rgb(0, 176, 240)">The non-parallel-axes condition is about rotation-only equations; translations can stand in for it.</span> In a noise-free check with every rotation axis exactly parallel, the separable solvers failed: Tsai–Lenz was off by 21°, Park–Martin returned NaN, and Horaud–Dornaika was off by 63°. The simultaneous solvers (Andreff, Daniilidis) recovered $R_X$ and the perpendicular part of $t_X$ exactly and returned an arbitrary height. For planar arms, SCARA robots and mobile bases that only yaw, the vertical offset of the camera must therefore be measured some other way.

## Solving AX = XB
---
<span style="color:rgb(255, 255, 0)"><b>Separable solvers</b></span> estimate $R_X$ from the rotation equation, then $t_X$ by linear least squares on the stacked translation equations $(R_{A_k} - I)\,t_X = R_X t_{B_k} - t_{A_k}$. Each block has rank 2, so at least two pairs are needed.

**Tsai–Lenz (1989).** With $P = 2\sin(\theta/2)\,n$ for each motion, solve by linear least squares over all pairs (Eqs. 12–14):

$$
\mathrm{Skew}(P_{A_k} + P_{B_k})\,P' = P_{B_k} - P_{A_k}, \qquad \theta_X = 2\arctan\lVert P'\rVert, \qquad P_X = \frac{2P'}{\sqrt{1 + \lVert P'\rVert^2}},
$$

where $P' = \tan(\theta_X/2)\,n_X$. Tsai & Lenz pair stations so that the interstation rotation angles, and the angles between interstation axes, are as large as possible (their Fig. 6). OpenCV's implementation also drops pairs whose $\lVert P\rVert$ lies outside $[0.3, 1.7]$, i.e. relative rotations below ≈ 17° or above ≈ 116°.

**Park–Martin (1994).** The axis relation $\alpha_k = R_X\beta_k$ between exponential coordinates is an orthogonal Procrustes problem on $SO(3)$ with a closed-form least-squares solution:

$$
R_X = (M^\top M)^{-1/2} M^\top, \qquad M = \sum_k \beta_k\,\alpha_k^\top .
$$

**Horaud–Dornaika closed form (1995).** With unit quaternions, $n_A = q\,n_B\,\bar q$ is equivalent to $n_A q - q\,n_B = 0$, which is linear in $q$: $(Q(n_A) - W(n_B))\,q = 0$, where $Q(\cdot)$ and $W(\cdot)$ are the $4\times4$ matrices of left and right quaternion multiplication. Minimizing the summed squares under $\lVert q\rVert = 1$ makes $q$ the eigenvector of the smallest eigenvalue of $\sum_k (Q(n_{A_k}) - W(n_{B_k}))^\top(Q(n_{A_k}) - W(n_{B_k}))$. This is the same construction as Horn's and Faugeras–Hebert's absolute-orientation solution, which Horaud & Dornaika cite.

<span style="color:rgb(255, 255, 0)"><b>Simultaneous solvers</b></span> estimate $R_X$ and $t_X$ together, so translations also inform rotation (hence the partial-calibration result above).

**Andreff et al. (1999, 2001).** The row-major identity $\mathrm{vec}(R_A R_X R_B^\top) = (R_A\otimes R_B)\,\mathrm{vec}(R_X)$ turns $AX = XB$ into a linear system in 12 unknowns per pair (stacked over all pairs):

$$
\begin{bmatrix} I_9 - R_A\otimes R_B & 0_{9\times3} \\ I_3\otimes t_B^\top & I_3 - R_A \end{bmatrix} \begin{bmatrix} \mathrm{vec}(R_X) \\ t_X \end{bmatrix} = \begin{bmatrix} 0_9 \\ t_A \end{bmatrix} .
$$

The recovered $V = \mathrm{vec}^{-1}$ is then rescaled to unit determinant, $R_X = \mathrm{sign}(\det V)\,\lvert\det V\rvert^{-1/3}\,V$ (their Lemma 1), and projected onto $SO(3)$ with the SVD.

<span style="color:rgb(0, 176, 240)">Because $\mathrm{vec}(R_X)$ is not constrained to $SO(3)$ inside the solve, the system can lose rank on motion sets that separable solvers handle fine.</span> Suppose every station-to-station motion rotates about one common point — for example, the camera is always aimed at the same target point from the same distance. Then every $A_k$ fixes that point $p$ (in the gripper frame), and $(\mathrm{vec}\,R_X,\ t_X - p)$ is a null vector of the stacked matrix (rank 11 of 12). The minimum-norm least-squares solution then shrinks $\mathrm{vec}(R_X)$ toward zero, and $t_X$ comes out near $p$, off by about the camera–target distance.

In a simulated eye-in-hand set of 10 stations at 0.5 m (OpenCV 4.14), the Andreff translation error was ≈ 500 mm, against 1.6 mm for Park–Martin. About half of the runs failed outright with OpenCV's "determinant(R) is null" error. Varying the viewing distance by ±20 % and the aim point by a few centimetres restored full rank (2.9 mm).

**Daniilidis (1999).** Each motion is a unit dual quaternion, and conjugate screw motions give linear equations in the 8 components of $\check q = q + \epsilon q'$. With $(a, a')$ the vector parts of $\check A$'s real and dual parts, and $(b, b')$ those of $\check B$, each pair contributes 6 rows (Eq. 31 as implemented in OpenCV):

$$
\begin{bmatrix} a - b & [a + b]_\times & 0 & 0_{3\times3} \\ a' - b' & [a' + b']_\times & a - b & [a + b]_\times \end{bmatrix} \begin{bmatrix} q \\ q' \end{bmatrix} = 0 .
$$

The stacked matrix has a 2-D null space. The unit-dual-quaternion constraints $q^\top q = 1$ and $q^\top q' = 0$ pick the right combination through a quadratic in the mixing ratio.

<span style="color:rgb(255, 255, 0)"><b>Nonlinear refinement.</b></span> Horaud & Dornaika also minimize the rotation and translation residuals jointly over $(q, t_X)$ with Levenberg–Marquardt:

$$
\lambda_1 \sum_k \lVert n_{A_k} - q\,n_{B_k}\bar q\rVert^2 + \lambda_2 \sum_k \lVert q\,t_{B_k}\bar q - (R_{A_k} - I)\,t_X - t_{A_k}\rVert^2 + \lambda\,(1 - q^\top q)^2 .
$$

In their stability study this nonlinear method "provides more accurate results in all the situations", while Tsai–Lenz and their closed form behave almost identically, which they attribute to both methods decoupling rotation from translation. Two later refinements go further:
- **Strobl & Hirzinger (2006)** propose a metric on $SE(3)$, with an error model, that adapts to the system's precision and handles both $AX = XB$ and $AX = ZB$.
- **Koide & Menegatti (2019)** skip per-image pose estimation and minimize the target's **reprojection error** directly. This is the hand–eye analogue of Zhang's refinement in [[Camera Calibration]].

<span style="color:rgb(255, 255, 0)"><b>Robot-world/hand-eye: AX = YB.</b></span> Keeping absolute poses instead of differencing stations gives $T_{bg}^{(i)}\,X = Y\,T_{tc}^{(i)}$, with $Y = T_{bt}$ and $T_{tc} = (T_{ct})^{-1}$. Each station contributes one equation (no correlated pairs), and $Y$ returns the target pose as a by-product. Formulations: Zhuang, Roth & Sudhakar ([1994](https://doi.org/10.1109/70.313105)); Dornaika & Horaud ([1998](https://doi.org/10.1109/70.704233)); Shah ([2013](https://doi.org/10.1115/1.4024473), Kronecker product); Li et al. (2010, dual quaternions and Kronecker product). OpenCV's `calibrateRobotWorldHandEye` implements the last two in the inverse form $AX = ZB$ (see the notation table).

| Method | Rotation representation | Rotation & translation | Key step | OpenCV flag |
|---|---|---|---|---|
| Tsai & Lenz (1989) | modified Rodrigues $2\sin(\theta/2)\,n$ | separable | linear LS on $\mathrm{Skew}(P_A{+}P_B)P' = P_B{-}P_A$ | `CALIB_HAND_EYE_TSAI` |
| Park & Martin (1994) | exponential coordinates (log map) | separable | Procrustes $(M^\top M)^{-1/2}M^\top$ | `CALIB_HAND_EYE_PARK` |
| Horaud & Dornaika (1995) | unit quaternion | separable closed form; joint nonlinear refinement | smallest eigenvector of $\sum (Q - W)^\top(Q - W)$ | `CALIB_HAND_EYE_HORAUD` |
| Andreff et al. (1999, 2001) | rotation matrix (vec, Kronecker) | simultaneous, linear | 12-unknown linear LS, then project to $SO(3)$ | `CALIB_HAND_EYE_ANDREFF` |
| Daniilidis (1999) | unit dual quaternion (screw) | simultaneous | SVD null space + quadratic constraint | `CALIB_HAND_EYE_DANIILIDIS` |
| Shah (2013); Li et al. (2010) | Kronecker; dual quaternion + Kronecker | $AX = YB$, absolute poses | also returns $T_{bt}$ | `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`, `_LI` |

## Accuracy and Data Collection
---
<span style="color:rgb(255, 255, 0)"><b>Tsai–Lenz error analysis.</b></span> From their accuracy formulas (Sec. III-B):
- *Observation 1:* the rotation error of $X$ is inversely proportional to the sine of the angle between interstation rotation axes.
- *Observation 2:* rotation and translation errors are both inversely proportional to the interstation rotation angle.
- *Observation 3:* the distance between the camera and the calibration block dominates the translation error. Their example: a 5 in lever arm with 3 mrad interstation-rotation error already gives 15 mil.
- *Observation 4:* the gripper's travel between stations matters for translation error; the camera's travel does not.
- *Observation 5:* the rotation error is linear in the orientation error of each station.

Non-systematic error also falls by $\sqrt N$ with $N$ stations. Their steps to improve accuracy (Sec. III-C) follow directly:
- maximize the angles between interstation axes and the interstation rotations;
- keep the target close (a small target and short-range optics);
- keep gripper travel between stations small;
- use redundant stations;
- calibrate the camera well and pre-calibrate the robot.

![[hand_eye_noise_sensitivity.png|750]]

*Synthetic check of the Tsai–Lenz observations with OpenCV's five solvers. Setup: median of 100 trials per point, noise 0.1° / 0.5 mm on PnP and 0.02° / 0.2 mm on kinematics.*
- *(a) Error falls roughly as 1/rotation (Park–Martin: 1.01° at ±5° to 0.079° at ±60°). Tsai–Lenz is missing at ±5° because OpenCV rejects the pairs as too small.*
- *(b) As rotation axes become parallel, the rotation-only solvers degrade (4.8–5.4° at a 1° cone) while the simultaneous ones stay ≈ 0.27°.*
- *(c) Translation error grows with camera–target distance (1.0 mm at 0.2 m to 4.6 mm at 1.5 m).*
- *(d) Error falls with the number of poses (0.46° at $N = 3$ to 0.11° at $N = 25$), at least as fast as $1/\sqrt N$.*

**In practice:**
- *Station design* — large, well-spread rotations; vary the viewing distance and aim point, so not every motion rotates about one point (the Andreff pitfall above); keep the target close and filling the image; take many stations.
- *Static capture and sync* — image only after the robot has settled, and check the screw invariants on every pair to catch timestamp offsets.
- *PnP quality* — use calibrated intrinsics, inspect per-view reprojection error, and drop bad views ([[Camera Calibration]]).
- *Kinematic quality* — only relative motion enters, but kinematic errors still bias $X$; Tsai & Lenz recommend pre-calibrating the robot.
- *Validation* — for eye-in-hand, $T_{bg}^{(i)}\,X\,T_{ct}^{(i)}$ should give the same target pose at every station, so its spread is a direct error estimate (code below). Also hold out stations.
- *Re-calibration* — repeat whenever a camera mount is touched; DROID made deliberate camera moves part of collection (below).

```python
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation as Rot

inv, log = np.linalg.inv, lambda R: Rot.from_matrix(R).as_rotvec()

def motion_pairs(T_bg, T_ct, eye_in_hand=True):           # per-station 4x4 poses -> list of (A, B)
    n = len(T_bg)
    return [((inv(T_bg[j]) @ T_bg[i]) if eye_in_hand else (T_bg[j] @ inv(T_bg[i])),
             T_ct[j] @ inv(T_ct[i])) for i in range(n) for j in range(i + 1, n)]

def park_martin(pairs):                                   # AX = XB -> X (4x4)
    M = sum(np.outer(log(B[:3, :3]), log(A[:3, :3])) for A, B in pairs)
    R = np.real(inv(sqrtm(M.T @ M)) @ M.T)                # orthogonal Procrustes: alpha_k = R beta_k
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    d = np.hstack([R @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    X = np.eye(4); X[:3, :3] = R; X[:3, 3] = np.linalg.lstsq(C, d, rcond=None)[0]
    return X

def screw_residuals(pairs):                               # conjugate motions: equal angle and slide
    out = []
    for A, B in pairs:
        a, b = log(A[:3, :3]), log(B[:3, :3])
        out.append((np.linalg.norm(a) - np.linalg.norm(b),
                    a @ A[:3, 3] / np.linalg.norm(a) - b @ B[:3, 3] / np.linalg.norm(b)))
    return np.array(out)                                  # large entries flag bad sync or a bad pose

def target_spread(T_bg, T_ct, X):                         # eye-in-hand: T_bt = T_bg X T_ct is constant
    T_bt = np.stack([G @ X @ C for G, C in zip(T_bg, T_ct)])
    return T_bt[:, :3, 3].std(axis=0)                     # per-axis spread of the target origin [m]
```

```python
import cv2                                                # OpenCV 4.x
# eye-in-hand: X = T_gc from FK poses T_bg and PnP poses T_ct
R_gc, t_gc = cv2.calibrateHandEye(R_bg, t_bg, R_ct, t_ct, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
# eye-to-hand: pass the inverted FK poses T_gb; the result is X = T_bc
R_bc, t_bc = cv2.calibrateHandEye(R_gb, t_gb, R_ct, t_ct, method=cv2.CALIB_HAND_EYE_PARK)
# robot-world/hand-eye (OpenCV's AX = ZB): returns T_tb and T_cg
R_tb, t_tb, R_cg, t_cg = cv2.calibrateRobotWorldHandEye(R_ct, t_ct, R_gb, t_gb)
```

## Markerless and Learned Camera-to-Robot Calibration
---
Classical hand–eye calibration needs a target and a dedicated routine. Learned methods use the robot itself as the target, which allows zero-setup and online re-estimation, mostly for external (eye-to-hand) cameras:
- **DREAM** ([Lee et al., 2020](https://arxiv.org/abs/1911.09231)) — a network detects the 2D projections of robot keypoints (such as joints) in a single RGB image, and PnP with known intrinsics and joint configuration recovers the camera extrinsics. It is trained only on synthetic data with domain randomization, and needs no offline calibration step.
- **CtRNet** ([Lu et al., 2023](https://arxiv.org/abs/2302.14332)) — an end-to-end differentiable pipeline combining deep learning and geometric vision. It uses foreground segmentation and differentiable rendering for self-supervised sim-to-real training on unlabeled real images, and runs online inside a visual-servoing loop.
- **EasyHeC** ([Chen et al., 2023](https://arxiv.org/abs/2305.01191)) — markerless; combines differentiable-rendering-based camera pose optimization with consistency-based joint-space exploration, removing hand-designed calibration poses.
- **Kalib** ([Tang et al., 2024](https://arxiv.org/abs/2408.10562)) — tracks a predefined reference point on the robot with visual foundation models, gets its 3D position from forward kinematics, and solves PnP. It needs only the kinematic chain and the reference point: no markers, retraining or meshes.

<span style="color:rgb(255, 255, 0)"><b>At dataset scale: DROID.</b></span> In DROID ([Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945)), collectors calibrated the two exterior cameras per scene with a checkerboard and OpenCV, and the collection GUI randomly prompted them to move and re-calibrate the cameras as scene augmentation. Because those calibrations "may not always be accurate", the authors re-audited them post hoc (App. G) in three stages:
1. Robot masks segmented by SAM, prompted with FK-projected keypoints, are compared with URDF-rendered masks; frames with IoU ≥ 0.7 count as well calibrated.
2. A tuned CtRNet-X re-estimates camera-to-base for more scenes.
3. A modified DUSt3R recovers camera-to-camera poses.

The release has camera-to-base calibrations for ≈ 36k scenes and camera-to-camera calibrations for all scenes. <span style="color:rgb(0, 176, 240)">Calibration quality is a property of the dataset: multi-view and 3D-aware policies trained on robot data inherit whatever extrinsic error the collection pipeline left behind.</span>

## Relationship to Other Notes
---
- [[Camera Calibration]] — supplies the intrinsics and the per-station PnP poses $T_{ct}$; its hand–eye section is a digest pointing here, and Koide–Menegatti's reprojection refinement mirrors Zhang's.
- [[Robot Kinematics]] — forward kinematics gives $T_{bg}$; $A$ and $B$ are one screw motion (Chasles) seen in two frames; Park–Martin works in the exponential coordinates defined there.
- [[SVD]] — Procrustes (Park–Martin), the Kronecker null space (Andreff), the dual-quaternion null space (Daniilidis) and the final projection onto $SO(3)$ are all SVD computations.
- [[Linear Algebra]] — rank-2 blocks per motion pair, Kronecker/vec identities, the smallest-eigenvector solution (Horaud–Dornaika), and the minimum-norm least-squares failure of the Andreff system on rank-deficient data.
- [[VLA]] — its cross-embodiment training mixtures include DROID, whose per-scene camera extrinsics come from target-based calibration plus learned re-calibration.
- [[Behavior Cloning]] — demonstrations recorded through miscalibrated cameras teach 3D-aware policies a systematic offset; calibrated extrinsics let preprocessing compensate when cameras move between sessions.

## References
---
- Georgia Tech ME 6406 *Machine Vision* (K.-M. Lee) — hand–eye reading: [Tsai & Lenz, 1989 (course copy)](http://kmlee.gatech.edu/me6406/handeye.pdf)
- [Chaumette — *Requirements for Sensor-based Control*, GdR Robotics Winter School "Robotics Principia", slides 9–14](https://project.inria.fr/roboticsprincipia/files/2019/01/Requirements_SBC.pdf)
- [Lynch & Park — *Modern Robotics: Mechanics, Planning, and Control*, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Ch. 3 Rigid-Body Motions
- [Shiu & Ahmad, 1989 — Calibration of Wrist-Mounted Robotic Sensors by Solving Homogeneous Transform Equations of the Form AX = XB, *IEEE Trans. Robotics and Automation* 5(1)](https://doi.org/10.1109/70.88014)
- [Tsai & Lenz, 1989 — A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration, *IEEE Trans. Robotics and Automation* 5(3)](https://doi.org/10.1109/70.34770)
- [Chen, 1991 — A Screw Motion Approach to Uniqueness Analysis of Head-Eye Geometry, CVPR](https://doi.org/10.1109/CVPR.1991.139677)
- [Park & Martin, 1994 — Robot Sensor Calibration: Solving AX = XB on the Euclidean Group, *IEEE Trans. Robotics and Automation* 10(5)](https://doi.org/10.1109/70.326576)
- [Horaud & Dornaika, 1995 — Hand-Eye Calibration, *IJRR* 14(3)](https://doi.org/10.1177/027836499501400301) · [arXiv:2311.12655](https://arxiv.org/abs/2311.12655)
- [Daniilidis, 1999 — Hand-Eye Calibration Using Dual Quaternions, *IJRR* 18(3)](https://doi.org/10.1177/02783649922066213)
- [Andreff, Horaud & Espiau, 1999 — On-line Hand-Eye Calibration, 3DIM](https://doi.org/10.1109/IM.1999.805374) · [2001 — Robot Hand-Eye Calibration Using Structure-from-Motion, *IJRR* 20(3)](https://doi.org/10.1177/02783640122067372) · [arXiv:2311.11808](https://arxiv.org/abs/2311.11808)
- [Zhuang, Roth & Sudhakar, 1994 — Simultaneous Robot/World and Tool/Flange Calibration by Solving Homogeneous Transformation Equations of the Form AX = YB, *IEEE Trans. Robotics and Automation* 10(4)](https://doi.org/10.1109/70.313105)
- [Dornaika & Horaud, 1998 — Simultaneous Robot-World and Hand-Eye Calibration, *IEEE Trans. Robotics and Automation* 14(4)](https://doi.org/10.1109/70.704233)
- [Shah, 2013 — Solving the Robot-World/Hand-Eye Calibration Problem Using the Kronecker Product, *J. Mechanisms and Robotics* 5(3)](https://doi.org/10.1115/1.4024473) · Li, Wang & Wu, 2010 — Simultaneous robot-world and hand-eye calibration using dual-quaternions and Kronecker product
- [Strobl & Hirzinger, 2006 — Optimal Hand-Eye Calibration, IROS](https://doi.org/10.1109/IROS.2006.282250)
- [Koide & Menegatti, 2019 — General Hand–Eye Calibration Based on Reprojection Error Minimization, *IEEE RA-L* 4(2)](https://doi.org/10.1109/LRA.2019.2893612)
- [OpenCV `calib3d` documentation (4.x)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — `calibrateHandEye`, `calibrateRobotWorldHandEye` · [implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_handeye.cpp)
- [Lee et al., 2020 — DREAM: Camera-to-Robot Pose Estimation from a Single Image](https://arxiv.org/abs/1911.09231) · [Lu et al., 2023 — CtRNet](https://arxiv.org/abs/2302.14332) · [Chen et al., 2023 — EasyHeC](https://arxiv.org/abs/2305.01191) · [Tang et al., 2024 — Kalib](https://arxiv.org/abs/2408.10562)
- [Khazatsky et al., 2024 — DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945)
