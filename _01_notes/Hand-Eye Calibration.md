### [[dl_moc]]  
# **Hand–Eye Calibration**

**Tags**:: #rl #math #vision
**Links**:: [[Camera Calibration]] · [[Pinhole and Perspective Projection]] · [[Robot Kinematics]] · [[SVD]] · [[Linear Algebra]] · [[VLA]] · [[Behavior Cloning]]


**Hand–eye calibration** finds out exactly where a camera sits relative to a robot. The "hand" is the robot's gripper and the "eye" is the camera. If the camera rides on the robot's wrist (*eye-in-hand*), we want the camera's pose in the gripper's frame. If the camera is fixed somewhere in the room (*eye-to-hand*), we want its pose in the robot base's frame. Without this, the robot sees an object in the camera but cannot tell where that object is in its own coordinates. Hand–eye calibration lets a robot (i) turn what the camera sees into robot coordinates and (ii) move the camera precisely ([Horaud & Dornaika, 1995](https://doi.org/10.1177/027836499501400301)). Every camera-guided grasp, every point-cloud policy and every multi-camera robot dataset depends on it.

The hard part is that we usually do not know exactly where the calibration target sits relative to the robot. So instead of absolute positions, the problem compares *motions*: how the gripper moved versus how the camera moved. This gives the matrix equation $AX = XB$ ([Shiu & Ahmad, 1989](https://doi.org/10.1109/70.88014); [Tsai & Lenz, 1989](https://doi.org/10.1109/70.34770)). The classic ways to solve it are Tsai–Lenz, Park–Martin ([1994](https://doi.org/10.1109/70.326576)), Horaud–Dornaika's quaternion method, Andreff et al.'s Kronecker-product method ([2001](https://doi.org/10.1177/02783640122067372)) and Daniilidis's dual quaternions ([1999](https://doi.org/10.1177/02783649922066213)).

The note follows two course sources:
- Georgia Tech **ME 6406 Machine Vision** (K.-M. Lee) assigns the Tsai–Lenz paper as its hand–eye reading ([course copy](http://kmlee.gatech.edu/me6406/handeye.pdf)). It supplies the robot-pose setup, the closed-form solution and the accuracy observations below.
- The GdR Robotics Winter School **Robotics Principia** ([Chaumette, *Requirements for Sensor-based Control*, slides 9–14](https://project.inria.fr/roboticsprincipia/files/2019/01/Requirements_SBC.pdf)) supplies the eye-in-hand / eye-to-hand derivation and the "why one motion is not enough" argument.

*Modern Robotics* ([Lynch & Park, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 3) provides the math of rotations and poses ($SE(3)$), as in [[Robot Kinematics]]. Horaud & Dornaika ([arXiv:2311.12655](https://arxiv.org/abs/2311.12655)) and Andreff et al. ([arXiv:2311.11808](https://arxiv.org/abs/2311.11808)) compare the methods, and the OpenCV `calib3d` docs and source fix the conventions used in practice.

**Notation.** Frames and transforms are written as in [[Robot Kinematics]] and [[Camera Calibration]]. The last column translates Tsai–Lenz, OpenCV and Andreff et al.

| Symbol | Meaning | Source equivalents · clashes |
|---|---|---|
| $\{b\}, \{g\}, \{c\}, \{t\}$ | coordinate frames of the robot base, gripper (flange), camera, and calibration target | Tsai–Lenz: RW, $G_i$, $C_i$, CW |
| $T_{ab}$ | pose of frame $b$ seen from frame $a$: $\mathbf x_a = T_{ab}\mathbf x_b$, and $T_{ab}T_{bc} = T_{ac}$ | OpenCV: ${}^{a}T_{b}$, named `b2a` (e.g. `gripper2base` $= T_{bg}$) |
| $T_{bg}^{(i)}$ | gripper pose at robot pose $i$, from forward kinematics | Tsai–Lenz: $H_{gi}$; OpenCV `R_gripper2base`, `t_gripper2base` |
| $T_{ct}^{(i)}$ | target pose seen by the camera at robot pose $i$, from PnP | Tsai–Lenz: $H_{ci}$; OpenCV `R_target2cam`, `t_target2cam` |
| $X$ | the unknown we want: $T_{gc}$ (eye-in-hand) or $T_{bc}$ (eye-to-hand) | Tsai–Lenz: $H_{cg}$ ("from $C_i$ to $G_i$"); OpenCV `cam2gripper` |
| $A, B$ | how the gripper / camera moved between robot poses $i$ and $j$ | Tsai–Lenz: $H_{gij}$, $H_{cij}$; Andreff et al. swap the roles ($A$ = camera motion) |
| $R_\bullet, t_\bullet$ | the rotation / translation part of a transform, e.g. $R_A$, $t_X$ | Tsai–Lenz: $T$ for translation |
| $\theta, n$ | rotation angle and rotation axis (a unit vector) of a motion | not a joint angle (cf. [[Robot Kinematics]]) |
| $P = 2\sin(\theta/2)\,n$ | Tsai–Lenz's way of writing a rotation as 3 numbers | Tsai–Lenz: $P_r$; OpenCV: `2*rot2quatMinimal` |
| $\alpha, \beta$ | rotation written as axis × angle ($\theta n$) for $R_A$, $R_B$ | Park–Martin; unrelated to CS231A's focal lengths in [[Camera Calibration]] |
| $d = n\cdot t$ | how far a motion slides along its own rotation axis | — |
| $q$; $\check q = q + \epsilon q'$ | unit quaternion; unit dual quaternion | Horaud–Dornaika; Daniilidis |
| $\mathrm{vec}(\cdot)$, $\otimes$ | stacking a matrix's rows into one long vector; Kronecker product | Andreff et al.; OpenCV stacks row by row |
| $Y$ | second unknown in $AX = YB$ (here $T_{bt}$) | OpenCV writes $AX = ZB$ with $A = T_{ct}$, $X = T_{tb}$, $Z = T_{cg}$, $B = T_{gb}$ |
| $N$ | number of robot poses at which a picture is taken ("stations") | Tsai–Lenz: $N$ |

## The Setup
---
<span style="color:rgb(255, 255, 0)"><b>Two ways to mount the camera.</b></span>
- **Eye-in-hand:** the camera rides on the gripper and looks at a target that stays fixed in the room. We want $X = T_{gc}$.
- **Eye-to-hand:** the camera stays fixed in the room and looks at a target held by the gripper. We want $X = T_{bc}$.

In both cases the robot moves to $N$ different poses (called **stations**). At each one we write down two things: where the gripper is, $T_{bg}^{(i)}$, from the robot's joint angles (forward kinematics, [[Robot Kinematics]]); and where the target is as seen by the camera, $T_{ct}^{(i)}$, from PnP using the calibrated camera ([[Camera Calibration]]; PnP inverts the projection of [[Pinhole and Perspective Projection]]).

![[hand_eye_calibration.svg|650]]

*Eye-in-hand. Going base → gripper → camera → target always ends at the same place, because the target does not move. That fixed (but unknown) $T_{bt}$ is removed by comparing two robot poses, which gives $AX = XB$.*

![[hand_eye_eye_to_hand.svg|650]]

*Eye-to-hand. Now the target rides on the gripper ($T_{gt}$ is fixed but unknown) and we want the fixed camera's pose in the base, $X = T_{bc}$. The same trick gives $AX = XB$, with a different $A$.*

**Why not just measure where the target is?** If we knew the target's exact pose in robot coordinates, one robot pose would be enough (eye-in-hand: $X = (T_{bg})^{-1}T_{bt}(T_{ct})^{-1}$). But Tsai & Lenz point out that it is very hard to place a target at a precisely known spot relative to the robot. So we treat the target's pose ($T_{bt}$, or $T_{gt}$) as an unknown and cancel it out by comparing robot poses.

**Only the robot's motions matter, not its absolute positions.** The answer depends on $T_{bg}^{(i)}$ only through how the gripper moved from one pose to another. Tsai & Lenz: "even if $H_{gi}$ may be bad, so long as $H_{gij}$ is good, there is no problem." A position error that is the same at nearby poses cancels out; an error in the motion *between* poses does not.

## From Two Robot Poses to AX = XB
---
<span style="color:rgb(255, 255, 0)"><b>Eye-in-hand.</b></span> The target does not move, so at every robot pose the chain base → gripper → camera → target gives the same answer: $T_{bg}^{(i)}\,X\,T_{ct}^{(i)} = T_{bt}$ (Chaumette, slide 10). Set pose $i$ equal to pose $j$ and rearrange:

$$
\underbrace{\big(T_{bg}^{(j)}\big)^{-1}T_{bg}^{(i)}}_{A\ \text{(gripper motion, from FK)}}\; X \;=\; X\; \underbrace{T_{ct}^{(j)}\big(T_{ct}^{(i)}\big)^{-1}}_{B\ \text{(camera motion, from PnP)}} .
$$

In plain words: "move the gripper, then step to the camera" must equal "step to the camera, then move the camera", because gripper and camera are bolted together.

<span style="color:rgb(255, 255, 0)"><b>Eye-to-hand.</b></span> Now the target is fixed to the gripper, so $X\,T_{ct}^{(i)} = T_{bg}^{(i)}\,T_{gt}$ at every robot pose (slide 13). The same steps give $A = T_{bg}^{(j)}\big(T_{bg}^{(i)}\big)^{-1}$ with the same $B$. Equivalently, give an eye-in-hand solver the *inverted* gripper poses $T_{gb}^{(i)}$; this is what the OpenCV documentation says to do.

**Rotation and translation parts.** Splitting $AX = XB$ into its rotation part and its translation part gives (slide 11; Tsai & Lenz Eq. 15):

$$
R_A R_X = R_X R_B, \qquad (R_A - I)\,t_X = R_X t_B - t_A .
$$

The rotation equation does not contain any translation. So we can solve for the rotation $R_X$ first and then for the translation $t_X$. Methods that do this are called *separable*.

![[hand_eye_axxb_square.svg|700]]

*$AX = XB$ as a square. Gripper and camera are one rigid piece, so going from $g_j$ to $c_i$ along the top and right ($A$, then $X$) equals going along the left and bottom ($X$, then $B$). The right panel previews what follows.*

## What AX = XB Tells Us, and How Many Poses We Need
---
*Proposition (screw invariants).* If $AX = XB$, then:
- $A$ and $B$ rotate by the same angle: $\theta_A = \theta_B$;
- their rotation axes are related by $n_A = R_X n_B$;
- they slide the same distance along their own axis: $n_A\cdot t_A = n_B\cdot t_B$.

*Proof.* $R_A = R_X R_B R_X^\top$ means $R_A$ is just $R_B$ seen from a turned frame, so the two have the same eigenvalues and hence the same angle. Also $R_A (R_X n_B) = R_X R_B n_B = R_X n_B$: the vector $R_X n_B$ is not changed by $R_A$, so it is $R_A$'s axis (Horaud & Dornaika, Eq. 18). For the slide, multiply $t_A = R_X t_B - (R_A - I)\,t_X$ on the left by $n_A^\top$. Since $n_A^\top(R_A - I) = 0$ and $n_A^\top R_X = (R_X^\top n_A)^\top = n_B^\top$, we get $n_A^\top t_A = n_B^\top t_B$. ∎

In the language of [[Robot Kinematics]] (Chasles' theorem: every rigid motion is a turn about an axis plus a slide along it), $A$ and $B$ are *the same screw motion, described from two different frames*. Chen (1991) based his analysis of when the solution is unique on this screw view.

<span style="color:rgb(0, 176, 240)">This gives a free sanity check:</span> for every pair of robot poses, compare the turning angle and the slide of $A$ and $B$. If they disagree by more than the noise, something is wrong before any solver even runs: the robot and camera clocks are out of sync, a pose is wrong, or two images got swapped.

**One motion is not enough.** For a single pair of robot poses, both the rotation equations ($\mathrm{Skew}(P_A + P_B)$ in Tsai–Lenz) and the translation equations ($(R_A - I)$) have rank 2. That leaves 2 things free: turning about $n_A$, and sliding along $n_A$. Tsai & Lenz (Lemma VII) conclude that the rotation cannot be found from only two robot poses; Chaumette (slide 11) notes that a third orientation is needed.

**Two motions with non-parallel rotation axes are enough** for a unique answer. Tsai & Lenz showed this with algebra and Chen with geometry, as summarized by Andreff et al. That means at least three robot poses. OpenCV requires three and recommends many more.

**What if the motions are poor?** Andreff et al. (2001, Table 1, Propositions 1–2, Lemmas 1–2) work out what can still be found:

| Motions available | Can be found | Cannot be found |
|---|---|---|
| one general motion | — | turning about $n_A$ and sliding along $n_A$ (2 unknowns) |
| ≥ 2 pure translations in different directions | $R_X$, from $t_A = R_X t_B$ | $t_X$ at all |
| planar motions: all rotation axes parallel, plus translations | $R_X$, and the part of $t_X$ across the axis | the part of $t_X$ along the shared axis (the "height") |
| ≥ 2 motions with non-parallel rotation axes | $R_X$, $t_X$ | — |

<span style="color:rgb(0, 176, 240)">The "non-parallel axes" rule is about the rotation equations alone; translations can make up for it.</span> In a noise-free test with every rotation axis exactly parallel, the separable solvers failed: Tsai–Lenz was off by 21°, Park–Martin returned NaN, and Horaud–Dornaika was off by 63°. The joint solvers (Andreff, Daniilidis) got $R_X$ and the across-axis part of $t_X$ exactly right, and returned a random height. So for planar arms, SCARA robots and mobile bases that only turn about the vertical, the camera's height has to be measured some other way.

## Solving AX = XB
---
<span style="color:rgb(255, 255, 0)"><b>Separable solvers: rotation first, then translation.</b></span> They find $R_X$ from the rotation equation. Then they find $t_X$ by least squares on all the translation equations stacked together, $(R_{A_k} - I)\,t_X = R_X t_{B_k} - t_{A_k}$. Each block has rank 2, so at least two pairs of poses are needed.

**Tsai–Lenz (1989).** Write each rotation as $P = 2\sin(\theta/2)\,n$. Then solve, by least squares over all pairs (Eqs. 12–14):

$$
\mathrm{Skew}(P_{A_k} + P_{B_k})\,P' = P_{B_k} - P_{A_k}, \qquad \theta_X = 2\arctan\lVert P'\rVert, \qquad P_X = \frac{2P'}{\sqrt{1 + \lVert P'\rVert^2}},
$$

where $P' = \tan(\theta_X/2)\,n_X$. Tsai & Lenz pick pairs of poses so that the rotation between them is as large as possible, and the axes of different pairs point in very different directions (their Fig. 6). OpenCV's version also throws away pairs with $\lVert P\rVert$ outside $[0.3, 1.7]$, i.e. rotations smaller than ≈ 17° or larger than ≈ 116°.

**Park–Martin (1994).** The axes satisfy $\alpha_k = R_X\beta_k$. Finding the rotation that best lines up one set of vectors with another is called an **orthogonal Procrustes problem**, and it has a direct formula:

$$
R_X = (M^\top M)^{-1/2} M^\top, \qquad M = \sum_k \beta_k\,\alpha_k^\top .
$$

**Horaud–Dornaika, closed form (1995).** With unit quaternions, $n_A = q\,n_B\,\bar q$ is the same as $n_A q - q\,n_B = 0$, which is linear in $q$: $(Q(n_A) - W(n_B))\,q = 0$. Here $Q(\cdot)$ and $W(\cdot)$ are the $4\times4$ matrices for multiplying a quaternion from the left and from the right. Making the total error as small as possible, with $\lVert q\rVert = 1$, gives $q$ = the eigenvector with the smallest eigenvalue of $\sum_k (Q(n_{A_k}) - W(n_{B_k}))^\top(Q(n_{A_k}) - W(n_{B_k}))$. This is the same trick as Horn's and Faugeras–Hebert's solution for aligning two point sets, which Horaud & Dornaika cite.

<span style="color:rgb(255, 255, 0)"><b>Joint solvers: rotation and translation together.</b></span> Because they solve both at once, the translations also help pin down the rotation (which is why they survive the parallel-axes case above).

**Andreff et al. (1999, 2001).** Using the rule $\mathrm{vec}(R_A R_X R_B^\top) = (R_A\otimes R_B)\,\mathrm{vec}(R_X)$ (with row-by-row stacking), $AX = XB$ becomes one linear system with 12 unknowns, stacked over all pairs:

$$
\begin{bmatrix} I_9 - R_A\otimes R_B & 0_{9\times3} \\ I_3\otimes t_B^\top & I_3 - R_A \end{bmatrix} \begin{bmatrix} \mathrm{vec}(R_X) \\ t_X \end{bmatrix} = \begin{bmatrix} 0_9 \\ t_A \end{bmatrix} .
$$

The 9 numbers found for the rotation are put back into a $3\times3$ matrix $V$. It is rescaled to determinant 1, $R_X = \mathrm{sign}(\det V)\,\lvert\det V\rvert^{-1/3}\,V$ (their Lemma 1), and then snapped to the nearest true rotation with the SVD.

<span style="color:rgb(0, 176, 240)">A trap with this method: inside the solve, the 9 rotation numbers are not forced to form a real rotation, so the system can break on motions that separable solvers handle fine.</span> Suppose every motion between robot poses turns about one shared point. That happens, for example, if the camera always aims at the same spot on the target from the same distance. Then every $A_k$ leaves that point $p$ (in the gripper frame) where it is, and $(\mathrm{vec}\,R_X,\ t_X - p)$ solves the equations with zero on the right (the stacked matrix has rank 11 instead of 12). The least-squares solver then shrinks the rotation numbers toward zero, and $t_X$ comes out near $p$: wrong by about the camera-to-target distance.

In a simulated eye-in-hand test with 10 robot poses at 0.5 m (OpenCV 4.14), Andreff's translation was off by ≈ 500 mm, against 1.6 mm for Park–Martin. About half of the runs crashed outright with OpenCV's "determinant(R) is null" error. Changing the viewing distance by ±20 % and the aim point by a few centimetres fixed it (2.9 mm).

**Daniilidis (1999).** Each motion is written as a unit dual quaternion. Because $A$ and $B$ are the same screw motion, this gives linear equations in the 8 numbers of $\check q = q + \epsilon q'$. Let $(a, a')$ be the vector parts of $\check A$'s real and dual parts, and $(b, b')$ those of $\check B$. Each pair of robot poses adds 6 rows (Eq. 31, as implemented in OpenCV):

$$
\begin{bmatrix} a - b & [a + b]_\times & 0 & 0_{3\times3} \\ a' - b' & [a' + b']_\times & a - b & [a + b]_\times \end{bmatrix} \begin{bmatrix} q \\ q' \end{bmatrix} = 0 .
$$

The stacked matrix leaves a 2-dimensional space of solutions. The two conditions every unit dual quaternion must meet, $q^\top q = 1$ and $q^\top q' = 0$, pick the right one by solving a quadratic equation.

<span style="color:rgb(255, 255, 0)"><b>Polishing the answer (nonlinear refinement).</b></span> Horaud & Dornaika also shrink the rotation and translation errors together over $(q, t_X)$, using Levenberg–Marquardt:

$$
\lambda_1 \sum_k \lVert n_{A_k} - q\,n_{B_k}\bar q\rVert^2 + \lambda_2 \sum_k \lVert q\,t_{B_k}\bar q - (R_{A_k} - I)\,t_X - t_{A_k}\rVert^2 + \lambda\,(1 - q^\top q)^2 .
$$

In their tests this method "provides more accurate results in all the situations". Tsai–Lenz and their closed form behaved almost the same as each other, which they put down to both solving rotation separately from translation. Two later methods go further:
- **Strobl & Hirzinger (2006)** define a way to measure error on $SE(3)$, with an error model, that adapts to how precise the system is and works for both $AX = XB$ and $AX = ZB$.
- **Koide & Menegatti (2019)** skip computing a camera pose per image and directly minimize the target's **reprojection error** (how far the predicted target points land from where they were seen). This is the hand–eye version of Zhang's final step in [[Camera Calibration]].

<span style="color:rgb(255, 255, 0)"><b>Solving for the target too: AX = YB.</b></span> Instead of comparing pairs of poses, keep each robot pose's absolute values: $T_{bg}^{(i)}\,X = Y\,T_{tc}^{(i)}$, with $Y = T_{bt}$ and $T_{tc} = (T_{ct})^{-1}$. Each robot pose gives one equation (no overlapping pairs), and $Y$ tells you where the target is as a bonus. Methods: Zhuang, Roth & Sudhakar ([1994](https://doi.org/10.1109/70.313105)); Dornaika & Horaud ([1998](https://doi.org/10.1109/70.704233)); Shah ([2013](https://doi.org/10.1115/1.4024473), Kronecker product); Li et al. (2010, dual quaternions and Kronecker product). OpenCV's `calibrateRobotWorldHandEye` implements the last two, written the other way round as $AX = ZB$ (see the notation table).

| Method | How rotation is written | Rotation & translation | Key step | OpenCV flag |
|---|---|---|---|---|
| Tsai & Lenz (1989) | $2\sin(\theta/2)\,n$ (modified Rodrigues) | separate | least squares on $\mathrm{Skew}(P_A{+}P_B)P' = P_B{-}P_A$ | `CALIB_HAND_EYE_TSAI` |
| Park & Martin (1994) | axis × angle (log map) | separate | Procrustes $(M^\top M)^{-1/2}M^\top$ | `CALIB_HAND_EYE_PARK` |
| Horaud & Dornaika (1995) | unit quaternion | separate (closed form); together (nonlinear polish) | smallest eigenvector of $\sum (Q - W)^\top(Q - W)$ | `CALIB_HAND_EYE_HORAUD` |
| Andreff et al. (1999, 2001) | rotation matrix (vec, Kronecker) | together, linear | 12-unknown least squares, then snap to a rotation | `CALIB_HAND_EYE_ANDREFF` |
| Daniilidis (1999) | unit dual quaternion (screw) | together | SVD solution space + quadratic condition | `CALIB_HAND_EYE_DANIILIDIS` |
| Shah (2013); Li et al. (2010) | Kronecker; dual quaternion + Kronecker | $AX = YB$, absolute poses | also returns $T_{bt}$ | `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`, `_LI` |

## Accuracy and Collecting Good Data
---
<span style="color:rgb(255, 255, 0)"><b>What Tsai & Lenz found makes errors big or small</b></span> (their accuracy formulas, Sec. III-B):
- *Observation 1:* the rotation error of $X$ goes like 1 / sin(angle between the rotation axes of different motions). Axes pointing in similar directions → big error.
- *Observation 2:* both rotation and translation errors go like 1 / (rotation angle between poses). Small turns → big error.
- *Observation 3:* the camera-to-target distance dominates the translation error. Their example: a 5 inch lever arm with a 3 mrad rotation error already gives 15 mil (0.015 inch).
- *Observation 4:* how far the gripper moves between poses matters for translation error; how far the camera moves does not.
- *Observation 5:* the rotation error grows in step with the orientation error of each robot pose.

Random errors also shrink like $1/\sqrt N$ with $N$ robot poses. Their advice follows directly:
- make the rotations between poses large, with axes pointing in many different directions;
- keep the target close (a small target and short-range optics);
- keep the gripper's travel between poses small;
- use many robot poses;
- calibrate the camera well, and calibrate the robot itself beforehand.

![[hand_eye_noise_sensitivity.png|750]]

*A simulated check of the Tsai–Lenz observations with OpenCV's five solvers. Each point is the median of 100 trials, with noise of 0.1° / 0.5 mm on PnP and 0.02° / 0.2 mm on forward kinematics.*
- *(a) Bigger rotations, smaller error, roughly as 1/rotation (Park–Martin: 1.01° at ±5° down to 0.079° at ±60°). Tsai–Lenz is missing at ±5° because OpenCV throws away those pairs as too small.*
- *(b) As the rotation axes become parallel, the rotation-first solvers get much worse (4.8–5.4° at a 1° cone), while the joint solvers stay at ≈ 0.27°.*
- *(c) Translation error grows with camera-to-target distance (1.0 mm at 0.2 m to 4.6 mm at 1.5 m).*
- *(d) More robot poses, smaller error (0.46° at $N = 3$ to 0.11° at $N = 25$), at least as fast as $1/\sqrt N$.*

**In practice:**
- *Choosing robot poses* — use large rotations in many directions. Vary the viewing distance and aim point, so not every motion turns about the same point (the Andreff trap above). Keep the target close and filling the picture. Take many poses.
- *Taking the pictures* — take each picture only after the robot has stopped moving, and run the screw check on every pair to catch clock offsets.
- *PnP quality* — use a calibrated camera, look at the reprojection error of each image, and drop bad images ([[Camera Calibration]]).
- *Robot accuracy* — only the robot's motions enter, but errors in its kinematics still bend the answer; Tsai & Lenz recommend calibrating the robot first.
- *Checking the result* — for eye-in-hand, $T_{bg}^{(i)}\,X\,T_{ct}^{(i)}$ should give the same target pose at every robot pose, so how much it spreads is a direct error estimate (code below). Also keep some poses aside for testing.
- *Redo it when things move* — recalibrate whenever a camera mount is touched; DROID even made deliberate camera moves part of data collection (below).

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

## Learned Methods That Need No Target
---
Classic hand–eye calibration needs a target and a special routine. Learned methods use the robot itself as the target. They need no setup and can re-check the calibration while running, mostly for cameras fixed in the room (eye-to-hand):
- **DREAM** ([Lee et al., 2020](https://arxiv.org/abs/1911.09231)) — a network finds the robot's keypoints (such as joints) in a single color image; PnP, with known intrinsics and joint angles, then gives the camera's pose. It is trained only on simulated images with domain randomization, and needs no separate calibration step.
- **CtRNet** ([Lu et al., 2023](https://arxiv.org/abs/2302.14332)) — a fully differentiable pipeline mixing deep learning and geometry. It learns from unlabeled real images by segmenting the robot and comparing that with a rendered robot (differentiable rendering), and runs live inside a visual-servoing loop.
- **EasyHeC** ([Chen et al., 2023](https://arxiv.org/abs/2305.01191)) — no markers; it fits the camera pose with differentiable rendering and automatically explores joint poses, so nobody has to hand-design calibration poses.
- **Kalib** ([Tang et al., 2024](https://arxiv.org/abs/2408.10562)) — follows one chosen point on the robot using visual foundation models, gets that point's 3D position from forward kinematics, and solves PnP. It needs only the robot's kinematic chain and the chosen point: no markers, no retraining, no 3D meshes.

<span style="color:rgb(255, 255, 0)"><b>At dataset scale: DROID.</b></span> In DROID ([Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945)), data collectors calibrated the two side cameras in every scene with a checkerboard and OpenCV. The collection app also randomly asked them to move and recalibrate the cameras, to add variety. Because those calibrations "may not always be accurate", the authors checked them again afterwards (App. G) in three stages:
1. They segmented the robot with SAM (prompted with keypoints projected from forward kinematics) and compared it with a robot mask rendered from its URDF model; frames with IoU ≥ 0.7 counted as well calibrated.
2. A tuned CtRNet-X re-estimated camera-to-base poses for more scenes.
3. A modified DUSt3R recovered camera-to-camera poses.

The release has camera-to-base calibrations for ≈ 36k scenes and camera-to-camera calibrations for all scenes. <span style="color:rgb(0, 176, 240)">Calibration quality is part of the dataset: multi-view and 3D-aware policies trained on robot data inherit whatever camera-placement errors the collection left behind.</span>

## Relationship to Other Notes
---
- [[Camera Calibration]] — provides the intrinsics and the PnP target poses $T_{ct}$ at each robot pose; its hand–eye section is a short summary pointing here, and Koide–Menegatti's reprojection polish mirrors Zhang's final step.
- [[Pinhole and Perspective Projection]] — PnP, which gives every $T_{ct}$, works by inverting the perspective projection explained there.
- [[Robot Kinematics]] — forward kinematics gives $T_{bg}$; $A$ and $B$ are one screw motion (Chasles' theorem) seen from two frames; Park–Martin works in the axis × angle coordinates defined there.
- [[SVD]] — Procrustes (Park–Martin), Andreff's Kronecker system, Daniilidis's solution space and the final snap to a true rotation are all SVD computations.
- [[Linear Algebra]] — rank 2 per motion pair, Kronecker/vec rules, the smallest-eigenvector solution (Horaud–Dornaika), and why least squares fails on the rank-deficient Andreff system.
- [[VLA]] — its cross-embodiment training data includes DROID, whose per-scene camera poses come from checkerboard calibration plus learned re-checking.
- [[Behavior Cloning]] — demonstrations recorded through badly calibrated cameras teach 3D-aware policies a fixed offset; calibrated camera poses let preprocessing correct for cameras that moved between sessions.

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
