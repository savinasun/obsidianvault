### [[dl_moc]]  
# **Camera Models and Calibration**

**Tags**:: #vision #math #rl
**Links**:: [[Pinhole and Perspective Projection]] · [[Robot Kinematics]] · [[Hand-Eye Calibration]] · [[Linear Algebra]] · [[SVD]] · [[VLA]] · [[Behavior Cloning]] · [[World Models]] · [[LDM]]


A **camera model** is a formula that tells you where a 3D point will show up in the picture, in pixels. **Camera calibration** means measuring the numbers inside that formula for one real camera. There are two kinds of numbers. The <span style="color:rgb(255, 255, 0)">intrinsics</span> describe the camera itself: focal length, where the image center is, and how much the lens bends straight lines. The <span style="color:rgb(255, 255, 0)">extrinsics</span> describe where the camera is and which way it points, as a rotation $R$ and a shift $t$. You need both whenever you want to measure real distances from images: turning a depth image into a 3D point cloud, stereo, 3D reconstruction (SfM, NeRF, 3DGS), placing a camera relative to a robot, or training a policy that thinks in 3D.

The main methods, oldest first: the Direct Linear Transform ([Abdel-Aziz & Karara, 1971](https://doi.org/10.14358/PERS.81.2.103)); Tsai's two-stage method ([Tsai, 1987](https://doi.org/10.1109/JRA.1987.1087109)); and Zhang's checkerboard method ([Zhang, 2000](https://doi.org/10.1109/34.888718)), which OpenCV's `calibrateCamera` still follows. For lens bending, the standard models are Brown–Conrady ([Conrady, 1919](https://doi.org/10.1093/mnras/79.5.384); Brown, 1966) and, for fisheye lenses, Kannala–Brandt ([Kannala & Brandt, 2006](https://doi.org/10.1109/TPAMI.2006.153)).

The note follows two computer-vision courses. Stanford **CS231A** ([Hata & Savarese, *Course Notes 1: Camera Models*](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)) gives the order: pinhole → $K$ → $[R \mid t]$ → DLT. Cyrill Stachniss's Bonn photogrammetry lectures ([*Camera Extrinsics and Intrinsics*](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-12-camera-params.pdf), [*Camera Calibration: Direct Linear Transform*](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-13-DLT.pdf)) give the parameter counting and the cases where calibration fails. Hartley & Zisserman's *Multiple View Geometry* (2nd ed., 2004, Ch. 6–7) is the reference text. Zhang's tech report ([MSR-TR-98-71](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)) covers checkerboard calibration, and the OpenCV `calib3d` documentation fixes the conventions used in practice.

**Notation.** Symbols follow H&Z and OpenCV. The last column translates the course and paper symbols and flags clashes with other notes.

| Symbol | Meaning | Source equivalents · clashes |
|---|---|---|
| $\mathbf X$; $\mathbf X_w$, $\mathbf X_c$ | a 3D point; the same point in world coordinates / camera coordinates | Zhang: $M$. Bold, unlike the transform $X$ in $AX = XB$ |
| $(X, Y, Z)$ | the point's camera coordinates; $Z$ is how far in front of the camera it is (depth) | — |
| $(u, v)$ | pixel position, counted from the top-left corner | Zhang: $m = [u, v]^\top$ |
| $\tilde{\mathbf x} = (u, v, 1)^\top$, $\tilde{\mathbf X} = (\mathbf X, 1)$ | homogeneous coordinates: the usual numbers with a 1 added | Zhang: $\tilde m$, $\tilde M$ |
| $(x, y) = (X/Z,\ Y/Z)$ | **normalized image coordinates**: position on an image plane 1 unit in front of the camera; equals the ray direction $K^{-1}\tilde{\mathbf x}$ | OpenCV: $(x', y')$ |
| $(x_d, y_d)$ | the same after lens distortion | OpenCV: $(x'', y'')$; Zhang: $(\breve x, \breve y)$ |
| $K$ | intrinsic matrix (5 free numbers) | CS231A: $K$, "camera matrix"; Zhang: $A$ |
| $f_x, f_y$ | focal lengths measured in pixels | CS231A, Zhang: $\alpha, \beta$; Bonn: camera constant $c$ with scale difference $m$ |
| $(c_x, c_y)$ | principal point: the pixel where the camera's center line hits the sensor | Zhang: $(u_0, v_0)$; Bonn: $(x_H, y_H)$ |
| $s$ | skew: nonzero only if the sensor's rows and columns are not at right angles | CS231A: axis angle $\theta$, $s = -\alpha\cot\theta$; Zhang: $\gamma$; Bonn: shear $s$ |
| $(R, t) = T_{cw}$ | extrinsics, world → camera: $\mathbf X_c = R\,\mathbf X_w + t$ | CS231A: $(R, T)$; [[Robot Kinematics]] writes $p$ for translation |
| $T_{wc} = T_{cw}^{-1}$ | camera pose, camera → world; the camera sits at $C = -R^\top t$ | NeRF/nerfstudio `transform_matrix` (in OpenGL axes) |
| $T_{ab}$ | pose of frame $b$ seen from frame $a$: $\mathbf X_a = T_{ab}\mathbf X_b$, and $T_{ab}T_{bc} = T_{ac}$ | same convention as [[Robot Kinematics]] |
| $P = K[R \mid t]$ | $3\times4$ projection matrix (11 free numbers) | CS231A: $M$. A matrix, not a probability |
| $\lambda$; $\mu$ | an unknown scale factor ($\lambda = Z_c$, or a homography's scale); the unknown scale of $B$ in Zhang's method | Zhang writes $\lambda$ for both |
| $\theta$ | skew angle in CS231A's $K$; in the fisheye section, the angle between a ray and the camera's center line | never a joint angle (cf. [[Robot Kinematics]]) |
| $X$ in $AX = XB$ | unknown hand–eye transform | not a point (points are bold) |

## From 3D Points to Pixels
---
<span style="color:rgb(255, 255, 0)"><b>Pinhole model in one line.</b></span> Light from a 3D point passes through the camera center and hits the image plane at distance $f$. Similar triangles give the landing spot: **sideways position divided by distance**, times $f$ (CS231A Eq. 1):

$$
(x', y') = \left(f\,\frac{X}{Z},\ f\,\frac{Y}{Z}\right).
$$

Why a pinhole works, why lenses replace it, and what this division does to a picture (far things shrink, parallel lines meet, depth is lost, weak-perspective and orthographic shortcuts) are explained in [[Pinhole and Perspective Projection]]. Here we only need the formula.

![[pinhole_camera_model.svg|600]]

*The pinhole model in the OpenCV convention. The ray from the 3D point through the camera center $C$ crosses the image plane ($z = f$) at pixel $(u, v)$. Intrinsics describe the camera → pixel step; extrinsics describe the world → camera step. Pixels are counted from the top-left corner.*

<span style="color:rgb(255, 255, 0)"><b>Intrinsics: from image plane to pixels.</b></span> A sensor counts pixels, not millimetres, so three more steps are needed (CS231A §4; the Bonn "image plane → sensor" step):
1. **Shift** the origin to the image corner. The pixel where the camera's center line hits the sensor is the **principal point** $(c_x, c_y)$.
2. **Change units** to pixels. If a pixel is $1/k$ wide and $1/l$ tall, the focal length becomes $f_x = fk$ and $f_y = fl$ pixels. These differ only if pixels are not square.
3. Allow **skew** $s$ in case the sensor's rows and columns are not at a right angle.

In homogeneous coordinates, all of this is one matrix $K$. The division by depth becomes "divide by the last number":

$$
Z \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \underbrace{\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}}_{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}, \qquad \text{CS231A form: } K = \begin{bmatrix} \alpha & -\alpha\cot\theta & c_x \\ 0 & \beta/\sin\theta & c_y \\ 0 & 0 & 1 \end{bmatrix},
$$

where $\theta$ is the angle between the sensor axes ($\theta = 90^\circ$ means $s = 0$). $K$ has **5 free numbers**: 2 focal lengths, 2 for the principal point, 1 skew. Modern sensors have almost no skew and nearly square pixels, and OpenCV's $K$ leaves skew out altogether.

**Going backwards.** $K$ can be inverted, so a pixel can be turned back into the **direction of the ray** it came from:

$$
\hat{\mathbf x} = K^{-1}\tilde{\mathbf x} = (X/Z,\ Y/Z,\ 1)^\top .
$$

<span style="color:rgb(0, 176, 240)">That is what calibration buys you: once $K$ is known, every pixel tells you a direction in 3D.</span>

**Field of view.** Focal length in pixels and field of view carry the same information, given the image width $W$: $\mathrm{FoV}_x = 2\arctan\frac{W}{2f_x}$. For example, $W = 640$ and $f_x = 320$ gives $90^\circ$.

**Where is the center of a pixel?** OpenCV puts pixel centers at whole numbers: the first pixel's center is $(0, 0)$. COLMAP puts the image *corner* at $(0, 0)$, so the first pixel's center is $(0.5, 0.5)$. The same camera therefore has $(c_x, c_y)$ half a pixel apart in the two tools.

## Extrinsics and the Projection Matrix
---
<span style="color:rgb(255, 255, 0)"><b>Extrinsics: where the camera is.</b></span> A point is usually given in some world frame. To use the formulas above, first rotate and shift it into the camera's frame (CS231A Eq. 9):

$$
\mathbf X_c = R\,\mathbf X_w + t, \qquad T_{cw} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in SE(3).
$$

This has 6 free numbers: 3 for the rotation and 3 for the shift. The math of rotations and poses is in [[Robot Kinematics]]. The camera center $C$ is the point that lands at the camera's own origin, so $RC + t = 0$, which gives $C = -R^\top t$. The **camera pose** is the reverse transform, camera → world:

$$
T_{wc} = T_{cw}^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}.
$$

Its rotation columns are the camera's own axes written in world coordinates. The third column, $R^\top e_3$, is the direction the camera looks.

<span style="color:rgb(0, 176, 240)">The most common calibration bug is not a math error but a mix-up of conventions:</span> using world → camera ($T_{cw}$) where camera → world ($T_{wc}$) was expected, or using the right matrix with the wrong axis directions. OpenCV and COLMAP extrinsics are $T_{cw}$. COLMAP's `images.txt` stores the rotation (as a quaternion) and the shift of world → camera, with the camera at $-R^\top t$. NeRF-style `transform_matrix` entries are $T_{wc}$, in OpenGL axes.

![[camera_frame_conventions.svg|650]]

*One physical camera under three axis conventions. OpenCV: x right, y down, z forward. OpenGL/Blender/NeRF/nerfstudio: y up and z backward, so the camera looks along −z. ROS REP-103 body frames: x forward, y left, z up; REP-103 adds a `_optical` frame in the OpenCV convention for cameras. To convert OpenCV → OpenGL, flip the sign of the y and z columns of $T_{wc}$.*

<span style="color:rgb(255, 255, 0)"><b>Projection matrix: the whole camera in one matrix.</b></span> Putting the two steps together gives (CS231A Eq. 10; the Bonn "DLT model"):

$$
\lambda\,\tilde{\mathbf x} = K\,[R \mid t]\,\tilde{\mathbf X}_w = P\,\tilde{\mathbf X}_w, \qquad P \in \mathbb{R}^{3\times4}, \quad \lambda = Z_c .
$$

In plain words: multiply the world point by $P$, then divide by the last number to get the pixel. $P$ has **11 free numbers**: 5 intrinsic + 3 rotation + 3 shift. The same count comes from its 12 entries minus 1, because multiplying $P$ by any number gives the same pixels. The reverse also holds: any $P$ whose left $3\times3$ block can be inverted splits back into $K[R \mid t]$ (the RQ step below). Also, $P\tilde C = 0$: the camera center is the one point that has no picture.

<span style="color:rgb(255, 255, 0)"><b>Back-projection: from pixel plus depth to 3D.</b></span> Projection loses depth, so you cannot go back from a pixel alone. But if a depth sensor gives the depth $Z$ at pixel $(u, v)$:

$$
\mathbf X_c = Z\,K^{-1}(u, v, 1)^\top, \qquad \mathbf X_w = T_{wc}\,\mathbf X_c .
$$

This is how an RGB-D image becomes a point cloud. Check what your sensor or simulator reports. **Z-depth** is the distance along the camera's center line; **range** $\|\mathbf X_c\|$ is the straight-line distance along the ray. They differ by the factor $\|K^{-1}\tilde{\mathbf x}\|$, which grows toward the picture edges.

**Simpler cameras.** When an object is small compared with its distance, the weak-perspective and orthographic models replace "divide by each point's depth" with one fixed zoom (CS231A App. B). They are handy approximations but are not used for calibration; see [[Pinhole and Perspective Projection#Weak Perspective and Orthographic Projection|Pinhole and Perspective Projection]].

## Lens Distortion
---
Real lenses bend light a little more or a little less than a perfect pinhole would, and the error grows with distance from the picture center. This makes straight lines look curved. <span style="color:rgb(255, 255, 0)"><b>Radial distortion</b></span> comes in two kinds (CS231A §3). **Barrel** distortion shrinks the edges (wide-angle and fisheye lenses). **Pincushion** distortion stretches the edges. Distortion is applied to the normalized coordinates $(x, y)$, before $K$. So the full camera is: world $\xrightarrow{[R\mid t]}$ camera $\xrightarrow{\div Z}$ normalized $\xrightarrow{\text{distort}}$ $(x_d, y_d)$ $\xrightarrow{K}$ pixels.

<span style="color:rgb(255, 255, 0)"><b>Brown–Conrady model</b></span> (OpenCV's default; called `plumb_bob` in ROS). With $r^2 = x^2 + y^2$ (squared distance from the center):

$$
\begin{aligned}
x_d &= x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2), \\
y_d &= y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 xy,
\end{aligned}
\qquad u = f_x x_d + s\,y_d + c_x,\quad v = f_y y_d + c_y .
$$

In plain words: each point is pushed toward or away from the center by an amount that grows with its distance $r$ (the $k$ terms), plus a small sideways push (the $p$ terms).
- The **radial** terms $k_i$ come from the curved shape of the lens.
- The **tangential** terms $p_i$ come from lens pieces that are not perfectly lined up. Conrady (1919) analyzed this effect and Brown (1966) wrote it in this form.

OpenCV orders the numbers as $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, \dots, s_4[, \tau_x, \tau_y]]]])$. The optional extras are a fraction-shaped radial model ($k_4..k_6$ below the line), thin-prism terms $s_i$, and a tilted sensor $(\tau_x, \tau_y)$. ROS `plumb_bob` is the 5 numbers $(k_1, k_2, t_1, t_2, k_3)$. Zhang (2000, §3.3) kept only $k_1, k_2$: radial bending dominates, mostly through $k_1$, and more terms can make the fit unstable.

![[lens_distortion_radial.png|700]]

*A grid of straight lines pushed through the radial model. Barrel distortion ($k_1 < 0$) pulls the edges in; pincushion ($k_1 > 0$) pushes them out. The shift grows like $r^3$: invisible at the center, strong at the corners.*

**Undoing distortion has no simple formula.** The model goes from ideal → distorted. Going back needs a small iterative solve for each point (`cv2.undistortPoints`), or a precomputed lookup table for whole images.

**The distortion numbers do not depend on image size.** They live in normalized coordinates. A camera calibrated at 320×240 keeps the same distortion numbers at 640×480; only $f_x, f_y, c_x, c_y$ need rescaling (OpenCV docs).

**Real radial distortion never folds back.** For a real lens, points farther from the center always land farther out. A fitted model that bends back near the corners is a failed calibration, even if it looks fine near the center (OpenCV docs). This usually happens when no calibration image covered the corners, so the polynomial was guessing there.

<span style="color:rgb(255, 255, 0)"><b>Fisheye models.</b></span> Let $\theta$ be the angle between an incoming ray and the camera's center line. A pinhole puts that ray at distance $r = f\tan\theta$ from the center. This blows up at $90^\circ$, so a pinhole (even with distortion terms) cannot show a field of view of $180^\circ$ or more, and it stretches the edges badly well before that. Fisheye lenses are built to follow other rules instead (Kannala & Brandt, Eqs. 1–5):
- **equidistance** $r = f\theta$ (the most common);
- **stereographic** $r = 2f\tan(\theta/2)$ (keeps small angles correct);
- **equisolid angle** $r = 2f\sin(\theta/2)$ (keeps areas correct);
- **orthogonal** $r = f\sin\theta$.

![[fisheye_projection_models.png|560]]

*Distance from the picture center vs. ray angle for the five classic rules. They agree near the center and spread apart past about 30°; only the pinhole curve blows up at 90°.*

**Kannala–Brandt** covers all of these with one polynomial that uses only odd powers:

$$
r(\theta) = k_1\theta + k_2\theta^3 + k_3\theta^5 + k_4\theta^7 + k_5\theta^9 .
$$

Even powers are dropped because $r$ can be extended as an odd function, and the authors found five terms enough. OpenCV's `fisheye` module is this model with the first number folded into $f$:

$$
\theta = \arctan r_u, \qquad \theta_d = \theta\,(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8), \qquad (x_d, y_d) = \frac{\theta_d}{r_u}(x, y),
$$

with $r_u = \sqrt{x^2 + y^2}$. So OpenCV's $k_1..k_4$ are *not* Kannala–Brandt's $k_1..k_4$.

Other models first project onto a ball around the camera and then onto the image from a shifted point: the **unified camera model** (calibrated from checkerboards by [Mei & Rives, 2007](https://doi.org/10.1109/ROBOT.2007.364084)) and the **double sphere** model ([Usenko et al., 2018](https://arxiv.org/abs/1807.08957)), which can be inverted with a formula and is cheap to compute. All of these assume every ray passes through one single center.

| Model | Numbers beyond $f, c$ | FoV $\ge 180^\circ$ | Going back (pixel → ray) | Implementations |
|---|---|---|---|---|
| Pinhole + Brown–Conrady | $k_1, k_2, p_1, p_2\,(, k_3)$ | no | iterative | OpenCV default, ROS `plumb_bob` |
| Kannala–Brandt (OpenCV fisheye) | $k_1..k_4$ | yes | solve a polynomial | OpenCV `fisheye`, ROS `equidistant` |
| Unified camera model | $\xi$ | yes | formula | Mei & Rives toolbox |
| Double sphere | $\xi, \alpha$ | yes | formula | Usenko et al. |

## Calibration with a 3D Rig: the DLT
---
<span style="color:rgb(255, 255, 0)"><b>Idea.</b></span> If you know some 3D points exactly and can see where they land in the picture, you can solve for $P$ (CS231A §5; Bonn DLT lecture). Classically the points sit on two or three checkerboards at right angles, so they are **not all on one plane**. Call the rows of $P$ $p_1^\top, p_2^\top, p_3^\top$. Then $u_i = p_1^\top\tilde{\mathbf X}_i / p_3^\top\tilde{\mathbf X}_i$, and likewise for $v_i$. Multiplying out the fraction gives two equations per point that are linear in the 12 entries of $P$:

$$
\begin{bmatrix} \tilde{\mathbf X}_i^\top & \mathbf 0^\top & -u_i\tilde{\mathbf X}_i^\top \\ \mathbf 0^\top & \tilde{\mathbf X}_i^\top & -v_i\tilde{\mathbf X}_i^\top \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \mathbf 0 \quad\Longrightarrow\quad A\,p = \mathbf 0, \quad A \in \mathbb{R}^{2n\times 12}.
$$

**How many points?** 11 unknowns and 2 equations per point means at least 6 points ($5\tfrac12$ to be exact).

**Solving with noise.** Find $p$ that makes $\|Ap\|^2$ as small as possible with $\|p\| = 1$. The answer is the right singular vector of $A$ with the smallest singular value ([[SVD]]; CS231A Eq. 13). This minimizes an algebraic quantity, not the real pixel error, so first rescale the data: move each point set so its average is at zero, and scale it so the typical distance from zero is $\sqrt2$ (image) or $\sqrt3$ (3D) (H&Z Alg. 7.1; the argument of [Hartley, 1997](https://doi.org/10.1109/34.601246)). Afterwards, polish the answer by minimizing the real pixel error.

**Splitting $P$ back into $K$, $R$, $t$.** Write $P = [M \mid p_4]$ with $M = KR$. That is a triangular matrix times a rotation, which is exactly what an **RQ decomposition** produces (the Bonn lecture uses the equivalent QR of $M^{-1}$). Fix the signs so the diagonal of $K$ is positive and $\det R = +1$, scale so that $K_{33} = 1$, then read off $t = K^{-1}p_4$ and $C = -M^{-1}p_4$. CS231A (Eqs. 15–16, after Forsyth & Ponce) gives the same answer as direct formulas in the rows $a_i^\top$ of $M$:

$$
\rho = \frac{\pm 1}{\|a_3\|}, \quad c_x = \rho^2\, a_1\!\cdot a_3, \quad c_y = \rho^2\, a_2\!\cdot a_3, \quad \cos\theta = -\frac{(a_1\times a_3)\cdot(a_2\times a_3)}{\|a_1\times a_3\|\,\|a_2\times a_3\|}, \quad \alpha = \rho^2\|a_1\times a_3\|\sin\theta, \quad \beta = \rho^2\|a_2\times a_3\|\sin\theta,
$$

with $r_3 = \rho\,a_3$, $r_1 = (a_2\times a_3)/\|a_2\times a_3\|$ and $r_2 = r_3\times r_1$.

**When it fails.** If all points lie on one plane ($Z_i = 0$), the three columns of $A$ that multiply $Z$ become zero and $A$ drops to rank 8: you can only recover the plane-to-picture mapping (a homography, 8 numbers), not the full camera. It also fails when the points and the camera center lie on a special curve, a twisted cubic (Bonn: "critical surfaces"; CS231A: points on the curve where two quadric surfaces meet). <span style="color:rgb(0, 176, 240)">The one-plane failure is why DLT rigs need carefully built non-flat targets, and it is exactly the gap Zhang's method fills.</span>

**If $K$ is already known: PnP.** Only the 6 extrinsic numbers remain, and 3 points are enough. Three points can give up to four answers; a fourth point picks the right one. This is called **PnP** (Perspective-n-Point; "spatial resection" in the Bonn lectures) and is `cv2.solvePnP`. It is how each calibration image's pose, and each hand–eye station's $T_{ct}$, is found.

## Zhang's Planar Method
---
[Zhang (2000)](https://doi.org/10.1109/34.888718) needs only a *flat* pattern: a checkerboard printed on paper and taped to something flat, photographed from a few angles. You do not need to know how the board or the camera moved. This is the method in almost every toolbox.

![[zhang_calibration_views.png|700]]

*Simulated calibration images of one checkerboard (9×6 inner corners) through a camera with barrel distortion. Tilted views pin down $K$; views that put corners near the picture edges pin down the distortion.*

<span style="color:rgb(255, 255, 0)"><b>Step 1: each image gives a homography.</b></span> Put the board on the plane $Z = 0$ of its own frame. Then the third column of $R$ never gets used:

$$
\lambda\,\tilde{\mathbf x} = K\,[r_1\ r_2\ r_3\ t]\,(X, Y, 0, 1)^\top = \underbrace{K\,[r_1\ r_2\ t]}_{H}\,(X, Y, 1)^\top .
$$

So the map from board to picture is a $3\times3$ matrix $H$ called a **homography** (8 free numbers). It is found for each image from at least 4 corners, with a DLT like the one above (Zhang App. A).

<span style="color:rgb(255, 255, 0)"><b>Step 2: each homography gives two clues about $K$.</b></span> Write $H = [h_1\ h_2\ h_3] = \lambda K[r_1\ r_2\ t]$. The first two columns of a rotation are perpendicular and have equal length. Translating that into $H$ gives (Zhang Eqs. 3–4):

$$
h_1^\top B\,h_2 = 0, \qquad h_1^\top B\,h_1 = h_2^\top B\,h_2, \qquad B = K^{-\top}K^{-1}.
$$

Why two clues per image: $H$ has 8 numbers, the board's pose uses 6, so 2 are left over to tell us about $K$. (Geometrically, $B$ is the *image of the absolute conic*, and the plane's two "circular points" $r_1 \pm i\,r_2$ must land on it; Zhang §2.4.)

<span style="color:rgb(255, 255, 0)"><b>Step 3: solve a linear system.</b></span> $B$ is symmetric, so it has only 6 different entries. Collect them as $b = (B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33})^\top$. Then $h_i^\top B h_j = v_{ij}^\top b$ with

$$
v_{ij} = \big(h_{i1}h_{j1},\ h_{i1}h_{j2} + h_{i2}h_{j1},\ h_{i2}h_{j2},\ h_{i3}h_{j1} + h_{i1}h_{j3},\ h_{i3}h_{j2} + h_{i2}h_{j3},\ h_{i3}h_{j3}\big)^\top .
$$

Stack $\begin{bmatrix} v_{12}^\top \\ (v_{11} - v_{22})^\top \end{bmatrix} b = \mathbf 0$ for all $n$ images to get $V b = \mathbf 0$, with $V \in \mathbb{R}^{2n\times 6}$. Solve it again with the smallest right singular vector. *Zhang §3.1:* 3 or more images fix $b$ (up to scale); 2 are enough if skew is set to 0; 1 image gives only $f_x, f_y$, assuming the principal point is known and skew is 0.

<span style="color:rgb(255, 255, 0)"><b>Step 4: read off $K$</b></span> (Zhang App. B, in this note's symbols; $\mu$ is the unknown scale of $B$, which Zhang calls $\lambda$):

$$
\begin{aligned}
c_y &= \frac{B_{12}B_{13} - B_{11}B_{23}}{B_{11}B_{22} - B_{12}^2}, &
\mu &= B_{33} - \frac{B_{13}^2 + c_y\,(B_{12}B_{13} - B_{11}B_{23})}{B_{11}}, &
f_x &= \sqrt{\mu / B_{11}}, \\
f_y &= \sqrt{\frac{\mu\,B_{11}}{B_{11}B_{22} - B_{12}^2}}, &
s &= -\frac{B_{12}\,f_x^2 f_y}{\mu}, &
c_x &= \frac{s\,c_y}{f_y} - \frac{B_{13}\,f_x^2}{\mu}.
\end{aligned}
$$

<span style="color:rgb(255, 255, 0)"><b>Step 5: each image's pose.</b></span> $r_1 = \lambda K^{-1}h_1$, $r_2 = \lambda K^{-1}h_2$, $r_3 = r_1\times r_2$, $t = \lambda K^{-1}h_3$, with $\lambda = 1/\|K^{-1}h_1\|$. Because of noise, $[r_1\ r_2\ r_3]$ is not quite a rotation, so replace it with the nearest rotation, $R = UV^\top$ from the SVD (Zhang App. C; the version that also fixes the determinant is in [[Robot Kinematics]]).

<span style="color:rgb(255, 255, 0)"><b>Step 6: polish everything at once.</b></span> The steps above minimize algebraic quantities, not real pixel error. If pixel noise is random and Gaussian, the best estimate minimizes the total **reprojection error**: the distance between where each corner was seen and where the model predicts it (Zhang Eq. 14):

$$
\min_{K,\,k_1,\,k_2,\,\{R_i,\,t_i\}}\ \sum_{i=1}^{n}\sum_{j=1}^{m} \big\|\, \mathbf x_{ij} - \breve{\mathbf x}(K, k_1, k_2, R_i, t_i, \mathbf X_j) \,\big\|^2 .
$$

Each rotation $R_i$ is stored as 3 numbers (an axis times an angle, through Rodrigues' formula; these are the exponential coordinates of [[Robot Kinematics]]). The problem is solved with the Levenberg–Marquardt method, starting from steps 1–5 with $k_1 = k_2 = 0$; Zhang reports it converges in 3–5 iterations. In his simulations (512×512 images, three planes, 0.5 px noise), $f_x, f_y$ came out within 0.3%, but $(c_x, c_y)$ was off by about 1 px. <span style="color:rgb(0, 176, 240)">The principal point is the hardest intrinsic to pin down.</span>

OpenCV's `calibrateCamera` follows the same recipe. It guesses $K$ from the flat views (principal point at the image center, focal lengths by least squares), finds each view's pose with `solvePnP`, then runs Levenberg–Marquardt over everything. By default it speeds this up with a Schur complement, using the sparse structure of bundle adjustment.

**Views that teach nothing new.** If the board only slides without turning ($R_2 = R_1$), or the second view's plane is parallel to the first (only spun around its own normal), the new clues repeat the old ones (*Zhang Prop. 1*). Tilt the board differently in each view.

**In practice:**
- *Cover the corners* — distortion is strongest far from the center, so some views must put corners near the picture edges; otherwise the polynomial guesses there (the "folds back" failure above).
- *Tilt the board* — strongly tilted views pin down $f$ and $(c_x, c_y)$; more face-on views add little (Prop. 1).
- *Lock the optics* — focus and zoom change the intrinsics. Turn off autofocus, and calibrate at the same resolution and sensor mode you will use (crop modes move $c_x, c_y$).
- *Choose a good target* — checkerboard corners can be located to a fraction of a pixel (`cv2.cornerSubPix`). ChArUco and AprilGrid boards add coded markers, so a board that is only partly in view can still be recognized, which helps when covering the edges.
- *Check the result* — the RMS reprojection error that calibration reports measures fit, not correctness. Compare the error per image, test on images not used for calibration, and check that straight lines look straight after undistortion.

## Stereo and Multi-Camera Rigs
---
Two or more cameras bolted together add one fixed extrinsic per extra camera, for example $T_{c_2c_1}$ for a stereo pair. They are calibrated together from pictures of the same target taken at the same time (`cv2.stereoCalibrate`). For two cameras, the relative pose is captured by the **essential matrix** $E$. It expresses a simple fact: a point seen in camera 1 must lie on a known line in camera 2 (the **epipolar constraint**):

$$
\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1 = 0, \qquad E = [t]_\times R, \qquad F = K_2^{-\top} E\,K_1^{-1} \quad \big(\text{in pixels: } \tilde{\mathbf x}_2^\top F\,\tilde{\mathbf x}_1 = 0\big).
$$

After the two images are aligned row by row (rectified), depth comes from **disparity** $d$ (how far a point shifts between the two images) and the **baseline** $b$ (the distance between the cameras):

$$
Z = \frac{f_x\,b}{d}, \qquad \delta Z \approx \frac{Z^2}{f_x\,b}\,\delta d .
$$

In plain words: a 1% error in $f_x$ or $b$ gives a 1% error in every depth, and matching noise hurts more and more with distance (it grows like $Z^2$). Robots with several cameras (head plus wrist) connect every camera to one common frame, usually the robot base, which is what [[Hand-Eye Calibration]] provides.

## Hand–Eye Calibration
---
A robot needs to know where its cameras are in *robot* coordinates. For a wrist camera (**eye-in-hand**) the unknown is $X = T_{gc}$, the camera's pose in the gripper frame. For a camera fixed in the room (**eye-to-hand**) it is $X = T_{bc}$, its pose in the robot base frame. At several robot poses, record the gripper pose $T_{bg}$ (from forward kinematics) and the target's pose in the camera $T_{ct}$ (from PnP, above). Removing the unknown target pose between two robot poses $i, j$ gives

$$
\underbrace{\big(T_{bg}^{(j)}\big)^{-1}T_{bg}^{(i)}}_{A\ \text{(gripper motion, from FK)}}\; X \;=\; X\; \underbrace{T_{ct}^{(j)}\big(T_{ct}^{(i)}\big)^{-1}}_{B\ \text{(camera motion, from PnP)}},
\qquad
R_A R_X = R_X R_B, \quad (R_A - I)\,t_X = R_X t_B - t_A .
$$

This needs at least two motions whose rotation axes are not parallel (so at least 3 robot poses).

![[hand_eye_calibration.svg|650]]

*Eye-in-hand. The loop through the unknown $X$ closes on the fixed but unknown target pose $T_{bt}$; removing it between two robot poses gives $AX = XB$.*

The full story is in [[Hand-Eye Calibration]]: the eye-to-hand case, which motions are good enough, the Tsai–Lenz / Park–Martin / Horaud–Dornaika / Andreff / Daniilidis solvers, robot-world calibration ($AX = YB$), accuracy tips, and learned methods that need no target.

## Camera Models in Learning Systems
---
<span style="color:rgb(255, 255, 0)"><b>Rays as network inputs.</b></span> To tell a network about the camera, turn $(K, T_{wc})$ into one ray per pixel: it starts at the camera center $o = C$ and points along $d \propto R_{wc}K^{-1}\tilde{\mathbf x}$. The **Plücker embedding** $(o\times d,\ d) \in \mathbb{R}^6$ ([Sitzmann et al., 2021](https://arxiv.org/abs/2106.02634)) describes the ray as a line, so it does not matter which point on the line you call the start: $(o + \lambda d)\times d = o\times d$.

CameraCtrl ([He et al., 2024](https://arxiv.org/abs/2404.02101)) feeds these per-pixel Plücker maps into latent video diffusion models (AnimateDiff, Stable Video Diffusion). Its argument is that raw $K, R, t$ numbers are hard for a network to connect to pixels, and their value ranges are very uneven. Its ablation favors Plücker on every metric: FVD 222.1 vs. 230.1, translation error 12.98 vs. 13.88, rotation error 1.29 vs. 1.51. <span style="color:rgb(0, 176, 240)">The camera reaches the network as a dense map lined up with the pixels, much like a positional encoding.</span>

<span style="color:rgb(255, 255, 0)"><b>3D reconstruction needs calibrated cameras.</b></span> NeRF ([Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934)) and 3D Gaussian Splatting ([Kerbl et al., 2023](https://arxiv.org/abs/2308.04079)) need $K$ and $T_{wc}$ for every image. These usually come from COLMAP structure-from-motion ([Schönberger & Frahm, 2016](https://doi.org/10.1109/CVPR.2016.445)). Structure-from-motion is calibration without a target: bundle adjustment minimizes the same reprojection error as Zhang's step 6, but the 3D points are unknown too. Watch the conventions from the figure above: COLMAP poses are world → camera in OpenCV axes; NeRF/nerfstudio poses are camera → world in OpenGL axes.

<span style="color:rgb(255, 255, 0)"><b>Networks that predict geometry directly.</b></span>
- DUSt3R ([Wang et al., 2024](https://arxiv.org/abs/2312.14132)) predicts a 3D point for every pixel (a **pointmap**) from a pair of images, "without prior information about camera calibration nor viewpoint poses"; the cameras can be recovered from the pointmaps afterward.
- VGGT ([Wang et al., 2025](https://arxiv.org/abs/2503.11651)) directly predicts camera parameters, point maps, depth maps and point tracks in a single pass taking under a second.
- From a single image, Perspective Fields ([Jin et al., 2023](https://arxiv.org/abs/2212.03239)) predict, for each pixel, which way is up and its latitude; GeoCalib ([Veicht et al., 2024](https://arxiv.org/abs/2409.06704)) combines a network with geometric optimization.

These matter for learning from internet video, where $K$ is never recorded.

<span style="color:rgb(255, 255, 0)"><b>Robot learning.</b></span>
- *Image augmentation must update $K$.* Resizing by a factor $\sigma$ multiplies $f_x, f_y, c_x, c_y$ by $\sigma$ (up to the half-pixel convention). Cropping at offset $(x_0, y_0)$ shifts $c_x \to c_x - x_0$ and $c_y \to c_y - y_0$. Distortion numbers do not change. A left-right flip mirrors the scene, which no rotation can undo, so any 3D labels and actions must be mirrored too.
- *3D policies inherit calibration errors.* Point-cloud policies turn depth into 3D through $K$ and $T_{wc}$, so a wrong extrinsic shifts everything the robot sees by a fixed offset, and its actions miss by that offset. [[Hand-Eye Calibration|Hand–eye]] accuracy has to match how precise the task is.
- *2D policies quietly learn the camera.* An image-based [[VLA]] or [[Behavior Cloning]] policy trained with cameras in fixed spots learns the calibration implicitly, so moving a camera changes what the policy sees (a distribution shift). The fixes: give the policy the camera (for example ray maps), randomize camera poses and intrinsics in simulation, or keep the mounts fixed.
- *Sim-to-real.* Simulated cameras are perfect pinholes unless you set them up otherwise. Matching or randomizing the real $K$ and distortion removes one easy source of the sim-to-real gap.

```python
import numpy as np

def project(X_w, K, R, t, dist=(0, 0, 0, 0, 0)):           # world points (N,3) -> pixels (N,2)
    k1, k2, p1, p2, k3 = dist                                # OpenCV coefficient order
    X_c = X_w @ R.T + t                                      # extrinsics: X_c = R X_w + t
    x, y = X_c[:, 0] / X_c[:, 2], X_c[:, 1] / X_c[:, 2]      # normalized image coordinates
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    yd = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    return np.stack([xd, yd, np.ones_like(xd)], -1) @ K[:2].T   # (fx xd + s yd + cx, fy yd + cy)

def backproject(depth, K, T_wc):                             # undistorted z-depth (H,W) -> world points
    H, W = depth.shape
    v, u = np.mgrid[:H, :W]                                  # OpenCV: integer coords are pixel centers
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    X_c = depth.reshape(-1, 1) * (pix @ np.linalg.inv(K).T)  # X_c = Z K^{-1} [u, v, 1]
    return X_c @ T_wc[:3, :3].T + T_wc[:3, 3]

def plucker_rays(K, T_wc, H, W):                             # per-pixel (o x d, d), shape (H, W, 6)
    v, u = np.mgrid[:H, :W]
    d = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T @ T_wc[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(T_wc[:3, 3], d.shape)                # camera center C = -R^T t
    return np.concatenate([np.cross(o, d), d], -1)
```

```python
import cv2, numpy as np                                      # OpenCV 4.x

# 1. intrinsics + distortion from checkerboard views (planar init -> solvePnP -> global LM)
pattern, square = (9, 6), 0.025                              # inner corners, square size [m]
board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
board[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * square   # board plane Z = 0
obj_pts, img_pts = [], []
for img in gray_images:
    found, corners = cv2.findChessboardCorners(img, pattern)
    if found:
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(board); img_pts.append(corners)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray_images[0].shape[::-1], None, None)
# rms = RMS reprojection error [px]; (rvecs[i], tvecs[i]) = T_ct of view i (axis-angle, translation)

# 2. hand-eye, eye-in-hand: X = T_gc from FK poses T_bg^(i) and PnP poses T_ct^(i)
R_gc, t_gc = cv2.calibrateHandEye(R_bg, t_bg, R_ct, t_ct, method=cv2.CALIB_HAND_EYE_PARK)
```

## Relationship to Other Notes
---
- [[Pinhole and Perspective Projection]] — explains the pinhole formula this note starts from, plus what perspective does to pictures and the weak-perspective / orthographic shortcuts; this note's first section is a short summary of it.
- [[Robot Kinematics]] — extrinsics, camera poses and $AX = XB$ use the same $SE(3)$ / $T_{ab}$ math; forward kinematics supplies $A$ in hand–eye calibration; Zhang's step 6 stores rotations as the exponential coordinates defined there and finds the nearest rotation with the same SVD step.
- [[SVD]] — every linear step here is "find the null space": the DLT for $P$, each view's homography and Zhang's $Vb = 0$ all take the last right singular vector; the nearest rotation is $UV^\top$.
- [[Linear Algebra]] — homogeneous coordinates and rank: the camera center is the null space of $P$, points on one plane make the DLT system lose rank, and RQ/QR factorization splits $P$ into $K$ and $R$.
- [[VLA]] — each of its cameras has its own intrinsics and extrinsics, which image-only policies learn implicitly; moving a camera and conditioning on the camera are calibration questions.
- [[Behavior Cloning]] — demonstrations are recorded through fixed cameras; crop and resize augmentations must keep $K$ consistent, and moving a camera shifts what the policy sees.
- [[World Models]] — video prediction that respects geometry (conditioned on actions or camera motion) needs a camera model; Plücker ray maps are how camera motion is fed to video generators.
- [[LDM]] — CameraCtrl adds per-pixel Plücker camera conditioning to latent video diffusion models (AnimateDiff, Stable Video Diffusion).
- [[Hand-Eye Calibration]] — places each camera relative to the robot from PnP poses and forward kinematics ($AX = XB$); split out of this note's hand–eye section, which remains as a short summary.

## References
---
- [Hata & Savarese — CS231A Course Notes 1: Camera Models, Stanford](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf) — pinhole, intrinsics with skew, extrinsics, DLT calibration and decomposition, distortion, weak perspective
- Stachniss — Photogrammetry & Robotics lectures, Univ. Bonn: [Camera Extrinsics and Intrinsics](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-12-camera-params.pdf) · [Camera Calibration: Direct Linear Transform](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-13-DLT.pdf)
- Hartley & Zisserman — *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge Univ. Press, 2004 — Ch. 6 Camera Models, Ch. 7 Computation of the Camera Matrix $P$
- [Zhang, 2000 — A Flexible New Technique for Camera Calibration, *IEEE TPAMI* 22(11)](https://doi.org/10.1109/34.888718) · [tech report MSR-TR-98-71](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
- [Abdel-Aziz & Karara, 1971 — Direct Linear Transformation from Comparator Coordinates into Object Space Coordinates in Close-Range Photogrammetry (reprinted *PE&RS* 81(2), 2015)](https://doi.org/10.14358/PERS.81.2.103)
- [Tsai, 1987 — A Versatile Camera Calibration Technique for High-Accuracy 3D Machine Vision Metrology Using Off-the-Shelf TV Cameras and Lenses, *IEEE J. Robotics and Automation* 3(4)](https://doi.org/10.1109/JRA.1987.1087109)
- [Conrady, 1919 — Decentred Lens-Systems, *MNRAS* 79(5)](https://doi.org/10.1093/mnras/79.5.384)
- Brown, 1966 — Decentering Distortion of Lenses, *Photogrammetric Engineering* 32(3), 444–462
- [Kannala & Brandt, 2006 — A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses, *IEEE TPAMI* 28(8)](https://doi.org/10.1109/TPAMI.2006.153)
- [Mei & Rives, 2007 — Single View Point Omnidirectional Camera Calibration from Planar Grids, ICRA](https://doi.org/10.1109/ROBOT.2007.364084)
- [Usenko et al., 2018 — The Double Sphere Camera Model](https://arxiv.org/abs/1807.08957)
- [Hartley, 1997 — In Defense of the Eight-Point Algorithm, *IEEE TPAMI* 19(6)](https://doi.org/10.1109/34.601246)
- Hand–eye calibration references (Shiu & Ahmad, Tsai & Lenz, Park & Martin, Horaud & Dornaika, Andreff et al., Daniilidis) moved to [[Hand-Eye Calibration]]
- [OpenCV `calib3d` documentation (4.x)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — pinhole and distortion model, `calibrateCamera`, `calibrateHandEye` · [`fisheye` module](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [COLMAP output format](https://colmap.github.io/format.html) · [ROS REP-103](https://www.ros.org/reps/rep-0103.html) · [nerfstudio data conventions](https://docs.nerf.studio/quickstart/data_conventions.html)
- [Sitzmann et al., 2021 — Light Field Networks](https://arxiv.org/abs/2106.02634) · [He et al., 2024 — CameraCtrl](https://arxiv.org/abs/2404.02101)
- [Mildenhall et al., 2020 — NeRF](https://arxiv.org/abs/2003.08934) · [Kerbl et al., 2023 — 3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) · [Schönberger & Frahm, 2016 — Structure-from-Motion Revisited](https://doi.org/10.1109/CVPR.2016.445)
- [Wang et al., 2024 — DUSt3R](https://arxiv.org/abs/2312.14132) · [Wang et al., 2025 — VGGT](https://arxiv.org/abs/2503.11651) · [Jin et al., 2023 — Perspective Fields](https://arxiv.org/abs/2212.03239) · [Veicht et al., 2024 — GeoCalib](https://arxiv.org/abs/2409.06704)
