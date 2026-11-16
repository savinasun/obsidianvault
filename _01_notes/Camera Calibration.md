### [[dl_moc]]  
# **Camera Models and Calibration**

**Tags**:: #vision #math #rl
**Links**:: [[Robot Kinematics]] · [[Linear Algebra]] · [[SVD]] · [[VLA]] · [[Behavior Cloning]] · [[World Models]] · [[LDM]]


A **camera model** is the function that sends a 3D point to a pixel; **camera calibration** estimates its parameters. They split into <span style="color:rgb(255, 255, 0)">intrinsics</span> — focal lengths, principal point, skew and lens distortion, fixed by the sensor and lens — and <span style="color:rgb(255, 255, 0)">extrinsics</span> — the camera's pose $(R, t)$ in a world frame. Every metric use of images needs both: back-projecting depth into point clouds, stereo and triangulation, SfM/NeRF/3DGS, registering a camera to a robot, and any 3D-aware policy. Landmarks: the photogrammetric Direct Linear Transform ([Abdel-Aziz & Karara, 1971](https://doi.org/10.14358/PERS.81.2.103)), Tsai's two-stage method ([Tsai, 1987](https://doi.org/10.1109/JRA.1987.1087109)), Zhang's planar-checkerboard method ([Zhang, 2000](https://doi.org/10.1109/34.888718)) — the recipe OpenCV's `calibrateCamera` still follows — the Brown–Conrady distortion model ([Conrady, 1919](https://doi.org/10.1093/mnras/79.5.384); Brown, 1966), and the Kannala–Brandt fisheye model ([Kannala & Brandt, 2006](https://doi.org/10.1109/TPAMI.2006.153)).

The note follows two computer-vision courses: Stanford **CS231A** ([Hata & Savarese, *Course Notes 1: Camera Models*](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)) for the pinhole → $K$ → $[R \mid t]$ → DLT progression, and Cyrill Stachniss's Bonn photogrammetry lectures ([*Camera Extrinsics and Intrinsics*](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-12-camera-params.pdf), [*Camera Calibration: Direct Linear Transform*](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-13-DLT.pdf)) for parameter counting and degenerate configurations. Hartley & Zisserman's *Multiple View Geometry* (2nd ed., 2004, Ch. 6–7) is the reference text, Zhang's tech report ([MSR-TR-98-71](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)) covers planar calibration, and the OpenCV `calib3d` documentation fixes the conventions used in practice.

**Notation.** Bold $\mathbf X$ is a 3D point, $\mathbf X_w$ / $\mathbf X_c$ its world / camera coordinates, $(X, Y, Z)$ its camera-frame components; $(u, v)$ is a pixel; a tilde marks homogeneous coordinates, $\tilde{\mathbf x} = (u, v, 1)^\top$, $\tilde{\mathbf X} = (\mathbf X, 1)$. $(x, y) = (X/Z, Y/Z)$ are **normalized image coordinates**, $(x_d, y_d)$ their distorted version. Intrinsic matrix $K$ with entries $(f_x, f_y, c_x, c_y, s)$; extrinsics $(R, t)$; projection matrix $P = K[R \mid t]$. Frames follow [[Robot Kinematics]]: $T_{ab}$ is the pose of frame $b$ in frame $a$, so $\mathbf X_a = T_{ab}\mathbf X_b$ and $T_{ab}T_{bc} = T_{ac}$; the extrinsics are $T_{cw}$ and the camera pose is $T_{wc}$; translation is $t$ (H&Z/OpenCV) where that note writes $p$. Source dictionary: CS231A writes $M = K[R\ T]$ with focal lengths $(\alpha, \beta)$ and skew as an axis angle $\theta$; Zhang writes $A$ for $K$, $(u_0, v_0)$ for the principal point and $\gamma$ for skew; the Bonn slides use camera constant $c$, scale difference $m$, shear $s$ and principal point $(x_H, y_H)$. Clashes: $P$ is a matrix, not a probability; $\theta$ in the fisheye section is an incidence angle, not a joint angle; in $AX = XB$, $X$ is an unknown transform, not a point.

## From 3D Points to Pixels
---
<span style="color:rgb(255, 255, 0)"><b>Pinhole model.</b></span> A barrier with a tiny aperture $O$ lets one ray per scene point reach a film at distance $f$; similar triangles give (CS231A Eq. 1)

$$
(x', y') = \left(f\,\frac{X}{Z},\ f\,\frac{Y}{Z}\right).
$$

Placing a *virtual* image plane at distance $f$ in front of $O$ removes the image inversion and changes nothing else. A real lens replaces the pinhole to gather more light; under the thin-lens (paraxial, $\sin\theta \approx \theta$) approximation the mapping keeps the same form, at the price of finite depth of field and of **distortion** where the approximation breaks down off-axis (CS231A §3).

![[pinhole_camera_model.svg|600]]

*The pinhole model in the OpenCV convention. The ray from the optical center $C$ to the 3D point crosses the image plane ($z = f$) at pixel $(u, v)$; intrinsics describe the camera → pixel step and extrinsics the world → camera step. Pixel coordinates start at the top-left corner.*

<span style="color:rgb(255, 255, 0)"><b>Intrinsics.</b></span> Going from the metric image plane to pixel indices takes three more steps (CS231A §4; the Bonn "image plane → sensor" affine map): shift the origin to the image corner by the **principal point** $(c_x, c_y)$, where the optical axis pierces the sensor; convert length to pixels with densities $k, l$ (pixels per unit length), giving $f_x = fk$ and $f_y = fl$ (unequal for non-square pixels); and allow a **skew** $s$ if the sensor axes are not perpendicular. In homogeneous coordinates the division by $Z$ becomes a scale factor:

$$
Z \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \underbrace{\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}}_{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}, \qquad \text{CS231A form: } K = \begin{bmatrix} \alpha & -\alpha\cot\theta & c_x \\ 0 & \beta/\sin\theta & c_y \\ 0 & 0 & 1 \end{bmatrix},
$$

where $\theta$ is the angle between the sensor axes ($\theta = 90^\circ \Rightarrow s = 0$). $K$ has **5 dof** (2 focal lengths, 2 offsets, 1 skew). Modern sensors have $s \approx 0$ and nearly square pixels; OpenCV's pinhole $K$ omits skew altogether. Because $K$ is upper-triangular and invertible,

$$
\hat{\mathbf x} = K^{-1}\tilde{\mathbf x} = (X/Z,\ Y/Z,\ 1)^\top
$$

turns a pixel back into the **direction of the ray** through it. <span style="color:rgb(0, 176, 240)">That is what calibration buys: a calibrated camera is a device that measures directions.</span>

**Field of view.** Focal length in pixels and field of view are interchangeable given the image width $W$: $\mathrm{FoV}_x = 2\arctan\frac{W}{2f_x}$ (e.g. $W = 640$, $f_x = 320$ gives $90^\circ$).

**Pixel-center convention.** OpenCV treats integer coordinates as pixel centers; COLMAP puts the image *corner* at $(0, 0)$ and the first pixel center at $(0.5, 0.5)$. The same camera therefore has $(c_x, c_y)$ half a pixel apart in the two tools.

## Extrinsics and the Projection Matrix
---
<span style="color:rgb(255, 255, 0)"><b>Extrinsics.</b></span> A rigid transform takes world coordinates to camera coordinates (CS231A Eq. 9):

$$
\mathbf X_c = R\,\mathbf X_w + t, \qquad T_{cw} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in SE(3),
$$

6 dof (3 rotation + 3 translation; the group structure is in [[Robot Kinematics]]). The camera center $C$ satisfies $RC + t = 0$, so $C = -R^\top t$, and the **camera pose** is the inverse transform

$$
T_{wc} = T_{cw}^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix},
$$

whose rotation columns are the camera axes expressed in world coordinates (the third column, $R^\top e_3$, is the viewing direction). <span style="color:rgb(0, 176, 240)">The most common calibration bug is not numerical but a convention mismatch: a world-to-camera $T_{cw}$ used where a camera-to-world $T_{wc}$ was expected, or the right matrix in the wrong axis convention.</span> OpenCV/COLMAP extrinsics are $T_{cw}$ (COLMAP's `images.txt` stores the quaternion and translation of world → camera, with center $-R^\top t$); NeRF-style `transform_matrix` entries are $T_{wc}$ in OpenGL axes.

![[camera_frame_conventions.svg|650]]

*One physical camera under three conventions. OpenCV: x right, y down, z forward. OpenGL/Blender/NeRF/nerfstudio: y up and z backward, so the camera looks along −z. ROS REP-103 body frames: x forward, y left, z up; REP-103 adds a `_optical` frame in the OpenCV convention for cameras. Converting OpenCV → OpenGL negates the y and z columns of $T_{wc}$.*

<span style="color:rgb(255, 255, 0)"><b>Projection matrix.</b></span> Composing the two gives the full linear camera (CS231A Eq. 10; the Bonn "DLT model"):

$$
\lambda\,\tilde{\mathbf x} = K\,[R \mid t]\,\tilde{\mathbf X}_w = P\,\tilde{\mathbf X}_w, \qquad P \in \mathbb{R}^{3\times4}, \quad \lambda = Z_c .
$$

$P$ has **11 dof**: 5 intrinsic + 3 rotation + 3 translation, which is also its 12 entries minus one overall scale. Conversely, any $P$ whose left $3\times3$ block is invertible factors as $K[R \mid t]$ (the RQ step below). Its right null vector is the camera center, $P\tilde C = 0$: the center is the only point with no image.

**Back-projection** inverts everything except depth, which central projection discards. With a depth $Z$ at pixel $(u, v)$:

$$
\mathbf X_c = Z\,K^{-1}(u, v, 1)^\top, \qquad \mathbf X_w = T_{wc}\,\mathbf X_c .
$$

This is how an RGB-D frame becomes a point cloud. Check whether a sensor or simulator reports **z-depth** $Z$ or **range** $\|\mathbf X_c\|$ along the ray: they differ by the factor $\|K^{-1}\tilde{\mathbf x}\|$, which grows toward the image edges.

**Simpler cameras.** When the depth variation of a scene is small relative to its distance $z_0$, the **weak-perspective** camera replaces $1/Z$ by a constant $f/z_0$ and the last row of $P$ becomes $(0, 0, 0, 1)$, an affine map; **orthographic** projection is the limit with the center at infinity (CS231A App. B). Both are good approximations for small, distant objects and are not used for calibration.

## Lens Distortion
---
Real lenses bend rays more or less than the pinhole predicts, as a function of distance from the optical axis. <span style="color:rgb(255, 255, 0)"><b>Radial distortion</b></span> is **barrel** when magnification decreases outward (wide-angle and fisheye lenses) and **pincushion** when it increases (CS231A §3). Distortion acts on the normalized coordinates $(x, y)$, before $K$, so the full camera is: world $\xrightarrow{[R\mid t]}$ camera $\xrightarrow{\div Z}$ normalized $\xrightarrow{\text{distort}}$ $(x_d, y_d)$ $\xrightarrow{K}$ pixels.

<span style="color:rgb(255, 255, 0)"><b>Brown–Conrady model</b></span> (OpenCV default; ROS `plumb_bob`). With $r^2 = x^2 + y^2$:

$$
\begin{aligned}
x_d &= x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2), \\
y_d &= y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 xy,
\end{aligned}
\qquad u = f_x x_d + s\,y_d + c_x,\quad v = f_y y_d + c_y .
$$

The radial terms $k_i$ come from lens curvature. The **tangential** (decentering) terms $p_i$ come from lens elements that are not centered on a common axis — an effect analyzed by Conrady (1919) and put into this form by Brown (1966). OpenCV orders the coefficients $(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, \dots, s_4[, \tau_x, \tau_y]]]])$, where the optional extensions are a rational radial model ($k_4..k_6$ in a denominator), thin-prism terms $s_i$, and a tilted sensor $(\tau_x, \tau_y)$. ROS `plumb_bob` is the 5-vector $(k_1, k_2, t_1, t_2, k_3)$. Zhang (2000, §3.3) kept only $k_1, k_2$: radial distortion dominates, mostly through the first term, and more elaborate models can make the estimation numerically unstable.

![[lens_distortion_radial.png|700]]

*A straight-line grid in normalized coordinates pushed through the radial model. Barrel distortion ($k_1 < 0$) pulls the periphery in and pincushion ($k_1 > 0$) pushes it out. The displacement grows like $r^3$: invisible at the center, dominant at the corners.*

**Undistortion has no closed form.** The model maps ideal → distorted coordinates; inverting it needs an iterative solve per point (`cv2.undistortPoints`) or a precomputed remap table for whole images.

**Coefficients are resolution-independent.** They live in normalized coordinates, so a camera calibrated at 320×240 keeps the same distortion coefficients at 640×480; only $f_x, f_y, c_x, c_y$ rescale (OpenCV docs).

**Radial distortion must be monotonic.** The radial mapping of a real lens is monotonic and bijective over the image. A fit that turns non-monotonic toward the corners is a calibration failure even when it looks good near the center (OpenCV docs). It typically comes from extrapolating the polynomial into regions no calibration view covered.

<span style="color:rgb(255, 255, 0)"><b>Fisheye models.</b></span> In terms of the angle $\theta$ between an incoming ray and the optical axis, the pinhole is $r = f\tan\theta$. That diverges at $90^\circ$, so no pinhole-plus-polynomial model can represent a field of view of $180^\circ$ or more, and it badly stretches the periphery well before that. Fisheye lenses are instead designed to follow one of the classical projections (Kannala & Brandt, Eqs. 1–5): **equidistance** $r = f\theta$ (the most common), **stereographic** $r = 2f\tan(\theta/2)$ (conformal), **equisolid angle** $r = 2f\sin(\theta/2)$ (area-preserving), and **orthogonal** $r = f\sin\theta$.

![[fisheye_projection_models.png|560]]

*Image radius vs. incidence angle for the five classical projections. All agree near the axis and separate past about 30°; only the perspective curve diverges at 90°.*

**Kannala–Brandt** covers all of them with one odd polynomial,

$$
r(\theta) = k_1\theta + k_2\theta^3 + k_3\theta^5 + k_4\theta^7 + k_5\theta^9,
$$

dropping even powers because $r$ extends to an odd function; the authors found five terms enough. OpenCV's `fisheye` module is this model with the leading coefficient absorbed into $f$:

$$
\theta = \arctan r_u, \qquad \theta_d = \theta\,(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8), \qquad (x_d, y_d) = \frac{\theta_d}{r_u}(x, y),
$$

with $r_u = \sqrt{x^2 + y^2}$, so its $k_1..k_4$ are *not* Kannala–Brandt's $k_1..k_4$. Other central models first project onto a unit sphere and then perspectively from a shifted center: the **unified camera model** (calibrated from planar grids by [Mei & Rives, 2007](https://doi.org/10.1109/ROBOT.2007.364084)) and the **double sphere** model ([Usenko et al., 2018](https://arxiv.org/abs/1807.08957)), which has a closed-form inverse and is cheap to evaluate. All of these assume a single center of projection.

| Model | Parameters beyond $f, c$ | FoV $\ge 180^\circ$ | Unprojection | Implementations |
|---|---|---|---|---|
| Pinhole + Brown–Conrady | $k_1, k_2, p_1, p_2\,(, k_3)$ | no | iterative | OpenCV default, ROS `plumb_bob` |
| Kannala–Brandt (OpenCV fisheye) | $k_1..k_4$ | yes | polynomial root-finding | OpenCV `fisheye`, ROS `equidistant` |
| Unified camera model | $\xi$ | yes | closed form | Mei & Rives toolbox |
| Double sphere | $\xi, \alpha$ | yes | closed form | Usenko et al. |

## Calibration with a 3D Rig: the DLT
---
**Setup** (CS231A §5; Bonn DLT lecture). Image a rig with $n$ known, **non-coplanar** points $\mathbf X_i$ — classically two or three orthogonal checkerboards — and detect their pixels $(u_i, v_i)$. With rows $p_1^\top, p_2^\top, p_3^\top$ of $P$, $u_i = p_1^\top\tilde{\mathbf X}_i / p_3^\top\tilde{\mathbf X}_i$ (and likewise $v_i$); cross-multiplying gives two equations linear in the 12 entries of $P$:

$$
\begin{bmatrix} \tilde{\mathbf X}_i^\top & \mathbf 0^\top & -u_i\tilde{\mathbf X}_i^\top \\ \mathbf 0^\top & \tilde{\mathbf X}_i^\top & -v_i\tilde{\mathbf X}_i^\top \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \mathbf 0 \quad\Longrightarrow\quad A\,p = \mathbf 0, \quad A \in \mathbb{R}^{2n\times 12}.
$$

11 dof need $n \ge 6$ points ($5\tfrac12$ to be exact). With noise, solve $\min_p \|Ap\|^2$ s.t. $\|p\| = 1$: the right singular vector of $A$ with the smallest singular value ([[SVD]]; CS231A Eq. 13). That minimizes an *algebraic* error, so condition the data first — translate each point set to zero centroid and scale to RMS distance $\sqrt2$ (image) or $\sqrt3$ (3D) (H&Z Alg. 7.1; the argument of [Hartley, 1997](https://doi.org/10.1109/34.601246)) — and refine afterwards by minimizing reprojection error.

**Decomposing $P$.** Write $P = [M \mid p_4]$ with $M = KR$: an upper-triangular matrix times a rotation, i.e. an **RQ decomposition** of $M$ (the Bonn lecture uses the equivalent QR of $M^{-1}$). Fix signs so that $\mathrm{diag}(K) > 0$ and $\det R = +1$, scale so that $K_{33} = 1$, then read off $t = K^{-1}p_4$ and $C = -M^{-1}p_4$. CS231A (Eqs. 15–16, after Forsyth & Ponce) gives the same result in closed form from the rows $a_i^\top$ of $M$:

$$
\rho = \frac{\pm 1}{\|a_3\|}, \quad c_x = \rho^2\, a_1\!\cdot a_3, \quad c_y = \rho^2\, a_2\!\cdot a_3, \quad \cos\theta = -\frac{(a_1\times a_3)\cdot(a_2\times a_3)}{\|a_1\times a_3\|\,\|a_2\times a_3\|}, \quad \alpha = \rho^2\|a_1\times a_3\|\sin\theta, \quad \beta = \rho^2\|a_2\times a_3\|\sin\theta,
$$

with $r_3 = \rho\,a_3$, $r_1 = (a_2\times a_3)/\|a_2\times a_3\|$, $r_2 = r_3\times r_1$.

**Degenerate configurations.** If all points lie on one plane ($Z_i = 0$), the three columns of $A$ that multiply $Z$ vanish and $A$ drops to rank 8: only the plane's 8-dof homography is determined. The DLT also fails when the points and the camera center lie on a twisted cubic (Bonn: "critical surfaces"; CS231A: points on the intersection curve of two quadrics). <span style="color:rgb(0, 176, 240)">The coplanar failure is why DLT rigs need precisely machined non-coplanar targets — and it is the gap Zhang's method fills.</span>

**Known intrinsics: PnP.** When $K$ is already known only the 6 extrinsics remain and 3 points suffice (P3P has up to four solutions; a fourth point disambiguates). This *spatial resection* (Bonn) is `cv2.solvePnP`, and it is how each calibration view's pose and each hand–eye station's $T_{ct}$ are obtained.

## Zhang's Planar Method
---
[Zhang (2000)](https://doi.org/10.1109/34.888718) needs only a *planar* pattern — printed, taped to something flat — seen in a few orientations; neither the board's nor the camera's motion has to be known. This is the method in essentially every toolbox.

![[zhang_calibration_views.png|700]]

*Synthetic calibration views of one 9×6-corner checkerboard through a camera with barrel distortion. Tilted views constrain $K$; corner-coverage views put detections at large $r$, which is what constrains distortion.*

**Each view gives a homography.** Put the board at $Z = 0$ in its own frame; the third column of $R$ drops out:

$$
\lambda\,\tilde{\mathbf x} = K\,[r_1\ r_2\ r_3\ t]\,(X, Y, 0, 1)^\top = \underbrace{K\,[r_1\ r_2\ t]}_{H}\,(X, Y, 1)^\top ,
$$

so the board-to-image map is a $3\times3$ homography $H$ (8 dof), estimated per view from ≥ 4 corners by a DLT like the one above (Zhang App. A).

**Each homography gives two constraints on $K$.** Write $H = [h_1\ h_2\ h_3] = \lambda K[r_1\ r_2\ t]$. Since $r_1, r_2$ are orthonormal (Zhang Eqs. 3–4):

$$
h_1^\top B\,h_2 = 0, \qquad h_1^\top B\,h_1 = h_2^\top B\,h_2, \qquad B = K^{-\top}K^{-1}.
$$

Counting: 8 homography dof minus 6 pose dof leaves 2 per view. Geometrically, $B$ is the *image of the absolute conic*, and the plane's two circular points $r_1 \pm i\,r_2$ must project onto it (Zhang §2.4).

**Linear solve.** $B$ is symmetric; collect $b = (B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33})^\top$, so that $h_i^\top B h_j = v_{ij}^\top b$ with

$$
v_{ij} = \big(h_{i1}h_{j1},\ h_{i1}h_{j2} + h_{i2}h_{j1},\ h_{i2}h_{j2},\ h_{i3}h_{j1} + h_{i1}h_{j3},\ h_{i3}h_{j2} + h_{i2}h_{j3},\ h_{i3}h_{j3}\big)^\top .
$$

Stacking $\begin{bmatrix} v_{12}^\top \\ (v_{11} - v_{22})^\top \end{bmatrix} b = \mathbf 0$ over $n$ views gives $V b = \mathbf 0$, $V \in \mathbb{R}^{2n\times 6}$, again solved by the smallest right singular vector. *Zhang §3.1:* $n \ge 3$ views determine $b$ up to scale in general; $n = 2$ suffices if skew is fixed to $0$; $n = 1$ recovers only $f_x, f_y$, with the principal point known and zero skew.

**Closed-form intrinsics** (Zhang App. B, rewritten in this note's symbols; $\mu$ is the unknown scale of $B$, Zhang's $\lambda$):

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

Then each view's extrinsics follow from $r_1 = \lambda K^{-1}h_1$, $r_2 = \lambda K^{-1}h_2$, $r_3 = r_1\times r_2$, $t = \lambda K^{-1}h_3$ with $\lambda = 1/\|K^{-1}h_1\|$. Noise leaves $[r_1\ r_2\ r_3]$ slightly non-orthogonal, so it is projected to the nearest rotation $R = UV^\top$ via the SVD (Zhang App. C; the $\det$-corrected version is in [[Robot Kinematics]]).

**Maximum-likelihood refinement.** The closed form minimizes an algebraic error. Under i.i.d. Gaussian pixel noise the MLE minimizes **reprojection error** over all parameters jointly (Zhang Eq. 14):

$$
\min_{K,\,k_1,\,k_2,\,\{R_i,\,t_i\}}\ \sum_{i=1}^{n}\sum_{j=1}^{m} \big\|\, \mathbf x_{ij} - \breve{\mathbf x}(K, k_1, k_2, R_i, t_i, \mathbf X_j) \,\big\|^2 .
$$

Each $R_i$ is parameterized by an axis-angle 3-vector through Rodrigues' formula (the exponential coordinates of [[Robot Kinematics]]), and the problem is solved with Levenberg–Marquardt from the closed-form start with $k_1 = k_2 = 0$; Zhang reports convergence in 3–5 iterations. In his simulations (512×512 images, three planes, $\sigma = 0.5$ px noise) the errors were below 0.3% in $f_x, f_y$ but about 1 px in $(c_x, c_y)$. <span style="color:rgb(0, 176, 240)">The principal point is the least well-determined intrinsic.</span> OpenCV's `calibrateCamera` runs the same pipeline: initialize $K$ from the planar views (principal point at the image center, focal lengths by least squares), get each view's pose with `solvePnP`, then run global LM — by default exploiting the bundle-adjustment sparsity through a Schur complement.

**Degenerate views.** If the board only translates ($R_2 = R_1$), or the second plane is parallel to the first (a rotation about the plane normal), the new homography's constraints are linear combinations of the old ones (*Zhang Prop. 1*). Change the board's **tilt** between views.

**In practice:**
- *Coverage* — distortion lives at large $r$; views must put corners near the image edges and corners, or the polynomial is extrapolated there (the non-monotonic failure above).
- *Tilt* — strongly tilted views constrain $f$ and $(c_x, c_y)$; more near-fronto-parallel views add little (Prop. 1).
- *Fixed optics* — focus and zoom change the intrinsics: lock autofocus, and calibrate at the resolution and sensor mode actually used (crop modes move $c_x, c_y$).
- *Targets* — checkerboard saddle points refine to sub-pixel accuracy (`cv2.cornerSubPix`); ChArUco and AprilGrid boards add coded markers so that partial views, needed for edge coverage, stay identifiable.
- *Validation* — the returned RMS reprojection error measures fit, not correctness: compare per-view errors, check held-out views, and look at undistorted straight lines.

## Stereo and Multi-Camera Rigs
---
A rigid rig adds one fixed extrinsic per extra camera, e.g. $T_{c_2c_1}$ for a stereo pair, estimated jointly from simultaneous views of the same target (`cv2.stereoCalibrate`). The relative pose is encoded by the **essential matrix** through the epipolar constraint

$$
\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1 = 0, \qquad E = [t]_\times R, \qquad F = K_2^{-\top} E\,K_1^{-1} \quad \big(\text{pixel form: } \tilde{\mathbf x}_2^\top F\,\tilde{\mathbf x}_1 = 0\big).
$$

After rectification, depth follows from disparity $d$ and baseline $b$:

$$
Z = \frac{f_x\,b}{d}, \qquad \delta Z \approx \frac{Z^2}{f_x\,b}\,\delta d .
$$

A relative error in $f_x$ or $b$ becomes the same relative error in every depth, and matching noise grows quadratically with distance. Multi-camera robots (head plus wrist cameras) chain every camera's extrinsics through a common frame — usually the robot base, which is what hand–eye calibration provides.

## Hand–Eye Calibration
---
A robot needs its cameras' poses in *robot* coordinates. In the **eye-in-hand** setup (wrist camera) the unknown is $X = T_{gc}$, the camera's pose in the gripper frame; in the **eye-to-hand** setup (static camera) it is $X = T_{bc}$, the camera's pose in the base frame.

![[hand_eye_calibration.svg|650]]

*Eye-in-hand. At every robot pose, forward kinematics gives $T_{bg}$ and PnP on a fixed target gives $T_{ct}$. The loop through the unknown $X$ closes on the fixed but unknown $T_{bt}$; eliminating $T_{bt}$ between two poses gives $AX = XB$.*

**Derivation.** The target does not move, so for every station $i$, $T_{bg}^{(i)}\,X\,T_{ct}^{(i)} = T_{bt}$. Equating two stations $i, j$:

$$
\underbrace{\big(T_{bg}^{(j)}\big)^{-1}T_{bg}^{(i)}}_{A\ \text{(gripper motion, from FK)}}\; X \;=\; X\; \underbrace{T_{ct}^{(j)}\big(T_{ct}^{(i)}\big)^{-1}}_{B\ \text{(camera motion, from PnP)}},
\qquad
R_A R_X = R_X R_B, \quad (R_A - I)\,t_X = R_X t_B - t_A .
$$

For eye-to-hand, pass the inverted FK poses $T_{gb}$ instead (OpenCV `calibrateHandEye` docs). The $AX = XB$ formulation is due to [Shiu & Ahmad (1989)](https://doi.org/10.1109/70.88014) and [Tsai & Lenz (1989)](https://doi.org/10.1109/70.34770).

**Rotation via the Lie algebra** ([Park & Martin, 1994](https://doi.org/10.1109/70.326576)). Since $R_A = R_X R_B R_X^\top$, the two rotations are conjugate: they have the **same angle**, and their axes are related by $R_X$. With exponential coordinates $\alpha = \log(R_A)^\vee$, $\beta = \log(R_B)^\vee$ this reads $\alpha = R_X\beta$, and over many motion pairs it is an orthogonal Procrustes problem with a closed-form solution:

$$
R_X = (M^\top M)^{-1/2} M^\top, \qquad M = \sum_i \beta_i\,\alpha_i^\top ;
$$

$t_X$ then follows by stacking the translation equations into one linear least-squares problem. <span style="color:rgb(0, 176, 240)">The equal-angle property is a free sanity check: if $\|\alpha_i\| \neq \|\beta_i\|$ beyond noise, robot and camera timestamps are out of sync or a pose is wrong.</span>

**Observability.** One motion leaves rotation about its own axis undetermined, so at least **two motions with non-parallel rotation axes** (≥ 3 poses) are required, and in practice many more (OpenCV docs). A small rotation has a poorly defined axis, so stations should differ by large, well-spread rotations. OpenCV implements separable solvers (Tsai–Lenz, Park–Martin, Horaud–Dornaika) and simultaneous ones (Andreff et al.; the dual-quaternion method of [Daniilidis, 1999](https://doi.org/10.1177/02783649922066213)); the **robot-world/hand-eye** variant also solves for the unknown base-to-target transform (`cv2.calibrateRobotWorldHandEye`).

## Camera Models in Learning Systems
---
**Rays as network inputs.** To condition a network on a camera, expand $(K, T_{wc})$ into one ray per pixel: origin $o = C$, direction $d \propto R_{wc}K^{-1}\tilde{\mathbf x}$. The **Plücker embedding** $(o\times d,\ d) \in \mathbb{R}^6$ ([Sitzmann et al., 2021](https://arxiv.org/abs/2106.02634)) identifies the ray as a line, independent of which point on it is used as origin: $(o + \lambda d)\times d = o\times d$. CameraCtrl ([He et al., 2024](https://arxiv.org/abs/2404.02101)) feeds per-pixel Plücker maps into latent video diffusion models (AnimateDiff, Stable Video Diffusion), arguing that raw $K, R, t$ values are hard to correlate with pixels and have unbalanced ranges. Its ablation favors Plücker on every metric (FVD 222.1 vs. 230.1, translation error 12.98 vs. 13.88, rotation error 1.29 vs. 1.51). <span style="color:rgb(0, 176, 240)">Intrinsics and extrinsics thus reach the network as a dense, pixel-aligned field — the same role positional encodings play.</span>

**Reconstruction needs calibrated cameras.** NeRF ([Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934)) and 3D Gaussian Splatting ([Kerbl et al., 2023](https://arxiv.org/abs/2308.04079)) take $K$ and $T_{wc}$ per image, usually from COLMAP structure-from-motion ([Schönberger & Frahm, 2016](https://doi.org/10.1109/CVPR.2016.445)). SfM is calibration without a target: bundle adjustment minimizes the same reprojection error as Zhang's refinement, with the 3D points as unknowns too. Mind the conventions figure: COLMAP poses are world → camera in OpenCV axes, NeRF/nerfstudio poses are camera → world in OpenGL axes.

**Feed-forward geometry.** DUSt3R ([Wang et al., 2024](https://arxiv.org/abs/2312.14132)) regresses per-pixel **pointmaps** from image pairs "without prior information about camera calibration nor viewpoint poses", and cameras can be recovered from the pointmaps afterward. VGGT ([Wang et al., 2025](https://arxiv.org/abs/2503.11651)) directly predicts camera parameters, point maps, depth maps and point tracks in a single feed-forward pass taking under a second. For a single image, Perspective Fields ([Jin et al., 2023](https://arxiv.org/abs/2212.03239)) predict a per-pixel up-vector and latitude, and GeoCalib ([Veicht et al., 2024](https://arxiv.org/abs/2409.06704)) combines a network with geometric optimization. These matter for learning from internet video, where $K$ is never recorded.

**Robot learning.**
- *Augmentation must update $K$.* Resizing by a factor $\sigma$ scales $f_x, f_y, c_x, c_y$ by $\sigma$ (up to the half-pixel convention); cropping at offset $(x_0, y_0)$ shifts $c_x \to c_x - x_0$, $c_y \to c_y - y_0$; distortion coefficients do not change. A horizontal flip mirrors the geometry, which no rotation undoes, so 3D labels and actions must be mirrored too.
- *3D policies inherit calibration error.* Point-cloud policies back-project depth through $K$ and $T_{wc}$, so an extrinsic error becomes a rigid, systematic offset between perceived and commanded positions. Hand–eye accuracy has to match the task's tolerance.
- *2D policies absorb the camera.* An image-based [[VLA]] or [[Behavior Cloning]] policy trained on fixed mounts learns the calibration implicitly, so moving a camera is a distribution shift. The remedies are to condition on the camera (e.g. ray maps), randomize camera poses and intrinsics in simulation, or keep mounts fixed.
- *Sim-to-real.* Simulated cameras are ideal pinholes unless configured otherwise; matching or randomizing the real $K$ and distortion removes one easily avoided source of gap.

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
- [[Robot Kinematics]] — extrinsics, camera poses and $AX = XB$ live in the same $SE(3)$ / $T_{ab}$ algebra; forward kinematics supplies $A$ in hand–eye calibration; Zhang's refinement parameterizes rotations by the exponential coordinates defined there and projects onto $SO(3)$ with the same SVD step.
- [[SVD]] — every linear step here is a null-space problem: the DLT for $P$, each view's homography, and Zhang's $Vb = 0$ all take the last right singular vector; the nearest rotation is $UV^\top$.
- [[Linear Algebra]] — homogeneous coordinates and rank: the camera center is the null space of $P$, coplanar points make the DLT system rank-deficient, and RQ/QR factorization splits $P$ into $K$ and $R$.
- [[VLA]] — its multi-camera observations $o_t$ each carry intrinsics and extrinsics that image-only policies absorb implicitly; camera-placement shift and camera conditioning are calibration questions.
- [[Behavior Cloning]] — demonstrations are recorded through fixed cameras; crop/resize augmentations must keep $K$ consistent, and moving a camera is an observation-distribution shift.
- [[World Models]] — geometrically consistent action- or camera-conditioned video prediction needs a camera model; Plücker ray maps are how camera motion is fed to video generators.
- [[LDM]] — CameraCtrl adds per-pixel Plücker camera conditioning to latent video diffusion models (AnimateDiff, Stable Video Diffusion).

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
- [Shiu & Ahmad, 1989 — Calibration of Wrist-Mounted Robotic Sensors by Solving Homogeneous Transform Equations of the Form AX = XB, *IEEE Trans. Robotics and Automation* 5(1)](https://doi.org/10.1109/70.88014)
- [Tsai & Lenz, 1989 — A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration, *IEEE Trans. Robotics and Automation* 5(3)](https://doi.org/10.1109/70.34770)
- [Park & Martin, 1994 — Robot Sensor Calibration: Solving AX = XB on the Euclidean Group, *IEEE Trans. Robotics and Automation* 10(5)](https://doi.org/10.1109/70.326576)
- [Daniilidis, 1999 — Hand-Eye Calibration Using Dual Quaternions, *IJRR* 18(3)](https://doi.org/10.1177/02783649922066213)
- [OpenCV `calib3d` documentation (4.x)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — pinhole and distortion model, `calibrateCamera`, `calibrateHandEye` · [`fisheye` module](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [COLMAP output format](https://colmap.github.io/format.html) · [ROS REP-103](https://www.ros.org/reps/rep-0103.html) · [nerfstudio data conventions](https://docs.nerf.studio/quickstart/data_conventions.html)
- [Sitzmann et al., 2021 — Light Field Networks](https://arxiv.org/abs/2106.02634) · [He et al., 2024 — CameraCtrl](https://arxiv.org/abs/2404.02101)
- [Mildenhall et al., 2020 — NeRF](https://arxiv.org/abs/2003.08934) · [Kerbl et al., 2023 — 3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) · [Schönberger & Frahm, 2016 — Structure-from-Motion Revisited](https://doi.org/10.1109/CVPR.2016.445)
- [Wang et al., 2024 — DUSt3R](https://arxiv.org/abs/2312.14132) · [Wang et al., 2025 — VGGT](https://arxiv.org/abs/2503.11651) · [Jin et al., 2023 — Perspective Fields](https://arxiv.org/abs/2212.03239) · [Veicht et al., 2024 — GeoCalib](https://arxiv.org/abs/2409.06704)
