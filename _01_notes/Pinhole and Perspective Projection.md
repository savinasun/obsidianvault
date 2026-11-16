### [[dl_moc]]  
# **Pinhole and Perspective Projection**

**Tags**:: #vision #math
**Links**:: [[Camera Calibration]] · [[Hand-Eye Calibration]] · [[Linear Algebra]] · [[Robot Kinematics]]


A camera turns the 3D world into a flat 2D picture. The simplest model of how it does this is the **pinhole camera**: light from each point in the world passes through one tiny hole and lands on the film. The rule that decides *where* each point lands is called **perspective projection**. In one sentence: a point's position on the picture is its sideways position divided by its distance. That single division explains why far things look small, why railway tracks seem to meet, and why one photo cannot tell you how far away something is. Every camera model in [[Camera Calibration]] starts from this rule and then adds corrections.

The note follows Stanford **CS231A** ([Hata & Savarese, *Course Notes 1: Camera Models*](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf), §2–4 and Appendix B) and Cyrill Stachniss's Bonn lecture *[Camera Extrinsics and Intrinsics](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-12-camera-params.pdf)* ("ideal perspective projection"). Hartley & Zisserman's *Multiple View Geometry* (2nd ed., 2004, Ch. 6) is the reference text.

**Notation.** The camera sits at the origin and looks along the $Z$ axis (OpenCV convention: $X$ right, $Y$ down, $Z$ forward).

| Symbol | Meaning | Source equivalents · clashes |
|---|---|---|
| $O$ | the pinhole, also called the camera center or optical center | H&Z: $C$; Bonn: projection center |
| $(X, Y, Z)$ | a 3D point, measured from the camera; $Z$ is how far in front of the camera it is (its **depth**) | CS231A: $(x, y, z)$ |
| $f$ | focal length: the distance from the pinhole to the image plane | Bonn: camera constant $c$ |
| $(x', y')$ | where the point lands on the image plane, in the same units as $f$ | CS231A: $P' = (x', y')$ |
| $(u, v)$ | the same position measured in pixels | — |
| $K$ | the small matrix that turns image-plane positions into pixels (see [[Camera Calibration]]) | Zhang: $A$ |
| $\tilde{\mathbf x}$, $\tilde{\mathbf X}$ | homogeneous coordinates: the usual numbers with a 1 added at the end | — |
| $\mathbf d$ | the direction of a family of parallel lines | not the disparity $d$ of stereo |

## The Pinhole Camera
---
<span style="color:rgb(255, 255, 0)"><b>Why a tiny hole makes a picture.</b></span> Every point on an object sends light in all directions. If you hold a piece of film in front of the object, each spot on the film gets light from *every* point, so you see only a blur. Put a wall with a tiny hole between them. Now each spot on the film can only be reached by the one ray that passes through the hole. Each object point therefore lights up exactly one film spot, and a picture appears (CS231A §2).

**The picture is upside down.** Rays cross at the hole, so the top of the object lands at the bottom of the film. To avoid the flip in the math, we pretend the film sits the same distance $f$ *in front of* the hole instead. This **virtual image plane** gives the same picture, just right side up.

![[pinhole_camera_model.svg|600]]

*The 3D pinhole model. The ray from the 3D point through the camera center $C$ crosses the image plane at distance $f$. Where it crosses is where the point appears in the picture.*

<span style="color:rgb(255, 255, 0)"><b>Small hole or big hole?</b></span> A smaller hole gives a sharper picture but lets in very little light, so the picture is dark. A bigger hole lets in more light but each film spot now gets light from several object points, so the picture blurs (CS231A §2). Real cameras solve this with a **lens**: it gathers lots of light and still bends all rays from one point onto one spot. The catch is that a lens does this perfectly only for objects at one distance (the *focus distance*), and it bends rays a little wrongly near the picture edges. That wrong bending is **lens distortion**, covered in [[Camera Calibration]]. For everything about *where* points land, a lens camera behaves like a pinhole camera (CS231A §3).

## Perspective Projection: Divide by Distance
---
<span style="color:rgb(255, 255, 0)"><b>The rule.</b></span> Look at the camera from the side. The point, the pinhole and the image form two triangles with the same shape (**similar triangles**). In the big triangle, the point is $Y$ to the side and $Z$ away. In the small triangle, the image is $y'$ to the side and $f$ away. Same shape means the same ratio, so $y'/f = Y/Z$. Doing the same for $X$ gives (CS231A Eq. 1)

$$
x' = f\,\frac{X}{Z}, \qquad y' = f\,\frac{Y}{Z}.
$$

In plain words: **image position = focal length × (sideways position ÷ distance).**

![[pinhole_similar_triangles.svg|700]]

*Side view of two trees of the same height. The far tree is twice as far away, so its image is half as tall. The dashed rays pass through the pinhole and land on the film (upside down) or on the virtual image plane (right side up).*

**From image units to pixels.** A digital camera measures positions in pixels, counted from the top-left corner. So we multiply by how many pixels fit in one unit of length, and shift by where the optical axis hits the sensor (the **principal point** $(c_x, c_y)$):

$$
u = f_x\,\frac{X}{Z} + c_x, \qquad v = f_y\,\frac{Y}{Z} + c_y .
$$

Here $f_x$ and $f_y$ are the focal length measured in pixels. These numbers, collected in the matrix $K$, are the camera's **intrinsics**. Finding them is the job of [[Camera Calibration]].

## What Perspective Does to a Picture
---
All of the following come from the one division by $Z$.

**1. Far things look smaller.** Image size is (real size) × $f$ ÷ (distance). Twice as far means half as big. This is why a person far down a road looks tiny.

**2. Parallel lines meet at a vanishing point.** Take railway tracks: two parallel lines running away from you. As you follow them farther, the gap between them stays the same in the world, but you divide it by a bigger and bigger $Z$. So the gap in the picture shrinks toward zero, and the two lines meet at one point, the **vanishing point**. It depends only on the lines' direction $\mathbf d = (d_X, d_Y, d_Z)$, not on where the lines are:

$$
\text{vanishing point} = \left(f\,\frac{d_X}{d_Z},\ f\,\frac{d_Y}{d_Z}\right).
$$

To see why, move along a line from a point $\mathbf X_0$ in direction $\mathbf d$. The position is $\mathbf X_0 + s\,\mathbf d$. As $s$ grows, the starting point stops mattering, and $X/Z$ tends to $d_X/d_Z$. Lines that run parallel to the picture ($d_Z = 0$) never meet: they stay parallel in the picture.

**3. The horizon is where the ground's vanishing points live.** Every set of parallel lines lying flat on the ground has its own vanishing point. All of these points sit on one line in the picture: the **horizon**. For a camera tilted down by an angle $\varphi$, the horizon sits $f\tan\varphi$ above the picture center.

![[perspective_vs_orthographic.png|800]]

*The same ground grid and three boxes, seen three ways.*
- *(a) A close camera with a wide lens: far boxes shrink, and the grid lines running away from us meet at a vanishing point on the horizon.*
- *(b) The camera moved 10× farther back and zoomed in 10×: perspective becomes weak and the lines are almost parallel.*
- *(c) Orthographic projection: no shrinking at all, and parallel lines stay parallel.*

**4. Straight lines stay straight, but lengths and angles change.** A straight edge in the world is still a straight edge in the picture (if the lens has no distortion). But equal lengths do not stay equal: the far half of a road looks shorter than the near half. Right angles do not stay right angles either.

**5. Depth is lost.** Every point along the same ray through the pinhole lands on the *same* pixel: $(X, Y, Z)$ and $(2X, 2Y, 2Z)$ give the same $X/Z$ and $Y/Z$. So a single photo cannot tell a small toy car up close from a real car far away. To get depth back you need extra information: a second camera (stereo), a depth sensor, motion, or knowing the true size of something. Going backwards from a pixel *with* a known depth $Z$ is easy (see back-projection in [[Camera Calibration]]).

<span style="color:rgb(0, 176, 240)">The amount of perspective depends on where the camera stands, not on the lens.</span> Perspective is strong when the scene's depth changes a lot compared with its distance from the camera. Zooming in only enlarges the middle of the picture; it does not change which things look bigger than others. That is why panel (b) looks "flat": the camera moved back, and the zoom just made the scene fill the frame again.

## Homogeneous Coordinates
---
Dividing by $Z$ is not a matrix multiplication, which makes the math awkward. **Homogeneous coordinates** fix this with a trick: add a 1 at the end of each point, and agree that a vector and any multiple of it mean the same point (CS231A §4.1.2).

- A picture point $(u, v)$ becomes $(u, v, 1)$. To go back, divide by the last number: $(a, b, w) \to (a/w, b/w)$.
- A 3D point $(X, Y, Z)$ becomes $(X, Y, Z, 1)$.

Now projection is a plain matrix product, and the division happens only at the very end:

$$
\begin{bmatrix} a \\ b \\ w \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} f_x X + c_x Z \\ f_y Y + c_y Z \\ Z \end{bmatrix}, \qquad (u, v) = \left(\frac{a}{w},\ \frac{b}{w}\right).
$$

Check: $a/w = f_x X/Z + c_x$, which is the pixel formula above.

**Points at infinity.** A homogeneous vector whose last number is 0, such as $(d_X, d_Y, d_Z, 0)$, is a point infinitely far away in direction $\mathbf d$. Projecting it gives $K\mathbf d$, which is exactly the vanishing point. So vanishing points are just the pictures of points at infinity.

## Focal Length and Field of View
---
**Field of view** (FoV) is how wide an angle the camera sees. For an image $W$ pixels wide:

$$
\mathrm{FoV}_x = 2\arctan\frac{W}{2 f_x}.
$$

A short focal length gives a wide view (wide-angle lens); a long focal length gives a narrow view (telephoto, "zoomed in"). Example: $W = 640$ and $f_x = 320$ gives $2\arctan 1 = 90^\circ$. Doubling $f_x$ to 640 narrows the view to about $53^\circ$.

## Weak Perspective and Orthographic Projection
---
<span style="color:rgb(255, 255, 0)"><b>Weak perspective.</b></span> If an object is small compared with its distance $z_0$ (a car 50 m away, not a car 1 m away), all its points have nearly the same depth. Then dividing each point by its own $Z$ is almost the same as dividing all of them by $z_0$. The projection becomes one fixed zoom factor (CS231A Appendix B):

$$
x' \approx \frac{f}{z_0}\,X, \qquad y' \approx \frac{f}{z_0}\,Y .
$$

There is no division by each point's own depth any more, so the math becomes linear and much simpler.

<span style="color:rgb(255, 255, 0)"><b>Orthographic.</b></span> Push the camera infinitely far away and you get $x' = X$, $y' = Y$: depth is simply ignored, and all rays are parallel. Architects' and engineers' drawings use this. Weak perspective is just an orthographic picture scaled by $f/z_0$.

| Model | Formula | Far things shrink? | Parallel lines meet? | Good for |
|---|---|---|---|---|
| Perspective (pinhole) | $x' = fX/Z$ | yes | yes | real cameras, always correct |
| Weak perspective | $x' = (f/z_0)\,X$ | only as a whole object | no | small objects far away |
| Orthographic | $x' = X$ | no | no | technical drawings, very distant views |

## A Tiny Example in Code
---
```python
import numpy as np

K = np.array([[500.0, 0, 320],                    # fx, skew, cx
              [0, 500.0, 240],                    # fy, cy
              [0, 0, 1]])

def project(P):                                   # 3D points (N,3) in camera frame -> pixels (N,2)
    h = P @ K.T                                   # homogeneous: (fx X + cx Z, fy Y + cy Z, Z)
    return h[:, :2] / h[:, 2:]                    # divide by the last number (the depth)

tree_near = np.array([[0.0, -2.0, 10.0]])         # top of a 2 m tree, 10 m away
tree_far  = np.array([[0.0, -2.0, 20.0]])         # same tree, 20 m away
print(project(tree_near), project(tree_far))      # [[320. 140.]] [[320. 190.]]: 100 px vs 50 px above center
print(project(np.array([[1.0, 1.0, 5.0], [2.0, 2.0, 10.0]])))   # same ray -> same pixel: depth is lost
d = np.array([0.0, 0.0, 1.0])                     # direction "straight ahead"
vp = K @ d; print(vp[:2] / vp[2])                 # vanishing point = principal point (320, 240)
```

## Relationship to Other Notes
---
- [[Camera Calibration]] — builds on this rule: adds the camera's position and rotation (extrinsics), lens distortion, and methods to measure $K$; its first section is a short summary pointing here.
- [[Hand-Eye Calibration]] — uses perspective projection indirectly: each robot pose's camera-to-target transform comes from PnP, which inverts this projection for known target points.
- [[Linear Algebra]] — homogeneous coordinates turn the division by depth into a matrix product; a projection is a rank-3 matrix, and the camera center is its null space.
- [[Robot Kinematics]] — the camera's position and rotation in the world is an $SE(3)$ transform, written in the same way as robot link poses.

## References
---
- [Hata & Savarese — CS231A Course Notes 1: Camera Models, Stanford](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf) — §2 pinhole cameras, §3 lenses, §4 digital image space and homogeneous coordinates, Appendix B weak perspective and orthographic models
- [Stachniss — Camera Extrinsics and Intrinsics, Photogrammetry & Robotics Lab, Univ. Bonn](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-12-camera-params.pdf) — ideal perspective projection, camera constant
- Hartley & Zisserman — *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge Univ. Press, 2004 — Ch. 6 Camera Models (vanishing points and points at infinity)
