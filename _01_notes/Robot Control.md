### [[dl_moc]]  
# **Robot Control**

**Tags**:: #robotics #math
**Links**:: [[Robot Kinematics]] · [[VLA]] · [[Behavior Cloning]] · [[Deep Reinforcement Learning]] · [[ODEs, SDEs, and Probability Paths]] · [[Linear Algebra]] · [[World Models]]


**Robot control** is the part of robotics that turns a goal ("be at this angle", "follow this path", "press with 10 N") into the torques and forces the motors actually produce. A controller works in a loop: it reads the sensors, compares what the robot *is* doing with what it *should* be doing, and sends a correction. The loop runs hundreds to a few thousand times per second. The textbook theory sorts controllers by what they try to control: **motion control** (positions and velocities), **force control** (pushing on things), **hybrid motion–force control** (both, in different directions), and **impedance control** (make the robot feel like a spring–damper). The landmark ideas are PD control plus gravity compensation ([Takegaki & Arimoto, 1981](https://doi.org/10.1115/1.3139651)), hybrid position/force control ([Raibert & Craig, 1981](https://doi.org/10.1115/1.3139652)) built on Mason's natural and artificial constraints ([Mason, 1981](https://doi.org/10.1109/TSMC.1981.4308708)), impedance control ([Hogan, 1985](https://doi.org/10.1115/1.3140702)), and operational-space control ([Khatib, 1987](https://doi.org/10.1109/JRA.1987.1087068)). For learning-based robots this layer matters because a policy almost never outputs torques directly. It outputs setpoints, and a controller from this note tracks them.

The note follows the Northwestern **Modern Robotics** course and book:
- *Modern Robotics: Mechanics, Planning, and Control* ([Lynch & Park, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics); free preprint PDF), **Ch. 11 Robot Control** is the spine. Section numbers (§11.x) and equation numbers (Eq. 11.x) below refer to it. Ch. 8 supplies the dynamics that the controllers use.
- The matching Coursera course, [*Modern Robotics, Course 4: Robot Motion Planning and Control*](https://www.coursera.org/learn/modernrobotics-course4), and the free [Chapter 11 lecture videos](https://hades.mech.northwestern.edu/index.php/Modern_Robotics_Videos) (11.1 Control System Overview → 11.7 Impedance Control) follow the same order.

**Notation.** The note uses the *Modern Robotics* symbols, which match [[Robot Kinematics]] ($\theta$ = joint angles, $\{s\}$ base frame, $\{b\}$ end-effector frame, $J_b$ body Jacobian). A tilde marks the controller's *model* of a quantity, which can be wrong.

| Symbol | Meaning | Source equivalents · clashes |
|---|---|---|
| $\theta, \dot\theta, \ddot\theta$ | joint angles, velocities, accelerations ($n$-vectors) | [[VLA]]: $q_t$; not network parameters as in the learning notes |
| $\theta_d(t)$ | desired (reference) joint trajectory | — |
| $\theta_e = \theta_d - \theta$ | joint error. Note the order: desired minus actual | some texts use $e = \theta - \theta_d$, which flips signs |
| $\tau$ | joint torques (or forces, for sliding joints) sent to the motors | not a time constant |
| $K_p, K_i, K_d$ | proportional, integral, derivative gains; scalars for one joint, matrices $k_pI$, $k_iI$, $k_dI$ for many | — |
| $\zeta, \omega_n$ | damping ratio and natural frequency of 2nd-order error dynamics | — |
| $\omega_d = \omega_n\sqrt{1-\zeta^2}$ | damped natural frequency (frequency of the decaying wobble) | not "desired $\omega$" |
| $\mathfrak t$ | time constant of a decaying exponential | MR: a script t |
| $s$ | complex root of a characteristic equation (Laplace variable) | not a frame label like $\{s\}$ |
| $M(\theta)$ | $n\times n$ joint-space mass matrix (symmetric, positive definite) | MR Ch. 8 |
| $h(\theta,\dot\theta)$ | everything in the dynamics except $M\ddot\theta$: Coriolis/centripetal, gravity $g(\theta)$, friction | $h = c(\theta,\dot\theta) + g(\theta)$ (+ friction) |
| $C(\theta,\dot\theta)$ | Coriolis matrix, with $c = C\dot\theta$ | — |
| $\tilde M, \tilde h, \tilde g$ | the controller's models of $M, h, g$ | — |
| $X, X_d \in SE(3)$ | actual and desired end-effector pose | [[Robot Kinematics]]: $T_{sb}$, $T_{sd}$ |
| $\mathcal V_b, \mathcal V_d$ | actual and desired end-effector twists, each in its own frame ($[\mathcal V_b] = X^{-1}\dot X$) | — |
| $X_e$ | pose error as a twist: $[X_e] = \log(X^{-1}X_d)$ | the IK error twist in [[Robot Kinematics]] |
| $\mathcal V_e$ | twist error: $[\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d - \mathcal V_b$ | — |
| $\Lambda(\theta), \eta(\theta,\mathcal V)$ | task-space mass matrix and bias wrench: $\Lambda = J^{-\top}MJ^{-1}$ | Khatib: operational-space inertia |
| $F_\text{tip}, F_d, F_e$ | wrench the robot applies to the world; desired wrench; force error $F_d - F_\text{tip}$ | — |
| $K_{fp}, K_{fi}, K_\text{damp}$ | force-control P and I gains; velocity damping gain | — |
| $A(\theta)$, $k$ | constraint matrix ($k\times 6$) with $A(\theta)\mathcal V = 0$; number of constrained directions | not the $A$ of $AX=XB$ in [[Hand-Eye Calibration]] |
| $\lambda$ | Lagrange multipliers (constraint forces) | not the DLS damping $\lambda$ in [[Robot Kinematics]] |
| $P(\theta)$ | projection onto wrenches that cause motion along the constraints | — |
| $M, B, K$ (§ impedance) | virtual mass, damping, stiffness the robot pretends to have | **clash:** MR reuses $M$ and $K$; here they are *virtual* values, not the mass matrix $M(\theta)$ or the gains $K_p$ |
| $Z(s), Y(s)$ | impedance (motion → force) and admittance (force → motion) | $Y = Z^{-1}$ |
| $k_t, I, G$ | motor torque constant, motor current, gear ratio | $I$ is also the identity matrix; context decides |

## The Control Problem
---
<span style="color:rgb(255, 255, 0)"><b>What a controller does.</b></span> The controller sits between the *desired behavior* and the *physics*. It reads sensors (joint encoders for angles; velocity, usually by differencing encoder readings and low-pass filtering them; and maybe a force–torque sensor at the wrist), and it outputs joint torques. *Modern Robotics* simplifies the hardware: each joint's amplifier, motor and gearing is treated as a perfect "torque source" that delivers whatever $\tau$ is asked for (§11.1). Section [[#Low-Level Joint Torque Control]] comes back to how real motors approximate this.

![[robot_control_stack.png|700]]

*The control stack. A learned policy or planner produces setpoints at tens of Hz. The robot controller (the subject of this note) turns them into joint torques at hundreds of Hz to a few kHz. The motor amplifier closes its own current loop even faster (MR §11.8 gives 10 kHz inner / 1 kHz outer as a typical example). Sensors close the loop.*

<span style="color:rgb(255, 255, 0)"><b>Open-loop vs. closed-loop control.</b></span> The two basic ways to compute a command differ in one thing: whether the controller looks at what the robot is actually doing.
- **Open-loop control** (also called **feedforward** control) computes the command from the plan alone, and possibly a model of the robot. It never reads sensors. Example: command the planned joint velocity, $\dot\theta = \dot\theta_d(t)$. MR calls this "a feedforward or open-loop controller, since no feedback (sensor data) is needed" (§11.3.1.1, Eq. 11.11).
- **Closed-loop control** (also called **feedback** control) measures the actual state with sensors, compares it with the desired state, and computes the command from the difference, the error $\theta_e = \theta_d - \theta$. Example: P control, $\dot\theta = K_p\theta_e$ (Eq. 11.12). The sensor reading feeds back into the command, which closes a loop (the arrow back to the controller in the figure above). MR §11.1: "Feedback is used in nearly all robot systems."

| Aspect | Open-loop (feedforward) | Closed-loop (feedback) |
|---|---|---|
| Uses sensor data | no | yes |
| Command depends on | the plan $\theta_d(t)$ (and a model $\tilde M, \tilde h$) | the error $\theta_e = \theta_d - \theta$ |
| When it acts | right away, before any error appears | only after an error has appeared |
| Model errors, pushes, friction | not corrected: errors pile up | corrected as they show up |
| Main risk | drifts away from the plan | lags behind; too-high gains oscillate or go unstable |

In plain words: open-loop is walking with your eyes closed using a memorized route; closed-loop is walking while watching where you are. Practical robot controllers **combine both**: the feedforward term does most of the work, and feedback cleans up what the model got wrong. "All practical controllers use feedback, as no model of robot and environment dynamics will be perfect" (MR §11.4.1.3). This note's main controllers are such combinations: feedforward + PI velocity control (Eq. 11.15) and computed torque (Eq. 11.35). The computed-torque figure below compares all three.

**Four kinds of goals.** The right controller depends on the task (MR Ch. 11 intro):
- **Motion control:** follow a path, e.g. move a spray-paint gun along a trajectory.
- **Force control:** push with a set force, e.g. hold a polishing wheel against a part.
- **Hybrid motion–force control:** force in some directions and motion in others, e.g. writing on a chalkboard (press *into* the board, move *along* it).
- **Impedance control:** behave like a spring, damper or mass when touched, e.g. a haptic display.

<span style="color:rgb(255, 255, 0)"><b>You cannot control motion and force in the same direction.</b></span> If the robot sets the motion in a direction, the environment decides the force there, and vice versa. Push a wall: you can choose how hard you push, but the wall decides that you don't move. This one fact shapes the force, hybrid and impedance sections.

## Error Dynamics
---
**Error and error dynamics.** For one joint, the error is $\theta_e(t) = \theta_d(t) - \theta(t)$. The differential equation that $\theta_e$ obeys once the controller is plugged in is the **error dynamics**. Designing a controller mostly means choosing the error dynamics you want: the error should shrink to zero, fast, without wild swings (§11.2).

**Judging an error response.** Start with error $\theta_e(0) = 1$ and zero error velocity, then watch. Three numbers describe what happens:
- **Steady-state error** $e_{ss}$: the error left over as $t \to \infty$.
- **Overshoot**: how far the error swings past its final value, as a percentage.
- **2% settling time**: the first time after which the error stays within 2% of its final value.

A good controller has little or no steady-state error, little or no overshoot, and a short settling time.

**Linear error dynamics and stability.** Most controllers in this note give linear error dynamics
$$
a_p\theta_e^{(p)} + \dots + a_2\ddot\theta_e + a_1\dot\theta_e + a_0\theta_e = 0 .
$$
*MR §11.2.2:* this is **stable** (every error decays to zero) exactly when every root $s$ of the characteristic equation $a_ps^p + \dots + a_1s + a_0 = 0$ has a negative real part. A quick test: all coefficients must be positive (after dividing by $a_p$). That is enough for orders 1 and 2. Order 3 also needs $a_2'a_1' > a_0'$ (with $a_i' = a_i/a_p$).

In plain words: the roots are the "decay rates" of the error. Roots on the left side of the complex plane mean decay; any root on the right side means the error blows up. This is the same "flow of a vector field" picture as in [[ODEs, SDEs, and Probability Paths]]: the error dynamics is a linear ODE $\dot x = Ax$, solved by $x(t) = e^{At}x(0)$.

**First order.** $\dot\theta_e + \frac{1}{\mathfrak t}\theta_e = 0$ has solution $\theta_e(t) = e^{-t/\mathfrak t}\theta_e(0)$. The **time constant** $\mathfrak t$ is when the error has dropped to about 37%. Solving $e^{-t/\mathfrak t} = 0.02$ gives $t = 3.91\mathfrak t$, so the 2% settling time is about $4\mathfrak t$. No overshoot, ever.

<span style="color:rgb(255, 255, 0)"><b>Second order: the mass–spring–damper.</b></span> The error behaves like a mass $m$ on a spring $k$ with a damper $b$: $m\ddot\theta_e + b\dot\theta_e + k\theta_e = 0$ (Eq. 11.4). The standard form is
$$
\ddot\theta_e + 2\zeta\omega_n\dot\theta_e + \omega_n^2\theta_e = 0, \qquad \omega_n = \sqrt{k/m},\quad \zeta = \frac{b}{2\sqrt{km}}, \qquad s_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}.
$$
In plain words: $\omega_n$ (natural frequency) says how fast the system wants to move, and $\zeta$ (damping ratio) says how much the damper fights oscillation. The value of $\zeta$ picks one of three behaviors (MR §11.2.2.2):

| Case | $\zeta$ | Roots | Error response $\theta_e(t)$ for $\theta_e(0)=1$, $\dot\theta_e(0)=0$ | Overshoot |
|---|---|---|---|---|
| overdamped | $> 1$ | two different real roots | sum of two decaying exponentials; the slower root $s_1$ dominates | none |
| critically damped | $= 1$ | one double real root $-\omega_n$ | $(1 + \omega_n t)\,e^{-\omega_n t}$ | none |
| underdamped | $< 1$ | complex pair $-\zeta\omega_n \pm j\omega_d$ | $\big(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\big)e^{-\zeta\omega_n t}$ | $e^{-\pi\zeta/\sqrt{1-\zeta^2}}$ |

The 2% settling time is about $4\mathfrak t$, with $\mathfrak t = 1/(\zeta\omega_n)$ for the underdamped and critically damped cases (and $\mathfrak t = -1/s_1$ for the slow root when overdamped). Underdamped overshoot peaks at $t_p = \pi/\omega_d$. Some numbers from MR: $\zeta = 0.1 \Rightarrow 73\%$ overshoot, $\zeta = 0.5 \Rightarrow 16\%$, $\zeta = 0.8 \Rightarrow 1.5\%$.

![[robot_control_error_response.png|700]]

*(a) Roots move as one gain grows (here the PI velocity controller below, with $K_p = 20$). Roots further left settle faster; roots further from the real axis overshoot and oscillate more. (b) The three cases. Critical damping ($\zeta = 1$) is the usual design target: the fastest response with no overshoot. The underdamped case settles at the same rate here (same $\zeta\omega_n = 10$) but swings past zero by 16%.*

## Motion Control with Velocity Inputs
---
**When the input is a velocity.** Sometimes you can command joint *velocities* directly: stepper motors (velocity = pulse frequency) or motor amplifiers in "velocity mode" (§11.3). Then the robot's dynamics can be ignored and $\dot\theta$ is the control.

**Feedforward only (open-loop).** Command $\dot\theta = \dot\theta_d(t)$ from the plan (Eq. 11.11). No sensors are needed, but small errors pile up and nothing removes them.

**P control.** $\dot\theta = K_p\theta_e$ (Eq. 11.12): the gain acts like a virtual spring pulling the joint toward the target. For a fixed target ("setpoint control") the error dynamics is $\dot\theta_e + K_p\theta_e = 0$: first order, time constant $1/K_p$, settling time $4/K_p$. But if the target moves at constant speed $\dot\theta_d = c$, the error settles at $c/K_p$, not zero. The joint always lags behind. Raising $K_p$ shrinks the lag, but real joints have speed limits, and a large gain can go unstable in a digital controller (the error changes a lot within one servo cycle).

**PI control.** Add the integral of the error (Eq. 11.13):
$$
\dot\theta = K_p\theta_e + K_i\int_0^t\theta_e(\mathrm t)\,d\mathrm t \quad\Longrightarrow\quad \ddot\theta_e + K_p\dot\theta_e + K_i\theta_e = 0 \quad (\text{for constant } \dot\theta_d).
$$
This is second-order error dynamics with $\omega_n = \sqrt{K_i}$ and $\zeta = K_p/(2\sqrt{K_i})$, stable for $K_p, K_i > 0$, and the lag goes to zero. In plain words: the integral keeps growing while any error remains, so it eventually supplies whatever command is needed. Choose $K_i = K_p^2/4$ for critical damping. Example (MR Fig. 11.7): $K_p = 20$, $K_i = 100$ gives a double root at $-10$ and a settling time of $4/(\zeta\omega_n) = 0.4$ s.

<span style="color:rgb(255, 255, 0)"><b>Feedforward plus feedback.</b></span> Feedback alone must wait for an error to appear before it acts. Adding the planned velocity gets the joint moving right away, and the feedback cleans up what is left (Eq. 11.15, MR's preferred velocity law):
$$
\dot\theta(t) = \dot\theta_d(t) + K_p\theta_e(t) + K_i\int_0^t\theta_e(\mathrm t)\,d\mathrm t .
$$
For $n$ joints the same law runs with gains $k_pI$, $k_iI$, and each joint behaves like the single-joint case (§11.3.2).

**Task space.** To track an end-effector pose $X_d(t) \in SE(3)$ instead (Eq. 11.16):
$$
\mathcal V_b(t) = [\mathrm{Ad}_{X^{-1}X_d}]\,\mathcal V_d(t) + K_pX_e(t) + K_i\int_0^tX_e(\mathrm t)\,d\mathrm t, \qquad [X_e] = \log(X^{-1}X_d), \qquad \dot\theta = J_b^\dagger(\theta)\,\mathcal V_b .
$$
In plain words: two poses cannot simply be subtracted. $X_e$ is the twist that would carry the current pose to the desired one in one unit of time, the same error twist that drives numerical IK in [[Robot Kinematics]]. The adjoint rewrites the planned twist $\mathcal V_d$ (given in the desired frame) in the current frame. The pseudoinverse turns the twist into joint speeds. Splitting the error into an orientation part $\log(R^\top R_d)$ and a position part $p_d - p$ (Eq. 11.18) makes the end-effector origin travel in a straight line. The twist version moves it along a screw-shaped arc (MR Fig. 11.10).

## Motion Control with Torque Inputs
---
Real robot controllers command **torques**, because then a model of the robot's dynamics can be used (§11.4).

**One link under gravity.** A motor turns a single link of inertia $M$, mass $m$ and center of mass at distance $r$, with viscous friction $b$ (Eq. 11.21):
$$
\tau = M\ddot\theta + mgr\cos\theta + b\dot\theta = M\ddot\theta + h(\theta, \dot\theta).
$$
MR's running example uses $M = 0.5$ kg·m², $m = 1$ kg, $r = 0.1$ m, $b = 0.1$ N·m·s/rad.

**PID control.** $\tau = K_p\theta_e + K_i\int\theta_e\,dt + K_d\dot\theta_e$ (Eq. 11.23). $K_p$ is a virtual spring, $K_d$ a virtual damper on the velocity error, and $K_i$ removes steady-state error.
- **PD, no gravity** ($K_i = 0$, $g = 0$): the error dynamics is $M\ddot\theta_e + (b + K_d)\dot\theta_e + K_p\theta_e = 0$. So $\zeta = \frac{b + K_d}{2\sqrt{K_pM}}$ and $\omega_n = \sqrt{K_p/M}$. Pick gains for critical damping, with $K_p$ as large as the hardware allows (torque limits, chattering, structural vibration, servo rate).
- **PD with gravity:** the link stops where the spring torque equals gravity, $K_p\theta_e = mgr\cos\theta$, so a steady-state error remains. A PD controller only makes torque when there is an error, and holding the link up *needs* torque.
- **PID with gravity:** with $K_i > 0$ the integral supplies the holding torque even at zero error. Differentiating gives third-order error dynamics $M\theta_e^{(3)} + (b + K_d)\ddot\theta_e + K_p\dot\theta_e + K_i\theta_e = \dot\tau_\text{dist}$ (Eq. 11.29). *MR §11.4.1.1:* stable iff
$$
K_d > -b, \qquad K_p > 0, \qquad \frac{(b + K_d)K_p}{M} > K_i > 0 .
$$
So $K_i$ has an *upper* limit too: too much integral action makes the system unstable. In practice many robot controllers set $K_i = 0$, or use **anti-windup**, which caps how large the error integral may grow.

![[robot_control_pd_vs_pid_gravity.png|700]]

*MR Fig. 11.13 setup, re-simulated: PD gains $K_p = 2.205$, $K_d = 2$ give $\zeta = 2.1/(2\sqrt{2.205\cdot0.5}) = 1$. (a) PD stalls about 0.41 rad short of level. PID ($K_i = 1$, below the bound $(b{+}K_d)K_p/M \approx 9.3$) reaches the goal but overshoots. (b) The integral term builds up the $mgr \approx 0.98$ N·m needed to hold the link level, so the P term can drop to zero.*

**Feedforward from a model.** Given the plan $(\theta_d, \dot\theta_d, \ddot\theta_d)$, compute the torque the model says is needed (Eq. 11.32): $\tau = \tilde M(\theta_d)\ddot\theta_d + \tilde h(\theta_d, \dot\theta_d)$. With a perfect model and a perfect start, the robot follows the plan exactly. Models are never perfect, so feedforward is always paired with feedback.

<span style="color:rgb(255, 255, 0)"><b>Computed torque control.</b></span> Ask for the *linear* error dynamics $\ddot\theta_e + K_d\dot\theta_e + K_p\theta_e + K_i\int\theta_e = 0$ along any trajectory. This needs the commanded acceleration $\ddot\theta = \ddot\theta_d + K_d\dot\theta_e + K_p\theta_e + K_i\int\theta_e$. Pass it through the dynamics model (Eq. 11.35; Eq. 11.37 for $n$ joints):
$$
\tau = \tilde M(\theta)\Big(\underbrace{\ddot\theta_d}_{\text{feedforward}} + \underbrace{K_p\theta_e + K_i\!\int\!\theta_e(t)\,dt + K_d\dot\theta_e}_{\text{feedback}}\Big) + \underbrace{\tilde h(\theta, \dot\theta)}_{\text{cancel nonlinear terms}} .
$$
In plain words: $\tilde h$ cancels gravity, Coriolis and friction; $\tilde M$ converts "desired acceleration" into "torque". What is left behaves like $n$ separate, linear mass–spring–dampers, which the error-dynamics section already knows how to tune. This is why the method is also called **inverse dynamics control** or **feedback linearization**. The MR software library implements it as `ComputedTorque` (and simulates it with `SimulateControl`).

![[robot_control_computed_torque.png|700]]

*MR Fig. 11.19 setup, re-simulated: Task 2, $\theta_d(t) = \pi/2 - (\pi/4)\cos t$, with a wrong model ($\tilde r = 0.08$ m instead of $0.1$ m). Feedforward alone drifts away, because the unmodeled gravity pulls against the motion. PID alone lags. Computed torque tracks best and uses less effort ($\int\tau^2 dt$) than feedback alone.*

```python
import numpy as np

M, m, r, b, g = 0.5, 1.0, 0.1, 0.1, 9.81           # true link (MR Sec. 11.4.1)
Mt, rt = 0.5, 0.08                                  # controller's (wrong) model
kp, ki, kd, dt = 2.205, 1.0, 2.0, 1e-4

th_d   = lambda t: np.pi/2 - np.pi/4*np.cos(t)      # Task 2 trajectory
dth_d  = lambda t: np.pi/4*np.sin(t)
ddth_d = lambda t: np.pi/4*np.cos(t)

th, dth, eint = th_d(0), 0.0, 0.0
for k in range(int(np.pi / dt)):
    t = k * dt
    e, edot = th_d(t) - th, dth_d(t) - dth
    eint += e * dt
    tau = Mt*(ddth_d(t) + kp*e + ki*eint + kd*edot) + m*g*rt*np.cos(th) + b*dth   # Eq. 11.35
    ddth = (tau - m*g*r*np.cos(th) - b*dth) / M                                    # true dynamics
    dth += ddth * dt; th += dth * dt
print(f"final tracking error: {th_d(np.pi) - th:+.4f} rad")
```

**Many joints: decentralized vs. centralized.** For $n$ joints the dynamics is $\tau = M(\theta)\ddot\theta + h(\theta, \dot\theta)$, and the joints push and pull on each other through $M(\theta)$ and $h$.
- **Decentralized control** runs an independent controller per joint. It works when the joints barely interact: gantry robots (diagonal $M$), or highly geared arms where the motors' own inertia dominates $M(\theta)$.
- **Centralized control** uses the full state, as computed torque does.

<span style="color:rgb(255, 255, 0)"><b>PD plus gravity compensation.</b></span> If the full model is too costly or too uncertain, keep only the gravity model (Eq. 11.38):
$$
\tau = K_p\theta_e + K_i\!\int\!\theta_e\,dt + K_d\dot\theta_e + \tilde g(\theta).
$$
*MR §11.4.2.2, after Takegaki & Arimoto (1981):* with no friction, perfect gravity compensation, $K_i = 0$ and a fixed target, the robot converges to $\theta_d$ from **any** starting state. *Proof sketch.* Define an "error energy" (spring energy plus kinetic energy) $V = \tfrac12\theta_e^\top K_p\theta_e + \tfrac12\dot\theta^\top M(\theta)\dot\theta$. Using the dynamics $M\ddot\theta + C\dot\theta = K_p\theta_e - K_d\dot\theta$ and the fact that $\dot M - 2C$ is skew-symmetric (MR Prop. 8.1.2),
$$
\dot V = -\dot\theta^\top K_p\theta_e + \dot\theta^\top(K_p\theta_e - K_d\dot\theta) + \tfrac12\dot\theta^\top(\dot M - 2C)\dot\theta = -\dot\theta^\top K_d\dot\theta \le 0 .
$$
The energy only goes down while the robot moves. If the robot stops away from the goal, the spring makes it move again. So by the Krasovskii–LaSalle invariance principle it must end at rest at $\theta_d$. In plain words: a spring plus a damper, with gravity cancelled, always settles at the spring's rest point, however complicated the arm is.

**Task-space motion control.** To control the end-effector pose there are two options (§11.4.3). (1) Turn the task trajectory into a joint trajectory with IK ($\dot\theta = J_b^\dagger\mathcal V_b$, $\ddot\theta = J_b^\dagger(\dot{\mathcal V}_b - \dot J_b\dot\theta)$) and use the joint-space controller. (2) Write the dynamics in task space, $F_b = \Lambda(\theta)\dot{\mathcal V}_b + \eta(\theta, \mathcal V_b)$ with $\Lambda = J^{-\top}MJ^{-1}$ (MR Eq. 8.89–8.91), and run computed torque there (Eq. 11.47):
$$
\tau = J_b^\top(\theta)\Big(\tilde\Lambda(\theta)\big(\tfrac{d}{dt}([\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d) + K_pX_e + K_i\!\int\! X_e\,dt + K_d\mathcal V_e\big) + \tilde\eta(\theta, \mathcal V_b)\Big).
$$
In plain words: the same recipe as joint-space computed torque, but with the end-effector's mass matrix $\Lambda$ ("how heavy the hand feels in each direction") and a wrench that $J_b^\top$ maps back to joint torques (the statics duality from [[Robot Kinematics]]). This is Khatib's **operational-space control**. Joint space makes joint limits easy and has no singularities; task space speaks the language of the objects the robot touches.

## Force Control
---
**The goal is a wrench, not a motion.** For tasks like pressing or polishing, the robot should apply a desired wrench $F_d$ no matter how the end-effector gets nudged (§11.5). This mirrors ideal motion control, where the motion stays the same no matter how the robot gets pushed. With the full dynamics $M\ddot\theta + c + g + b + J^\top F_\text{tip} = \tau$ (Eq. 11.48) and a robot that barely moves, the velocity and acceleration terms drop out, leaving $\tau = g(\theta) + J^\top(\theta)F_\text{tip}$.

**Without a force sensor.** Command $\tau = \tilde g(\theta) + J^\top(\theta)F_d$ (Eq. 11.50). This needs a good gravity model and accurate joint torques, which is easy for a direct-drive motor (torque ∝ current) and hard through high-friction gearing.

**With a wrist force–torque sensor.** Add PI feedback on the force error $F_e = F_d - F_\text{tip}$ (Eq. 11.51, 11.54):
$$
\tau = \tilde g(\theta) + J^\top(\theta)\Big(F_d + K_{fp}F_e + K_{fi}\!\int\! F_e(t)\,dt - K_\text{damp}\,\mathcal V\Big).
$$
- **No D term** (MR footnote): force readings are noisy, so their derivatives are nearly meaningless. Also, torque reaches the end-effector force directly, so there is no "integration" for a D term to anticipate, unlike in motion control.
- **The damping term is a safety net.** If there is nothing to push against, a pure force controller accelerates the arm in a failing attempt to create force. $-K_\text{damp}\mathcal V$ limits that speed.

With perfect gravity compensation the error dynamics is $K_{fp}F_e + K_{fi}\int F_e = 0$. A constant disturbance, such as a gravity-model error, still gives $K_{fp}\dot F_e + K_{fi}F_e = 0$, so $F_e \to 0$.

## Hybrid Motion–Force Control
---
**Pick $n$ of $2n$.** In an $n$-dimensional task space there are $n$ motion values and $n$ force values. The robot may choose any $n$ of them; the environment decides the rest. It must not choose a motion and a force in the "same direction" (§11.6).

<span style="color:rgb(255, 255, 0)"><b>Natural and artificial constraints</b></span> ([Mason, 1981](https://doi.org/10.1109/TSMC.1981.4308708)). When the environment is rigid in $k$ directions, contact itself decides the split. **Natural constraints** come from the environment: you can't move into the board, and you can't apply force along a frictionless surface. **Artificial constraints** are the goals the robot picks in the remaining directions. MR's example is an eraser sliding on a frictionless chalkboard, with twist $(\omega_x, \omega_y, \omega_z, v_x, v_y, v_z)$ and wrench $(m_x, m_y, m_z, f_x, f_y, f_z)$:

| Direction | Natural constraint (environment) | Artificial constraint (robot's goal) |
|---|---|---|
| rotate about $x$ | $\omega_x = 0$ | $m_x = 0$ |
| rotate about $y$ | $\omega_y = 0$ | $m_y = 0$ |
| rotate about $z$ | $m_z = 0$ | $\omega_z = 0$ |
| slide along $x$ | $f_x = 0$ | $v_x = k_1$ |
| slide along $y$ | $f_y = 0$ | $v_y = 0$ |
| push into the board ($z$) | $v_z = 0$ | $f_z = k_2 < 0$ |

So the eraser slides at speed $k_1$ while pressing with force $k_2$. Every direction gets exactly one goal: a motion where the environment fixes the force, a force where it fixes the motion.

**The controller.** Write the $k$ natural velocity constraints as $A(\theta)\mathcal V = 0$, with $A(\theta) \in \mathbb R^{k\times6}$. The constraint wrench is $A^\top(\theta)\lambda$. Solving for the Lagrange multipliers $\lambda$ gives a projection (Eq. 11.60)
$$
P(\theta) = I - A^\top(A\Lambda^{-1}A^\top)^{-1}A\Lambda^{-1},
$$
of rank $6 - k$. $P$ keeps the part of a wrench that moves the end-effector along the constraints; $I - P$ keeps the part that pushes against them. The hybrid controller sends each controller's output through its own projection (Eq. 11.61):
$$
\tau = J_b^\top(\theta)\Big(P(\theta)\big[\underbrace{\tilde\Lambda(\theta)(\tfrac{d}{dt}([\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d) + K_pX_e + K_i\!\textstyle\int\! X_e + K_d\mathcal V_e)}_{\text{motion control}}\big] + (I - P(\theta))\big[\underbrace{F_d + K_{fp}F_e + K_{fi}\!\textstyle\int\! F_e}_{\text{force control}}\big] + \underbrace{\tilde\eta(\theta, \mathcal V_b)}_{\text{Coriolis and gravity}}\Big).
$$
In plain words: the motion controller only acts in the free directions, and the force controller only acts in the blocked ones. Because the two subspaces don't interfere, each half keeps the stability analysis it had on its own.

**The catch.** You must know $A(\theta)$, meaning which directions are blocked, at every moment, and real contact is uncertain. Remedies from MR: estimate the constraint directions online from force feedback; lower the gains so motion control is "soft" and force control tolerates errors; or build passive compliance into the robot.

## Impedance Control
---
**Impedance.** Hybrid control in rigid environments asks for extremes: motion control wants **high impedance** (pushes barely change the motion), and force control wants **low impedance** (motions barely change the force). Impedance control instead asks the end-effector to behave like a chosen mass–spring–damper ([Hogan, 1985](https://doi.org/10.1115/1.3140702); MR §11.7):
$$
M\ddot x + B\dot x + Kx = f_\text{ext} \qquad \text{(Eq. 11.64)},
$$
where $x$ is the task-space position, $M, B, K$ are the *virtual* mass, damping and stiffness, and $f_\text{ext}$ is the force from a person or object. In Laplace form, the **impedance** $Z(s) = F(s)/X(s) = Ms^2 + Bs + K$ maps motion to force. The spring dominates at low frequency and the mass at high frequency. The **admittance** $Y(s) = Z^{-1}(s)$ maps force to motion. Stiffness (or compliance) control is the special case where only the spring is rendered.

**Two ways to build it:**
- <span style="color:rgb(255, 255, 0)"><b>Impedance control</b></span> *measures motion, outputs force* (Eq. 11.65):
$$
\tau = J^\top(\theta)\Big(\underbrace{\tilde\Lambda(\theta)\ddot x + \tilde\eta(\theta, \dot x)}_{\text{cancel the arm's own dynamics}} - \underbrace{(M\ddot x + B\dot x + Kx)}_{\text{virtual environment}}\Big).
$$
Acceleration measurements are noisy, so the $\tilde\Lambda\ddot x$ term is often dropped and $M = 0$ is set (the user then feels the arm's real mass, so such arms are built light). It struggles with stiff virtual walls: a large $K$ is a high effective gain, and with delays and quantization it oscillates. It is good at low impedance on a light, **backdrivable** arm.
- <span style="color:rgb(255, 255, 0)"><b>Admittance control</b></span> *measures force, outputs motion* (Eq. 11.66): read $f_\text{ext}$ with a wrist sensor, compute $\ddot x_d = M^{-1}(f_\text{ext} - B\dot x - Kx)$, convert with $\ddot\theta_d = J^\dagger(\ddot x_d - \dot J\dot\theta)$, and track $\ddot\theta_d$ with inverse dynamics. It struggles with a small virtual mass (small forces cause huge accelerations), but a highly geared, stiff robot does well at rendering stiff environments.

**Choosing the right controller.** The table below summarizes the four goals:

| Controller | Robot sets | Environment sets | Needs | Typical use |
|---|---|---|---|---|
| motion (PD, computed torque) | motion | force | dynamics model (or just gravity) | free-space moves, pick-and-place |
| force | force | motion | force sensor or good torque control; damping for safety | pressing, polishing |
| hybrid motion–force | motion in $6-k$ directions, force in $k$ | the rest | constraint model $A(\theta)$ | wiping, writing, door opening |
| impedance / admittance | the *relationship* between force and motion | the actual force and motion | backdrivable arm (impedance) or wrist sensor (admittance) | contact-rich tasks, working near people, haptics |

## Low-Level Joint Torque Control
---
The note so far assumed each joint delivers exactly the torque requested. MR §11.8 lists how electric-motor joints get close to this, each fixing a problem of the previous one:

| Setup | How torque is controlled | Gain | Cost |
|---|---|---|---|
| direct-drive motor, current control | $\tau = k_tI$; amplifier loop on measured current (e.g. 10 kHz inner vs. 1 kHz outer loop) | clean torque control | ungeared motors must be large; impractical at moving joints of a serial arm |
| geared motor, current control | output torque nominally $Gk_tI$, gear ratio $G > 1$ | small motor running at efficient high speed | gear **backlash** and **friction** make actual torque uncertain |
| geared motor + strain gauges | strain gauges on a harmonic drive measure output torque; local loop corrects current | compensates gear friction | more complex; harmonic drive compliance adds dynamics |
| **series elastic actuator (SEA)** | a soft torsional spring after the gearhead; torque $= k\,\Delta\phi$ from measured spring deflection ([Pratt & Williamson, 1995](https://doi.org/10.1109/IROS.1995.525827)) | naturally soft joint, safe around people; spring protects gears from impacts | harder to control fast, high-frequency motion |

NASA's Robonaut 2, the first humanoid in space (2011), uses SEAs, e.g. in its hip joint (MR Fig. 11.24).

**Other topics** (MR §11.9). **Robust control** guarantees performance despite bounded model errors. **Adaptive control** estimates inertial parameters while running. **Iterative learning control** (ILC) repeats a task and corrects the feedforward from the last run's error, which is useful for one repeated trajectory. **Passive compliance** and flexible links mean the motor angle and the true link angle differ. **Variable-impedance actuators** set stiffness mechanically (e.g. a second motor adjusting a nonlinear spring), avoiding the bandwidth limits of feedback.

## Control in Learning-Based Robotics
---
**Policies output setpoints; controllers make them real.** A learned policy runs at tens of Hz, far too slow and too unstable to command raw torques on most arms. So the policy's "action" is usually a setpoint for one of the controllers above:
- **Joint position targets + PD.** In legged locomotion with massively parallel RL, "the actions are interpreted as desired joint positions sent to the motors. There, a PD controller produces motor torques" ([Rudin et al., 2021](https://arxiv.org/abs/2109.11978)). Moving to a smaller robot meant reducing the PD gains. The gains are part of the action space: they decide how stiff, and how forgiving of policy errors, the robot is.
- **Position vs. velocity actions.** Diffusion Policy found that a position-control action space consistently beats velocity control for its policy, and that position control suffers less from compounding error, which suits predicting action sequences ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)). In this note's terms: a velocity command integrates its errors (the P-control lag above); a position setpoint tracked by feedback does not drift.
- **Impedance as the action space.** VICES has the RL policy output end-effector setpoints *and* impedance (stiffness) parameters, with the arm's kinematics and dynamics compensated by an operational-space-style controller. Across path following, door opening and surface wiping it improved sample efficiency, kept energy use low, kept the robot safe, and transferred across robot models in simulation ([Martín-Martín et al., 2019](https://arxiv.org/abs/1906.08880)).

This is the control half of the "action representation" choice in [[VLA]] and [[Behavior Cloning]]. [[Robot Kinematics]] covers the geometric half (joint space vs. task space). The same action chunk ("move the end-effector 2 cm down") produces different contact forces under a stiff position controller than under a soft impedance controller.

**Classic ideas with learning cousins.** Computed torque is "use a model for the bulk, feedback for the residual": the same split as residual RL or a learned dynamics model ([[World Models]]) plus a feedback controller. ILC is learning from repetition for a single trajectory. Adaptive control learns a few physical parameters online. Lyapunov arguments like the PD + gravity proof above are what learned controllers usually lack, which is why a well-tuned low-level controller is the safety layer under most policies.

## Relationship to Other Notes
---
- [[Robot Kinematics]] — supplies everything this note builds on: the Jacobian $J_b$, the error twist $\log(X^{-1}X_d)$, the adjoint, the pseudoinverse, and statics $\tau = J^\top F$, which all task-space and force controllers here use. Kinematics assumes commands are achieved; control is how they get achieved.
- [[VLA]] — its action chunks (joint targets, end-effector deltas, gripper commands) are setpoints for the controllers in this note; the controller type and gains decide what an action physically does.
- [[Behavior Cloning]] — demonstrations are recorded through a controller, and replaying them needs the same one; position-control actions reduce the compounding drift that velocity actions integrate.
- [[Deep Reinforcement Learning]] — RL policies for locomotion and manipulation act through PD or impedance controllers; the choice of action space (joint targets, impedance parameters) shapes exploration and sim-to-real transfer.
- [[World Models]] — a learned dynamics model plays the role of $\tilde M, \tilde h$ in computed torque or model-predictive control; model error then has to be absorbed by feedback, as with $\tilde r \ne r$ above.
- [[ODEs, SDEs, and Probability Paths]] — error dynamics are linear ODEs $\dot x = Ax$ with solution $e^{At}x(0)$; stability is the sign of the eigenvalues' real parts.
- [[Linear Algebra]] — eigenvalues decide stability; the hybrid controller's $P(\theta)$ is a projection; positive-definite gain and mass matrices make the energy argument work.
- [[Hand-Eye Calibration]] — task-space and vision-guided control both rely on accurate camera-to-robot transforms; there, $A$ and $X$ mean different things than the constraint matrix and pose here.

## References
---
- [Lynch & Park — *Modern Robotics: Mechanics, Planning, and Control*, Cambridge Univ. Press, 2017](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Ch. 11 Robot Control (spine of this note); Ch. 8 Dynamics of Open Chains (§8.6 task-space dynamics, §8.7 constrained dynamics)
- [*Modern Robotics, Course 4: Robot Motion Planning and Control* (Northwestern / Coursera)](https://www.coursera.org/learn/modernrobotics-course4) and [Modern Robotics videos, Ch. 11](https://hades.mech.northwestern.edu/index.php/Modern_Robotics_Videos)
- [Takegaki & Arimoto, 1981 — A New Feedback Method for Dynamic Control of Manipulators, *J. Dyn. Sys., Meas., Control* 103:119–125](https://doi.org/10.1115/1.3139651)
- [Raibert & Craig, 1981 — Hybrid Position/Force Control of Manipulators, *J. Dyn. Sys., Meas., Control* 103:126–133](https://doi.org/10.1115/1.3139652)
- [Mason, 1981 — Compliance and Force Control for Computer Controlled Manipulators, *IEEE Trans. SMC* 11:418–432](https://doi.org/10.1109/TSMC.1981.4308708)
- [Hogan, 1985 — Impedance Control: An Approach to Manipulation, Part I—Theory, *J. Dyn. Sys., Meas., Control* 107:1–7](https://doi.org/10.1115/1.3140702)
- [Khatib, 1987 — A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation, *IEEE J. Robotics and Automation* 3:43–53](https://doi.org/10.1109/JRA.1987.1087068)
- [Pratt & Williamson, 1995 — Series Elastic Actuators, IROS](https://doi.org/10.1109/IROS.1995.525827)
- [Martín-Martín et al., 2019 — Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks](https://arxiv.org/abs/1906.08880)
- [Rudin et al., 2021 — Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
- [Chi et al., 2023 — Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
