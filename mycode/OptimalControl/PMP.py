"""
missile_pursuit_dynamic.py

A dynamic (animated) visualization of a missile pursuing a moving target.
Here the missile uses a simple Pure Pursuit law: at each timestep it turns
to point directly at the current target position, with a maximum turn rate.

The target moves at constant speed and heading. The missile moves at constant
speed but can change its heading up to a specified max angular velocity.

Matplotlib FuncAnimation is used to animate the trajectories in real time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
dt = 0.01             # time step [s]
max_time = 30.0       # max simulation time [s]
t_vals = np.arange(0, max_time, dt)

# Target parameters
v_t = 1.0                                  # target speed
phi = np.deg2rad(225.0)                    # target heading (225°)
target_vel = np.array([np.cos(phi), np.sin(phi)])

# Missile parameters
v_m = 2.0                                  # missile speed
max_turn_rate = np.deg2rad(30.0)           # max turn rate [rad/s]

# Initial states
x_m = np.array([0.0, 0.0])                 # missile position
heading = np.deg2rad(45.0)                 # missile initial heading
x_t = np.array([10.0, 10.0])               # target position

# Data storage for plotting
missile_path = [x_m.copy()]
target_path = [x_t.copy()]

# Simulation loop
for t in t_vals[1:]:
    # --- target update (straight line) ---
    x_t = x_t + v_t * dt * target_vel

    # --- missile guidance (pure pursuit) ---
    # direction to target
    vec_to_target = x_t - x_m
    desired_heading = np.arctan2(vec_to_target[1], vec_to_target[0])
    # heading error
    delta = (desired_heading - heading + np.pi) % (2*np.pi) - np.pi
    # limit turn rate
    turn = np.clip(delta, -max_turn_rate*dt, max_turn_rate*dt)
    heading = heading + turn

    # missile update
    x_m = x_m + v_m * dt * np.array([np.cos(heading), np.sin(heading)])

    missile_path.append(x_m.copy())
    target_path.append(x_t.copy())

    # check for intercept
    if np.linalg.norm(x_t - x_m) < 0.1:
        print(f"Intercept at t = {t:.2f}s, position = {x_m}")
        break

missile_path = np.array(missile_path)
target_path = np.array(target_path)
sim_time = np.arange(len(missile_path)) * dt

# --- set up animation ---
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(min(missile_path[:,0].min(), target_path[:,0].min()) - 5,
            max(missile_path[:,0].max(), target_path[:,0].max()) + 5)
ax.set_ylim(min(missile_path[:,1].min(), target_path[:,1].min()) - 5,
            max(missile_path[:,1].max(), target_path[:,1].max()) + 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Missile Pursuit Animation')

missile_line, = ax.plot([], [], 'r-', lw=2, label='Missile path')
target_line, = ax.plot([], [], 'b--', lw=1, label='Target path')
missile_dot,  = ax.plot([], [], 'ro')
target_dot,   = ax.plot([], [], 'bo')
ax.legend()
ax.grid(True)
ax.axis('equal')

def init():
    missile_line.set_data([], [])
    target_line.set_data([], [])
    missile_dot.set_data([], [])
    target_dot.set_data([], [])
    return missile_line, target_line, missile_dot, target_dot

def update(frame):
    # frame is index in arrays
    missile_line.set_data(missile_path[:frame,0], missile_path[:frame,1])
    target_line.set_data(target_path[:frame,0], target_path[:frame,1])
    missile_dot.set_data(missile_path[frame,0], missile_path[frame,1])
    target_dot.set_data(target_path[frame,0], target_path[frame,1])
    return missile_line, target_line, missile_dot, target_dot

anim = FuncAnimation(fig, update,
                     frames=len(missile_path),
                     init_func=init,
                     interval=dt*1000,
                     blit=True)

# To save the animation, uncomment:
# anim.save('missile_pursuit.gif', writer='imagemagick', fps=30)

plt.show()