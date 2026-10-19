# Physics

World axes: +x to the right, +y up. The landing pad is centred at the origin. Angles are radians. Zero is upright. A positive angle is counter-clockwise and points the nose toward -x.

The state is the centre-of-mass position, linear velocity, angle and angular velocity. Semi-implicit Euler advances it by a fixed `dt` of 0.02 seconds:

```
nose = (-sin(angle), cos(angle))
ax = main * main_accel * nose_x
ay = main * main_accel * nose_y - gravity
alpha = torque_command * attitude_torque / inertia
```

`main` is 0 or 1 in discrete mode and may be fractional for a future continuous controller. `torque_command` is +1 for the left attitude thruster and -1 for the right. Default gravity is 1.62 m/s². Default main-engine acceleration is 4.0 m/s², so the engine can lift the rocket but cannot hover without pulsing. Default torque is 0.8 N·m and inertia is 1.0 kg·m².

No drag is applied. The model is a vacuum. The main engine's line of thrust passes through the centre of mass, so it does not add torque. The attitude thrusters are modelled as a pure couple: a left-side thruster that exhausts to the left produces counter-clockwise torque and no net force. Flames are drawn from the scene graph and are never read by the integrator.

The same `dt` is used in training and in the dashboard. Playback speed changes how many steps run per wall-clock second. It does not change `dt`.

## Terrain and contact

The pad, 8 metres wide, is flat at y = 0. Outside it, `terrain_height` rises continuously from the pad edge. A foot or a hull point at or below the terrain, within 2 cm, is contact.

A landing requires all of the following on that contact step:

- both feet touching
- both feet inside the pad
- hull clear of the terrain
- horizontal speed at most 1.0 m/s
- downward speed at most 1.5 m/s
- upward speed at most 0.5 m/s
- absolute wrapped tilt at most 0.30 rad
- absolute angular velocity at most 0.40 rad/s

Any other contact is a crash. Landing and crash cannot both be true. Flying above the ceiling or beyond ±100 m truncates the episode, as does the step limit. Truncation adds neither +100 nor -1000.

The upward-speed and angular-velocity checks are part of stable contact. A vehicle that is still climbing or spinning is not treated as landed.
