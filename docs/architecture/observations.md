# Observations

The policy sees eight float32 values.

| Index | Quantity | Normalisation | Raw unit |
| --- | --- | --- | --- |
| 0 | Horizontal position relative to the pad | metres / 20 | m |
| 1 | Altitude of the centre of mass | metres / 20 | m |
| 2 | Horizontal velocity | (m/s) / 10 | m/s |
| 3 | Vertical velocity, positive up | (m/s) / 10 | m/s |
| 4 | Wrapped angle | radians / pi | rad, 0 upright, positive nose-left |
| 5 | Angular velocity | (rad/s) / 2 | rad/s, positive counter-clockwise |
| 6 | Left foot contact | 0 or 1 | boolean sensor |
| 7 | Right foot contact | 0 or 1 | boolean sensor |

The box bounds are wide (±50 on the scaled position and velocity channels, ±1 on the angle channel). Typical flights stay far inside them. Raw values are stored in `info["telemetry"]` and in trajectory files. The agent does not receive the reward weights, the remaining step budget, the terrain map, or which controller is in use.

Distance in the reward is the distance from the centre of mass to the pad centre `(0, 0)`, in metres. Speed is the magnitude of the velocity vector. Tilt is the absolute wrapped angle from upright.
