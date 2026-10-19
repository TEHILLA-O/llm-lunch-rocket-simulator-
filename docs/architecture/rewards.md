# Reward

Default mode is `delta`.

```
distance_reward = 5 * (previous_distance - current_distance)
speed_reward    = 3 * (previous_speed - current_speed)
upright_reward  = 2 * (previous_tilt - current_tilt)
```

A transition that ends in a safe landing adds exactly 100. A transition that ends in a crash adds exactly -1000. Those two constants live in `simulation/rewards/shaping.py` and are not experiment fields. The API rejects a request that tries to set them.

The reward function does not take an action, a fuel counter or a timestep. Using the engine changes the next state, and the next state changes the score. There is no extra penalty for firing or for taking another step.

Summed across an episode in delta mode, the dense terms telescope to `5*(d0-dT) + 3*(s0-sT) + 2*(tilt0-tiltT)`. A cycle that returns to the same distance, speed and tilt is worth zero. The same identity means a rocket can collect the shaping reward by arriving near the pad, slow and upright, and then waiting until truncation. That behaviour is monitored with `timeout_near_pad` during evaluation. The default formula is unchanged.

Optional mode `potential` uses `weight * (previous - gamma * current)` with the same weights. It is potential-based shaping. It matches the requested formula only when gamma is 1. Experiments record the mode so a run cannot change formula quietly.

Truncation does not add a terminal term.
