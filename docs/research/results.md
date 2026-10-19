# Results

No multi-seed landing study has been run.

A pipeline check on 19 October 2026 trained one PPO agent for 512 steps (seed 1, two epochs, episode cap 80, one evaluation episode per checkpoint). It is not a landing result:

- experiment `9d6d6d83eb15`
- checkpoints at 0, 256 and 512 steps
- 6 training episodes, all truncated, none landed, none crashed
- mean training return about 6.4
- evaluation landing rate 0.0 at every checkpoint

The 80-step cap is about 1.6 seconds, which is not long enough for a typical start to reach the ground. The zeros above show that the logger, checkpoints and evaluator ran. They do not measure caution, hovering or landing skill.

The automated smoke test trains a PPO model for a few hundred steps, saves it, loads it and completes one evaluation episode. That test is allowed to land or crash. Its purpose is to show that the training stack runs. A landing rate from that test would not be a finding.

When a study is run, write the seed list, step count, mean and standard deviation here, and point at the experiment id in `data/lunarmind.sqlite`. Leave failed or incomplete runs in the record.

The reward analysis that does not need a trained policy is in `protocol.md`: the dense terms telescope, so oscillation cancels, and truncation near the pad can outscore a risky landing attempt under the -1000 crash penalty.
