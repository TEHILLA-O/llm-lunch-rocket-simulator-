# Research protocol

These questions are the evaluation plan. Numbers belong in `results.md` only after a run writes them. A single smoke test is not a result about landing skill.

1. **Steps until consistent landing.** Train several seeds to the milestone list (0, 10k, 50k, 100k, 250k, 500k, 1M). At each checkpoint, evaluate on a fixed seed list disjoint from training and report mean landing rate with the standard deviation across seeds.

2. **Crash penalty and caution.** Compare the default -1000 penalty with an otherwise identical labelled experiment only if the constant in `shaping.py` is intentionally changed for that study. Until then, use `timeout_near_pad` and the truncation rate as the caution measurement under the published penalty.

3. **Hovering.** Count evaluation episodes that truncate within 5 m of the pad and below 2 m/s. That is the hover-like timeout the telescoping reward allows.

4. **Which term matters.** Ablate by setting one of the dense weights to zero in a new experiment. Do not change the terminal rewards. Compare landing rate, not the raw return, because the return scale changes with the weights.

5. **Generalisation.** Evaluate checkpoints on seeds that were not used for training. The evaluation helper already offsets eval seeds by 100000.

6. **Gravity.** Repeat a trained configuration at other gravity values. Report the training gravity and the test gravity. A mismatch is a different task.

7. **Thruster strength.** Repeat with a different `main_accel` or `attitude_torque`, again as a new experiment.

8. **Reward variant.** Compare `delta` with optional `potential` at the same PPO gamma. Keep terminal rewards fixed.

9. **LLM versus random.** Run the comparison endpoint on the same seeds. If `OPENAI_API_KEY` is unset, record that the LLM was not run. Do not fill in a score.

10. **PPO versus LLM.** Same comparison, with the checkpoint's training configuration noted. A warning is stored when the checkpoint was trained under different physics.

Report the mean and the standard deviation across episodes or seeds. Do not report the best episode as the result.

The PPO policy is an MLP on the eight normalised observations. The LLM controller is not part of that training loop.
