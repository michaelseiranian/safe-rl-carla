# Safe Reinforcement Learning for Autonomous Driving in CARLA

MSc dissertation code, University of Manchester (2025). Author: Michael Seiranian.

This repository contains my implementation and extension of **Lag-U**, the uncertainty-augmented Lagrangian safe reinforcement learning method introduced by Zhang et al. (2024), for autonomous driving in the CARLA simulator.

> **Credit:** Lag-U is not my method. It was introduced in Zhang, Liu, Li, Lin and Li, *Safe Reinforcement Learning in Autonomous Driving With Epistemic Uncertainty Estimation*, IEEE Transactions on Intelligent Transportation Systems 25(10), 2024 ([DOI](https://doi.org/10.1109/TITS.2024.3397700)). My work here is the CARLA implementation on a TD3 backbone, the curriculum and training pipeline, and the analysis of how the method behaved.

## What is in this repository

- **Lag-U on a TD3 backbone (PyTorch):** ensemble reward critics, a separate cost critic, uncertainty-driven exploration and an uncertainty-adaptive safety budget.
- **Baselines:** a fixed-budget Lagrangian method (LAG) and unconstrained TD3.
- **Training pipeline:** a two-GPU setup on a Slurm cluster, with the CARLA simulator and the learner on separate GPUs, a staged curriculum, checkpoint recovery and Weights & Biases tracking.
- **Evaluation:** route completion, cost per km (collisions, lane invasions and red-light violations) and episode duration.

Built on the TransFuser framework (Chitta et al.), with CARLA 0.9.10.1, Python 3.7 and PyTorch 1.11.0.

## Results (MSc dissertation)

Held-out route evaluation, one training run per method. Each number is the mean over route labels of the best checkpoint per label ("Top-1" in the dissertation):

| Method | Route completion | Cost per km |
| --- | --- | --- |
| TD3 (unconstrained) | 74.8% | 144.9 |
| LAG (fixed-budget Lagrangian) | 44.3% | 109.8 |
| Lag-U (this implementation) | 49.1% | 51.0 |

Please read these with care:

- They come from a single training run per method, with no confidence intervals.
- The best checkpoint per label was chosen by ranking on the same evaluation (completion first, then cost per km), so the numbers are optimistic for every method.
- The evaluation ran with Lag-U's deployment-time uncertainty guard switched off (the evaluation job `jobs/run_policy_eval_manifest.sh` passes `--disable-guard`).
- Later analysis showed that the uncertainty-adaptive safety budget never activated during training: value estimates sat at a clip of 200, so the uncertainty ratio stayed below the threshold. The gap above therefore can't be credited to the adaptive budget.
- Late in training, the Lag-U policy also learned to brake hard and stall, keeping its cost low while making little progress along the route.

An earlier version of this README showed placeholder results (for example, 88.0% route completion). Those numbers were not from the dissertation and have been removed.

## Status and known issues

- **Not runnable end to end:** the CARLA environment wrapper (`safe_rl/env/carla_cmdp_env.py` and `ped_spawner.py`) is not in this repository. A `.gitignore` rule (`env/`) excluded it, and the files were later lost.
- **Curriculum:** the dissertation's final runs used a 15-stage, 459k-step curriculum. The files in `curricula/` may not match those runs exactly.
- **Follow-up study:** an independent, pre-registered, multi-seed re-test of Lag-U on Safety-Gymnasium is in this repository: the port and analysis code in `lagu/`, and the pre-registration and its decision log in `prereg.md` (tag `prereg-v1`). A paper is in preparation.

## Citation

If you use this code, please cite the original Lag-U paper:

```bibtex
@article{zhang2024lagu,
  title   = {Safe Reinforcement Learning in Autonomous Driving With Epistemic Uncertainty Estimation},
  author  = {Zhang, Zheng and Liu, Qi and Li, Yanjie and Lin, Ke and Li, Linyu},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {25},
  number  = {10},
  pages   = {13653--13666},
  year    = {2024},
  doi     = {10.1109/TITS.2024.3397700}
}
```

## Acknowledgements

Lag-U is by Zhang et al. (2024). This code builds on TransFuser (Chitta et al.) and on the CARLA leaderboard and scenario runner.

## License

MIT
