# Safe Reinforcement Learning for Autonomous Driving in CARLA

A safety-aware reinforcement learning system for autonomous driving built on the CARLA simulator, featuring uncertainty-aware Lagrangian methods and curriculum learning.

## Overview

This project implements a constrained reinforcement learning approach for training autonomous driving agents with explicit safety guarantees. The system achieves **88% route completion rate on unseen evaluation routes** while maintaining low collision rates through uncertainty-aware constraint enforcement.

### Key Features

- **Uncertainty-Aware Safety**: Novel LagU (Lagrangian with Uncertainty) algorithm that adapts safety budgets based on epistemic uncertainty
- **Curriculum Learning**: Progressive difficulty training across multiple CARLA towns and scenarios
- **Lagrangian Constraint Handling**: Dual variable optimization for cost constraint enforcement
- **GPU Cluster Support**: Slurm integration for distributed training on remote compute clusters
- **Comprehensive Evaluation**: Automated metrics tracking and comparative analysis across baselines

## Project Structure

```
safe-rl-carla/
├── safe_rl/                    # Core safe RL implementation
│   ├── algorithms/             # RL algorithms (LagU, LAG, TD3)
│   │   ├── lagu.py            # Main uncertainty-aware Lagrangian algorithm
│   │   ├── lag.py             # Fixed-budget Lagrangian baseline
│   │   └── td3.py             # Reward-only TD3 baseline
│   ├── env/                   # CARLA environment wrappers
│   │   ├── carla_cmdp_env.py  # Constrained MDP environment
│   │   └── ped_spawner.py     # Traffic/pedestrian spawning
│   ├── eval/                  # Evaluation scripts
│   │   ├── run_policy_eval.py
│   │   └── record_unseen_videos.py
│   ├── utils/                 # Utilities (replay buffer, etc.)
│   └── train_lagu.py          # Main curriculum training script
├── jobs/                       # Slurm job scripts for cluster training
│   ├── train_curriculum.sh    # Main curriculum training job
│   ├── train_parallel.sh      # Parallel training with multiple servers
│   └── eval_routes.sh         # Evaluation job scripts
├── curricula/                  # Curriculum learning manifests
│   ├── manifest.csv           # 6-stage curriculum definition
│   ├── eval/                  # Evaluation route splits (seen/unseen)
│   ├── simple/                # Bootstrap stage routes
│   └── medium/                # Intermediate difficulty routes
├── tools/                      # Analysis and visualization tools
│   ├── eval_report.py         # Comprehensive evaluation reports
│   └── plot_routes.py         # Route visualization
├── team_code_safe_rl/         # Leaderboard integration
│   └── safe_rl_agent.py       # Agent wrapper for CARLA Leaderboard
├── leaderboard/               # CARLA Leaderboard framework
├── scenario_runner/           # CARLA scenario execution engine
└── team_code_autopilot/       # Privileged autopilot (for data gen)
```

## Algorithms

### LagU: Lagrangian with Uncertainty

The primary contribution is the **LagU** algorithm, which combines:

1. **Ensemble Q-Critics**: 3 Q-function ensembles for epistemic uncertainty quantification
2. **Uncertainty-Aware Exploration**: √Var exploration bonus during early training
3. **Adaptive Safety Budget**: Budget tightens when uncertainty is high (risk-sensitive)
4. **Lagrangian Constraint Handling**: Dual variable λ optimized per-episode
5. **Safety Guard**: Hard brake when uncertainty exceeds threshold at evaluation

**Key Innovation**: The adaptive budget mechanism `δ = δ₀ * min(1, T_thresh / (√Var / |Q|))`
ensures the agent is more conservative when uncertain, preventing dangerous exploratory actions.

### Baselines

- **LAG**: Fixed-budget Lagrangian (no uncertainty awareness)
- **TD3**: Standard reward-maximizing TD3 (no cost constraints)

## Setup

### Prerequisites

- CARLA 0.9.10.1
- Python 3.7+
- CUDA-capable GPU (recommended: 2 GPUs for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/michaelseiranian/safe-rl-carla.git
cd safe-rl-carla

# Setup CARLA 0.9.10.1
chmod +x setup_carla.sh
./setup_carla.sh

# Create conda environment
conda env create -f environment.yml
conda activate tfuse

# Install additional dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
```

### Environment Variables

Set up your environment paths:

```bash
export CARLA_ROOT=/path/to/carla
export WORK_DIR=/path/to/safe-rl-carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${WORK_DIR}/scenario_runner":"${WORK_DIR}/leaderboard":${PYTHONPATH}
```

## Training

### Local Training

To train on a single machine:

```bash
# Launch CARLA server
cd ${CARLA_ROOT}
./CarlaUE4.sh -world-port=2000 -opengl &

# Train with curriculum learning
cd ${WORK_DIR}
python safe_rl/train_lagu.py \
    --manifest curricula/manifest.csv \
    --baseline lagu \
    --port 2000 \
    --logdir checkpoints/lagu_experiment
```

### Curriculum-Based Training

The curriculum spans 6 progressive stages (1.5M total steps):

1. **Bootstrap** (Town01, Town02): Straight roads - 200k steps
2. **Medium Intersections** (Town03): T-junctions - 250k steps
3. **Night Driving** (Town05): Low visibility - 300k steps
4. **Roundabouts** (Town06): Complex navigation - 350k steps
5. **Dense Traffic** (Town10HD): High traffic density - 400k steps

Configure curricula in [curricula/manifest.csv](curricula/manifest.csv):

```csv
routes_xml,scenarios_json,steps,label,stagnation
path/to/routes.xml,path/to/scenarios.json,100000,stage_name,400
```

### GPU Cluster Training (Slurm)

For training on a compute cluster:

```bash
# Submit curriculum training job
sbatch jobs/train_curriculum.sh
```

**Resource Requirements**:
- 2 GPUs (GPU0: simulator, GPU1: learner)
- 32-48GB RAM
- 8-16 CPU cores

The Slurm script handles:
- Automatic CARLA map preloading
- Port conflict avoidance (random RPC ports)
- WandB experiment tracking across curriculum stages
- Fault tolerance and cleanup

## Evaluation

### Running Policy Evaluation

Evaluate a trained checkpoint:

```bash
# Launch CARLA server
./CarlaUE4.sh -world-port=2000 -opengl &

# Run evaluation on seen/unseen routes
python safe_rl/eval/run_policy_eval.py \
    --checkpoint checkpoints/lagu_ckpt_2000000.pth \
    --routes curricula/eval/manifest_seen.csv \
    --port 2000 \
    --output eval_logs/lagu_seen.csv
```

### Evaluation Metrics

- **Route Completion (%)**: Primary success metric
- **Cost per km**: Safety violations per distance traveled
  - Collisions (weight: 1.0)
  - Lane invasions (weight: 0.5)
  - Red light violations (weight: 0.2)
- **Episode Duration**: Time efficiency

### Generating Reports

Comprehensive evaluation reports with rankings and visualizations:

```bash
python tools/eval_report.py \
    --eval_dirs eval_logs/lagu_seen eval_logs/lag_seen eval_logs/td3_seen \
    --labels "LagU" "LAG" "TD3" \
    --output reports/comparison_seen.html
```

Generates:
- Top-K ranking by route completion → cost/km → time
- Per-algorithm performance distributions
- Comparative box plots and scatter plots

## Configuration

### Key Hyperparameters (LagU)

```python
# Ensemble configuration
M = 3                          # Number of Q-function ensembles

# Uncertainty parameters
T_thresh = 3.0                 # Uncertainty threshold for budget adaptation
uncertainty_bonus_coef = 0.02  # Exploration bonus coefficient
safety_guard_thresh = 300.0    # Hard brake threshold at eval time

# Lagrangian parameters
lambda_lr = 1e-4              # Dual variable learning rate
delta_0 = 0.05                # Base safety budget (5% cost tolerance)

# Training
actor_lr = 3e-4               # Actor learning rate
critic_lr = 3e-4              # Critic learning rate
actor_dropout = 0.5           # Token dropout for robustness
replay_capacity = 500_000     # Experience replay size
```

### Environment Configuration

```python
# Observation space: vector-only (no images)
- Speed (1D)
- Guidance (cross-track error, heading error, traffic light, distance) (4D)
- Nearby actors (bounding boxes: MAX_ACTORS=10 × 6 features) (60D)

# Action space: continuous control
- Steer ∈ [-1, 1]
- Accel ∈ [-1, 1] (negative = brake)

# Cost function
cost = 1.0 × collisions + 0.5 × lane_invasions + 0.2 × red_lights

# Reward shaping modes
- "speed": Speed-dominant with progress bonus
- "blend": Balanced progress + target speed
- "progress": Progress-primary with speed Gaussian shaping
```

## Results

### Performance Summary

| Algorithm | Route Completion (%) | Cost/km | Collisions/100km |
|-----------|---------------------|---------|------------------|
| **LagU**  | **88.0**           | 0.12    | 2.4              |
| LAG       | 82.5               | 0.18    | 3.6              |
| TD3       | 91.2               | 0.45    | 9.0              |

**Key Findings**:
- LagU achieves best balance of completion and safety
- TD3 highest completion but 3.75× higher collision rate
- Uncertainty awareness critical for safe exploration

### Ablation Studies

The adaptive budget mechanism is crucial:
- **With adaptive budget**: 88% completion, 0.12 cost/km
- **Without adaptive budget (LAG)**: 82.5% completion, 0.18 cost/km

## Troubleshooting

### Common Issues

**CARLA server crashes during training**:
- The curriculum script automatically restarts servers between stages
- Ensure sufficient GPU memory (8GB+ recommended)
- Use `-opengl` flag for headless servers

**Stagnation detection triggered**:
- Curriculum increases stagnation limits progressively (400 → 600 episodes)
- Temporary λ soft resets (×0.25) help unstick stuck policies
- Check replay buffer isn't full of stagnant data

**Uncertainty estimates too high**:
- Ensemble critics need 2000+ policy updates to converge
- Warmup period skips adaptive budget for first 2k steps
- Increase ensemble size M if variance remains high

## Acknowledgments

This project is built on top of the **TransFuser** autonomous driving framework:

- [Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving", PAMI 2023](http://www.cvlibs.net/publications/Chitta2022PAMI.pdf)
- Original TransFuser repository: [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser)

We leverage TransFuser's infrastructure (CARLA leaderboard integration, scenario runner, environment wrappers) but replace the imitation learning approach with constrained reinforcement learning.

### Related Work

Other recent work on safe/constrained RL for autonomous driving:
- [Shalev-Shwartz et al., "Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving", arXiv 2016]
- [Chow et al., "Risk-Constrained Reinforcement Learning with Percentile Risk Criteria", JMLR 2018]
- [Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020]

## License

This project inherits the MIT License from the TransFuser base framework. See [LICENSE](LICENSE) for details.

---

**Author**: Michael Seiranian

**Built with**: CARLA 0.9.10.1 | PyTorch 1.11.0 | Python 3.7
