export SAFE_RL_CKPT_PATH=/path/to/checkpoints/lagu_ckpt_2000000.pth
bash leaderboard/scripts/local_evaluation.sh \
    --agent          team_code_safe_rl/safe_rl_agent.py \
    --agent-config   "" \
    --routes         curricula/eval_routes.xml \
    --scenarios      leaderboard/data/longest6/eval_scenarios.json \
    --checkpoint     results/final_longest6.json \
    --resume         0
