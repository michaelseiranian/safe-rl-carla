# msc-safe-rl-carla/train_agent.py:
import argparse
from leaderboard.leaderboard.leaderboard_evaluator_local import LeaderboardEvaluator
from leaderboard.leaderboard.utils.statistics_manager_local import StatisticsManager


def main():
    parser = argparse.ArgumentParser()

    # ── files & agent ────────────────────────────────────────────────────────
    parser.add_argument('--agent',       required=True)
    parser.add_argument('--routes',      required=True)
    parser.add_argument('--scenarios',   required=True)
    parser.add_argument('--agent-config', default="")
    parser.add_argument('--checkpoint',   default='./simulation_results.json')
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--track',        default='SENSORS')

    # ── repetitions & timeouts ───────────────────────────────────────────────
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--timeout',     default="60.0")
    parser.add_argument('--debug',       type=int, default=0)
    parser.add_argument('--record',      default="")

    # ── NEW: networking flags that LeaderboardEvaluator requires ─────────────
    parser.add_argument('--host', default='localhost',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA RPC port')
    parser.add_argument('--trafficManagerPort',  default='8000')
    parser.add_argument('--trafficManagerSeed',  default='0')
    # ─────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    stats     = StatisticsManager()
    evaluator = LeaderboardEvaluator(args, stats)
    evaluator.run(args)


if __name__ == '__main__':
    main()
