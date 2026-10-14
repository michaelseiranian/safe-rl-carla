"""Seed and layout registry of the method study (prereg_method.md, section 4 and Appendix C).

Two kinds of integers are registered here:

* a *training seed* sets the network initialisation, replay sampling and the first training layout
  (the trainers call env.reset(seed=seed) once, then let the environment's RNG run; a resumed run
  re-seeds with seed * 1000 + start, see lagu/train.py);
* a *layout id* is the integer passed to env.reset(seed=...) for an evaluation or probe episode.
  In Safety-Gymnasium it fixes the hazard, vase and goal positions of that episode.

USED_LAYOUTS lists every layout range that any file in the repository or the session scratchpad
passed to env.reset before method-v0. It was built by grepping 'reset(seed=' and 'seed_base' in
/Users/michaelseiranian/Desktop/safe-rl-carla, its worktree safe-rl-carla-method and the scratchpad
(2026-10-11), reading every hit, and resolving the episode count where the code takes it from the
command line by reserving the whole thousand (e.g. 20000-20999 for scripts that reset on 20000 + ep).
The study's own blocks (NEW_LAYOUTS) must be disjoint from it and from each other; check() asserts
this and is run by safe2/tests/test_seeds.py.

reset() is the ONLY place where safe2 code calls env.reset: it maps a named block and an index to a
layout id and asserts at run time that the id lies in that block (review item 27). A test asserts
that no other file in safe2/ contains '.reset('.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    lo: int          # first id
    hi: int          # last id, inclusive
    use: str

    def overlaps(self, other: "Block") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def __contains__(self, x: int) -> bool:
        return self.lo <= x <= self.hi


# --- training seeds -------------------------------------------------------------------------
SELECTION_SEEDS = (0, 1, 2)                    # Stage 0 (via P1 B1), P1, P2
EXTRA_SELECTION_SEEDS = (3, 4, 5)              # only the top-3 baseline configs, before B* is frozen
CONFIRMATORY_SEEDS = tuple(range(300, 316))    # B* and BVS*: all 16; other arms a prefix (300-311, 300-307)
BLIND_FROM = 300                               # outcomes of seeds >= 300 are sealed until unblinding
CONTINUATION_SEEDS = (300, 301, 302)           # 3M continuations (descriptive)
EXTRA_CONTINUATION_SEEDS = (303, 304)          # 3M continuations funded only by compute freed by a kill
DEV_TRAINING_SEEDS = tuple(range(900, 1000))   # smoke runs and unit tests (never analysed)
STUDY_TRAINING_SEEDS = SELECTION_SEEDS + EXTRA_SELECTION_SEEDS + CONFIRMATORY_SEEDS
# training seeds of every earlier run (results/ file names and scratchpad --seed flags)
USED_TRAINING_SEEDS = frozenset({0, 1, 2, 7, 9} | set(range(100, 120)))

# --- layout ids used before method-v0 -------------------------------------------------------
USED_LAYOUTS = (
    Block(0, 2, "first training layout of seeds 0-2; lookahead-filter obs_layout/sim1/sim2, benches"),
    Block(7, 7, "first training layout of scratch seed 7 (probe-author, probe-fix, probe-reverify)"),
    Block(9, 9, "first training layout of scratch seed 9"),
    Block(100, 119, "first training layout of seeds 100-119 (gate, gate_literal, gate_x, gate_literal_x)"),
    Block(10_000, 10_099, "lagu in-training evaluation EVAL_SEED + k (k < eval episodes)"),
    Block(11_000, 11_099, "lagu held-out block 11000 + 20i + k and every analysis on it "
                          "(uncertainty_anticipation, gate_dose, dose-verify, g3/forensics, governor-feas)"),
    Block(12_000, 12_999, "probe-author-verify state episodes 12000 + k"),
    Block(15_000, 15_999, "feas-bandcond shield_probe BASE = 15000 + k"),
    Block(20_000, 20_999, "BVS design scripts 20000 + ep (lookahead-filter sim1-4, sceptic-bvs sim3_*), "
                          "g3/reviewer-forensics probe 20000-20002"),
    Block(30_000, 30_999, "diag-critic a8_snap_probe 30000 + k"),
    Block(50_100, 50_104, "redteam/fidelity a3: 50000 + seed, seeds 100-104"),
    Block(60_100, 60_110, "redteam/fidelity a4: 60000 + seed, seeds 100-110"),
    Block(424_242, 424_242, "redteam/usefulness-review signspot"),
    Block(777_777, 777_777, "redteam/usefulness signprobe"),
)

# --- layout ids of this study -----------------------------------------------------------------
TRAIN_EVAL_BASE = 10_000      # in-training evaluation, 10000 + k, k < 20: reused on purpose (selection data only)
TRAIN_EVAL_EPISODES = 20
HELDOUT_BASE = 70_000         # confirmatory held-out block 70000 + 20 i + k
HELDOUT_STRIDE = 20
HELDOUT_SNAPSHOTS = 10        # snapshots at N * {0.82, 0.84, ..., 1.00}
HELDOUT_EPISODES = 20
G0A_BASE, G0A_EPISODES = 90_000, 30      # Stage 0, oracle ceiling (same 30 layouts for each actor)
G0B_BASE, G0B_MAX_EPISODES = 90_100, 100 # G0b(ii) oracle labels; also the PHS quality check (B* actors)
S01_BASE, S01_EPISODES = 11_000, 10      # S0-1 extensions: the lagu held-out layouts 11000-11009 (reused,
                                         # exploratory, on the Lag-U study's existing runs only)
DEV_BASE, DEV_LAST = 90_500, 90_999      # smoke runs and unit tests that roll out episodes

NEW_LAYOUTS = (
    Block(HELDOUT_BASE, HELDOUT_BASE + HELDOUT_STRIDE * HELDOUT_SNAPSHOTS - 1,
          "confirmatory held-out block (every arm, every confirmatory seed, R3 re-evaluation; "
          "every evaluation on it is sealed whatever the training seed)"),
    Block(G0A_BASE, G0A_BASE + G0A_EPISODES - 1, "G0a oracle ceiling"),
    Block(G0B_BASE, G0B_BASE + G0B_MAX_EPISODES - 1, "G0b(ii) oracle-label rollouts; PHS V quality check"),
    Block(DEV_BASE, DEV_LAST, "development: smoke runs and unit tests"),
)
REUSED_LAYOUTS = (Block(TRAIN_EVAL_BASE, TRAIN_EVAL_BASE + TRAIN_EVAL_EPISODES - 1,
                        "in-training evaluation (selection data; same block as lagu)"),)
S01_LAYOUTS = Block(S01_BASE, S01_BASE + S01_EPISODES - 1, "S0-1 extensions on existing Lag-U runs")


def heldout_layout(i: int, k: int) -> int:
    """Layout of episode k of held-out snapshot i."""
    if not (0 <= i < HELDOUT_SNAPSHOTS and 0 <= k < HELDOUT_EPISODES):
        raise ValueError(f"held-out index out of range: i={i}, k={k}")
    return HELDOUT_BASE + HELDOUT_STRIDE * i + k


def heldout_targets(end_step: int) -> list[int]:
    """Snapshot steps of the held-out endpoint: end_step * {0.82, 0.84, ..., 1.00}."""
    return [round(end_step * (0.80 + 0.02 * (i + 1))) for i in range(HELDOUT_SNAPSHOTS)]


def resume_reseed(seed: int, start: int) -> int:
    """Layout of the first episode after a resume (lagu/train.py convention, kept by safe2)."""
    return seed * 1000 + start


def check(max_steps: int = 3_000_000, ckpt_every: int = 50_000) -> None:
    """Assert that the study's layout blocks are disjoint from every earlier use and from each other,
    that no study seed's first or resumed training layout falls inside an evaluation or probe block,
    and that the confirmatory seeds were never used to train anything before."""
    for new in NEW_LAYOUTS:
        for old in USED_LAYOUTS + REUSED_LAYOUTS:
            assert not new.overlaps(old), f"{new.use} overlaps {old.use}"
    for i, a in enumerate(NEW_LAYOUTS):
        for b in NEW_LAYOUTS[i + 1:]:
            assert not a.overlaps(b), f"{a.use} overlaps {b.use}"
    assert not set(CONFIRMATORY_SEEDS) & USED_TRAINING_SEEDS, "a confirmatory seed was used before"
    assert not set(EXTRA_SELECTION_SEEDS) & USED_TRAINING_SEEDS, "an extra selection seed was used before"
    assert not set(DEV_TRAINING_SEEDS) & (USED_TRAINING_SEEDS | set(STUDY_TRAINING_SEEDS)), "dev seed reused"
    study_seeds = STUDY_TRAINING_SEEDS + DEV_TRAINING_SEEDS
    for s in study_seeds:
        firsts = [s] + [resume_reseed(s, step + 1) for step in range(ckpt_every, max_steps + 1, ckpt_every)]
        for x in firsts:
            for blk in NEW_LAYOUTS + REUSED_LAYOUTS + (S01_LAYOUTS,):
                assert x not in blk, f"training layout {x} of seed {s} lies in {blk.use}"


# --- the only env.reset in safe2 ------------------------------------------------------------------
_EVAL_BLOCKS = {
    "train_eval": REUSED_LAYOUTS[0],
    "heldout": NEW_LAYOUTS[0],
    "g0a": NEW_LAYOUTS[1],
    "g0b": NEW_LAYOUTS[2],
    "dev": NEW_LAYOUTS[3],
    "s01": S01_LAYOUTS,
}
_BASES = {"train_eval": TRAIN_EVAL_BASE, "heldout": HELDOUT_BASE, "g0a": G0A_BASE, "g0b": G0B_BASE,
          "dev": DEV_BASE, "s01": S01_BASE}


def layout(block: str, i: int) -> int:
    """Layout id of index i in a named evaluation block (heldout: i = 20 * snapshot + k)."""
    if block not in _EVAL_BLOCKS:
        raise ValueError(f"unknown layout block {block!r}")
    x = _BASES[block] + int(i)
    if i < 0 or x not in _EVAL_BLOCKS[block]:
        raise ValueError(f"index {i} is outside block {block!r} ({_EVAL_BLOCKS[block].use})")
    return x


def reset(env, block: str, i: int | None = None, start: int | None = None):
    """env.reset for every safe2 rollout; asserts the layout's block at run time.

    block "train":  first training episode of training seed i (layout = i);
          "resume": first episode after resuming seed i after step start - 1 (layout = i * 1000 + start);
          "next":   any later training episode (no seed: the environment's own RNG continues);
          otherwise an evaluation block of layout(); i is the index inside it."""
    if block == "next":
        if i is not None or start is not None:
            raise ValueError("'next' takes no index")
        return env.reset()
    if block in ("train", "resume"):
        if i not in STUDY_TRAINING_SEEDS and i not in DEV_TRAINING_SEEDS:
            raise ValueError(f"training seed {i} is not registered")
        x = i if block == "train" else resume_reseed(i, int(start))
        for blk in NEW_LAYOUTS + REUSED_LAYOUTS + (S01_LAYOUTS,):
            if x in blk:
                raise ValueError(f"training layout {x} lies in {blk.use}")
        return env.reset(seed=x)
    return env.reset(seed=layout(block, i))
