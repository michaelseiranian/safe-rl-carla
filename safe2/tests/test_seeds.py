"""safe2.seeds: the study's layout blocks are disjoint from every earlier use, every safe2 reset goes
through safe2.seeds.reset (which asserts the block at run time), and no safe2 code resets an
environment on a literal layout outside the registered blocks."""
import glob
import os
import re

import pytest

from safe2 import seeds as sd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_registry_is_disjoint():
    sd.check()


def test_heldout_block_and_snapshot_targets():
    assert sd.heldout_layout(0, 0) == 70_000 and sd.heldout_layout(9, 19) == 70_199
    assert len({sd.heldout_layout(i, k) for i in range(10) for k in range(20)}) == 200
    with pytest.raises(ValueError):
        sd.heldout_layout(10, 0)
    with pytest.raises(ValueError):
        sd.heldout_layout(0, 20)
    assert sd.heldout_targets(1_000_000) == list(range(820_000, 1_000_001, 20_000))


def test_block_overlap_logic_catches_the_old_mistake():
    old_bvs_design = next(b for b in sd.USED_LAYOUTS if 20_000 in b)
    proposed_in_shortlist = sd.Block(20_000, 20_199, "held-out block 20000 + 20i + k")
    assert proposed_in_shortlist.overlaps(old_bvs_design)
    assert not sd.NEW_LAYOUTS[0].overlaps(old_bvs_design)


def test_confirmatory_seeds_are_fresh():
    assert sd.CONFIRMATORY_SEEDS == tuple(range(300, 316)) and sd.BLIND_FROM == 300
    assert not set(sd.CONFIRMATORY_SEEDS) & sd.USED_TRAINING_SEEDS


def test_safe2_code_resets_only_on_registered_layouts():
    """Every literal base in 'reset(seed=<int> ...' inside safe2/ lies in a new or reused block."""
    lit = re.compile(r"reset\(\s*seed\s*=\s*(\d[\d_]*)")
    allowed = sd.NEW_LAYOUTS + sd.REUSED_LAYOUTS
    bad = []
    for path in glob.glob(os.path.join(ROOT, "safe2", "**", "*.py"), recursive=True):
        if os.path.basename(path) == "test_seeds.py":
            continue
        with open(path) as f:
            for i, line in enumerate(f, 1):
                for m in lit.finditer(line):
                    x = int(m.group(1).replace("_", ""))
                    if not any(x in b for b in allowed):
                        bad.append(f"{os.path.relpath(path, ROOT)}:{i}: {x}")
    assert not bad, "unregistered layouts: " + "; ".join(bad)


class FakeEnv:
    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return seed, {}


def test_reset_helper_maps_blocks_and_refuses_outside_ids():
    env = FakeEnv()
    sd.reset(env, "train", 300)
    sd.reset(env, "resume", 300, start=100_001)
    sd.reset(env, "next")
    sd.reset(env, "train_eval", 19)
    sd.reset(env, "heldout", 20 * 9 + 19)
    sd.reset(env, "g0a", 29)
    sd.reset(env, "g0b", 99)
    sd.reset(env, "s01", 9)
    sd.reset(env, "dev", 499)
    assert env.seeds == [300, 300 * 1000 + 100_001, None, 10_019, 70_199, 90_029, 90_199, 11_009, 90_999]
    for block, i in (("train_eval", 20), ("heldout", 200), ("g0a", 30), ("g0b", 100), ("s01", 10),
                     ("dev", 500), ("heldout", -1), ("nope", 0)):
        with pytest.raises(ValueError):
            sd.reset(env, block, i)
    with pytest.raises(ValueError):
        sd.reset(env, "train", 7)                      # a training seed used before this study
    with pytest.raises(ValueError):
        sd.reset(env, "next", 3)
    assert sd.CONTINUATION_SEEDS == (300, 301, 302) and sd.EXTRA_CONTINUATION_SEEDS == (303, 304)


def test_every_reset_in_safe2_goes_through_the_helper():
    """Review item 27: '.reset(' may appear only in safe2/seeds.py (and in tests' fakes)."""
    bad = []
    for path in glob.glob(os.path.join(ROOT, "safe2", "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, ROOT)
        if rel == os.path.join("safe2", "seeds.py") or os.sep + "tests" + os.sep in path:
            continue
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if ".reset(" in line and "sd.reset(" not in line and "seeds.reset(" not in line:
                    bad.append(f"{rel}:{i}")
    assert not bad, "env.reset outside safe2.seeds.reset: " + "; ".join(bad)
