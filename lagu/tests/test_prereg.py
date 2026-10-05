"""prereg.md (repository root) names every hypothesis and the pre-registered seed ranges, and the
facts it states about the code (CSV columns, held-out layouts) match the code."""
import re
from dataclasses import fields
from pathlib import Path

from lagu import evaluate, train
from lagu.agent import Config

PREREG = Path(__file__).resolve().parents[2] / "prereg.md"
TEXT = PREREG.read_text(encoding="utf-8")


def section(title: str) -> str:
    return TEXT.split(title, 1)[1].split("\n## ", 1)[0]


def test_prereg_has_hypotheses_and_seed_ranges():
    for h in ("H1", "H2", "H3", "H4", "H5"):
        m = re.search(rf"^### {h} .*?(?=^### |^## )", TEXT, re.M | re.S)
        assert m, f"no section for {h}"
        assert "*Confirm.*" in m.group(0) and "*Refute" in m.group(0), f"{h} lacks confirm/refute"
    seeds = section("## 4. Seeds")
    for label, lo, hi in (("Selection", 0, 2), ("Confirmatory", 100, 114), ("Extension", 115, 119)):
        assert re.search(rf"{label} seeds {lo}\s*[-–]\s*{hi}\b", seeds), f"{label} seeds {lo}-{hi} missing"


def test_prereg_column_names_exist():
    """Every backticked snake_case name that is not an arm, a tag or a Config field is a CSV column."""
    tags = {"tt", "omsw", "lit", "sw", "fix", "epi", "epigg"}
    other = set(train.ARMS) | {f.name for f in fields(Config)} | tags
    names = {t for t in re.findall(r"`([a-z][a-z0-9_]*)`", TEXT)} - other
    assert names and names <= set(train.FIELDS), sorted(names - set(train.FIELDS))


def test_divergence_ignores_columns_undefined_by_construction():
    """jc (non-episodic duals) and would_engage (M = 1) are NaN in healthy runs; they must not make
    a run 'diverged', and the columns that do must exist."""
    s11 = section("## 11.")
    rule = s11.split("*diverges*", 1)[1].split("Values that are undefined", 1)[0]
    cols = set(re.findall(r"`([a-z_]+)`", rule))
    assert cols and cols <= set(train.FIELDS)
    assert not cols & {"jc", "would_engage"} and not any(c.endswith("_avg") for c in cols)
    assert "`jc`" in s11 and "`would_engage`" in s11


def test_heldout_layouts_match_code():
    assert train.EVAL_SEED == 10_000 and "EVAL_SEED = 10000" in TEXT
    assert evaluate.HELDOUT_SEED == train.EVAL_SEED + 1000 and evaluate.STRIDE == 20
    assert "EVAL_SEED + 1000 + 20i + k" in TEXT


def test_analysis_commit_and_omnisafe_reference():
    assert re.search(r"^\*\*Analysis code commit\*\*", TEXT, re.M)
    ref = section("## 14.")
    for task, nums in (("SafetyPointGoal1-v0", ("25.27", "28.00", "18.76", "12.17")),
                       ("SafetyCarGoal1-v0", ("7.31", "33.83", "27.28", "9.50")),
                       ("SafetyPointCircle1-v0", ("83.07", "7.83", "70.95", "0.00"))):
        row = next(ln for ln in ref.splitlines() if ln.startswith(f"| {task} "))
        assert [c.split("±")[0].strip() for c in row.split("|")[2:6]] == list(nums), row
