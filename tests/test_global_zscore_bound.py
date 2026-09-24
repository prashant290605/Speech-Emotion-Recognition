import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_figures import Data, LADDER, PAIRS
from report_global_zscore_bound import calculate_global_bounds


def test_global_zscore_bound_covers_every_direction_and_competitor():
    summary = calculate_global_bounds(Data(), n_boot=100)
    competitors = {rung for rung in LADDER if rung not in {"none", "zscore"}}

    assert summary["family_size"] == len(PAIRS) * len(competitors) == 8
    assert set(summary["directions"]) == set(PAIRS)
    for item in summary["directions"].values():
        assert set(item["comparisons"]) == competitors
        assert item["comparisons"][item["limiting_rung"]]["global_upper"] == max(
            result["global_upper"] for result in item["comparisons"].values()
        )
        assert all(
            result["n_seeds"] == 5
            for result in item["comparisons"].values()
        )
