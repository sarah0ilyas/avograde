import itertools
import pandas as pd
from avograde.data.splits import temporal_split, spatial_split, spatiotemporal_split


def _panel():
    rows = list(itertools.product([f"c{i:02d}" for i in range(30)], range(2005, 2021)))
    df = pd.DataFrame(rows, columns=["region", "year"])
    df["y"] = 0.0
    return df


def test_temporal_has_no_future_leak():
    s = temporal_split(_panel(), "year", val_start=2017, test_start=2019)
    assert s.train["year"].max() < s.test["year"].min()


def test_spatial_has_no_region_overlap():
    s = spatial_split(_panel(), "region", seed=0)
    assert not (set(s.train["region"]) & set(s.test["region"]))


def test_spatiotemporal_has_neither_leak():
    s = spatiotemporal_split(_panel(), "region", "year", test_start=2019, seed=0)
    assert s.train["year"].max() < s.test["year"].min()
    assert not (set(s.train["region"]) & set(s.test["region"]))
