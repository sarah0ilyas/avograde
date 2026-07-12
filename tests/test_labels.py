import itertools
import pandas as pd
from avograde.data.labels import build_table, split_by_fruit


def _raw():
    # Mimic the Mendeley sheet: 20 fruit x 6 photos each, 5-stage index label.
    rows = []
    for fruit in range(20):
        for day in range(6):
            rows.append({"Filename": f"f{fruit:02d}_d{day}.jpg",
                         "Sample": f"avo_{fruit:02d}",
                         "Ripening Index": (day % 5) + 1})
    return pd.DataFrame(rows)


def test_build_table_maps_labels_and_paths():
    t = build_table(_raw(), "/imgs")
    assert set(t["label_idx"]) == {0, 1, 2, 3, 4}
    assert t["path"].iloc[0].startswith("/imgs/")
    assert t.attrs["classes"] == [1, 2, 3, 4, 5]


def test_split_is_by_fruit_not_photo():
    t = build_table(_raw(), "/imgs")
    s = split_by_fruit(t, seed=0)
    train_fruit = set(s.train["sample_id"])
    test_fruit = set(s.test["sample_id"])
    assert not (train_fruit & test_fruit)   # no fruit leaks across splits
