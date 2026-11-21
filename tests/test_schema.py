from xsids.schema import ALL_FEATURES, FEATURE_GROUPS, HARD_DROP_COLS


def test_43_unique_features():
    assert len(ALL_FEATURES) == 43
    assert len(set(ALL_FEATURES)) == 43


def test_groups_do_not_overlap():
    seen = set()
    for cols in FEATURE_GROUPS.values():
        assert not (seen & set(cols))
        seen |= set(cols)


def test_hard_drop_cols_are_known():
    assert all(c in ALL_FEATURES for c in HARD_DROP_COLS)
    assert "MIN_TTL" in FEATURE_GROUPS["ttl"] and "IN_BYTES" in FEATURE_GROUPS["volume"]
