import numpy as np
import pandas as pd

from xsids.data.overlap import overlap_ceiling, vector_hash


def test_vector_hash_ignores_storage_dtype():
    a = pd.DataFrame({"x": np.array([1, 2, 300], dtype="int16"), "y": np.array([0.5, 1.0, 2.0], dtype="float32")})
    b = pd.DataFrame({"x": np.array([1, 2, 300], dtype="int64"), "y": np.array([0.5, 1.0, 2.0], dtype="float64")})
    assert (vector_hash(a, ["x", "y"]) == vector_hash(b, ["x", "y"])).all()
    assert len(set(vector_hash(a, ["x", "y"]))) == 3


def test_overlap_ceiling_counts_mass_lost_to_majority_label():
    # vector A: 8 luồng lớp 0, 2 luồng lớp 1 -> 2 luồng lớp 1 không thể đoán đúng
    # vector B: chỉ lớp 1 (5 luồng)
    h = np.array([1, 1, 2])
    lab = np.array([0, 1, 1])
    w = np.array([8.0, 2.0, 5.0])
    r = overlap_ceiling(h, lab, w)
    assert np.isclose(r["accuracy_ceiling"], 13 / 15)
    assert np.isclose(r["recall_ceiling"]["1"], 5 / 7)
    assert np.isclose(r["recall_ceiling"]["0"], 1.0)
    assert np.isclose(r["shared_share"]["1"], 2 / 7)
    assert r["n_vectors"] == 2


def test_overlap_ceiling_is_one_without_shared_vectors():
    r = overlap_ceiling(np.array([1, 2, 3]), np.array(["a", "b", "a"]))
    assert r["accuracy_ceiling"] == 1.0 and all(v == 0 for v in r["shared_share"].values())
