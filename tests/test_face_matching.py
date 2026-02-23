import unittest

import numpy as np

from core.face_matching import match_embedding


class FaceMatchingTests(unittest.TestCase):
    def test_best_match_facenet_embeddings(self):
        query = np.arange(128, dtype=np.float32) + 1
        student, score = match_embedding(
            query, {"S02": -query, "S01": query.copy()}, threshold=.8
        )
        self.assertEqual(student, "S01")
        self.assertAlmostEqual(score, 1)
        self.assertIsInstance(score, float)

    def test_unknown_retains_best_score(self):
        self.assertEqual(
            match_embedding([1, 0], {"S01": [0, 1], "S02": [-1, 0]}, threshold=.5),
            ("Unknown", 0.),
        )

    def test_inclusive_configurable_threshold(self):
        known = {"S01": [0, 1]}
        self.assertEqual(match_embedding([1, 0], known, threshold=0), ("S01", 0.))
        self.assertEqual(match_embedding([1, 0], known, threshold=.01), ("Unknown", 0.))
        self.assertEqual(match_embedding([1], {"S01": [-1]}, threshold=-1), ("S01", -1.))
        self.assertEqual(match_embedding([1], {"S01": [1]}, threshold=1), ("S01", 1.))

    def test_empty_collection(self):
        self.assertEqual(match_embedding([1, 2], {}, threshold=.8), ("Unknown", None))

    def test_ties_are_independent_of_mapping_order(self):
        for known in ({"S02": [1, 0], "S01": [1, 0]},
                      {"S01": [1, 0], "S02": [1, 0]}):
            self.assertEqual(match_embedding([1, 0], known, threshold=.8), ("S01", 1.))

    def test_negative_scores_select_best(self):
        student, score = match_embedding(
            [1, 0], {"S01": [-1, 0], "S02": [-1, 1]}, threshold=-.8
        )
        self.assertEqual(student, "S02")
        self.assertAlmostEqual(score, -1 / np.sqrt(2))

    def test_invalid_query_even_with_empty_collection(self):
        for query in (None, [], [0, 0], [[1, 2]], [np.nan, 1], [np.inf, 1], [1j, 1]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                match_embedding(query, {}, threshold=.8)

    def test_invalid_known_embeddings_are_not_skipped(self):
        for value in (None, [], [0, 0], [1, 2, 3], [np.nan, 1], ["1", "2"]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "S02"):
                match_embedding([1, 0], {"S01": [1, 0], "S02": value}, threshold=.8)

    def test_invalid_thresholds(self):
        for threshold in (None, "0.8", True, np.bool_(True), np.nan, np.inf, -1.01, 1.01):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                match_embedding([1, 0], {}, threshold=threshold)

    def test_invalid_collections_and_ids(self):
        for known in (None, [], [("S01", [1, 0])], {1: [1, 0]},
                      {"": [1, 0]}, {" ": [1, 0]}, {"Unknown": [1, 0]}):
            with self.subTest(known=known), self.assertRaises(ValueError):
                match_embedding([1, 0], known, threshold=.8)

    def test_inputs_are_not_modified(self):
        query = np.array([1., 2.])
        original = query.copy()
        known = {"S01": query[::-1]}
        match_embedding(query, known, threshold=.8)
        np.testing.assert_array_equal(query, original)
        self.assertEqual(list(known), ["S01"])


if __name__ == "__main__":
    unittest.main()
