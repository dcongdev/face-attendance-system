import unittest

import numpy as np

from core.similarity import cosine_similarity


class CosineSimilarityTests(unittest.TestCase):
    def test_known_angles(self):
        for first, second, expected in [
            ([1, 2, 3], [1, 2, 3], 1),
            ([1, 0], [0, 1], 0),
            ([1, 2], [-1, -2], -1),
            ([1, 0], [1, 1], 1 / np.sqrt(2)),
        ]:
            with self.subTest(first=first, second=second):
                self.assertAlmostEqual(cosine_similarity(first, second), expected)

    def test_facenet_float32_embeddings(self):
        first = np.arange(128, dtype=np.float32) - 64
        second = first[::-1].copy()
        expected = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
        result = cosine_similarity(first, second)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected, places=6)
        self.assertEqual(result, cosine_similarity(second, first))

    def test_scale_invariance_and_extreme_magnitudes(self):
        for scale in (1e-300, 1, 1e300):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(
                    cosine_similarity(np.array([1, 2]) * scale, [3, 6]), 1
                )

    def test_does_not_modify_inputs(self):
        first = np.array([1., 2., 3.])
        original = first.copy()
        cosine_similarity(first, first[::-1])
        np.testing.assert_array_equal(first, original)

    def test_mismatched_dimensions(self):
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_zero_vectors_in_either_position(self):
        for first, second in [([0, 0], [1, 2]), ([1, 2], [0, 0]), ([0], [0])]:
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "zero vectors"):
                    cosine_similarity(first, second)

    def test_invalid_inputs_in_either_position(self):
        invalid = [
            None, 1, [], [[1, 2]], [[1], [2, 3]], "text",
            ["1", "2"], [True, False], [1 + 2j, 3],
            [np.nan, 1], [np.inf, 1], [-np.inf, 1],
            np.array([1, 2], dtype=object),
        ]
        for value in invalid:
            for first, second in [(value, [1, 2]), ([1, 2], value)]:
                with self.subTest(value=value, first_invalid=first is value):
                    with self.assertRaises(ValueError):
                        cosine_similarity(first, second)


if __name__ == "__main__":
    unittest.main()
