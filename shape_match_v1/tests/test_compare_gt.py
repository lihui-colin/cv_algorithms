import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("compare_gt", Path(__file__).resolve().parents[1] / "scripts/compare_gt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(index, x, y=0, angle=0):
    return dict(index=index, model="test", row=y, column=x, angle_deg=angle,
                scale_row=1, scale_column=1, score=1)


class ComparisonTests(unittest.TestCase):
    def test_reordering(self):
        result = module.compare([row(9, 10.3, .4), row(8, 0)], [row(0, 0), row(1, 10)])
        self.assertEqual(result["associated"], 2)
        self.assertAlmostEqual(result["position_rmse_px"], (.25 / 2) ** .5)
        self.assertEqual(result["comparisons"][0]["prediction_index"], 8)

    def test_missing_and_extra(self):
        result = module.compare([row(0, 50)], [row(0, 0)])
        self.assertEqual(result["associated"], 0)
        self.assertEqual(result["extra"], 1)
        self.assertIsNone(result["position_rmse_px"])

    def test_ambiguity(self):
        with self.assertRaises(ValueError):
            module.compare([row(0, 0), row(1, .1)], [row(0, 0)])

    def test_no_double_use(self):
        with self.assertRaises(ValueError):
            module.compare([row(0, 0)], [row(0, -.1), row(1, .1)])

    def test_empty_reference(self):
        with self.assertRaises(ValueError):
            module.compare([], [])

    def test_label_correction_is_explicit(self):
        with self.assertRaises(ValueError):
            module.compare([row(0, 0)], [row(0, 0)], bundled_labels=True)


if __name__ == "__main__":
    unittest.main()
