import unittest
from src.field_lookup import choose_field


def box(x, y, size=0.002):
    return {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
            ],
        },
        "properties": {"area_ha": size * size * 1e6},
    }


class FieldLookupTests(unittest.TestCase):
    def test_inside_beats_nearby(self):
        near, inside = box(0.004, 0), box(0, 0)
        result = choose_field([near, inside], 0.001, 0.001)
        self.assertIs(result["feature"], inside)
        self.assertEqual(result["match"], "contains")

    def test_nearest_uses_boundary_not_center(self):
        result = choose_field([box(0.003, 0), box(0.002, 0)], 0.001, 0)
        self.assertEqual(result["match"], "nearest")
        self.assertEqual(result["distance_m"], 223)

    def test_outside_radius_and_empty(self):
        self.assertIsNone(choose_field([box(1, 1)], 0, 0)["feature"])
        self.assertIsNone(choose_field([], 0, 0)["feature"])

    def test_boundary_is_contained(self):
        self.assertEqual(choose_field([box(0, 0)], 0, 0)["match"], "contains")

    def test_nested_prefers_smaller(self):
        small = box(0.001, 0.001)
        self.assertIs(
            choose_field([box(0, 0, 0.02), small], 0.0015, 0.0015)["feature"], small
        )
