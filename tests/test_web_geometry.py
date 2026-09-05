"""Контуры с пересечениями и незамкнутыми границами не попадают в хранилище."""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.pipeline import AnalysisPipeline
from src.web_geometry import validate_geometry


class GeometryTests(unittest.TestCase):
    def geometry(self, ring):
        return {'type': 'Polygon', 'coordinates': [ring]}

    def test_valid_field_area(self):
        area = validate_geometry(self.geometry([[40,47],[40.01,47],[40.01,47.01],[40,47.01],[40,47]]))
        self.assertGreater(area, 80)
        self.assertLess(area, 90)

    def test_reject_invalid_shapes(self):
        rings = [
            [[40,47],[40.01,47],[40.01,47.01],[40,47.01]],
            [[40,47],[40.01,47.01],[40.01,47],[40,47.01],[40,47]],
            [[40,47],[40.01,47],[40.02,47],[40,47]],
            [[40,47],[float('nan'),47],[40,48],[40,47]],
            [[40,47],[40.01,47],[40,47],[40,47.01],[40,47]],
        ]
        for ring in rings:
            with self.subTest(ring=ring), self.assertRaises(ValueError):
                validate_geometry(self.geometry(ring))


class MissingClimatologyTests(unittest.TestCase):
    def test_analysis_uses_other_years_without_climatology_columns(self):
        dates = pd.to_datetime(['2023-06-01','2023-06-02','2023-06-03','2024-06-02'])
        data = pd.DataFrame({'anon_polygon_id':['A']*4, 'date':dates, 'year':dates.year, 'doy':dates.dayofyear, 'primary_ndvi':[.3,.4,.5,np.nan], 'crop_type':['x']*4, 'era5_temp_c':[20]*4, 'era5_precip_mm':[1]*4})
        pipeline = AnalysisPipeline(data)
        with patch.object(pipeline, 'predict_targets', return_value=pd.DataFrame({'prediction':[.2]},index=[3])):
            result = pipeline.analyze_polygon('A',2024)
        self.assertAlmostEqual(result.iloc[0]['climatology_mean_calc'], .4)
        self.assertTrue(np.isfinite(result.iloc[0]['ndvi_zscore_calc']))


if __name__ == '__main__':
    unittest.main()
