"""Choose a containing OSM ring, otherwise the closest boundary within 500 m."""

import math


def choose_field(features, lat, lon, radius=500):
    candidates = []
    for feature in features:
        ring = feature["geometry"]["coordinates"][0]
        xy = [
            ((x - lon) * 111320 * math.cos(math.radians(lat)), (y - lat) * 111320)
            for x, y in ring
        ]
        inside = False
        distance = float("inf")
        for (x, y), (u, v) in zip(xy, xy[1:]):
            if (y > 0) != (v > 0) and 0 < x + (u - x) * (-y) / (v - y):
                inside = not inside
            dx, dy = u - x, v - y
            t = (
                max(0, min(1, -(x * dx + y * dy) / (dx * dx + dy * dy)))
                if dx or dy
                else 0
            )
            distance = min(distance, math.hypot(x + t * dx, y + t * dy))
        contained = inside or distance < 0.01
        if contained or distance <= radius:
            candidates.append(
                (
                    not contained,
                    0 if contained else distance,
                    feature["properties"]["area_ha"],
                    feature,
                )
            )
    if not candidates:
        return {"feature": None, "radius_m": radius}
    outside, distance, _, feature = min(candidates, key=lambda c: c[:3])
    return {
        "feature": feature,
        "match": "nearest" if outside else "contains",
        "distance_m": round(distance),
        "radius_m": radius,
    }
