"""Валидация замкнутого контура и приблизительная площадь в гектарах."""
import math


def validate_geometry(geometry):
    if not isinstance(geometry, dict) or geometry.get('type') != 'Polygon':
        raise ValueError('Ожидается GeoJSON Polygon')
    rings = geometry.get('coordinates')
    if not isinstance(rings, list) or len(rings) != 1:
        raise ValueError('Нужен один внешний контур')
    ring = rings[0]
    if not isinstance(ring, list) or not 4 <= len(ring) <= 1001:
        raise ValueError('Нужно от 3 до 1000 вершин')
    for p in ring:
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p):
            raise ValueError('Некорректные координаты')
        if not (-180 <= p[0] <= 180 and -85 <= p[1] <= 85):
            raise ValueError('Координаты вне диапазона')
    if ring[0] != ring[-1]:
        raise ValueError('Замкните контур')
    points = ring[:-1]
    if len(set(map(tuple, points))) != len(points):
        raise ValueError('Повторяющиеся вершины')
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):
        return abs(cross(a,b,c)) < 1e-12 and min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
    for i in range(len(points)):
        a,b = ring[i:i+2]
        for j in range(i+1,len(points)):
            if j == i+1 or (i == 0 and j == len(points)-1):
                continue
            c,d = ring[j:j+2]
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                raise ValueError('Границы пересекаются. Переместите вершины')
    origin=points[0]
    scale=math.cos(math.radians(sum(p[1] for p in points)/len(points)))
    xy=[((p[0]-origin[0])*111320*scale,(p[1]-origin[1])*111320) for p in ring]
    area=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(xy,xy[1:])))/20000
    if area < .001:
        raise ValueError('Площадь слишком мала')
    return round(area,3)
