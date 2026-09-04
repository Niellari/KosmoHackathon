const state = { polygons: [], current: null, map: null, drawPoints: [], drawLayer: null };
const $ = (id) => document.getElementById(id);

function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function initMap() {
  if (!window.L) {
    $('map').innerHTML = '<div class="map-fallback">Базовая карта недоступна. Остальной анализ работает локально.</div>';
    return;
  }
  state.map = L.map('map', { zoomControl: false }).setView([47.23, 39.72], 8);
  L.control.zoom({ position: 'topright' }).addTo(state.map);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap'
  }).addTo(state.map);
}

async function loadPolygons() {
  const payload = await api('/api/polygons');
  state.polygons = payload.items;
  const select = $('polygonSelect');
  select.innerHTML = state.polygons
    .filter(item => item.source === 'competition_dataset')
    .map(item => `<option value="${item.id}">${item.id}</option>`).join('');
  updatePolygonControls();
}

function updatePolygonControls() {
  const polygon = state.polygons.find(item => item.id === $('polygonSelect').value);
  if (!polygon) return;
  state.current = polygon;
  $('cropType').textContent = polygon.crop_type;
  $('yearSelect').innerHTML = [...polygon.years].reverse()
    .map(year => `<option value="${year}">${year}</option>`).join('');
}

function pathFor(points, key, x, y) {
  let path = '';
  let drawing = false;
  points.forEach((point, index) => {
    const value = point[key];
    if (value == null) { drawing = false; return; }
    path += `${drawing ? 'L' : 'M'}${x(index).toFixed(1)},${y(value).toFixed(1)} `;
    drawing = true;
  });
  return path;
}

function renderChart(points) {
  const width = 1000, height = 300;
  const margin = { left: 45, right: 16, top: 16, bottom: 32 };
  const values = points.flatMap(p => [p.observed, p.filled, p.climatology]).filter(v => v != null && Number.isFinite(v));
  const min = Math.min(-0.1, ...values), max = Math.max(0.9, ...values);
  const x = i => margin.left + i * (width - margin.left - margin.right) / Math.max(1, points.length - 1);
  const y = v => margin.top + (max - v) * (height - margin.top - margin.bottom) / (max - min || 1);
  const ticks = [min, min + (max-min)/2, max];
  const grid = ticks.map(v => `<line class="grid-line" x1="${margin.left}" x2="${width-margin.right}" y1="${y(v)}" y2="${y(v)}"/><text class="axis-label" x="4" y="${y(v)+4}">${v.toFixed(2)}</text>`).join('');
  const labels = [0, Math.floor(points.length/2), points.length-1].map(i => `<text class="axis-label" text-anchor="middle" x="${x(i)}" y="${height-5}">${points[i]?.date.slice(5) || ''}</text>`).join('');
  const anomalies = points.map((p,i) => p.zscore != null && p.zscore < -1 ? `<circle class="anomaly-point" cx="${x(i)}" cy="${y(p.filled)}" r="4"><title>${p.date}: ${p.explanation}</title></circle>` : '').join('');
  $('chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${grid}${labels}<path class="line-norm" d="${pathFor(points,'climatology',x,y)}"/><path class="line-filled" d="${pathFor(points,'filled',x,y)}"/><path class="line-observed" d="${pathFor(points,'observed',x,y)}"/>${anomalies}</svg>`;
}

function renderAnomalies(points) {
  const anomalous = points.filter(p => p.zscore != null && p.zscore < -1).sort((a,b) => a.zscore-b.zscore).slice(0, 8);
  if (!anomalous.length) {
    $('anomalyList').innerHTML = '<div class="empty">Негативных отклонений не обнаружено</div>';
    return;
  }
  $('anomalyList').innerHTML = anomalous.map(point => {
    const critical = point.zscore < -2;
    return `<div class="anomaly-item"><strong>${point.date}</strong><span class="badge ${critical ? '' : 'stress'}">${point.status}</span><span>${point.explanation}</span></div>`;
  }).join('');
}

async function analyze() {
  const polygonId = $('polygonSelect').value;
  const year = $('yearSelect').value;
  $('analyzeButton').disabled = true;
  $('analyzeButton').textContent = 'Анализируем…';
  try {
    const result = await api(`/api/series/${encodeURIComponent(polygonId)}?year=${year}`);
    $('viewTitle').textContent = `${result.polygon_id} · ${result.year}`;
    $('observations').textContent = result.summary.observations;
    $('restored').textContent = result.summary.restored;
    $('anomalies').textContent = result.summary.anomalies;
    $('meanNdvi').textContent = result.summary.mean_ndvi?.toFixed(3) ?? '—';
    renderChart(result.points);
    renderAnomalies(result.points);
  } catch (error) {
    toast(`Ошибка анализа: ${error.message}`);
  } finally {
    $('analyzeButton').disabled = false;
    $('analyzeButton').textContent = 'Проанализировать';
  }
}

function startDrawing() {
  if (!state.map) { toast('Для рисования нужна загруженная карта'); return; }
  state.drawPoints = [];
  if (state.drawLayer) state.map.removeLayer(state.drawLayer);
  $('saveButton').disabled = true;
  toast('Кликайте по карте. После трёх точек можно сохранить.');
  state.map.getContainer().style.cursor = 'crosshair';
  state.map.off('click', addDrawPoint);
  state.map.on('click', addDrawPoint);
}

function addDrawPoint(event) {
  state.drawPoints.push([event.latlng.lat, event.latlng.lng]);
  if (state.drawLayer) state.map.removeLayer(state.drawLayer);
  state.drawLayer = L.polygon(state.drawPoints, { color: '#eb7b3b', fillOpacity: .22 }).addTo(state.map);
  $('saveButton').disabled = state.drawPoints.length < 3;
}

async function savePolygon() {
  const coordinates = state.drawPoints.map(([lat,lng]) => [lng,lat]);
  coordinates.push(coordinates[0]);
  try {
    const result = await api('/api/polygons', {
      method: 'POST',
      body: JSON.stringify({ name: 'Пользовательский контур', geometry: { type: 'Polygon', coordinates: [coordinates] } })
    });
    state.map.off('click', addDrawPoint);
    state.map.getContainer().style.cursor = '';
    $('saveButton').disabled = true;
    toast(`Контур ${result.id} сохранён. Для спутникового анализа подключите провайдер данных.`);
  } catch (error) { toast(`Не удалось сохранить: ${error.message}`); }
}

async function boot() {
  initMap();
  $('polygonSelect').addEventListener('change', updatePolygonControls);
  $('analyzeButton').addEventListener('click', analyze);
  $('drawButton').addEventListener('click', startDrawing);
  $('saveButton').addEventListener('click', savePolygon);
  try {
    const [meta] = await Promise.all([api('/api/meta'), loadPolygons()]);
    $('datasetMeta').textContent = `${meta.rows.toLocaleString('ru-RU')} строк · ${meta.polygons} полигонов · ${meta.date_min} — ${meta.date_max}`;
    await analyze();
  } catch (error) { toast(`Ошибка загрузки: ${error.message}`); }
}

boot();
