const $ = (id) => document.getElementById(id);
const state = {
  polygons: [],
  fields: [],
  layers: new Map(),
  current: null,
  map: null,
  drawing: false,
  closed: false,
  vertices: [],
  markers: [],
  line: null,
  points: [],
  periods: [],
  selectedPeriod: -1,
  analysisToken: 0,
  shapeKind: "polygon",
  activeTool: "select",
  lookupToken: 0,
  freehandPointer: null,
  vertexEditing: false,
  history: [],
};
const escapeHtml = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (value, digits = 1) =>
  Number.isFinite(value)
    ? value.toLocaleString("ru-RU", { maximumFractionDigits: digits })
    : "—";
let toastTimer;
function toast(message) {
  $("toastMessage").textContent = message;
  $("toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(closeToast, 5500);
}
function closeToast() {
  clearTimeout(toastTimer);
  $("toast").classList.remove("show");
  $("toastMessage").textContent = "";
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.text();
      const parsed = new DOMParser().parseFromString(body, "text/html");
      throw new Error(
        parsed.querySelector("pre")?.textContent ||
          (body.startsWith("<") ? `Ошибка запроса (${response.status})` : body),
      );
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}
function initMap() {
  if (!window.L) {
    $("map").innerHTML =
      '<div class="map-fallback">Карта недоступна. Проверьте интернет и обновите страницу.</div>';
    $("discoverButton").disabled = true;
    return;
  }
  $("map").replaceChildren();
  state.map = L.map("map", { zoomControl: false }).setView([46.85, 40.32], 12);
  initNavigationControls();
  window.AgroWeather?.init(state.map, state.maptilerKey);
  L.control.scale({ position: "bottomleft", imperial: false }).addTo(state.map);
  let savedBasemap = "standard";
  try { savedBasemap = localStorage.getItem("agropulse.basemap") || "standard"; } catch {}
  setBasemap(savedBasemap);
  document.querySelectorAll("[data-basemap]").forEach(button => {
    button.onclick = () => setBasemap(button.dataset.basemap);
  });
  state.map.on("click", (event) => {
    if (state.activeTool === "lookup") { lookupField(event.latlng); return; }
    if (state.drawing && !state.closed && state.shapeKind !== "freehand") {
      if (state.vertices.length >= 1000) {
        toast("Достигнут предел: 1000 вершин");
        return;
      }
      rememberDraft();
      if (state.shapeKind !== "polygon" && state.vertices.length === 1) {
        state.vertices = basicShape(state.vertices[0], event.latlng);
        state.closed = true;
      } else state.vertices.push([event.latlng.lat, event.latlng.lng]);
      renderDraft();
    }
  });
  state.map.on("mousemove", (event) => {
    if (
      state.drawing &&
      !state.closed &&
      state.shapeKind !== "polygon" &&
      state.shapeKind !== "freehand" &&
      state.vertices.length === 1
    ) {
      state.line.setLatLngs(basicShape(state.vertices[0], event.latlng));
    }
  });
  state.map.on("zoomend", () => {
    if (state.drawing && state.closed) renderDraft();
  });
  initFreehand();
}
const BASEMAPS = {
  satellite: { source: "satellite", palette: "standard" },
  standard: { source: "osm", palette: "standard" },
  light: { source: "osm", palette: "light" },
  dark: { source: "osm", palette: "dark" },
  topo: { source: "topo", palette: "standard" },
};
function setBasemap(requested) {
  if (!state.map) return;
  const id = Object.hasOwn(BASEMAPS, requested) ? requested : "standard";
  const config = BASEMAPS[id];
  if (id === "satellite" && !state.maptilerKey) {
    $("basemapStatus").textContent = window.AgroI18n?.translate("Спутник недоступен: проверьте локальную настройку ключа MapTiler.") || "Спутник недоступен: проверьте локальную настройку ключа MapTiler.";
    if (!state.basemapLayer) setBasemap("standard");
    return;
  }
  state.basemapId = id;
  $("basemapStatus").textContent = "";
  // Reuse OSM tiles across palette changes, without touching vector overlays.
  if (state.basemapSource !== config.source) {
    if (state.basemapLayer) state.basemapLayer.remove();
    const topo = config.source === "topo";
    const satellite = config.source === "satellite";
    const osmCredit = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    const layer = L.tileLayer(satellite
      ? `https://api.maptiler.com/tiles/satellite-v2/{z}/{x}/{y}.jpg?key=${encodeURIComponent(state.maptilerKey)}`
      : topo
      ? "https://tile.opentopomap.org/{z}/{x}/{y}.png"
      : "https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      maxNativeZoom: topo ? 17 : 19,
      attribution: satellite ? '<a href="https://www.maptiler.com/"><img class="maptiler-logo" src="https://api.maptiler.com/resources/logo.svg" alt="MapTiler" /></a> &copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> | ' + osmCredit : osmCredit + (topo ? ', SRTM | <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)' : ""),
    });
    state.basemapLayer = layer;
    state.basemapSource = config.source;
    let warned = false;
    layer.on("tileerror", () => {
      if (state.basemapLayer !== layer || warned) return;
      warned = true;
      $("basemapStatus").textContent = window.AgroI18n?.translate("Часть карты не загрузилась. Попробуйте другой слой или проверьте интернет.") || "Часть карты не загрузилась. Попробуйте другой слой или проверьте интернет.";
    });
    layer.addTo(state.map);
  }
  state.map.getPane("tilePane").dataset.palette = config.palette;
  document.querySelectorAll("[data-basemap]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.basemap === id));
  });
  try { localStorage.setItem("agropulse.basemap", id); } catch {}
}
function initNavigationControls() {
  const control = L.control({ position: "bottomright" });
  control.onAdd = () => {
    const container = L.DomUtil.create("div", "map-navigation");
    container.innerHTML = `<div class="map-zoom-group"><button type="button" data-nav="in"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M12 5v14"/></svg></button><button type="button" data-nav="out"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14"/></svg></button></div><button type="button" class="map-locate" data-nav="locate"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m20 4-6 16-3-7-7-3Z"/></svg></button>`;
    L.DomEvent.disableClickPropagation(container);
    L.DomEvent.disableScrollPropagation(container);
    const zoomIn = container.querySelector('[data-nav="in"]');
    const zoomOut = container.querySelector('[data-nav="out"]');
    const locate = container.querySelector('[data-nav="locate"]');
    const labels = () => {
      [[zoomIn, "Приблизить"], [zoomOut, "Отдалить"], [locate, "Моё местоположение"]].forEach(([button, label]) => {
        const text = window.AgroI18n?.translate(label) || label;
        button.title = text;
        button.setAttribute("aria-label", text);
      });
    };
    labels();
    document.addEventListener("agropulse:languagechange", labels);
    zoomIn.onclick = () => state.map.zoomIn();
    zoomOut.onclick = () => state.map.zoomOut();
    const updateZoom = () => {
      zoomIn.disabled = state.map.getZoom() >= state.map.getMaxZoom();
      zoomOut.disabled = state.map.getZoom() <= state.map.getMinZoom();
    };
    state.map.on("zoomend zoomlevelschange", updateZoom);
    updateZoom();
    locate.onclick = () => locateUser(locate);
    return container;
  };
  control.addTo(state.map);
}
function locateUser(button) {
  if (button.disabled) return;
  if (!navigator.geolocation) {
    toast("Геолокация недоступна в этом браузере. Используйте HTTPS или localhost.");
    return;
  }
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  const finish = () => {
    button.disabled = false;
    button.setAttribute("aria-busy", "false");
  };
  // One-shot request, only on an explicit click; no tracking or server persistence.
  navigator.geolocation.getCurrentPosition(position => {
    finish();
    const point = [position.coords.latitude, position.coords.longitude];
    if (state.locationLayer) state.locationLayer.remove();
    state.locationLayer = L.featureGroup([
      L.circle(point, { radius: position.coords.accuracy, color: "#4285ed", weight: 1, fillOpacity: 0.08, interactive: false }),
      L.circleMarker(point, { radius: 7, color: "#fff", weight: 3, fillColor: "#4285ed", fillOpacity: 1, interactive: false }),
    ]).addTo(state.map);
    state.map.setView(point, 16);
  }, error => {
    finish();
    toast(error.code === 1
      ? "Доступ к местоположению запрещён. Разрешите его в настройках браузера."
      : error.code === 3
        ? "Не удалось определить местоположение вовремя. Попробуйте ещё раз."
        : "Местоположение недоступно. Проверьте настройки геолокации устройства.");
  }, { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 });
}
function initFreehand() {
  const container = state.map.getContainer();
  const consume = (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  const append = (event) => {
    const ll = state.map.mouseEventToLatLng(event);
    const last = state.vertices.at(-1);
    if (
      last &&
      state.map
        .latLngToContainerPoint(last)
        .distanceTo(state.map.latLngToContainerPoint(ll)) < 4
    )
      return;
    if (state.vertices.length >= 1000) return;
    state.vertices.push([ll.lat, ll.lng]);
    state.line.setLatLngs(state.vertices);
  };
  container.addEventListener(
    "pointerdown",
    (event) => {
      if (
        !state.drawing ||
        state.closed ||
        state.shapeKind !== "freehand" ||
        event.button !== 0 ||
        state.freehandPointer !== null ||
        event.target.closest(".leaflet-control")
      )
        return;
      consume(event);
      rememberDraft();
      state.vertices = [];
      state.freehandPointer = event.pointerId;
      container.setPointerCapture(event.pointerId);
      append(event);
    },
    true,
  );
  container.addEventListener(
    "pointermove",
    (event) => {
      if (event.pointerId !== state.freehandPointer) return;
      consume(event);
      append(event);
    },
    true,
  );
  container.addEventListener(
    "pointerup",
    (event) => {
      if (event.pointerId !== state.freehandPointer) return;
      consume(event);
      append(event);
      releaseFreehand();
      // A near-identical final point is redundant: Polygon closes the ring itself.
      if (
        state.vertices.length > 3 &&
        state.map
          .latLngToContainerPoint(state.vertices[0])
          .distanceTo(state.map.latLngToContainerPoint(state.vertices.at(-1))) <
          6
      )
        state.vertices.pop();
      if (state.vertices.length < 3) {
        state.vertices = [];
        state.history.pop();
        toast(
          "Обведите область с зажатой кнопкой мыши — нужно минимум три точки.",
        );
      } else {
        state.closed = true;
        state.map.dragging.enable();
        state.vertexEditing = !draftInfo().valid;
        if (!draftInfo().valid)
          toast(
            "Контур пересекает сам себя. Исправьте вершины или нажмите «Назад» и обведите снова.",
          );
      }
      renderDraft();
    },
    true,
  );
  container.addEventListener(
    "pointercancel",
    () => {
      if (state.freehandPointer === null) return;
      releaseFreehand();
      const previous = state.history.pop();
      if (previous) {
        state.vertices = previous.vertices;
        state.closed = previous.closed;
      }
      renderDraft();
    },
    true,
  );
}
function releaseFreehand() {
  const pointer = state.freehandPointer;
  state.freehandPointer = null;
  const container = state.map.getContainer();
  if (pointer !== null && container.hasPointerCapture(pointer))
    container.releasePointerCapture(pointer);
}
function polygonStyle(item) {
  const selected = state.current?.id === item.id;
  return {
    color: selected ? "#264b22" : "#57854b",
    weight: selected ? 3 : 1.7,
    fillColor: selected ? "#cbe17e" : "#84af63",
    fillOpacity: selected ? 0.45 : 0.18,
  };
}
function allFields() {
  return [...(state.showOsmFields === false ? [] : state.fields), ...state.polygons.filter((p) => p.geometry)];
}
function renderMap() {
  if (!state.map) return;
  state.layers.forEach((layer) => state.map.removeLayer(layer));
  state.layers.clear();
  allFields().forEach((item) => {
    const layer = L.geoJSON(item.geometry, { style: polygonStyle(item) }).addTo(
      state.map,
    );
    const tip = document.createElement("span");
    tip.textContent = `${item.name || item.id} · ${number(item.area_ha)} га`;
    layer.bindTooltip(tip);
    layer.on("click", () => {
      if (!state.drawing && !["pan", "lookup"].includes(state.activeTool)) selectField(item);
    });
    state.layers.set(item.id, layer);
  });
}
function fitFields() {
  if (!state.map || !state.layers.size) return;
  const bounds = L.latLngBounds([]);
  state.layers.forEach((layer) => bounds.extend(layer.getBounds()));
  state.map.fitBounds(bounds, { padding: [50, 70], maxZoom: 14 });
}
function fieldThumbnail(geometry) {
  const polygons =
    geometry.type === "MultiPolygon"
      ? geometry.coordinates
      : [geometry.coordinates];
  const rings = polygons.flat();
  // Match the map's Web Mercator projection, with north at the top.
  const projected = rings.map((ring) =>
    ring.map(([lng, lat]) => {
      const phi =
        (Math.max(-85.051129, Math.min(85.051129, lat)) * Math.PI) / 180;
      return [
        (lng * Math.PI) / 180,
        -Math.log(Math.tan(Math.PI / 4 + phi / 2)),
      ];
    }),
  );
  const points = projected.flat();
  if (!points.length || points.some((p) => !p.every(Number.isFinite)))
    return "";
  const minX = Math.min(...points.map((p) => p[0])),
    maxX = Math.max(...points.map((p) => p[0]));
  const minY = Math.min(...points.map((p) => p[1])),
    maxY = Math.max(...points.map((p) => p[1]));
  const scale = 56 / Math.max(maxX - minX, maxY - minY, 1e-12);
  const cx = (minX + maxX) / 2,
    cy = (minY + maxY) / 2;
  const path = projected
    .map(
      (ring) =>
        ring
          .map(
            ([x, y], i) =>
              `${i ? "L" : "M"}${(36 + (x - cx) * scale).toFixed(2)},${(36 + (y - cy) * scale).toFixed(2)}`,
          )
          .join(" ") + " Z",
    )
    .join(" ");
  return `<svg viewBox="0 0 72 72" aria-hidden="true" focusable="false"><path d="${path}" fill="#cbe17e" fill-rule="evenodd" stroke="#365f2d" stroke-width="1.6" stroke-linejoin="round" /></svg>`;
}
function renderList() {
  const saved = state.polygons.filter((p) => p.geometry);
  $("fieldCount").textContent = saved.length;
  $("fieldList").innerHTML = saved.length
    ? saved
        .map(
          (p) =>
            `<button class="field-item ${state.current?.id === p.id ? "active" : ""}" data-id="${escapeHtml(p.id)}"><span class="field-thumbnail">${fieldThumbnail(p.geometry)}</span><span class="field-card-text"><strong>${escapeHtml(p.name || p.id)}</strong><small>${number(p.area_ha)} <span>га</span></small></span><span class="field-card-arrow" aria-hidden="true">↗</span></button>`,
        )
        .join("")
    : '<div class="fields-empty"><span aria-hidden="true">⬠</span><strong>Здесь появятся ваши поля</strong><p>Сохраните контур на карте — он появится в каталоге.</p></div>';
  $("fieldList")
    .querySelectorAll("button")
    .forEach(
      (button) =>
        (button.onclick = () =>
          selectField(saved.find((p) => p.id === button.dataset.id), { fromCatalog: true })),
    );
}
async function loadPolygons() {
  state.polygons = (await api("/api/polygons")).items;
  renderList();
  renderMap();
  const selected = $("polygonSelect").value;
  $("polygonSelect").innerHTML =
    '<option value="">Выберите ряд</option>' +
    state.polygons
      .filter((p) => p.source === "competition_dataset")
      .map(
        (p) =>
          `<option value="${escapeHtml(p.id)}">${escapeHtml(p.id)}</option>`,
      )
      .join("");
  $("polygonSelect").value = selected;
}
function clearAnalysis() {
  $("chartsEmpty").hidden = false;
  clearTimeout(state.pollTimer);
  $("jobProgress").hidden = true;
  $("dataNotes").hidden = true;
  state.analysisToken++;
  state.points = [];
  state.periods = [];
  state.selectedPeriod = -1;
  $("chartPanel").hidden = true;
  document.querySelector(".workspace").classList.remove("has-chart");
  $("fieldSummary").hidden = true;
  $("anomalyList").replaceChildren();
}
// Leaflet's default scale is 100 px wide; keep roughly 500 m on that ruler.
function catalogZoom(latitude) {
  const metersPerPixel = 156543.03392 * Math.cos(latitude * Math.PI / 180);
  return Math.max(0, Math.min(19, Math.floor(Math.log2(metersPerPixel * 100 / 500))));
}
function selectField(item, { fromCatalog = false } = {}) {
  if (state.drawing || !item) return;
  setFieldsOpen(false);
  clearAnalysis();
  state.current = item;
  $("detailPanel").hidden = false;
  setFieldTab("info");
  if (!$("fieldDialog").open) $("fieldDialog").showModal();
  document.body.classList.add("has-detail");
  const dataset = item.source === "competition_dataset",
    saved = item.source === "user_geometry";
  $("fieldTitle").textContent = item.name || item.id;
  $("plotShapePreview").hidden = !item.geometry;
  $("plotShapePreview").innerHTML = item.geometry ? fieldThumbnail(item.geometry) : "";
  $("fieldSource").textContent = dataset
    ? "КОНКУРСНЫЙ ВРЕМЕННОЙ РЯД"
    : saved
      ? "МОЁ ПОЛЕ"
      : "КОНТУР OPENSTREETMAP";
  $("fieldDescription").textContent = dataset
    ? "География скрыта организаторами"
    : saved
      ? "Граница сохранена на этом сервере"
      : "Сельскохозяйственная территория · ODbL";
  $("fieldArea").textContent = item.geometry
    ? `≈ ${number(item.area_ha)} га`
    : "Нет координат";
  $("cropType").textContent = item.crop_type || "Неизвестна";
  ["fieldName", "nameLabel", "saveFieldButton"].forEach(
    (id) => ($(id).hidden = dataset),
  );
  $("saveFieldButton").textContent = saved
    ? "Сохранить название"
    : "Добавить в мои поля";
  $("fieldName").value = item.name || "";
  $("deleteButton").hidden = !saved;
  $("analysisControls").hidden = false;
  $("realPeriod").hidden = dataset;
  $("yearSelect").hidden = !dataset;
  $("yearLabel").hidden = !dataset;
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  $("startDate").max = yesterday;
  $("endDate").max = yesterday;
  const previousYear = new Date().getFullYear() - 1;
  $("startDate").value = `${previousYear}-04-01`;
  $("endDate").value = `${previousYear}-10-31`;
  $("yearSelect").innerHTML = [...(item.years || [])]
    .sort((a, b) => b - a)
    .map((year) => `<option>${year}</option>`)
    .join("");
  $("analysisNotice").textContent = dataset
    ? "Выберите сезон, чтобы сравнить наблюдения, восстановленный NDVI и историческую норму."
    : "Получим спутниковые наблюдения, историю поля и погоду. Первый сбор может занять несколько минут; готовые данные сохраняются.";
  $("analyzeButton").disabled = false;
  $("analyzeButton").textContent = "Проанализировать";
  renderList();
  state.layers.forEach((layer, id) => layer.setStyle(polygonStyle({ id })));
  const layer = state.layers.get(item.id);
  if (layer) {
    layer.bringToFront();
    const mobile = window.innerWidth < 800;
    state.map.fitBounds(layer.getBounds(), {
      paddingTopLeft: mobile ? [30, 60] : [320, 70],
      paddingBottomRight: mobile ? [30, 120] : [340, 50],
      maxZoom: fromCatalog ? catalogZoom(layer.getBounds().getCenter().lat) : 16,
    });
  }
  if (saved) resumeAnalysis(item, state.analysisToken);
}
function closeDetail() {
  $("fieldDialog").close();
  clearTimeout(state.pollTimer);
  state.current = null;
  state.analysisToken++;
  $("detailPanel").hidden = true;
  document.body.classList.remove("has-detail");
  renderList();
  state.layers.forEach((layer) => layer.setStyle(polygonStyle({})));
}
async function saveGeometry(geometry, name) {
  return api("/api/polygons", {
    method: "POST",
    body: JSON.stringify({ geometry, name: name.trim() || "Новое поле" }),
  });
}
async function saveSelected() {
  if (!state.current?.geometry) return;
  const item = state.current;
  $("saveFieldButton").disabled = true;
  try {
    const saved =
      item.source === "user_geometry"
        ? await api(`/api/polygons/${encodeURIComponent(item.id)}`, {
            method: "PATCH",
            body: JSON.stringify({ name: $("fieldName").value }),
          })
        : await saveGeometry(item.geometry, $("fieldName").value);
    await loadPolygons();
    selectField(saved);
    toast(
      item.source === "user_geometry"
        ? "Название сохранено"
        : "Поле добавлено в «Мои поля»",
    );
  } catch (error) {
    toast(error.message);
  } finally {
    $("saveFieldButton").disabled = false;
  }
}
async function deleteSelected() {
  const item = state.current;
  if (!item || item.source !== "user_geometry" || state.deletingField) return;
  if (!confirm(`Удалить «${item.name || item.id}» из моих полей?`)) return;
  state.deletingField = true;
  $("deleteButton").disabled = true;
  try {
    await api(`/api/polygons/${encodeURIComponent(item.id)}`, {
      method: "DELETE",
    });
    if (state.current?.id === item.id) { clearAnalysis(); closeDetail(); }
    await loadPolygons();
    toast("Сохранённый контур удалён");
  } catch (error) {
    toast(error.message);
  } finally {
    state.deletingField = false;
    $("deleteButton").disabled = false;
  }
}
let searchVersion = 0;
let searchTimer;
function savedSearchMatches(query) {
  const normalized = query.toLocaleLowerCase("ru").replaceAll("ё", "е");
  return state.polygons.filter(p => p.geometry && (p.name || "").toLocaleLowerCase("ru").replaceAll("ё", "е").includes(normalized)).slice(0,20);
}
function showSearchResults(items, message = "") {
  const results = $("searchResults");
  results.replaceChildren();
  let lastGroup = "";
  items.forEach(item => {
    const group = item.geometry ? "Ваши участки" : "Города и страны";
    if (group !== lastGroup) {
      const heading = document.createElement("p"); heading.className = "search-note"; heading.textContent = group;
      results.append(heading); lastGroup = group;
    }
    const button = document.createElement("button");
    button.type = "button";
    const title = document.createElement("strong"), subtitle = document.createElement("small");
    title.textContent = item.name;
    subtitle.textContent = item.geometry ? `Сохранённый участок · ${number(item.area_ha)} га` : "Город / регион / страна";
    button.append(title, subtitle);
    button.onclick = () => {
      if (state.drawing) {
        if (state.vertices.length && !confirm("Отменить несохранённый контур и перейти к результату поиска?")) return;
        stopDrawing();
      }
      if (state.activeTool === "lookup") setTool("select");
      searchVersion++;
      results.hidden = true;
      $("regionSearch").value = item.name;
      if (item.geometry) selectField(item);
      else {
        clearAnalysis(); closeDetail(); setFieldsOpen(false);
        if (item.bounds?.length === 4) {
          const [south,north,west,east] = item.bounds;
          state.map.fitBounds([[south,west],[north,east]], {padding:[75,90],maxZoom:13});
        } else state.map.setView([item.lat,item.lon],10);
      }
    };
    results.append(button);
  });
  if (message) {
    const note = document.createElement("p"); note.className = "search-note"; note.textContent = message; results.append(note);
  }
  results.hidden = false;
}
function suggestSavedPlots() {
  const version = ++searchVersion;
  clearTimeout(searchTimer);
  const query = $("regionSearch").value.trim();
  if (query.length < 2) { $("searchResults").hidden = true; return; }
  const matches = savedSearchMatches(query);
  showSearchResults(matches, "Ищем города и страны…");
  searchTimer = setTimeout(() => {
    if (version === searchVersion) searchRegion(null);
  }, 700);
}
async function searchRegion(event) {
  event?.preventDefault();
  clearTimeout(searchTimer);
  if (!state.map) return;
  const version = ++searchVersion;
  const query = $("regionSearch").value.trim();
  if (query.length < 2) return;
  const matches = savedSearchMatches(query);
  showSearchResults(matches, "Ищем города и страны…");
  try {
    const payload = await api(
      `/api/places/suggest?q=${encodeURIComponent(query)}`,
    );
    if (version !== searchVersion) return;
    const items = [...savedSearchMatches(query), ...payload.items];
    showSearchResults(items, items.length ? "Геопоиск: Photon · OpenStreetMap" : "Ничего не найдено. Попробуйте другое название.");
  } catch (error) {
    if (version === searchVersion) showSearchResults(savedSearchMatches(query), `Города временно недоступны. ${error.message}`);
  } finally {
    $("searchButton").disabled = false;
  }
}
function convertFields(payload) {
  return payload.features.map((f) => ({
    ...f.properties,
    id: String(f.properties.polygon_id),
    name: f.properties.name || `Поле ${f.properties.polygon_id}`,
    source: "OpenStreetMap",
    geometry: f.geometry,
    years: [],
  }));
}
function discoverFields() {
  if (!state.map || state.drawing) return;
  if (state.activeTool === "lookup") { setTool("select"); return; }
  clearAnalysis(); closeDetail(); setFieldsOpen(false);
  setTool("lookup");
  state.map.getContainer().style.cursor = "crosshair";
  toast("Нажмите внутри участка. Если контура под точкой нет, ищем ближайший в пределах 500 м. Esc — отмена.");
}
async function lookupField(point) {
  if (state.lookupPending) return;
  state.lookupPending = true;
  const token = ++state.lookupToken;
  updateLookupButton(true);
  try {
    const payload = await api(
      `/api/fields?lat=${point.lat}&lon=${point.lng}`,
    );
    if (token !== state.lookupToken || state.activeTool !== "lookup") return;
    if (!payload.feature) { toast("Рядом не найден подходящий контур OSM. Выберите другую точку или обведите участок вручную."); return; }
    setTool("select");
    startDrawing("polygon");
    state.vertices = payload.feature.geometry.coordinates[0].slice(0,-1).map(([lng,lat])=>[lat,lng]);
    state.closed = true;
    renderDraft();
    state.map.fitBounds(state.line.getBounds(), {padding:[90,100],maxZoom:16});
    toast(payload.match === "contains" ? "Найден контур OSM под точкой. Проверьте границу и сохраните участок." : `Предложен ближайший контур OSM — ${payload.distance_m} м от точки. Проверьте, что это нужный участок.`);
  } catch (error) {
    if (token === state.lookupToken) toast(error.message);
  } finally {
    state.lookupPending = false;
    if (token === state.lookupToken) updateLookupButton(false);
  }
}

// Проверяем геометрию сразу при редактировании; сервер повторяет проверку при сохранении.
function draftInfo() {
  const p = state.vertices;
  if (p.length < 3)
    return { area: 0, valid: false, error: "Добавьте минимум три вершины." };
  const cross = (a, b, c) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const on = (a, b, c) =>
    Math.abs(cross(a, b, c)) < 1e-12 &&
    Math.min(a[0], b[0]) <= c[0] &&
    c[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= c[1] &&
    c[1] <= Math.max(a[1], b[1]);
  const edges = state.closed ? p.length : p.length - 1;
  for (let i = 0; i < edges; i++)
    for (let j = i + 1; j < edges; j++) {
      if (j === i + 1 || (state.closed && i === 0 && j === edges - 1)) continue;
      const a = p[i],
        b = p[(i + 1) % p.length],
        c = p[j],
        d = p[(j + 1) % p.length];
      if (
        (cross(a, b, c) * cross(a, b, d) < 0 &&
          cross(c, d, a) * cross(c, d, b) < 0) ||
        on(a, b, c) ||
        on(a, b, d) ||
        on(c, d, a) ||
        on(c, d, b)
      )
        return {
          area: 0,
          valid: false,
          error: "Границы пересекаются. Перетащите вершины.",
        };
    }
  if (new Set(p.map((v) => v.join(","))).size !== p.length)
    return { area: 0, valid: false, error: "Вершины не должны повторяться." };
  const scale = Math.cos(
      ((p.reduce((s, v) => s + v[0], 0) / p.length) * Math.PI) / 180,
    ),
    xy = p.map((v) => [
      (v[1] - p[0][1]) * 111320 * scale,
      (v[0] - p[0][0]) * 111320,
    ]);
  const area =
    Math.abs(
      xy.reduce((s, a, i) => {
        const b = xy[(i + 1) % xy.length];
        return s + a[0] * b[1] - b[0] * a[1];
      }, 0),
    ) / 20000;
  return {
    area,
    valid: area >= 0.001,
    error: area < 0.001 ? "Площадь слишком мала. Раздвиньте вершины." : "",
  };
}
function rememberDraft() {
  state.history.push({
    vertices: state.vertices.map((p) => [...p]),
    closed: state.closed,
  });
  if (state.history.length > 100) state.history.shift();
}
function basicShape(first, last) {
  const a = state.map.project(first),
    b = state.map.project(last);
  const points =
    state.shapeKind === "rectangle"
      ? [a, L.point(b.x, a.y), b, L.point(a.x, b.y)]
      : Array.from({ length: 64 }, (_, i) => {
          const angle = (i * Math.PI) / 32,
            radius = a.distanceTo(b);
          return L.point(
            a.x + radius * Math.cos(angle),
            a.y + radius * Math.sin(angle),
          );
        });
  return points.map((p) => {
    const ll = state.map.unproject(p);
    return [ll.lat, ll.lng];
  });
}
function updateLookupButton(busy = false) {
  const button = $("discoverButton");
  const label = busy ? "Ищем участок… Нажмите для отмены" : state.activeTool === "lookup"
    ? "Выберите точку на карте · нажмите для отмены" : "Найти участок · выберите точку на карте";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-busy", String(busy));
}
function setTool(tool) {
  if (state.activeTool === "lookup" && tool !== "lookup") {
    state.lookupToken++;
    state.map.getContainer().style.cursor = "";
  }
  state.activeTool = tool;
  updateLookupButton();
  $("discoverButton").setAttribute("aria-pressed", String(tool === "lookup"));
  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.tool === tool));
  });
}
function startDrawing(kind = "polygon") {
  if (!state.map) return;
  if (
    state.drawing &&
    state.vertices.length &&
    !confirm("Заменить несохранённую фигуру новой?")
  )
    return;
  setFieldsOpen(false);
  state.shapeKind = kind;
  releaseFreehand();
  if (kind === "freehand") state.map.dragging.disable();
  else state.map.dragging.enable();
  state.vertexEditing = false;
  state.history = [];
  setTool(kind);
  clearAnalysis();
  closeDetail();
  state.drawing = true;
  state.closed = false;
  state.vertices = [];
  document.body.classList.add("drawing");
  $("drawToolbar").hidden = false;
  state.map.doubleClickZoom.disable();
  state.map.getContainer().style.cursor = "crosshair";
  $("discoverButton").disabled = true;
  renderDraft();
}
function renderDraft() {
  if (state.line) state.map.removeLayer(state.line);
  state.markers.forEach((m) => state.map.removeLayer(m));
  state.markers = [];
  const info = draftInfo();
  state.line = (state.closed ? L.polygon : L.polyline)(state.vertices, {
    color: info.error && state.vertices.length >= 3 ? "#bb4d45" : "#365f2d",
    weight: 3,
    fillColor: "#cbe17e",
    fillOpacity: 0.35,
    interactive: false,
    dashArray: state.closed ? null : "7 5",
  }).addTo(state.map);
  if (!state.closed || state.vertexEditing)
    state.vertices.forEach((p, i) => {
      const marker = L.marker(p, {
        draggable: true,
        icon: L.divIcon({
          className: `vertex ${i === 0 ? "first" : ""}`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        }),
        title:
          i === 0
            ? "Первая вершина. Нажмите, чтобы замкнуть"
            : "Перетащите вершину",
      }).addTo(state.map);
      marker.on("click", () => {
        if (i === 0 && !state.closed) closePolygon();
      });
      marker.on("dragstart", rememberDraft);
      marker.on("drag", () => {
        const v = marker.getLatLng();
        state.vertices[i] = [v.lat, v.lng];
        state.line.setLatLngs(state.vertices);
        updateDraftInfo();
      });
      marker.on("dragend", renderDraft);
      state.markers.push(marker);
    });
  if (state.closed && !state.vertexEditing) renderTransformHandles();
  state.map.getContainer().style.cursor = state.closed ? "" : "crosshair";
  updateDraftInfo();
}
function renderTransformHandles() {
  const map = state.map,
    zoom = map.getZoom();
  const points = state.vertices.map((p) => map.project(p, zoom));
  const bounds = L.bounds(points),
    min = bounds.min,
    max = bounds.max,
    center = bounds.getCenter();
  const toLL = (p) => map.unproject(p, zoom);
  const box = L.rectangle(L.latLngBounds(toLL(min), toLL(max)), {
    color: "#365f2d",
    weight: 1,
    dashArray: "4 5",
    fill: false,
    interactive: false,
  }).addTo(map);
  state.markers.push(box);
  const handles = [
    ["move", center, "Переместить фигуру", "✥"],
    ["rotate", L.point(center.x, min.y - 38), "Повернуть фигуру", "↻"],
    ...[
      [0, 0],
      [0.5, 0],
      [1, 0],
      [1, 0.5],
      [1, 1],
      [0.5, 1],
      [0, 1],
      [0, 0.5],
    ].map(([x, y]) => [
      "scale",
      L.point(min.x + x * (max.x - min.x), min.y + y * (max.y - min.y)),
      "Растянуть фигуру",
      "",
      x,
      y,
    ]),
  ];
  const stem = L.polyline(
    [toLL(L.point(center.x, min.y)), toLL(handles[1][1])],
    {
      color: "#365f2d",
      weight: 1,
      interactive: false,
    },
  ).addTo(map);
  state.markers.push(stem);
  handles.forEach(([kind, origin, title, label, hx, hy]) => {
    const marker = L.marker(toLL(origin), {
      draggable: true,
      title,
      zIndexOffset: 1000,
      icon: L.divIcon({
        className: `transform-handle ${kind}`,
        html: label,
        iconSize: kind === "scale" ? [14, 14] : [28, 28],
        iconAnchor: kind === "scale" ? [7, 7] : [14, 14],
      }),
    }).addTo(map);
    marker.on("dragstart", () => {
      rememberDraft();
      state.markers.forEach((other) => {
        if (other !== marker) map.removeLayer(other);
      });
    });
    marker.on("drag", () => {
      const target = map.project(marker.getLatLng(), zoom);
      const anchor =
        kind === "scale"
          ? L.point(
              hx === 0.5 ? center.x : hx === 0 ? max.x : min.x,
              hy === 0.5 ? center.y : hy === 0 ? max.y : min.y,
            )
          : center;
      const angle =
        Math.atan2(target.y - center.y, target.x - center.x) -
        Math.atan2(origin.y - center.y, origin.x - center.x);
      const sx =
        kind === "scale" && hx !== 0.5
          ? Math.max(0.02, (target.x - anchor.x) / (origin.x - anchor.x || 1))
          : 1;
      const sy =
        kind === "scale" && hy !== 0.5
          ? Math.max(0.02, (target.y - anchor.y) / (origin.y - anchor.y || 1))
          : 1;
      state.vertices = points.map((p) => {
        let next;
        if (kind === "move") next = p.add(target.subtract(origin));
        else if (kind === "rotate") {
          const d = p.subtract(center);
          next = center.add(
            L.point(
              d.x * Math.cos(angle) - d.y * Math.sin(angle),
              d.x * Math.sin(angle) + d.y * Math.cos(angle),
            ),
          );
        } else
          next = L.point(
            anchor.x + (p.x - anchor.x) * sx,
            anchor.y + (p.y - anchor.y) * sy,
          );
        const ll = toLL(next);
        return [ll.lat, ll.lng];
      });
      state.line.setLatLngs(state.vertices);
      updateDraftInfo();
    });
    marker.on("dragend", renderDraft);
    state.markers.push(marker);
  });
}
function updateDraftInfo() {
  const info = draftInfo();
  $("drawArea").textContent = `≈ ${number(info.area)} га`;
  $("drawToolbar").classList.toggle(
    "invalid",
    !info.valid && state.vertices.length >= 3,
  );
  $("drawTitle").textContent = state.closed
    ? "Проверьте границу поля"
    : state.shapeKind === "freehand"
      ? "Обведите поле движением мыши"
      : state.shapeKind === "polygon"
        ? `Отметьте вершины поля · ${state.vertices.length}`
        : state.shapeKind === "circle"
          ? "Круг · центр и радиус"
          : "Прямоугольник · два угла";
  $("drawHint").textContent =
    (state.vertices.length >= 3 ? info.error : "") ||
    (state.closed
      ? state.vertexEditing
        ? "Перетаскивайте вершины. Вернитесь к трансформации кнопкой «Фигура»."
        : "✥ — перемещение · ↻ — поворот · маркеры по краям — растяжение."
      : state.shapeKind === "freehand"
        ? "Зажмите левую кнопку, обведите область и отпустите — контур замкнётся автоматически."
        : state.shapeKind === "polygon"
          ? "Нажмите на первую точку, чтобы замкнуть границу."
          : state.shapeKind === "circle"
            ? "Первый клик — центр, второй — край круга."
            : "Нажмите на два противоположных угла прямоугольника.");
  $("saveButton").disabled = !state.closed || !info.valid;
  $("closePolygonButton").disabled = state.closed || state.vertices.length < 3;
  $("closePolygonButton").hidden =
    state.shapeKind !== "polygon" || state.closed;
  $("verticesButton").hidden =
    !state.closed || !["polygon", "freehand"].includes(state.shapeKind);
  $("verticesButton").textContent = state.vertexEditing ? "Фигура" : "Вершины";
  $("undoButton").disabled = !state.history.length;
  state.line?.setStyle({
    color: !info.valid && state.vertices.length >= 3 ? "#bb4d45" : "#365f2d",
  });
}
function closePolygon() {
  if (state.vertices.length < 3) {
    toast("Добавьте минимум три вершины");
    return;
  }
  rememberDraft();
  state.closed = true;
  renderDraft();
}
function stopDrawing() {
  releaseFreehand();
  state.map.dragging.enable();
  setTool("select");
  state.history = [];
  state.drawing = false;
  state.closed = false;
  state.vertices = [];
  if (state.line) state.map.removeLayer(state.line);
  state.line = null;
  state.markers.forEach((m) => state.map.removeLayer(m));
  state.markers = [];
  document.body.classList.remove("drawing");
  $("drawToolbar").hidden = true;
  $("discoverButton").disabled = false;
  state.map.doubleClickZoom.enable();
  state.map.getContainer().style.cursor = "";
}
async function saveDraft() {
  if (!state.closed || !draftInfo().valid || state.savingDraft) return;
  state.savingDraft = true;
  const ring = state.vertices.map(([lat, lng]) => [lng, lat]);
  ring.push([...ring[0]]);
  $("saveButton").disabled = true;
  try {
    const saved = await saveGeometry(
      { type: "Polygon", coordinates: [ring] },
      `Моё поле ${state.polygons.filter((p) => p.geometry).length + 1}`,
    );
    stopDrawing();
    await loadPolygons();
    selectField(saved);
    toast("Замкнутый контур сохранён");
  } catch (error) {
    toast(error.message);
    updateDraftInfo();
  } finally {
    state.savingDraft = false;
  }
}

function anomalyPeriods(points) {
  const periods = [];
  let active = null;
  points.forEach((p, i) => {
    const anomalous = p.zscore != null && p.zscore < -1;
    if (!anomalous) {
      active = null;
      return;
    }
    if (!active || new Date(p.date) - new Date(points[i - 1].date) > 86400000) {
      active = { start: i, end: i, worst: p };
      periods.push(active);
    } else {
      active.end = i;
      if (p.zscore < active.worst.zscore) active.worst = p;
    }
  });
  return periods;
}
async function analyze() {
  const field = state.current;
  if (!field) return;
  if (field.source !== "competition_dataset") return startRealAnalysis();
  const year = $("yearSelect").value;
  clearAnalysis();
  const token = state.analysisToken;
  $("analyzeButton").disabled = true;
  $("analyzeButton").textContent = "Анализируем…";
  $("analysisNotice").textContent =
    "Восстанавливаем пропуски и сравниваем ряд с сезонной нормой…";
  try {
    const result = await api(
      `/api/series/${encodeURIComponent(field.id)}?year=${year}`,
    );
    if (token !== state.analysisToken) return;
    displayResult(result, `${field.id} · ${year}`);
  } catch (error) {
    if (token === state.analysisToken) {
      $("analysisNotice").textContent =
        "Не удалось завершить анализ. Попробуйте ещё раз.";
      toast(error.message);
    }
  } finally {
    if (token === state.analysisToken) {
      $("analyzeButton").disabled = false;
      $("analyzeButton").textContent = "Проанализировать";
    }
  }
}
function displayResult(result, title) {
  state.points = result.points
    .slice()
    .sort((a, b) => a.date.localeCompare(b.date));
  state.periods = anomalyPeriods(state.points);
  $("chartTitle").textContent = `${title} · NDVI`;
  $("chartPanel").hidden = false;
  $("chartsEmpty").hidden = true;
  $("chartBody").hidden = false;
  $("toggleChart").setAttribute("aria-expanded", "true");
  $("chartChevron").textContent = "⌄";
  document.querySelector(".workspace").classList.add("has-chart");
  $("fieldSummary").hidden = false;
  $("fieldSummary").innerHTML = [
    [result.summary.observations, "наблюдений"],
    [result.summary.restored, "восстановлено"],
    [state.periods.length, "периодов отклонений"],
    [number(result.summary.mean_ndvi, 3), "средний NDVI"],
  ]
    .map(
      ([value, label]) =>
        `<div><strong>${escapeHtml(value)}</strong><small>${label}</small></div>`,
    )
    .join("");
  $("analysisNotice").textContent = state.points.some((p) => p.zscore != null)
    ? "Анализ завершён. Выберите период отклонения, чтобы выделить его на графике."
    : "Для оценки отклонений недостаточно исторических данных.";
  renderAnomalies();
  renderChart();
  $("jobProgress").hidden = true;
  if (result.method) {
    $("dataNotes").hidden = false;
    $("dataNotes").replaceChildren();
    [
      result.method,
      ...(result.warnings || []),
      result.weather_note,
      result.attribution,
    ]
      .filter(Boolean)
      .forEach((note) => {
        const p = document.createElement("p");
        p.textContent = note;
        $("dataNotes").append(p);
      });
    $("chartTitle").textContent = `${title} · ${result.start} — ${result.end}`;
    if (result.summary.missing)
      $("analysisNotice").textContent +=
        ` Без восстановления: ${result.summary.missing} дат.`;
  }
}
async function resumeAnalysis(field, token) {
  try {
    const { job } = await api(
      `/api/polygons/${encodeURIComponent(field.id)}/analyses/latest`,
    );
    if (token !== state.analysisToken || !job) return;
    $("startDate").value = job.start;
    $("endDate").value = job.end;
    watchJob(job.id, token);
  } catch (error) {
    if (token === state.analysisToken)
      toast("Не удалось проверить предыдущий анализ");
  }
}
async function startRealAnalysis() {
  let field = state.current;
  const start = $("startDate").value,
    end = $("endDate").value;
  if (!start || !end || start > end || end > $("endDate").max) {
    toast("Проверьте даты: окончание должно быть не позже вчерашнего дня");
    return;
  }
  clearAnalysis();
  let token = state.analysisToken;
  $("analyzeButton").disabled = true;
  $("analyzeButton").textContent = "Создаём задание…";
  try {
    if (field.source !== "user_geometry") {
      field = await saveGeometry(
        field.geometry,
        $("fieldName").value || field.name || "Моё поле",
      );
      if (token !== state.analysisToken) return;
      await loadPolygons();
      if (token !== state.analysisToken) return;
      selectField(field);
      clearAnalysis();
      token = state.analysisToken;
      $("startDate").value = start;
      $("endDate").value = end;
      $("analyzeButton").disabled = true;
    }
    const payload = await api(
      `/api/polygons/${encodeURIComponent(field.id)}/analyses`,
      { method: "POST", body: JSON.stringify({ start, end }) },
    );
    if (token !== state.analysisToken) return;
    watchJob(payload.job.id, token);
  } catch (error) {
    if (token === state.analysisToken) {
      $("analysisNotice").textContent = error.message;
      $("analyzeButton").disabled = false;
      $("analyzeButton").textContent = "Повторить анализ";
    }
  }
}
async function watchJob(id, token) {
  if (token !== state.analysisToken) return;
  try {
    const job = await api(`/api/analyses/${encodeURIComponent(id)}`);
    if (token !== state.analysisToken) return;
    $("analysisNotice").textContent = job.message || job.stage;
    $("jobProgress").hidden = !["queued", "running"].includes(job.status);
    $("jobProgress").value = job.progress;
    if (job.status === "completed") {
      displayResult(job.result, state.current.name || state.current.id);
      $("analyzeButton").disabled = false;
      $("analyzeButton").textContent = "Проверить обновления";
    } else if (job.status === "failed") {
      $("analyzeButton").disabled = false;
      $("analyzeButton").textContent = "Повторить анализ";
    } else {
      $("analyzeButton").disabled = true;
      $("analyzeButton").textContent =
        job.status === "queued" ? "В очереди…" : "Собираем данные…";
      state.pollTimer = setTimeout(() => watchJob(id, token), 2000);
    }
  } catch (error) {
    if (token === state.analysisToken) {
      $("analysisNotice").textContent =
        "Связь с сервером прервалась. Сбор продолжится в фоне; откройте поле повторно.";
      $("jobProgress").hidden = true;
      $("analyzeButton").disabled = false;
      $("analyzeButton").textContent = "Проверить анализ";
    }
  }
}
function renderAnomalies() {
  const p = state.points;
  $("anomalyList").innerHTML = state.periods.length
    ? state.periods
        .map(
          (period, i) =>
            `<button class="anomaly-item ${period.worst.zscore < -2 ? "critical" : ""} ${i === state.selectedPeriod ? "selected" : ""}" data-period="${i}"><strong>${escapeHtml(p[period.start].date)}${period.end !== period.start ? " — " + escapeHtml(p[period.end].date) : ""}</strong>${period.worst.zscore < -2 ? "Критическое отклонение" : "Угнетение растительности"}<span>${escapeHtml(period.worst.explanation)}</span></button>`,
        )
        .join("")
    : '<p class="muted small">Негативные периоды не выявлены в доступных данных.</p>';
  $("anomalyList")
    .querySelectorAll("button")
    .forEach(
      (b) =>
        (b.onclick = () => {
          state.selectedPeriod = +b.dataset.period;
          $("chartBody").hidden = false;
          $("toggleChart").setAttribute("aria-expanded", "true");
          $("chartChevron").textContent = "⌄";
          renderAnomalies();
          renderChart();
        }),
    );
}
function renderChart() {
  const points = state.points;
  if (!points.length) {
    $("chart").innerHTML = '<p class="empty">Нет наблюдений за этот сезон</p>';
    return;
  }
  const w = 900,
    h = 210,
    left = 40,
    right = 16,
    top = 12,
    bottom = 28;
  const range = points
    .flatMap((p) => [
      p.observed,
      p.filled,
      p.climatology,
      p.climatology != null && p.climatology_std != null
        ? p.climatology + p.climatology_std
        : null,
      p.climatology != null && p.climatology_std != null
        ? p.climatology - p.climatology_std
        : null,
    ])
    .filter(Number.isFinite);
  const min = Math.min(-0.1, ...range),
    max = Math.max(0.9, ...range),
    start = Date.parse(points[0].date),
    end = Date.parse(points.at(-1).date);
  const x = (i) =>
      left +
      ((Date.parse(points[i].date) - start) / Math.max(86400000, end - start)) *
        (w - left - right),
    y = (v) => top + ((max - v) / (max - min)) * (h - top - bottom);
  function line(key) {
    let d = "",
      open = false;
    points.forEach((p, i) => {
      if (!Number.isFinite(p[key])) {
        open = false;
        return;
      }
      d += `${open ? "L" : "M"}${x(i)},${y(p[key])} `;
      open = true;
    });
    return d;
  }
  let bands = "",
    segment = [];
  const flush = () => {
    if (segment.length) {
      bands += `<path fill="#d7e4c7" opacity=".65" d="M${segment.map((i) => `${x(i)},${y(points[i].climatology + points[i].climatology_std)}`).join(" L")} L${segment
        .slice()
        .reverse()
        .map(
          (i) =>
            `${x(i)},${y(points[i].climatology - points[i].climatology_std)}`,
        )
        .join(" L")} Z"/>`;
      segment = [];
    }
  };
  points.forEach((p, i) => {
    if (Number.isFinite(p.climatology) && Number.isFinite(p.climatology_std))
      segment.push(i);
    else flush();
  });
  flush();
  const ticks = [min, (min + max) / 2, max]
    .map(
      (v) =>
        `<line x1="${left}" x2="${w - right}" y1="${y(v)}" y2="${y(v)}" stroke="#e6ece3"/><text x="2" y="${y(v) + 4}" fill="#77877b" font-size="11">${v.toFixed(2)}</text>`,
    )
    .join("");
  const periods = state.periods
    .map(
      (p, i) =>
        `<rect x="${x(p.start) - 2}" y="${top}" width="${Math.max(4, x(p.end) - x(p.start) + 4)}" height="${h - top - bottom}" fill="${p.worst.zscore < -2 ? "#d58075" : "#e4b56b"}" opacity="${state.selectedPeriod === i ? 0.45 : 0.16}"/>`,
    )
    .join("");
  const dots = points
    .map((p, i) =>
      Number.isFinite(p.observed)
        ? `<circle cx="${x(i)}" cy="${y(p.observed)}" r="2.7" fill="#2f704b"/>`
        : "",
    )
    .join("");
  const labels = [
    ...new Set([0, Math.floor(points.length / 2), points.length - 1]),
  ]
    .map(
      (i) =>
        `<text x="${x(i)}" y="${h - 5}" text-anchor="middle" fill="#77877b" font-size="11">${points[i].date.slice(5)}</text>`,
    )
    .join("");
  $("chart").innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Исходные наблюдения NDVI, восстановленный ряд, историческая норма и периоды отклонений" tabindex="0">${ticks}${bands}${periods}<path d="${line("climatology")}" fill="none" stroke="#99ad86" stroke-dasharray="4 4"/><path d="${line("filled")}" fill="none" stroke="#c58d34" stroke-width="2"/>${dots}${labels}<line id="chartCursor" x1="0" x2="0" y1="${top}" y2="${h - bottom}" stroke="#52705d" stroke-dasharray="3 3" visibility="hidden"/></svg>`;
  const svg = $("chart").querySelector("svg");
  let current = 0;
  function inspect(i) {
    current = Math.max(0, Math.min(points.length - 1, i));
    const p = points[current];
    $("chartReadout").textContent =
      `${p.date} · NDVI: ${number(p.observed, 3)} · Ряд: ${number(p.filled, 3)} · Z: ${number(p.zscore, 2)} · ${number(p.temperature)} °C · ${number(p.precipitation)} мм`;
    const cursor = $("chartCursor");
    cursor.setAttribute("x1", x(current));
    cursor.setAttribute("x2", x(current));
    cursor.setAttribute("visibility", "visible");
  }
  svg.onpointermove = (e) => {
    const local =
      ((e.clientX - svg.getBoundingClientRect().left) /
        svg.getBoundingClientRect().width) *
      w;
    let nearest = 0;
    points.forEach((_, i) => {
      if (Math.abs(x(i) - local) < Math.abs(x(nearest) - local)) nearest = i;
    });
    inspect(nearest);
  };
  svg.onkeydown = (e) => {
    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
      inspect(current + (e.key === "ArrowRight" ? 1 : -1));
    }
  };
}
function initMovableTools() {
  const panel = document.querySelector(".shape-tools");
  const grip = $("moveTools");
  const workspace = document.querySelector(".workspace");
  const storageKey = "agropulse.tools-position.v1";
  let drag = null;
  const position = () => {
    const rect = panel.getBoundingClientRect(), bounds = workspace.getBoundingClientRect();
    return {x: rect.left - bounds.left, y: rect.top - bounds.top};
  };
  const place = (x,y) => {
    const bounds = workspace.getBoundingClientRect();
    const rect = panel.getBoundingClientRect();
    panel.style.left = `${Math.max(8, Math.min(x, bounds.width - rect.width - 8))}px`;
    panel.style.top = `${Math.max(8, Math.min(y, bounds.height - rect.height - 8))}px`;
  };
  const save = () => { try { localStorage.setItem(storageKey, JSON.stringify(position())); } catch {} };
  const reset = () => {
    panel.style.left = ""; panel.style.top = "";
    try { localStorage.removeItem(storageKey); } catch {}
  };
  $("resetTools").onclick = () => { reset(); toast("Панель инструментов возвращена на исходное место"); };
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) place(saved.x, saved.y);
  } catch {}
  grip.addEventListener("pointerdown", event => {
    if (event.button !== 0 || drag) return;
    event.preventDefault();
    grip.focus({preventScroll:true});
    drag = {id:event.pointerId, clientX:event.clientX, clientY:event.clientY, ...position()};
    grip.setPointerCapture(event.pointerId);
    panel.classList.add("is-moving");
  });
  grip.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.id) return;
    place(drag.x + event.clientX - drag.clientX, drag.y + event.clientY - drag.clientY);
  });
  const finish = (event, cancelled = false) => {
    if (!drag || event.pointerId !== drag.id) return;
    if (cancelled) place(drag.x, drag.y);
    const id = drag.id;
    drag = null;
    panel.classList.remove("is-moving");
    if (grip.hasPointerCapture(id)) grip.releasePointerCapture(id);
    save();
  };
  grip.addEventListener("pointerup", event => finish(event));
  grip.addEventListener("pointercancel", event => finish(event, true));
  grip.addEventListener("lostpointercapture", event => finish(event, true));
  grip.addEventListener("dblclick", reset);
  grip.addEventListener("keydown", event => {
    if (event.key === "Home") { event.preventDefault(); reset(); return; }
    const directions = {ArrowLeft:[-1,0], ArrowRight:[1,0], ArrowUp:[0,-1], ArrowDown:[0,1]};
    if (!directions[event.key]) return;
    event.preventDefault();
    const [dx,dy] = directions[event.key], step = event.shiftKey ? 40 : 10, p = position();
    place(p.x + dx*step, p.y + dy*step); save();
  });
  window.addEventListener("resize", () => {
    const p = position(); place(p.x,p.y);
  });
}
function initSettings() {
  try { state.showOsmFields = localStorage.getItem("agropulse.show-osm") !== "false"; }
  catch { state.showOsmFields = true; }
  const checkbox = $("showOsmFields");
  checkbox.checked = state.showOsmFields;
  checkbox.onchange = () => {
    state.showOsmFields = checkbox.checked;
    try { localStorage.setItem("agropulse.show-osm", String(checkbox.checked)); } catch {}
    if (!checkbox.checked && state.current && state.fields.some(p => p.id === state.current.id)) {
      clearAnalysis(); closeDetail();
    }
    renderMap();
  };
  const setOpen = (open, returnFocus = false) => {
    if (open) $("settingsPanel").showModal();
    else $("settingsPanel").close();
    $("settingsButton").setAttribute("aria-expanded", String(open));
    if (open) $("closeSettings").focus({preventScroll:true});
    else if (returnFocus) $("settingsButton").focus({preventScroll:true});
  };
  $("settingsButton").onclick = () => setOpen(!$("settingsPanel").open);
  $("closeSettings").onclick = () => setOpen(false, true);
  $("settingsPanel").addEventListener("click", event => {
    const rect = $("settingsPanel").getBoundingClientRect();
    if (event.target === $("settingsPanel") && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) setOpen(false, true);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && $("settingsPanel").open) {
      event.preventDefault(); event.stopImmediatePropagation(); setOpen(false, true);
    }
  }, true);
  window.AgroI18n?.init();
}
function deleteSelectionOnKey(event) {
  if (event.key !== "Delete" || event.repeat || event.isComposing || event.ctrlKey || event.altKey || event.metaKey) return;
  const target = event.target;
  if (target?.isContentEditable || target?.closest?.("input, textarea, select, [role='textbox'], [contenteditable]:not([contenteditable='false'])")) return;
  if (state.savingDraft || state.deletingField) return;
  if (state.drawing) {
    event.preventDefault();
    stopDrawing();
    toast("Несохранённый контур удалён");
  } else if (state.current?.source === "user_geometry") {
    event.preventDefault();
    deleteSelected();
  } else if (state.current?.geometry) {
    event.preventDefault();
    const id = state.current.id;
    clearAnalysis(); closeDetail();
    state.fields = state.fields.filter(item => item.id !== id);
    renderMap();
    toast("Несохранённый контур убран с карты");
  }
}
function setFieldTab(tab, focus = false) {
  const charts = tab === "charts";
  $("fieldInfoPane").hidden = charts;
  $("fieldChartsPane").hidden = !charts;
  [$("fieldInfoTab"), $("fieldChartsTab")].forEach((button, index) => {
    const selected = charts === (index === 1);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  if (charts && state.points.length) renderChart();
}
function initFieldDialog() {
  const dialog = $("fieldDialog");
  document.body.append(dialog);
  // Keep the existing chart and analysis handlers; only change their container.
  $("fieldChartsPane").append($("chartPanel"), $("fieldSummary"), $("anomalyList"), $("dataNotes"));
  $("fieldInfoTab").onclick = () => setFieldTab("info");
  $("fieldChartsTab").onclick = () => setFieldTab("charts");
  $("goToFieldInfo").onclick = () => setFieldTab("info", true);
  dialog.querySelector('[role="tablist"]').addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const charts = event.key === "End" || (event.key !== "Home" && $("fieldChartsTab").getAttribute("aria-selected") !== "true");
    setFieldTab(charts ? "charts" : "info", true);
  });
  dialog.addEventListener("cancel", event => { event.preventDefault(); closeDetail(); });
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) closeDetail();
  });
}
async function boot() {
  initFieldDialog();
  try { state.maptilerKey = (await api("/api/map-config")).maptiler_key; } catch { state.maptilerKey = ""; }
  $("closeToast").onclick = closeToast;
  initSettings();
  initMovableTools();
  setFieldsOpen(false);
  initMap();
  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.disabled = !state.map;
    button.onclick = () => {
      if (button.dataset.tool === "pan") {
        if (
          state.drawing &&
          state.vertices.length &&
          !confirm(
            "Сбросить несохранённый контур и перейти к перемещению карты?",
          )
        )
          return;
        stopDrawing();
        clearAnalysis();
        closeDetail();
        setTool("pan");
        state.map.getContainer().style.cursor = "grab";
      } else if (button.dataset.tool === "select") {
        if (state.drawing && state.closed) {
          state.vertexEditing = false;
          setTool("select");
          renderDraft();
        } else if (
          !state.drawing ||
          !state.vertices.length ||
          confirm("Отменить незавершённую фигуру?")
        )
          stopDrawing();
      } else startDrawing(button.dataset.tool);
    };
  });
  $("verticesButton").onclick = () => {
    state.vertexEditing = !state.vertexEditing;
    renderDraft();
  };
  $("cancelDraw").onclick = stopDrawing;
  $("closePolygonButton").onclick = closePolygon;
  $("undoButton").onclick = () => {
    const previous = state.history.pop();
    if (!previous) return;
    state.vertices = previous.vertices;
    state.closed = previous.closed;
    if (state.shapeKind === "freehand" && !state.closed)
      state.map.dragging.disable();
    renderDraft();
  };
  $("saveButton").onclick = saveDraft;
  $("saveFieldButton").onclick = saveSelected;
  $("deleteButton").onclick = deleteSelected;
  $("closeDetail").onclick = closeDetail;
  $("analyzeButton").onclick = analyze;
  $("discoverButton").onclick = discoverFields;
  $("searchForm").onsubmit = searchRegion;
  $("regionSearch").addEventListener("input", suggestSavedPlots);
  $("regionSearch").addEventListener("keydown", event => {
    if (event.key === "ArrowDown" && !$("searchResults").hidden) {
      event.preventDefault(); $("searchResults").querySelector("button")?.focus();
    }
  });
  $("polygonSelect").onchange = () =>
    selectField(state.polygons.find((p) => p.id === $("polygonSelect").value));
  $("yearSelect").onchange = () => {
    clearAnalysis();
    $("analyzeButton").disabled = false;
    $("analyzeButton").textContent = "Проанализировать";
    $("analysisNotice").textContent = "Сезон изменён. Запустите анализ.";
  };
  [$("startDate"), $("endDate")].forEach(
    (input) =>
      (input.onchange = () => {
        clearAnalysis();
        $("analyzeButton").disabled = false;
        $("analyzeButton").textContent = "Проанализировать";
        $("analysisNotice").textContent =
          "Период изменён. Запустите анализ для выбранных дат.";
      }),
  );
  $("toggleFields").onclick = () => {
    const open = document.body.classList.contains("fields-collapsed");
    if (open && state.drawing) {
      if (
        state.vertices.length &&
        !confirm("Отменить несохранённый контур и открыть список полей?")
      )
        return;
      stopDrawing();
    }
    if (open) {
      clearAnalysis();
      closeDetail();
    }
    setFieldsOpen(open);
  };
  $("closeFields").onclick = () => {
    setFieldsOpen(false);
    $("toggleFields").focus();
  };
  $("toggleChart").onclick = () => {
    const hidden = !$("chartBody").hidden;
    $("chartBody").hidden = hidden;
    $("toggleChart").setAttribute("aria-expanded", String(!hidden));
    $("chartChevron").textContent = hidden ? "⌃" : "⌄";
  };
  document.addEventListener("keydown", (e) => {
    if ($("settingsPanel").open) return;
    if ($("fieldDialog").open && e.key === "Escape") {
      e.preventDefault(); closeDetail(); return;
    }
    deleteSelectionOnKey(e);
    if (e.key === "Escape") {
      searchVersion++;
      if (state.activeTool === "lookup") setTool("select");
      setFieldsOpen(false);
      if (state.drawing) stopDrawing();
      $("searchResults").hidden = true;
    }
  });
  document.addEventListener("click", (e) => {
    if (!$("searchForm").contains(e.target)) { searchVersion++; $("searchResults").hidden = true; }
  });
  const results = await Promise.allSettled([
    loadPolygons(),
    api("/api/fields"),
  ]);
  if (results[0].status === "rejected") {
    toast("Не удалось загрузить поля. Обновите страницу.");
    $("fieldList").innerHTML = '<p class="empty">Сервер недоступен</p>';
  }
  if (results[1].status === "fulfilled") {
    state.fields = convertFields(results[1].value);
    renderMap();
    fitFields();
  } else
    toast("Готовые контуры недоступны. Можно нарисовать поле самостоятельно.");
}
function setFieldsOpen(open) {
  document.body.classList.toggle("fields-collapsed", !open);
  $("toggleFields").setAttribute("aria-expanded", String(open));
  $("fieldsPanel").inert = !open;
}
boot();
