// Run with: node --test tests/test_shape_editor.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
function setup() {
  const point = (x, y) => ({
    x,
    y,
    add(p) {
      return point(x + p.x, y + p.y);
    },
    subtract(p) {
      return point(x - p.x, y - p.y);
    },
    distanceTo(p) {
      return Math.hypot(x - p.x, y - p.y);
    },
  });
  const markers = [];
  const layer = () => ({
    addTo() {
      return this;
    },
    on(event, fn) {
      this[event] = fn;
      return this;
    },
  });
  const nodes = new Map();
  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    document: {
      getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, { classList: { toggle() {} } });
        return nodes.get(id);
      },
    },
    L: {
      point,
      divIcon: (x) => x,
      latLngBounds: (...x) => x,
      rectangle: layer,
      polyline: layer,
      bounds(points) {
        const min = point(
          Math.min(...points.map((p) => p.x)),
          Math.min(...points.map((p) => p.y)),
        );
        const max = point(
          Math.max(...points.map((p) => p.x)),
          Math.max(...points.map((p) => p.y)),
        );
        return {
          min,
          max,
          getCenter: () => point((min.x + max.x) / 2, (min.y + max.y) / 2),
        };
      },
      marker(ll, options) {
        const m = {
          ...layer(),
          ll,
          options,
          getLatLng() {
            return this.ll;
          },
        };
        markers.push(m);
        return m;
      },
    },
  });
  vm.runInContext(
    fs.readFileSync("web/app.js", "utf8").replace(/boot\(\);\s*$/, ""),
    context,
  );
  vm.runInContext(
    `state.map = {getZoom:()=>10, project:p=>L.point(p.lng ?? p[1],p.lat ?? p[0]), unproject:p=>({lat:p.y,lng:p.x}), removeLayer() {}};
    state.line={setLatLngs(){},setStyle(){}};`,
    context,
  );
  return { run: (code) => vm.runInContext(code, context), markers };
}
test("freehand closes a stroke, supports vertex editing and restores cancelled strokes", () => {
  const { run } = setup();
  run(`
    const handlers = {};
    const surface = {addEventListener:(name,fn)=>handlers[name]=fn,
      setPointerCapture(){},hasPointerCapture:()=>true,releasePointerCapture(){}};
    state.map.getContainer=()=>surface;
    state.map.mouseEventToLatLng=e=>({lat:e.y,lng:e.x});
    state.map.latLngToContainerPoint=p=>state.map.project(p);
    state.map.dragging={enable(){},disable(){}};
    renderDraft=()=>{};
    initFreehand();
    state.drawing=true; state.shapeKind='freehand';
    const event=(x,y)=>({x,y,pointerId:1,button:0,target:{closest:()=>null},preventDefault(){},stopImmediatePropagation(){}});
    handlers.pointerdown(event(0,0));
    handlers.pointermove(event(20,0));
    handlers.pointermove(event(20,20));
    handlers.pointerup(event(0,20));
  `);
  assert.equal(run("state.closed && draftInfo().valid"), true);
  assert.equal(run("state.vertices.length"), 4);
  assert.equal(run("state.freehandPointer"), null);
  assert.equal(
    run(`updateDraftInfo(); document.getElementById('verticesButton').hidden`),
    false,
  );
  run(
    `state.closed=false;state.vertices=[];handlers.pointerdown(event(0,0));handlers.pointermove(event(10,10));handlers.pointercancel();`,
  );
  assert.equal(run("state.vertices.length"), 0);
  assert.equal(run("state.freehandPointer"), null);
});
test("rectangle and circle produce closed-area-ready vertices", () => {
  const { run } = setup();
  assert.equal(
    run(
      `state.shapeKind='rectangle'; state.vertices=basicShape([0,0],{lat:2,lng:3}); state.closed=true; draftInfo().valid`,
    ),
    true,
  );
  assert.equal(run("state.vertices.length"), 4);
  assert.equal(
    run(
      `state.shapeKind='circle'; state.vertices=basicShape([0,0],{lat:0,lng:3}); draftInfo().valid`,
    ),
    true,
  );
  assert.equal(run("state.vertices.length"), 64);
});
test("move, stretch and rotate use the original geometry and preserve valid contours", () => {
  for (const kind of ["move", "rotate", "scale"]) {
    const { run, markers } = setup();
    run(
      `state.vertices=[[0,0],[0,4],[2,4],[2,0]]; state.closed=true; renderTransformHandles();`,
    );
    const m = markers.find((m) => m.options.icon.className.endsWith(kind));
    m.dragstart();
    m.ll = { lat: m.ll.lat + 1, lng: m.ll.lng + 1 };
    m.drag();
    assert.equal(run("draftInfo().valid"), true);
    assert.equal(run("state.history.length"), 1);
    assert.equal(run("state.history[0].vertices[0][0]"), 0);
    assert.notEqual(
      run("JSON.stringify(state.vertices)"),
      "[[0,0],[0,4],[2,4],[2,0]]",
    );
  }
});
