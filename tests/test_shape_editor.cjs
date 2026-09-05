// Run with: node --test tests/test_shape_editor.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
test("Delete removes draft, routes saved selection and ignores text input", () => {
  const {run} = setup();
  run(`let removed=0, requested=0;
    stopDrawing=()=>{removed++;state.drawing=false;};
    deleteSelected=()=>requested++;
    toast=()=>{};
    const key={key:'Delete',target:{closest:()=>null},preventDefault(){}};
    state.drawing=true; deleteSelectionOnKey(key);`);
  assert.equal(run('removed'),1);
  run(`state.current={id:'saved',source:'user_geometry'};deleteSelectionOnKey(key);`);
  assert.equal(run('requested'),1);
  run(`deleteSelectionOnKey({...key,target:{closest:()=>({})}});
    deleteSelectionOnKey({...key,repeat:true});
    state.savingDraft=true;deleteSelectionOnKey(key);`);
  assert.equal(run('requested'),1);
});
test("saved deletion asks for confirmation before any request", async () => {
  const {run} = setup();
  run(`let prompts=0, requests=0; state.current={id:'saved',name:'Test',source:'user_geometry'};
    confirm=()=>{prompts++;return false;}; api=async()=>{requests++;};`);
  await run('deleteSelected()');
  assert.equal(run('prompts'),1);
  assert.equal(run('requests'),0);
});
test("location is requested once on click and recovers after denial", () => {
  const {run} = setup();
  run(`let calls=0, failure, message;
    navigator={geolocation:{getCurrentPosition(success,error,options){calls++;failure=error;}}};
    toast=text=>{message=text;};
    const locationButton={disabled:false,setAttribute(){}};
    locateUser(locationButton);locateUser(locationButton);`);
  assert.equal(run('calls'), 1);
  assert.equal(run('locationButton.disabled'), true);
  run('failure({code:1})');
  assert.equal(run('locationButton.disabled'), false);
  assert.match(run('message'), /запрещён/);
});
test("location success centers the map and replaces the previous marker", () => {
  const {run} = setup();
  run(`let centered, removed=0;
    navigator={geolocation:{getCurrentPosition(success){success({coords:{latitude:55,longitude:37,accuracy:20}});}}};
    L.circle=()=>({});L.circleMarker=()=>({});
    L.featureGroup=()=>({addTo(){return this;}});
    state.map={setView(point,zoom){centered={point,zoom};}};
    state.locationLayer={remove(){removed++;}};
    locateUser({disabled:false,setAttribute(){}});`);
  assert.equal(run('removed'), 1);
  assert.equal(run('centered.zoom'), 16);
  assert.equal(run('centered.point[0]'), 55);
});
test("basemap palettes reuse tiles and switching sources preserves plot selection", () => {
  const {run} = setup();
  run(`let created=0,removed=0;const pane={dataset:{}};
    document.querySelectorAll=()=>[];
    state.map.getPane=()=>pane;
    state.current={id:'kept'};
    L.tileLayer=(url,options)=>{created++;return {on(){},addTo(){},remove(){removed++;}};};
    setBasemap('standard');setBasemap('dark');`);
  assert.equal(run('created'),1);
  assert.equal(run('pane.dataset.palette'),'dark');
  run("setBasemap('topo')");
  assert.equal(run('created'),2);
  assert.equal(run('removed'),1);
  assert.equal(run('state.current.id'),'kept');
  run("setBasemap('invalid')");
  assert.equal(run('state.basemapId'),'standard');
});
test("catalog zoom keeps a roughly 500 m scale at different latitudes", () => {
  const {run} = setup();
  for (const latitude of [0, 46.85, 55.75, 65]) {
    const zoom = run(`catalogZoom(${latitude})`);
    const meters = 156543.03392 * Math.cos(latitude * Math.PI / 180) * 100 / 2 ** zoom;
    assert.ok(meters >= 500 && meters < 1000);
  }
});
test("satellite requires a key and uses an unfiltered attributed layer", () => {
  const {run} = setup();
  run(`window={};let tileUrl, tileOptions; const pane={dataset:{}};
    document.querySelectorAll=()=>[]; state.map.getPane=()=>pane;
    state.basemapLayer={remove(){}};state.basemapSource='osm';
    setBasemap('satellite');`);
  assert.equal(run('state.basemapSource'), 'osm');
  run(`state.maptilerKey='test-key';
    L.tileLayer=(url,options)=>{tileUrl=url;tileOptions=options;return {on(){},addTo(){}};};
    setBasemap('satellite');`);
  assert.match(run('tileUrl'), /satellite-v2/);
  assert.match(run('tileOptions.attribution'), /MapTiler/);
  assert.equal(run('pane.dataset.palette'), 'standard');
});
test("field tabs switch panels without clearing analysis", () => {
  const {run} = setup();
  run(`const tabNodes={};
    document.getElementById=id=>tabNodes[id] ||= {hidden:false,setAttribute(key,value){this[key]=value;},focus(){}};
    let chartRenders=0;renderChart=()=>chartRenders++;
    state.points=[{date:'2024-05-01',observed:.6}];
    setFieldTab('charts');`);
  assert.equal(run('tabNodes.fieldInfoPane.hidden'),true);
  assert.equal(run('tabNodes.fieldChartsPane.hidden'),false);
  assert.equal(run('tabNodes.fieldChartsTab["aria-selected"]'),'true');
  assert.equal(run('chartRenders'),1);
  run("setFieldTab('info')");
  assert.equal(run('tabNodes.fieldChartsPane.hidden'),true);
  assert.equal(run('state.points.length'),1);
});
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
