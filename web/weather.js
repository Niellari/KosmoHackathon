/* MapTiler GFS scalar tiles: decode source channels before applying our palette.
   This pane deliberately sits below plot vectors and outside basemap CSS filters. */
(() => {
  const palettes = {
    temperature: [[-30,88,67,157],[-10,68,137,211],[0,100,196,218],[10,112,198,151],[20,241,211,111],[30,240,141,82],[40,197,62,91]],
    precipitation: [[0,129,161,203],[1,103,140,190],[3,77,113,167],[10,51,81,139],[25,34,57,111],[50,23,39,85]],
  };
  function color(value, mode) {
    const stops = palettes[mode];
    if (mode === "precipitation" && value < 0.1) return [0,0,0,0];
    if (mode === "precipitation") {
      const band = stops.filter(stop => value >= stop[0]).at(-1) || stops[0];
      return band.slice(1).concat(220);
    }
    let hi = stops.findIndex(stop => stop[0] >= value);
    if (hi < 0) hi = stops.length - 1;
    const b = stops[hi], a = stops[Math.max(0,hi-1)];
    const t = a === b ? 0 : Math.max(0,Math.min(1,(value-a[0])/(b[0]-a[0])));
    return [1,2,3].map(i => Math.round(a[i]+(b[i]-a[i])*t)).concat(255);
  }
  function decode(byte, spec) { return spec.min + byte / 255 * (spec.max - spec.min); }
  function rainStrength(value) {
    return Number.isFinite(value) && value >= 0.1 ? Math.min(1,Math.log1p(value)/Math.log(26)) : 0;
  }
  function rainClouds(width,height,sample,origin={x:0,y:0}) {
    const clouds=[];
    // World-aligned spacing avoids random repositioning each time the map moves.
    const startX=-((origin.x%104+104)%104),startY=-((origin.y%96+96)%96);
    for(let y=startY;y<height+48;y+=96) for(let x=startX;x<width+48;x+=104) {
      let cx=x,cy=y,strength=rainStrength(sample(x,y));
      // Narrow rainy bands must not disappear between the regular cloud anchors.
      if(!strength) for(let dy=-40;dy<=40;dy+=16) for(let dx=-40;dx<=40;dx+=16) {
        const candidate=rainStrength(sample(x+dx,y+dy));
        if(candidate>strength) {strength=candidate;cx=x+dx;cy=y+dy;}
      }
      if (strength) clouds.push({x:cx,y:cy,strength});
    }
    return clouds;
  }
  // Marching squares, including a deterministic center test for saddle cells.
  // Corners run clockwise: top-left, top-right, bottom-right, bottom-left.
  function contourSegments(values, level) {
    if (values.some(v => !Number.isFinite(v))) return [];
    const corners = [[0,0],[1,0],[1,1],[0,1]], hits = [];
    for (let i=0;i<4;i++) {
      const j=(i+1)%4, a=values[i], b=values[j];
      if ((a < level) === (b < level)) continue;
      const t=(level-a)/(b-a);
      hits.push([corners[i][0]+t*(corners[j][0]-corners[i][0]),corners[i][1]+t*(corners[j][1]-corners[i][1])]);
    }
    if (hits.length === 2) return [[hits[0],hits[1]]];
    if (hits.length === 4) return (values.reduce((a,b)=>a+b,0)/4 >= level) === (values[0] >= level)
      ? [[hits[0],hits[1]],[hits[2],hits[3]]] : [[hits[0],hits[3]],[hits[1],hits[2]]];
    return [];
  }
  function framesFor(variable, now = Date.now()) {
    const frames = (variable?.keyframes || []).filter(f => /^[\w-]+$/.test(f.id) && Number.isFinite(Date.parse(f.timestamp))).sort((a,b) => Date.parse(a.timestamp)-Date.parse(b.timestamp));
    if (!frames.length) return [];
    const closest = frames.reduce((a,b) => Math.abs(Date.parse(a.timestamp)-now) <= Math.abs(Date.parse(b.timestamp)-now) ? a : b);
    return frames.filter(f => Date.parse(f.timestamp) >= Date.parse(closest.timestamp) && Date.parse(f.timestamp) <= now + 24*3600000 && Date.parse(f.timestamp) >= now-3600000);
  }
  function init(map, key) {
    const el = id => document.getElementById(id);
    const tr = text => window.AgroI18n?.translate(text) || text;
    const pane = map.createPane("weatherPane");
    pane.style.zIndex = 350;
    pane.style.pointerEvents = "none";
    let enabled = false, mode = "temperature", catalog = null, loadedAt = 0, layer = null, generation = 0, controller;
    let statusText = "", cleanupOverlay = () => {}, redrawOverlay = () => {};
    const status = text => { statusText = text; el("weatherStatus").textContent = tr(text); };
    const remove = () => { cleanupOverlay(); cleanupOverlay = () => {}; redrawOverlay = () => {}; if (layer) layer.remove(); layer = null; };
    const variable = () => catalog?.variables?.find(v => v.metadata?.weather_variable?.variable_id === (mode === "temperature" ? "temperature-2m:gfs" : "precipitation-1h:gfs"));
    function populateTimes() {
      const previous = el("weatherTime").value;
      el("weatherTime").replaceChildren();
      framesFor(variable()).forEach(frame => {
        const option = document.createElement("option");
        option.value = frame.timestamp;
        option.textContent = new Date(frame.timestamp).toLocaleString(document.documentElement.lang === "en" ? "en-GB" : "ru-RU", { month:"short", day:"numeric", hour:"2-digit", minute:"2-digit", timeZoneName:"short" });
        el("weatherTime").append(option);
      });
      if ([...el("weatherTime").options].some(o => o.value === previous)) el("weatherTime").value = previous;
      el("weatherTime").disabled = !el("weatherTime").options.length;
    }
    function render() {
      remove();
      const v = variable(), spec = v?.metadata?.weather_variable?.decoding;
      const frame = v?.keyframes?.find(f => f.timestamp === el("weatherTime").value);
      if (!enabled || !frame || !spec || !["R","B"].includes(spec.channels?.toUpperCase())) { status("Нет актуальных погодных данных."); return; }
      const selectedMode = mode;
      const rawTiles = new Map();
      const channel = spec.channels.toUpperCase() === "R" ? 0 : 2;
      const matrices = v.tile_matrix_set?.items || [];
      const maxNativeZoom = Math.max(...matrices.map(m => m.zoom_level));
      if (!Number.isFinite(maxNativeZoom)) { status("Нет актуальных погодных данных."); return; }
      const Tiles = L.GridLayer.extend({
        createTile(coords, done) {
          const canvas = document.createElement("canvas");
          canvas.width = canvas.height = 512;
          const img = new Image();
          img.crossOrigin = "anonymous";
          let finished = false;
          const finish = error => { if (finished) return; finished = true; clearTimeout(timeout); done(error,canvas); };
          const timeout = setTimeout(() => { img.src = ""; finish(new Error("Weather tile timeout")); },15000);
          img.onerror = () => finish(new Error("Weather tile unavailable"));
          img.onload = () => {
            try {
              const ctx = canvas.getContext("2d", { willReadFrequently:true });
              ctx.drawImage(img,0,0,512,512);
              const pixels = ctx.getImageData(0,0,512,512), data = pixels.data;
              rawTiles.set(`${coords.x}/${coords.y}`,new Uint8ClampedArray(data));
              // Retain only a bounded working set while panning at regional scale.
              if (rawTiles.size > 64) rawTiles.delete(rawTiles.keys().next().value);
              for (let i=0;i<data.length;i+=4) {
                if (!data[i+3]) continue;
                const rgba = color(decode(data[i+channel],spec),selectedMode);
                if (selectedMode === "temperature") rgba[3] = 36;
                else rgba[3] = 0; // Draw rain bands at viewport resolution, without tile upscaling blur.
                data.set(rgba,i);
              }
              ctx.putImageData(pixels,0,0); finish(null);
            } catch { finish(new Error("Weather tile decoding failed")); }
          };
          img.src = `https://api.maptiler.com/tiles/${encodeURIComponent(frame.id)}/${coords.z}/${coords.x}/${coords.y}.png?key=${encodeURIComponent(key)}`;
          return canvas;
        },
      });
      // A fixed source zoom keeps viewport sampling and tile coordinates identical.
      const next = new Tiles({ pane:"weatherPane", tileSize:256, minNativeZoom:maxNativeZoom, maxNativeZoom, maxZoom:19, opacity:Number(el("weatherOpacity").value)/100, keepBuffer:1, attribution:'<a href="https://www.maptiler.com/"><img class="maptiler-logo" src="https://api.maptiler.com/resources/logo.svg" alt="MapTiler" /></a> · Weather / GFS' });
      layer = next;
      let failed = false;
      const overlay = document.createElement("canvas");
      overlay.className = "weather-isotherms leaflet-zoom-hide";
      overlay.style.cssText = "position:absolute;pointer-events:none;z-index:5";
      overlay.setAttribute("aria-hidden","true");
      pane.append(overlay);
      let animation=0, lastFrame=0;
      const reducedMotion=window.matchMedia("(prefers-reduced-motion: reduce)");
      const stopAnimation=()=>{cancelAnimationFrame(animation);animation=0;};
      function sample(x,y) {
        const p = map.project(map.containerPointToLatLng([x,y]),maxNativeZoom);
        const world = 512 * 2**maxNativeZoom;
        const px=((p.x*2-0.5)%world+world)%world, py=p.y*2-0.5;
        const ix=Math.floor(px), iy=Math.floor(py), fx=px-ix, fy=py-iy;
        const read = (xx,yy) => {
          xx=(xx+world)%world;
          const tile=rawTiles.get(`${Math.floor(xx/512)}/${Math.floor(yy/512)}`);
          const offset=((yy%512)*512+(xx%512))*4;
          return tile && tile[offset+3] ? decode(tile[offset+channel],spec) : NaN;
        };
        const a=read(ix,iy),b=read(ix+1,iy),c=read(ix,iy+1),d=read(ix+1,iy+1);
        return a*(1-fx)*(1-fy)+b*fx*(1-fy)+c*(1-fx)*fy+d*fx*fy;
      }
      function drawViewport() {
        stopAnimation();
        if (layer !== next) return;
        const size=map.getSize(), ratio=Math.min(window.devicePixelRatio || 1,2);
        overlay.width=Math.round(size.x*ratio); overlay.height=Math.round(size.y*ratio);
        overlay.style.width=`${size.x}px`; overlay.style.height=`${size.y}px`;
        L.DomUtil.setPosition(overlay,map.containerPointToLayerPoint([0,0]));
        const ctx=overlay.getContext("2d"); ctx.scale(ratio,ratio);
        const step=16, cols=Math.ceil(size.x/step)+1, rows=Math.ceil(size.y/step)+1;
        const values=[], valid=[];
        for (let y=0;y<rows;y++) for (let x=0;x<cols;x++) {
          const value=sample(Math.min(x*step,size.x),Math.min(y*step,size.y));
          values.push(value); if (Number.isFinite(value)) valid.push(value);
        }
        if (failed) return;
        if (!valid.length) { status("Нет погодных данных для видимой области. Попробуйте обновить слой."); return; }
        if (selectedMode === "precipitation") {
          const clouds=rainClouds(size.x,size.y,sample,map.project(map.containerPointToLatLng([0,0]),map.getZoom()));
          const mask=document.createElement("canvas");mask.width=size.x;mask.height=size.y;
          const maskCtx=mask.getContext("2d");
          const bands=document.createElement("canvas");bands.width=size.x;bands.height=size.y;
          const bandsCtx=bands.getContext("2d");
          // Clip droplets to data-supported wet areas (never missing tiles).
          // Classify after interpolation, so neither outer edges nor intensity boundaries blur.
          const bandStep=4;
          for(let y=0;y<size.y;y+=bandStep) for(let x=0;x<size.x;x+=bandStep) {
            const value=sample(x+bandStep/2,y+bandStep/2);
            const strength=rainStrength(value);
            if(!strength) continue;
            maskCtx.fillStyle="white";
            maskCtx.fillRect(x,y,bandStep,bandStep);
            const rgba=color(value,"precipitation");
            bandsCtx.fillStyle=`rgba(${rgba[0]},${rgba[1]},${rgba[2]},${rgba[3]/255})`;
            bandsCtx.fillRect(x,y,bandStep,bandStep);
          }
          const rainCanvas=document.createElement("canvas");rainCanvas.width=size.x;rainCanvas.height=size.y;
          const rainCtx=rainCanvas.getContext("2d");
          function paintRain(time) {
            ctx.clearRect(0,0,size.x,size.y);
            ctx.globalAlpha=Number(el("weatherOpacity").value)/100;
            ctx.imageSmoothingEnabled=false;
            ctx.drawImage(bands,0,0);
            rainCtx.clearRect(0,0,size.x,size.y);
            rainCtx.globalCompositeOperation="source-over";
            rainCtx.lineCap="round";
            for(const c of clouds) {
              const count=2+Math.round(c.strength*6);
              for(let i=0;i<count;i++) {
                const phase=((time/1100)+(i*0.618))%1;
                const x=c.x-40+(i*31%83),y=c.y-40+phase*88;
                const alpha=Math.sin(phase*Math.PI)*0.95;
                rainCtx.strokeStyle=`rgba(248,253,255,${alpha})`;
                rainCtx.fillStyle=rainCtx.strokeStyle;
                rainCtx.lineWidth=1.1;
                rainCtx.beginPath();rainCtx.moveTo(x,y-7-c.strength*5);rainCtx.lineTo(x,y);rainCtx.stroke();
                rainCtx.beginPath();rainCtx.arc(x,y,1.5+c.strength*0.6,0,Math.PI*2);rainCtx.fill();
              }
            }
            rainCtx.globalCompositeOperation="destination-in";rainCtx.drawImage(mask,0,0);
            ctx.drawImage(rainCanvas,0,0);
          }
          function tick(time) {
            if(layer!==next || document.hidden || reducedMotion.matches) {animation=0;return;}
            if(time-lastFrame>=40) {paintRain(time);lastFrame=time;}
            animation=requestAnimationFrame(tick);
          }
          paintRain(400);
          if(clouds.length && !document.hidden && !reducedMotion.matches) animation=requestAnimationFrame(tick);
          if (valid.length < values.length) status("Погодные данные покрывают не всю видимую область. Нельзя подтвердить отсутствие осадков.");
          else status(valid.some(v=>v>=0.1) ? "Темнее синий — сильнее осадки. Капли — условная анимация; дождь и снег не разделены." : "В видимой области на выбранный час осадки не прогнозируются. Это не означает отсутствие облаков.");
          return;
        }
        const min=valid.reduce((a,b)=>Math.min(a,b),Infinity),max=valid.reduce((a,b)=>Math.max(a,b),-Infinity), labels=[];
        let segments=0;
        for (let level=Math.ceil(min/2)*2;level<=max;level+=2) {
          ctx.beginPath(); const candidates=[];
          for (let y=0;y<rows-1;y++) for (let x=0;x<cols-1;x++) {
            const corners=[values[y*cols+x],values[y*cols+x+1],values[(y+1)*cols+x+1],values[(y+1)*cols+x]];
            for (const [a,b] of contourSegments(corners,level)) {
              const width=Math.min(step,size.x-x*step),height=Math.min(step,size.y-y*step);
              const ax=x*step+a[0]*width,ay=y*step+a[1]*height,bx=x*step+b[0]*width,by=y*step+b[1]*height;
              ctx.moveTo(ax,ay);ctx.lineTo(bx,by);segments++;
              candidates.push([(ax+bx)/2,(ay+by)/2]);
            }
          }
          ctx.globalAlpha=0.45+Number(el("weatherOpacity").value)/170;
          ctx.strokeStyle="#ffffff";ctx.lineWidth=3.5;ctx.stroke();
          ctx.strokeStyle=level<=0 ? "#2b609e" : "#9b5139";ctx.lineWidth=1.5;ctx.stroke();
          for (const [x,y] of candidates) {
            if (x<35 || y<30 || x>size.x-35 || y>size.y-30 || labels.some(p=>Math.hypot(p[0]-x,p[1]-y)<140)) continue;
            labels.push([x,y]);
            ctx.font="600 12px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";
            const text=`${level>0?"+":""}${level}°C`;
            ctx.lineWidth=5;ctx.strokeStyle="#fff";ctx.strokeText(text,x,y);
            ctx.fillStyle="#334b53";ctx.fillText(text,x,y);
          }
        }
        status(segments ? "Изотермы через 2 °C. Подписи показывают температуру воздуха." : "Здесь температура почти одинаковая. Отдалите карту, чтобы увидеть изотермы.");
        // Always show one contextual value, even if no contour crosses a small plot.
        const center=sample(size.x/2,size.y/2);
        el("weatherValue").textContent=Number.isFinite(center) ? `${tr("В центре карты")} ≈ ${center.toLocaleString(document.documentElement.lang === "en" ? "en-GB" : "ru-RU",{minimumFractionDigits:1,maximumFractionDigits:1})} °C` : "";
      }
      redrawOverlay=drawViewport;
      const hideOverlay=()=>{stopAnimation();overlay.style.visibility="hidden";};
      const refreshOverlay=()=>{overlay.style.visibility="visible";drawViewport();};
      map.on("movestart",hideOverlay); map.on("moveend resize",refreshOverlay);
      const visibilityChanged=()=>{if(document.hidden) stopAnimation();else refreshOverlay();};
      document.addEventListener("visibilitychange",visibilityChanged);
      reducedMotion.addEventListener("change",refreshOverlay);
      cleanupOverlay=()=>{stopAnimation();document.removeEventListener("visibilitychange",visibilityChanged);reducedMotion.removeEventListener("change",refreshOverlay);map.off("movestart",hideOverlay);map.off("moveend resize",refreshOverlay);overlay.remove();};
      status("Загрузка погодного слоя…");
      next.on("tileerror", () => { if (layer !== next) return; failed = true; status("Часть погоды не загрузилась. Проверьте интернет или лимит MapTiler."); el("weatherRetry").hidden = false; });
      next.on("load", () => { if (layer === next && !failed) drawViewport(); });
      next.addTo(map);
      el("weatherLegend").style.background = `linear-gradient(90deg,${palettes[mode].map(s => `rgb(${s.slice(1).join(",")})`).join(",")})`;
      el("weatherLegend").replaceChildren();
      palettes[mode].forEach((stop,index) => {
        const label = document.createElement("span");
        label.textContent = stop[0] + (index === palettes[mode].length-1 ? (mode === "temperature" ? " °C" : " " + tr("мм/ч")) : "");
        el("weatherLegend").append(label);
      });
    }
    async function load() {
      const token = ++generation;
      controller?.abort();
      remove();
      el("weatherValue").textContent="";
      el("weatherRetry").hidden = true;
      status("Загрузка погодного слоя…");
      if (!key) { status("Погода недоступна: не настроен ключ MapTiler."); return; }
      controller = new AbortController();
      const timer = setTimeout(() => controller.abort(),15000);
      try {
        if (!catalog || Date.now()-loadedAt > 15*60000) {
          const response = await fetch(`https://api.maptiler.com/weather/latest.json?key=${encodeURIComponent(key)}`, { signal:controller.signal });
          if (!response.ok) throw new Error("Weather unavailable");
          const result = await response.json();
          if (token !== generation || !enabled) return;
          catalog = result; loadedAt = Date.now();
        }
        if (token !== generation || !enabled) return;
        populateTimes(); render();
      } catch {
        if (token !== generation || !enabled) return;
        status("Не удалось загрузить погоду. Проверьте интернет или лимит MapTiler.");
        el("weatherRetry").hidden = false;
      } finally { clearTimeout(timer); }
    }
    function toggle(value) {
      enabled = value;
      el("weatherToggle").setAttribute("aria-pressed",String(value));
      el("weatherToggle").setAttribute("aria-expanded",String(value));
      el("weatherPanel").hidden = !value;
      if (value) load();
      else { ++generation; controller?.abort(); remove(); }
    }
    el("weatherToggle").onclick = () => toggle(!enabled);
    el("weatherClose").onclick = () => { toggle(false); el("weatherToggle").focus(); };
    el("weatherRetry").onclick = () => { catalog = null; load(); };
    el("weatherTime").onchange = () => { el("weatherRetry").hidden = true; el("weatherValue").textContent=""; render(); };
    el("weatherOpacity").oninput = () => { layer?.setOpacity(Number(el("weatherOpacity").value)/100); redrawOverlay(); };
    document.querySelectorAll("[data-weather]").forEach(button => {
      button.onclick = () => {
        mode = button.dataset.weather;
        document.querySelectorAll("[data-weather]").forEach(b => b.setAttribute("aria-pressed",String(b === button)));
        load();
      };
    });
    document.addEventListener("agropulse:languagechange", () => { status(statusText); if (enabled) { populateTimes(); redrawOverlay(); } });
  }
  window.AgroWeather = { init, color, decode, framesFor, contourSegments, rainStrength, rainClouds };
})();
