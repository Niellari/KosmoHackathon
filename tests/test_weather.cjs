const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const context = {window:{}};
vm.createContext(context);
vm.runInContext(fs.readFileSync('web/weather.js','utf8'),context);
const {decode,color,framesFor} = context.window.AgroWeather;
const {contourSegments} = context.window.AgroWeather;
test('rain clouds only appear in wet data, with bounded increasing intensity',()=>{
  const {rainStrength,rainClouds}=context.window.AgroWeather;
  assert.equal(rainStrength(NaN),0);
  assert.equal(rainStrength(0),0);
  assert.ok(rainStrength(10)>rainStrength(1));
  assert.equal(rainStrength(100),1);
  assert.equal(rainClouds(300,200,()=>0).length,0);
  assert.equal(rainClouds(300,200,()=>NaN).length,0);
  const clouds=rainClouds(300,200,(x)=>x>150?5:0);
  assert.ok(clouds.length>0);
  assert.ok(clouds.every(c=>c.x>150));
});
test('isotherms interpolate linear gradients and skip missing/constant cells',()=>{
  const lines=contourSegments([10,20,20,10],15);
  assert.equal(JSON.stringify(lines),'[[[0.5,0],[0.5,1]]]');
  assert.equal(contourSegments([10,10,10,10],10).length,0);
  assert.equal(contourSegments([10,NaN,20,10],15).length,0);
  assert.equal(contourSegments([10,20,10,20],15).length,2);
  assert.equal(contourSegments([-10,0,0,-10],-5).length,1);
});
test('MapTiler scalar channels decode their advertised ranges',()=>{
  assert.equal(decode(127,{min:-127,max:128}),0);
  assert.equal(decode(255,{min:0,max:50}),50);
  assert.equal(decode(0,{min:0,max:50}),0);
});
test('dry precipitation is transparent, temperature and rain have colors',()=>{
  assert.equal(color(0,'precipitation')[3],0);
  assert.equal(color(1,'precipitation')[3],220);
  assert.ok(color(10,'precipitation')[0]<color(1,'precipitation')[0]);
  assert.ok(color(25,'precipitation')[2]<color(3,'precipitation')[2]);
  assert.equal(color(-50,'temperature')[0],88);
  assert.equal(color(80,'temperature')[0],197);
});
test('forecast selection uses nearest hour and excludes stale/malformed frames',()=>{
  const now=Date.parse('2026-09-05T12:10:00Z');
  const variable={keyframes:[
    {id:'past',timestamp:'2026-09-04T12:00:00Z'},
    {id:'now',timestamp:'2026-09-05T12:00:00Z'},
    {id:'next',timestamp:'2026-09-05T13:00:00Z'},
    {id:'invalid/path',timestamp:'2026-09-05T14:00:00Z'},
    {id:'future',timestamp:'2026-09-07T12:00:00Z'},
  ]};
  assert.equal(framesFor(variable,now).map(f=>f.id).join(','),'now,next');
  assert.equal(framesFor({keyframes:[variable.keyframes[0]]},now).length,0);
  assert.equal(framesFor(null,now).length,0);
});
test('weather is opt-in and closing cancels a pending catalog without restoring overlay',async()=>{
  const elements = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id,{hidden:true,style:{},attributes:{},setAttribute(k,v){this.attributes[k]=v;},focus(){}});
    return elements.get(id);
  };
  let calls=0, signal, resolveRequest;
  const sandbox={window:{}, AbortController,setTimeout,clearTimeout,
    document:{getElementById:el,querySelectorAll:()=>[],addEventListener(){}},
    fetch:(_url,options)=>{calls++;signal=options.signal;return new Promise(resolve=>{resolveRequest=resolve;});},
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync('web/weather.js','utf8'),sandbox);
  sandbox.window.AgroWeather.init({createPane:()=>({style:{}})},'test-key');
  assert.equal(calls,0);
  el('weatherToggle').onclick();
  assert.equal(calls,1);
  assert.equal(el('weatherPanel').hidden,false);
  el('weatherClose').onclick();
  assert.equal(signal.aborted,true);
  resolveRequest({ok:true,json:async()=>({variables:[]})});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(el('weatherPanel').hidden,true);
  assert.equal(el('weatherToggle').attributes['aria-pressed'],'false');
});
