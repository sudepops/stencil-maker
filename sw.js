const CACHE='stencil-maker-v3';
const ASSETS=['seg.js','./','index.html','style.css','app.js','worker.js','lib/jszip.min.js','manifest.json','icons/icon-192.png','icons/icon-512.png','icons/icon-maskable-512.png','icons/apple-touch-icon.png'];
// AI cut-out files (~16MB, ~7.5MB gzipped): precached too, but best-effort so a slow network can't block the app update
const AI_ASSETS=['lib/ort/ort.wasm.min.js','lib/ort/ort-wasm-simd-threaded.mjs','lib/ort/ort-wasm-simd-threaded.wasm','models/u2netp.onnx'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS).then(()=>Promise.allSettled(AI_ASSETS.map(a=>c.add(a)))).then(()=>self.skipWaiting())));});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));});
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;e.respondWith(caches.match(e.request,{ignoreSearch:true}).then(r=>r||fetch(e.request).then(res=>{if(res.ok&&new URL(e.request.url).origin===location.origin){const cp=res.clone();caches.open(CACHE).then(c=>c.put(e.request,cp));}return res;}).catch(()=>caches.match('index.html'))));});
