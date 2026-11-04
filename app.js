'use strict';
const $=id=>document.getElementById(id);
const PRESETS={
 'Classic black/gray/white':['#ffffff','#d9d9d9','#a6a6a6','#7a7a7a','#4d4d4d','#262626','#000000'],
 'Sepia':['#f5ecd9','#e3cfa6','#c9a979','#a37f52','#7a5a37','#523b23','#2b1d10'],
 'Pop art':['#fff200','#ff7ac8','#00b7eb','#ff2a1a','#1b3fd1','#000000'],
 'Street (spray basics)':['#ffffff','#f6d400','#ff8a00','#e3001b','#0057b8','#1c1c1c'],
 'Earth tones':['#efe4cf','#c9b28a','#9c8a5a','#6b7a3a','#7a4a2a','#3d2b1f','#1a1410'],
 'Ocean blues':['#eaf6fb','#a9dcef','#5fb3d9','#2b7fb8','#174f86','#0b2a4a','#050f1c'],
 'Warm sunset':['#fff1c1','#ffd166','#f8961e','#f3722c','#d62828','#7b1e3a','#2d0f1f'],
 'Neon':['#f2ff00','#39ff14','#00fff0','#ff00e6','#8a00ff','#101010']};
const DEFAULTS={layers:4,palette:'auto',custom:null,detail:3,width:12,bridge:'0.0625',rmbg:false,labels:true};
let batch={...DEFAULTS},images=[],cur=0,selSw=0;
const worker=new Worker('worker.js');let jobId=0;const pending=new Map();
worker.onmessage=e=>{const m=e.data,p=pending.get(m.id);if(!p)return;if(m.progress!==undefined){p.onp&&p.onp(m.progress,m.stage);return;}pending.delete(m.id);m.error?p.rej(new Error(m.error)):p.res(m);};
let chain=Promise.resolve();
function runWorker(msg,onp){const id=++jobId;const job=()=>new Promise((res,rej)=>{pending.set(id,{res,rej,onp});worker.postMessage({...msg,id},[msg.data.buffer]);});
 const p=chain.then(job);chain=p.catch(()=>{});return p;}
const settingsOf=i=>images[i]&&images[i].override||batch;
function hexToRgb(h){return[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));}
const rgbHex=c=>'#'+c.map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
function resample(list,k){if(list.length===k)return list.slice();const out=[];for(let i=0;i<k;i++){const t=k===1?0:i*(list.length-1)/(k-1),a=Math.floor(t),b=Math.min(list.length-1,a+1),f=t-a;
 const A=hexToRgb(list[a]),B=hexToRgb(list[b]);out.push(rgbHex(A.map((v,j)=>v+(B[j]-v)*f)));}return out;}
function paletteFor(s,autoColors){const k=s.layers;if(s.palette==='auto')return null;
 if(s.palette==='gray')return[...Array(k).keys()].map(i=>{const v=Math.round(240-220*i/(k-1));return rgbHex([v,v,v]);});
 if(s.palette==='custom')return resample(s.custom&&s.custom.length?s.custom:(autoColors||['#ffffff','#000000']),k);
 return resample(PRESETS[s.palette],k);}
function toCfg(s){const d=s.detail-1,dpi=Math.min(100,Math.floor(1600/s.width));
 const bridge=s.bridge==='auto'?Math.min(0.125,Math.max(3/64,s.width/12/16)):+s.bridge;
 return{layers:s.layers,width_in:s.width,dpi,margin_in:0.5,bridge_in:bridge,blur:[2,2,1,1,0][d],mode_r:[3,2,2,1,1][d],min_feature_in:[0.08,0.06,0.04,0.03,0.02][d],
  simplify:[1.2,0.9,0.7,0.5,0.4][d],remove_bg:s.rmbg,labels:s.labels,palette:paletteFor(s,images[cur]&&images[cur].autoColors)};}
async function decode(file,wpx){let src;try{src=await createImageBitmap(file,{imageOrientation:'from-image'});}catch(e){
 src=await new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error('Cannot read '+file.name));im.src=URL.createObjectURL(file);});}
 const W=src.width||src.naturalWidth,H=src.height||src.naturalHeight,h=Math.max(1,Math.round(H*wpx/W));
 const c=document.createElement('canvas');c.width=wpx;c.height=h;const x=c.getContext('2d',{willReadFrequently:true});x.imageSmoothingQuality='high';x.drawImage(src,0,0,wpx,h);src.close&&src.close();
 return x.getImageData(0,0,wpx,h);}
async function process(i,onp){const im=images[i],s=settingsOf(i),cfg=toCfg(s);if(s.palette==='custom'||s.palette==='auto'){}
 if(i!==cur)cfg.palette=paletteFor(s,im.autoColors);const key=JSON.stringify(cfg);if(im.result&&im.key===key)return im.result;
 const wpx=Math.round(cfg.width_in*cfg.dpi),img=await decode(im.file,wpx);if(i===cur)curSrc=img;
 const copy=new Uint8ClampedArray(img.data);
 const r=await runWorker({w:img.width,h:img.height,data:copy,cfg,name:im.name},onp);
 if(!cfg.palette)im.autoColors=r.colors;
 r.compositeUrl=await renderComposite(r);r.previews=[];for(const L of r.layers){r.previews.push(await maskPng(L.mask,r.w,r.h));delete L.mask;}
 r.cfg=cfg;r.settings={...s};im.result=r;im.key=key;return r;}
let curSrc=null;
function canvasBlob(c){return new Promise(r=>c.toBlob(r,'image/png'));}
async function renderComposite(r){const c=document.createElement('canvas');c.width=r.w;c.height=r.h;const x=c.getContext('2d'),id=x.createImageData(r.w,r.h),d=id.data;
 const base=r.base?hexToRgb(r.base):[255,255,255];for(let i=0;i<r.w*r.h;i++){d.set(base,i*4);d[i*4+3]=255;}
 for(const L of r.layers){const c2=hexToRgb(L.color);for(let i=0;i<L.mask.length;i++)if(L.mask[i])d.set(c2,i*4);}
 x.putImageData(id,0,0);r.compositeBlob=await canvasBlob(c);return URL.createObjectURL(r.compositeBlob);}
async function maskPng(m,w,h){const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d'),id=x.createImageData(w,h),d=id.data;
 for(let i=0;i<m.length;i++){const v=m[i]?0:255;d[i*4]=d[i*4+1]=d[i*4+2]=v;d[i*4+3]=255;}x.putImageData(id,0,0);return canvasBlob(c);}
// ---------- UI
const sel=$('palette');[['auto','Auto (from photo)'],['gray','Grayscale'],...Object.keys(PRESETS).map(k=>[k,k]),['custom','Custom']].forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;sel.appendChild(o);});
function editable(){return images.length&&images[cur].override?images[cur].override:batch;}
function showControls(){const s=editable();$('layers').value=s.layers;$('layersV').textContent=s.layers;sel.value=s.palette;$('detail').value=s.detail;$('detailV').textContent=['Bold','Simple','Normal','Fine','Finest'][s.detail-1];
 $('width').value=s.width;$('widthV').textContent=s.width;$('bridge').value=s.bridge;$('rmbg').checked=s.rmbg;$('labels').checked=s.labels;
 const own=images.length&&images[cur].override;$('scopeTxt').textContent=images.length?(own?'Settings for: this photo only':'Settings for: all photos'):'Settings for: all photos';
 $('resetOne').hidden=!own;$('ownBtn').hidden=!images.length||own;drawSwatches();}
function drawSwatches(){const s=editable(),im=images[cur],cols=paletteFor(s,im&&im.autoColors)||(im&&im.result&&im.result.colors)||null,box=$('swatches');box.innerHTML='';
 if(!cols){box.textContent='Colors appear after the preview.';return;}
 cols.forEach((c,i)=>{const b=document.createElement('button');b.style.background=c;b.title=c+(i===0&&(!im||!im.result||im.result.base)?' (base coat)':'');b.setAttribute('aria-label','Paint color '+(i+1)+' '+c);
  if(i===selSw)b.style.outline='3px solid #d62828';b.onclick=()=>{selSw=i;const p=$('colorPick');p.value=c;p.oninput=()=>setCustom(i,p.value);p.click();drawSwatches();};box.appendChild(b);});}
function setCustom(i,c){const s=editable(),im=images[cur];const cols=paletteFor(s,im&&im.autoColors)||(im&&im.result&&im.result.colors)||resample(['#ffffff','#000000'],s.layers);
 s.custom=cols.slice();s.custom[i]=c;s.palette='custom';showControls();schedule();}
const own=document.createElement('button');own.id='ownBtn';own.className='small';own.textContent='Only this photo';own.hidden=true;$('resetOne').after(own);
own.onclick=()=>{images[cur].override={...batch,custom:batch.custom&&batch.custom.slice()};showControls();};
$('resetOne').onclick=()=>{images[cur].override=null;showControls();schedule();};
function bind(id,key,conv,ev='input'){$(id).addEventListener(ev,()=>{const s=editable();s[key]=conv($(id));if(key==='layers'&&s.custom)s.custom=resample(s.custom,s.layers);showControls();schedule();});}
bind('layers','layers',e=>+e.value);bind('detail','detail',e=>+e.value);bind('width','width',e=>+e.value);
bind('palette','palette',e=>e.value,'change');bind('bridge','bridge',e=>e.value,'change');bind('rmbg','rmbg',e=>e.checked,'change');bind('labels','labels',e=>e.checked,'change');
let timer=null,gen=0;function schedule(){clearTimeout(timer);timer=setTimeout(preview,350);}
async function preview(){if(!images.length)return;const g=++gen,i=cur;$('pstatus').textContent='Working…';
 try{const r=await process(i,(f,st)=>{if(g===gen)$('pstatus').textContent=st+'…';});if(g!==gen||i!==cur)return;showResult(r);}catch(e){if(g===gen)$('pstatus').textContent='Error: '+e.message;}}
function showResult(r){const c=$('comp'),img=new Image();img.onload=()=>{c.width=r.w;c.height=r.h;c.getContext('2d').drawImage(img,0,0);};img.src=r.compositeUrl;
 const isl=r.layers.reduce((a,l)=>a+l.islands,0);$('pstatus').textContent=`${r.layers.length} sheets${r.base?' + base coat '+r.base:''} · ${r.cfg.width_in}in wide · ${(r.ms/1000).toFixed(1)}s`+(isl?` · ${isl} loose pieces!`:' · all sheets one piece');
 const L=$('layerList');L.innerHTML='';r.layers.forEach((l,i)=>{if(!l.url)l.url=URL.createObjectURL(new Blob([l.svg],{type:'image/svg+xml'}));
  const f=document.createElement('figure');f.innerHTML=`<img alt="layer ${i+1}" src="${l.url}"><figcaption>Layer ${i+1} · paint <span style="display:inline-block;width:12px;height:12px;background:${l.color};border:1px solid #0005"></span> ${l.color} · ${l.bridges} bridges</figcaption>`;L.appendChild(f);});drawSwatches();}
function showImage(){$('preview').hidden=!images.length;if(!images.length)return;$('imgName').textContent=`${cur+1}/${images.length} · ${images[cur].name}`;showControls();
 const r=images[cur].result;if(r&&images[cur].key===JSON.stringify(toCfg(settingsOf(cur))))showResult(r);else{$('layerList').innerHTML='';schedule();}}
$('prev').onclick=()=>{if(images.length){cur=(cur+images.length-1)%images.length;showImage();}};
$('next').onclick=()=>{if(images.length){cur=(cur+1)%images.length;showImage();}};
$('pick').addEventListener('change',e=>{const fs=[...e.target.files];if(!fs.length)return;const used=new Set(images.map(i=>i.name));
 for(const f of fs){let n=(f.name||'photo').replace(/\.[^.]+$/,'').replace(/[^\w.-]+/g,'_')||'photo',b=n,k=2;while(used.has(n))n=b+'_'+k++;used.add(n);images.push({file:f,name:n,override:null,result:null});}
 e.target.value='';$('run').disabled=false;cur=images.length-fs.length;showImage();});
// long-press eyedropper on preview
let lp=null;$('comp').addEventListener('pointerdown',e=>{const c=$('comp'),rect=c.getBoundingClientRect();lp=setTimeout(()=>{if(!curSrc)return;
 const x=Math.floor((e.clientX-rect.left)/rect.width*curSrc.width),y=Math.floor((e.clientY-rect.top)/rect.height*curSrc.height),i=(y*curSrc.width+x)*4,d=curSrc.data;
 setCustom(selSw,rgbHex([d[i],d[i+1],d[i+2]]));$('pstatus').textContent='Picked color for swatch '+(selSw+1);},550);});
['pointerup','pointerleave','pointercancel','pointermove'].forEach(t=>$('comp').addEventListener(t,ev=>{if(t!=='pointermove'||Math.abs(ev.movementX)+Math.abs(ev.movementY)>4)clearTimeout(lp);}));
$('comp').addEventListener('contextmenu',e=>e.preventDefault());
// batch
let zipBlob=null;
$('run').onclick=async()=>{$('run').disabled=true;$('dl').disabled=true;$('share').disabled=true;$('barWrap').hidden=false;const n=images.length,t0=performance.now();let fails=0;
 for(let i=0;i<n;i++){const set=f=>{$('bar').style.width=((i+f)/n*100).toFixed(1)+'%';$('barTxt').textContent=`Photo ${i+1} of ${n}`;};set(0);
  try{await process(i,f=>set(f));}catch(e){fails++;images[i].error=e.message;}set(1);if(i===cur)showImage();}
 $('barTxt').textContent=`Done: ${n-fails} of ${n} photos in ${((performance.now()-t0)/1000).toFixed(1)}s`+(fails?` (${fails} failed)`:'');
 $('run').disabled=false;$('dl').disabled=false;$('share').disabled=false;zipBlob=null;window.__batchDone=true;};
async function makeZip(){if(zipBlob)return zipBlob;const z=new JSZip();
 for(const im of images){const r=im.result;if(!r)continue;const f=z.folder(im.name);
  r.layers.forEach((l,i)=>{f.file(`layer${i+1}.svg`,l.svg);f.file(`layer${i+1}_preview.png`,r.previews[i]);});f.file('composite_preview.png',r.compositeBlob);
  const lines=[`Stencil Maker settings for ${im.name}`,JSON.stringify({...r.settings,custom:undefined,bridge_in:r.cfg.bridge_in,dpi:r.cfg.dpi},null,1),
   r.base?`Base coat (paint the whole surface first): ${r.base}`:'No base coat (background removed)',
   ...r.layers.map((l,i)=>`layer${i+1}.svg  paint ${l.color}  bridges ${l.bridges}  loose pieces ${l.islands}`)];f.file('settings.txt',lines.join('\n')+'\n');}
 zipBlob=await z.generateAsync({type:'blob',compression:'DEFLATE',compressionOptions:{level:6}},m=>{$('barTxt').textContent=`Zipping ${m.percent.toFixed(0)}%`;});return zipBlob;}
const zipName=()=>`stencils_${new Date().toISOString().slice(0,10)}.zip`;
$('dl').onclick=async()=>{const b=await makeZip(),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=zipName();document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),60000);$('barTxt').textContent='ZIP ready';};
if(navigator.canShare&&navigator.canShare({files:[new File(['x'],'t.zip',{type:'application/zip'})]})){$('share').hidden=false;
 $('share').onclick=async()=>{const f=new File([await makeZip()],zipName(),{type:'application/zip'});try{await navigator.share({files:[f],title:'Stencils'});}catch(e){}};}
if('serviceWorker'in navigator&&/^https?:/.test(location.protocol))navigator.serviceWorker.register('sw.js').catch(()=>{});
showControls();
