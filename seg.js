// Stencil Maker subject cut-out worker: U2-Net-p (Apache-2.0) salient-object model on onnxruntime-web (MIT), fully on-device.
'use strict';
importScripts('lib/ort/ort.wasm.min.js');
ort.env.wasm.wasmPaths=new URL('lib/ort/',self.location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
let sess=null;const S=320;
async function session(){if(!sess)sess=await ort.InferenceSession.create(new URL('models/u2netp.onnx',self.location.href).href,{executionProviders:['wasm'],graphOptimizationLevel:'all'});return sess;}
function resizeArea(d,w,h){// RGBA w*h -> float RGB S*S (box filter)
 const o=new Float32Array(S*S*3),c=new Float32Array(S*S);
 for(let y=0;y<h;y++){const ty=Math.min(S-1,Math.floor(y*S/h));for(let x=0;x<w;x++){const tx=Math.min(S-1,Math.floor(x*S/w)),t=ty*S+tx,i=(y*w+x)*4;o[t*3]+=d[i];o[t*3+1]+=d[i+1];o[t*3+2]+=d[i+2];c[t]++;}}
 for(let t=0;t<S*S;t++){const k=c[t]||1;o[t*3]/=k;o[t*3+1]/=k;o[t*3+2]/=k;}return o;}
function upsample(p,w,h){// bilinear S*S -> w*h
 const o=new Float32Array(w*h);for(let y=0;y<h;y++){const fy=Math.max(0,Math.min(S-1,(y+0.5)*S/h-0.5)),y0=Math.floor(fy),y1=Math.min(S-1,y0+1),ay=fy-y0;
  for(let x=0;x<w;x++){const fx=Math.max(0,Math.min(S-1,(x+0.5)*S/w-0.5)),x0=Math.floor(fx),x1=Math.min(S-1,x0+1),ax=fx-x0;
   o[y*w+x]=(p[y0*S+x0]*(1-ax)+p[y0*S+x1]*ax)*(1-ay)+(p[y1*S+x0]*(1-ax)+p[y1*S+x1]*ax)*ay;}}return o;}
function comps(M,w,h,val){const n=w*h,lab=new Int32Array(n).fill(-1),sizes=[],border=[],st=new Int32Array(n);
 for(let s=0;s<n;s++){if(M[s]!==val||lab[s]>=0)continue;const id=sizes.length;let sp=0,sz=0,bd=false;st[sp++]=s;lab[s]=id;
  while(sp){const p=st[--sp];sz++;const x=p%w,y=(p-x)/w;if(!x||!y||x===w-1||y===h-1)bd=true;for(const q of[x>0?p-1:-1,x<w-1?p+1:-1,y>0?p-w:-1,y<h-1?p+w:-1])if(q>=0&&M[q]===val&&lab[q]<0){lab[q]=id;st[sp++]=q;}}
  sizes.push(sz);border.push(bd);}return{lab,sizes,border};}
self.onmessage=async e=>{const{id,w,h,data}=e.data;const t0=Date.now();
 try{const s=await session();const r=resizeArea(data,w,h);let mx=0;for(let i=0;i<r.length;i++)if(r[i]>mx)mx=r[i];mx=mx||1;
  const mean=[0.485,0.456,0.406],sd=[0.229,0.224,0.225],x=new Float32Array(3*S*S);
  for(let t=0;t<S*S;t++)for(let c=0;c<3;c++)x[c*S*S+t]=(r[t*3+c]/mx-mean[c])/sd[c];
  const out=await s.run({[s.inputNames[0]]:new ort.Tensor('float32',x,[1,3,S,S])});const p=out[s.outputNames[0]].data;
  let lo=1e9,hi=-1e9;for(let i=0;i<p.length;i++){if(p[i]<lo)lo=p[i];if(p[i]>hi)hi=p[i];}const pn=new Float32Array(S*S);for(let i=0;i<pn.length;i++)pn[i]=(p[i]-lo)/((hi-lo)||1);
  const up=upsample(pn,w,h),n=w*h,M=new Uint8Array(n);for(let i=0;i<n;i++)M[i]=up[i]>0.4?1:0;
  // keep the main subject(s): components at least 10% of the largest
  let C=comps(M,w,h,1),big=0;for(const z of C.sizes)big=Math.max(big,z);for(let i=0;i<n;i++)if(M[i]&&C.sizes[C.lab[i]]<big*0.1)M[i]=0;
  // fill small holes inside the subject (<0.3% of the photo); keep real gaps (e.g. between arm and body)
  C=comps(M,w,h,0);for(let i=0;i<n;i++)if(!M[i]&&!C.border[C.lab[i]]&&C.sizes[C.lab[i]]<n*0.003)M[i]=1;
  let fg=0;for(let i=0;i<n;i++)fg+=M[i];
  self.postMessage({id,mask:M,fg:fg/n,ms:Date.now()-t0},[M.buffer]);
 }catch(err){self.postMessage({id,error:String(err&&err.message||err)});}};
