// Stencil Maker worker: all image processing runs here (client-side, no server).
'use strict';
function rng(seed){let s=seed>>>0;return()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};}
function srgb2lin(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4);}
const LIN=new Float32Array(256);for(let i=0;i<256;i++)LIN[i]=srgb2lin(i);
function fLab(t){return t>0.008856?Math.cbrt(t):7.787*t+16/116;}
function toLab(rgba,n){const L=new Float32Array(n*3);for(let i=0;i<n;i++){const r=LIN[rgba[i*4]],g=LIN[rgba[i*4+1]],b=LIN[rgba[i*4+2]];
 const x=fLab((0.4124*r+0.3576*g+0.1805*b)/0.95047),y=fLab(0.2126*r+0.7152*g+0.0722*b),z=fLab((0.0193*r+0.1192*g+0.9505*b)/1.08883);
 L[i*3]=116*y-16;L[i*3+1]=500*(x-y);L[i*3+2]=200*(y-z);}return L;}
function boxBlur(rgba,w,h){const out=new Uint8ClampedArray(rgba.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){for(let c=0;c<3;c++){let s=0,k=0;
 for(let dy=-1;dy<=1;dy++){const yy=y+dy;if(yy<0||yy>=h)continue;for(let dx=-1;dx<=1;dx++){const xx=x+dx;if(xx<0||xx>=w)continue;s+=rgba[(yy*w+xx)*4+c];k++;}}
 out[(y*w+x)*4+c]=s/k;}out[(y*w+x)*4+3]=rgba[(y*w+x)*4+3];}return out;}
function kmeans(lab,n,k,alpha){const R=rng(1);const idx=[];for(let i=0;i<n;i++)if(!alpha||alpha[i*4+3]>127)idx.push(i);
 const S=Math.min(30000,idx.length),samp=new Float32Array(S*3);for(let i=0;i<S;i++){const j=idx[Math.floor(R()*idx.length)];samp.set(lab.subarray(j*3,j*3+3),i*3);}
 const C=new Float32Array(k*3);let j0=Math.floor(R()*S);C.set(samp.subarray(j0*3,j0*3+3),0);const D=new Float32Array(S).fill(1e30);
 for(let c=1;c<k;c++){let tot=0;for(let i=0;i<S;i++){const a=samp[i*3]-C[(c-1)*3],b=samp[i*3+1]-C[(c-1)*3+1],d=samp[i*3+2]-C[(c-1)*3+2];const dd=a*a+b*b+d*d;if(dd<D[i])D[i]=dd;tot+=D[i];}
  let r=R()*tot,p=0;for(;p<S-1;p++){r-=D[p];if(r<=0)break;}C.set(samp.subarray(p*3,p*3+3),c*3);}
 const as=new Int32Array(S);for(let it=0;it<40;it++){const sum=new Float64Array(k*3),cnt=new Float64Array(k);
  for(let i=0;i<S;i++){let best=0,bd=1e30;for(let c=0;c<k;c++){const a=samp[i*3]-C[c*3],b=samp[i*3+1]-C[c*3+1],d=samp[i*3+2]-C[c*3+2];const dd=a*a+b*b+d*d;if(dd<bd){bd=dd;best=c;}}
   as[i]=best;cnt[best]++;sum[best*3]+=samp[i*3];sum[best*3+1]+=samp[i*3+1];sum[best*3+2]+=samp[i*3+2];}
  let mv=0;for(let c=0;c<k;c++)if(cnt[c]){for(let q=0;q<3;q++){const v=sum[c*3+q]/cnt[c];mv=Math.max(mv,Math.abs(v-C[c*3+q]));C[c*3+q]=v;}}if(mv<0.1)break;}
 return C;}
function modeFilter(lbl,w,h,k,r){// sliding-window majority filter (labels -1..k-1)
 const out=new Int8Array(lbl.length),cnt=new Int32Array(k+1),K=k+1;
 for(let y=0;y<h;y++){cnt.fill(0);const y0=Math.max(0,y-r),y1=Math.min(h-1,y+r);
  const col=(x,s)=>{const xx=x<0?0:x>=w?w-1:x;for(let yy=y-r;yy<=y+r;yy++){const yc=yy<0?0:yy>=h?h-1:yy;cnt[lbl[yc*w+xx]+1]+=s;}};
  for(let x=-r;x<=r;x++)col(x,1);
  for(let x=0;x<w;x++){let b=0;for(let c=1;c<K;c++)if(cnt[c]>cnt[b])b=c;out[y*w+x]=b-1;col(x-r,-1);col(x+r+1,1);}}return out;}
function morph(m,w,h,val){// val=1 dilate, 0 erode (3x3), out-of-bounds ignored
 const o=new Uint8Array(m.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let r=val?0:1;
 for(let dy=-1;dy<=1&&r!==val;dy++){const yy=y+dy;if(yy<0||yy>=h)continue;for(let dx=-1;dx<=1;dx++){const xx=x+dx;if(xx<0||xx>=w)continue;if(m[yy*w+xx]===val){r=val;break;}}}o[y*w+x]=r;}return o;}
function label(m,w,h,val,eight){const lab=new Int32Array(m.length).fill(-1),sizes=[],border=[];const st=new Int32Array(m.length);
 for(let s=0;s<m.length;s++){if(m[s]!==val||lab[s]>=0)continue;const id=sizes.length;let sp=0,sz=0,bd=false;st[sp++]=s;lab[s]=id;
  while(sp){const p=st[--sp];sz++;const x=p%w,y=(p-x)/w;if(x===0||y===0||x===w-1||y===h-1)bd=true;
   for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;if(!eight&&dx&&dy)continue;const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;const q=yy*w+xx;if(m[q]===val&&lab[q]<0){lab[q]=id;st[sp++]=q;}}}
  sizes.push(sz);border.push(bd);}return{lab,sizes,border};}
function clean(m,w,h,minpx){m=morph(morph(m,w,h,0),w,h,1);m=morph(morph(m,w,h,1),w,h,0);
 let L=label(m,w,h,1,true);for(let i=0;i<m.length;i++)if(L.lab[i]>=0&&L.sizes[L.lab[i]]<minpx)m[i]=0;
 L=label(m,w,h,0,false);for(let i=0;i<m.length;i++)if(L.lab[i]>=0&&L.sizes[L.lab[i]]<minpx&&!L.border[L.lab[i]])m[i]=1;return m;}
function stamp(m,w,h,x0,y0,x1,y1,bw){const r=Math.max(1,bw/2),n=Math.ceil(Math.hypot(x1-x0,y1-y0))*2+1,ri=Math.ceil(r);
 for(let s=0;s<=n;s++){const cx=x0+(x1-x0)*s/n,cy=y0+(y1-y0)*s/n;for(let dy=-ri;dy<=ri;dy++)for(let dx=-ri;dx<=ri;dx++){if(dx*dx+dy*dy>r*r+0.5)continue;
  const xx=Math.round(cx+dx),yy=Math.round(cy+dy);if(xx>=0&&yy>=0&&xx<w&&yy<h)m[yy*w+xx]=0;}}}
// material (0) must be one 4-connected piece; everything touching the image edge joins the sheet margin
function bridge(m,w,h,bw,minA){let nb=0;minA=minA||0;
 {const L=label(m,w,h,0,false);for(let i=0;i<m.length;i++){const c=L.lab[i];if(c>=0&&!L.border[c]&&L.sizes[c]<Math.max(minA*0.5,(3*bw)*(3*bw)))m[i]=1;}}// fill tiny material islands instead of bridging them
for(let it=0;it<30;it++){const L=label(m,w,h,0,false);const main=L.border;
 const isl=[];for(let c=0;c<L.sizes.length;c++)if(!main[c])isl.push(c);if(!isl.length)return{m,nb,left:0};
 // nearest main-material pixel for every pixel (propagated BFS, Euclidean to source)
 const N=w*h,src=new Int32Array(N).fill(-1),dist=new Float32Array(N).fill(1e30);let q=new Int32Array(N),qh=0,qt=0;
 const anyMain=L.border.some(b=>b);
 for(let i=0;i<N;i++){const c=L.lab[i];if(c>=0&&main[c]){src[i]=i;dist[i]=0;q[qt++]=i;}}
 if(!anyMain){// no material touches the edge: treat the image frame itself as the sheet
  for(let x=0;x<w;x++)for(const y of[0,h-1]){const i=y*w+x;src[i]=-2-i;}}
 const inq=new Uint8Array(N);for(let i=0;i<qt;i++)inq[q[i]]=1;
 while(qh!==qt){const p=q[qh];qh=(qh+1)%N;inq[p]=0;const s=src[p],sx=s%w,sy=(s-sx)/w,x=p%w,y=(p-x)/w;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;const t=yy*w+xx;
   const d=(xx-sx)*(xx-sx)+(yy-sy)*(yy-sy);if(d<dist[t]){dist[t]=d;src[t]=s;if(!inq[t]){inq[t]=1;q[qt]=t;qt=(qt+1)%N;}}}}
 const best=new Map();for(let i=0;i<N;i++){const c=L.lab[i];if(c<0||main[c])continue;let b=best.get(c);
  if(!b){b={i,d:dist[i],sx:0,sy:0,n:0};best.set(c,b);}if(dist[i]<b.d){b.d=dist[i];b.i=i;}b.sx+=i%w;b.sy+=(i-i%w)/w;b.n++;}
 for(const[c,b]of best){const x0=b.i%w,y0=(b.i-x0)/w;let x1,y1;if(src[b.i]>=0){x1=src[b.i]%w;y1=(src[b.i]-x1)/w;}else{// nearest frame edge
   const ds=[x0,w-1-x0,y0,h-1-y0],k=ds.indexOf(Math.min(...ds));x1=k===0?0:k===1?w-1:x0;y1=k===2?0:k===3?h-1:y0;}
  if(minA&&b.n<minA*1.5&&Math.sqrt(b.d)>8*bw){for(let i=0;i<N;i++)if(L.lab[i]===c)m[i]=1;continue;}
  stamp(m,w,h,x0,y0,x1,y1,bw);nb++;
  if(b.n>(60*bw)*(60*bw)){// large island: second bridge on the far side
   const cx=b.sx/b.n,cy=b.sy/b.n,vx=x0-cx,vy=y0-cy;let bi=-1,bd=1e30;
   for(let i=0;i<N;i++){if(L.lab[i]!==c)continue;const x=i%w,y=(i-x)/w;if((x-cx)*vx+(y-cy)*vy<0&&dist[i]<bd&&src[i]>=0){bd=dist[i];bi=i;}}
   if(bi>=0){const xa=bi%w,ya=(bi-xa)/w,s=src[bi];stamp(m,w,h,xa,ya,s%w,(s-s%w)/w,bw);nb++;}}}}
 const L=label(m,w,h,0,false);return{m,nb,left:L.border.filter(b=>!b).length};}
// pixel-edge contour tracing + Douglas-Peucker simplification
function trace(m,w,h,tol){const W=w+1;const nx1=new Int32Array((w+1)*(h+1)).fill(-1),nx2=new Int32Array((w+1)*(h+1)).fill(-1);
 const add=(a,b)=>{if(nx1[a]<0)nx1[a]=b;else nx2[a]=b;};const cut=(x,y)=>x>=0&&y>=0&&x<w&&y<h&&m[y*w+x]===1;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){if(!m[y*w+x])continue;const v=y*W+x;
  if(!cut(x,y-1))add(v,v+1);if(!cut(x+1,y))add(v+1,v+1+W);if(!cut(x,y+1))add(v+1+W,v+W);if(!cut(x-1,y))add(v+W,v);}
 const paths=[];for(let s=0;s<nx1.length;s++){while(nx1[s]>=0){const pts=[];let v=s,pd=-1;
  for(;;){let n=nx1[v];if(nx2[v]>=0){// saddle: turn right (keeps cut regions 8-connected)
    const vx=v%W,vy=(v-vx)/W,cand=[nx1[v],nx2[v]];let pick=0;if(pd>=0){const px=pd%W,py=(pd-px)/W,dx=vx-px,dy=vy-py;
     for(let k=0;k<2;k++){const cx=cand[k]%W-vx,cy=(cand[k]-cand[k]%W)/W-vy;if(dx*cy-dy*cx>0)pick=k;}}
    n=cand[pick];if(pick===0){nx1[v]=nx2[v];}nx2[v]=-1;}else nx1[v]=-1;
   pts.push(v);pd=v;v=n;if(v===s&&nx1[s]<0)break;if(v===s)break;}
  const P=pts.map(p=>[p%W,(p-p%W)/W]);// keep corners only
  const C=[];for(let i=0;i<P.length;i++){const a=P[(i+P.length-1)%P.length],b=P[i],c=P[(i+1)%P.length];if((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])!==0)C.push(b);}
  if(C.length>=3)paths.push(simplify(C,tol));}}
 return paths.filter(p=>p.length>=3).map(smoothPath).join('');}
function dp(P,a,b,tol,keep){let md=0,mi=-1;const[x1,y1]=P[a],[x2,y2]=P[b],L=Math.hypot(x2-x1,y2-y1)||1;
 for(let i=a+1;i<b;i++){const d=Math.abs((x2-x1)*(y1-P[i][1])-(x1-P[i][0])*(y2-y1))/L;if(d>md){md=d;mi=i;}}
 if(md>tol){keep[mi]=1;dp(P,a,mi,tol,keep);dp(P,mi,b,tol,keep);}}
function simplify(C,tol){if(C.length<8||tol<=0)return C;let far=0,fd=0;for(let i=1;i<C.length;i++){const d=(C[i][0]-C[0][0])**2+(C[i][1]-C[0][1])**2;if(d>fd){fd=d;far=i;}}
 const P=C.concat([C[0]]),keep=new Uint8Array(P.length);keep[0]=keep[far]=keep[P.length-1]=1;dp(P,0,far,tol,keep);dp(P,far,P.length-1,tol,keep);
 const out=[];for(let i=0;i<C.length;i++)if(keep[i])out.push(C[i]);return out;}
const hex=c=>'#'+c.map(v=>Math.max(0,Math.min(255,Math.round(v))).toString(16).padStart(2,'0')).join('');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function regMark(x,y,pointDown){// hollow 0.2in triangle, tip points toward the image
 return `<g transform="translate(${x.toFixed(4)} ${y.toFixed(4)}) rotate(${pointDown?270:90}) scale(0.005)"><path d="M -20 0 L 20 20 L 20 -20 Z" stroke-width="2"/></g>`;}
function svg(d,cfg,h,li,n,color,name){const m=cfg.margin_in,s=1/cfg.dpi,W=cfg.width_in+2*m,H=h*s+2*m;
 const marks=[[m,m/2,true],[W-m,m/2,true],[m,H-m/2,false],[W-m,H-m/2,false]].map(a=>regMark(...a)).join('');
 const fs=Math.min(0.18,m*0.4);let legend='';
 if(cfg.labels)legend=`<text x="${(W/2).toFixed(4)}" y="${(m/2+fs/3).toFixed(4)}" text-anchor="middle" font-family="Arial" font-size="${fs.toFixed(3)}" fill="none" stroke="blue" stroke-width="0.01">${esc(name)} - layer ${li} of ${n}</text>`+
  `<text x="${(W/2).toFixed(4)}" y="${(H-m/2+fs/3).toFixed(4)}" text-anchor="middle" font-family="Arial" font-size="${fs.toFixed(3)}" fill="none" stroke="blue" stroke-width="0.01">paint ${color}</text>`+
  `<rect x="${(W/2+fs*5).toFixed(4)}" y="${(H-m/2-fs/2).toFixed(4)}" width="${(fs*1.5).toFixed(4)}" height="${fs.toFixed(4)}" fill="${color}" stroke="blue" stroke-width="0.01"/>`;
 return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${W.toFixed(4)}in" height="${H.toFixed(4)}in" viewBox="0 0 ${W.toFixed(4)} ${H.toFixed(4)}">\n`+
 `<rect id="sheet outline" x="0.005" y="0.005" width="${(W-0.01).toFixed(4)}" height="${(H-0.01).toFixed(4)}" fill="none" stroke="#999" stroke-width="0.01"/>\n`+
 `<g id="legend">${legend}</g>\n<g id="registration marks" style="fill:none;stroke:red;stroke-width:0.01">${marks}</g>\n`+
 `<g id="image" transform="translate(${m} ${m}) scale(${s.toFixed(6)})"><path fill="#bbbbbb" stroke="red" stroke-width="${(0.01*cfg.dpi).toFixed(3)}" fill-rule="evenodd" d="${d}"/></g>\n</svg>\n`;}
function removeBg(d,w,h){// flood-fill plain background from the photo edges (colour distance in Lab)
 const n=w*h,lab=toLab(d,n),seen=new Uint8Array(n),st=[];const ref=[0,0,0];let c=0;
 for(let x=0;x<w;x++)for(const y of[0,h-1]){const i=y*w+x;for(let q=0;q<3;q++)ref[q]+=lab[i*3+q];c++;}
 for(let y=0;y<h;y++)for(const x of[0,w-1]){const i=y*w+x;for(let q=0;q<3;q++)ref[q]+=lab[i*3+q];c++;}
 for(let q=0;q<3;q++)ref[q]/=c;const ok=i=>{const a=lab[i*3]-ref[0],b=lab[i*3+1]-ref[1],e=lab[i*3+2]-ref[2];return a*a+b*b+e*e<400;};
 for(let x=0;x<w;x++)for(const y of[0,h-1])st.push(y*w+x);for(let y=0;y<h;y++)for(const x of[0,w-1])st.push(y*w+x);
 let removed=0;while(st.length){const p=st.pop();if(seen[p]||!ok(p))continue;seen[p]=1;d[p*4+3]=0;removed++;const x=p%w,y=(p-x)/w;
  if(x>0)st.push(p-1);if(x<w-1)st.push(p+1);if(y>0)st.push(p-w);if(y<h-1)st.push(p+w);}
 if(removed>n*0.9)for(let i=0;i<n;i++)d[i*4+3]=255;}// nothing sensible left: keep everything
// ---- v2 cleanup: edge-preserving smoothing, label-map despeckle, sliver removal
function bilateral(rgba,w,h,sigS,sigR){const R=Math.max(1,Math.ceil(sigS*2)),n=w*h,out=new Uint8ClampedArray(rgba.length);
 const r8=new Int32Array(n),g8=new Int32Array(n),b8=new Int32Array(n);for(let i=0;i<n;i++){r8[i]=rgba[i*4];g8[i]=rgba[i*4+1];b8[i]=rgba[i*4+2];}
 const ox=[],oy=[],ow=[];for(let dy=-R;dy<=R;dy++)for(let dx=-R;dx<=R;dx++){const d2=dx*dx+dy*dy;if(d2>R*R+R)continue;ox.push(dx);oy.push(dy);ow.push(Math.exp(-d2/(2*sigS*sigS)));}
 const K=ox.length,OX=Int32Array.from(ox),OY=Int32Array.from(oy),OW=Float32Array.from(ow),OF=new Int32Array(K);for(let k=0;k<K;k++)OF[k]=OY[k]*w+OX[k];
 const LUT=new Float32Array(766);for(let i=0;i<766;i++){const t=i/(sigR*1.7);LUT[i]=Math.exp(-t*t/2);}
 for(let y=0;y<h;y++){const yin=y>=R&&y<h-R;for(let x=0;x<w;x++){const i=y*w+x,r0=r8[i],g0=g8[i],b0=b8[i];let sr=0,sg=0,sb=0,sw=0;
  if(yin&&x>=R&&x<w-R){for(let k=0;k<K;k++){const p=i+OF[k],r=r8[p],g=g8[p],b=b8[p];let d=r-r0;if(d<0)d=-d;let e=g-g0;if(e<0)e=-e;let f=b-b0;if(f<0)f=-f;
    const wt=OW[k]*LUT[d+e+f];sr+=wt*r;sg+=wt*g;sb+=wt*b;sw+=wt;}}
  else{for(let k=0;k<K;k++){const xx=x+OX[k],yy=y+OY[k];if(xx<0||yy<0||xx>=w||yy>=h)continue;const p=yy*w+xx,r=r8[p],g=g8[p],b=b8[p];let d=r-r0;if(d<0)d=-d;let e=g-g0;if(e<0)e=-e;let f=b-b0;if(f<0)f=-f;
    const wt=OW[k]*LUT[d+e+f];sr+=wt*r;sg+=wt*g;sb+=wt*b;sw+=wt;}}
  const c=i*4;out[c]=sr/sw;out[c+1]=sg/sw;out[c+2]=sb/sw;out[c+3]=rgba[c+3];}}return out;}
// chamfer (3-4) distance in px from every pixel to the nearest pixel where src[i] is truthy
function distTo(src,w,h){const n=w*h,D=new Int32Array(n);for(let i=0;i<n;i++)D[i]=src[i]?0:1e8;
 for(let y=0;y<h;y++){const r=y*w;for(let x=0;x<w;x++){const i=r+x;let d=D[i];if(d===0)continue;let t;
  if(x>0){t=D[i-1]+3;if(t<d)d=t;}if(y>0){t=D[i-w]+3;if(t<d)d=t;if(x>0){t=D[i-w-1]+4;if(t<d)d=t;}if(x<w-1){t=D[i-w+1]+4;if(t<d)d=t;}}D[i]=d;}}
 for(let y=h-1;y>=0;y--){const r=y*w;for(let x=w-1;x>=0;x--){const i=r+x;let d=D[i];if(d===0)continue;let t;
  if(x<w-1){t=D[i+1]+3;if(t<d)d=t;}if(y<h-1){t=D[i+w]+3;if(t<d)d=t;if(x<w-1){t=D[i+w+1]+4;if(t<d)d=t;}if(x>0){t=D[i+w-1]+4;if(t<d)d=t;}}D[i]=d;}}
 const F=new Float32Array(n);for(let i=0;i<n;i++)F[i]=D[i]/3;return F;}
// relabel pixels set to -2 from their nearest labelled neighbours (BFS)
function fillUnassigned(lbl,w,h){const n=w*h,q=new Int32Array(n);let qt=0,qh=0;
 for(let i=0;i<n;i++)if(lbl[i]>=0){const x=i%w;if((x>0&&lbl[i-1]===-2)||(x<w-1&&lbl[i+1]===-2)||(i>=w&&lbl[i-w]===-2)||(i<n-w&&lbl[i+w]===-2))q[qt++]=i;}
 while(qh<qt){const p=q[qh++],x=p%w,v=lbl[p];const nb=[x>0?p-1:-1,x<w-1?p+1:-1,p-w,p+w];for(const t of nb){if(t<0||t>=n||lbl[t]!==-2)continue;lbl[t]=v;q[qt++]=t;}}
 for(let i=0;i<n;i++)if(lbl[i]===-2)lbl[i]=0;}
// per-label morphological opening: removes slivers thinner than 2r; freed pixels go to neighbours
function removeSlivers(lbl,w,h,k,r){if(r<1)return;const n=w*h,m=new Uint8Array(n),core=new Uint8Array(n),drop=new Uint8Array(n);
 for(let t=0;t<k;t++){let any=false;for(let i=0;i<n;i++){const o=lbl[i]===t;m[i]=o?0:1;if(o)any=true;}if(!any)continue;
  const de=distTo(m,w,h);for(let i=0;i<n;i++)core[i]=de[i]>r?1:0;const dd=distTo(core,w,h);
  for(let i=0;i<n;i++)if(lbl[i]===t&&dd[i]>r+0.5)drop[i]=1;}
 for(let i=0;i<n;i++)if(drop[i])lbl[i]=-2;fillUnassigned(lbl,w,h);}
// merge connected regions smaller than minA px into the label they share the most border with
function mergeSmall(lbl,w,h,k,minA){const n=w*h;for(let it=0;it<8;it++){const cid=new Int32Array(n).fill(-1),sizes=[],st=new Int32Array(n);
  for(let s=0;s<n;s++){if(cid[s]>=0||lbl[s]<0)continue;const id=sizes.length,v=lbl[s];let sp=0,sz=0;st[sp++]=s;cid[s]=id;
   while(sp){const p=st[--sp];sz++;const x=p%w;const nb=[x>0?p-1:-1,x<w-1?p+1:-1,p>=w?p-w:-1,p<n-w?p+w:-1];for(const q of nb)if(q>=0&&cid[q]<0&&lbl[q]===v){cid[q]=id;st[sp++]=q;}}sizes.push(sz);}
  const small=new Int32Array(sizes.length).fill(-1);let ns=0;for(let c=0;c<sizes.length;c++)if(sizes[c]<minA)small[c]=ns++;if(!ns)return;
  const bc=new Float64Array(ns*k);for(let p=0;p<n;p++){const c=cid[p];if(c<0||small[c]<0)continue;const x=p%w;const nb=[x>0?p-1:-1,x<w-1?p+1:-1,p>=w?p-w:-1,p<n-w?p+w:-1];
   for(const q of nb)if(q>=0&&lbl[q]>=0&&cid[q]!==c)bc[small[c]*k+lbl[q]]+=sizes[cid[q]]>=minA?2:1;}
  const to=new Int32Array(ns).fill(-1);for(let s=0;s<ns;s++){let b=-1,bv=0;for(let t=0;t<k;t++)if(bc[s*k+t]>bv){bv=bc[s*k+t];b=t;}to[s]=b;}
  let ch=0;for(let p=0;p<n;p++){const c=cid[p];if(c<0||small[c]<0)continue;const t=to[small[c]];if(t>=0&&t!==lbl[p]){lbl[p]=t;ch++;}}if(!ch)return;}}
// smooth closed outline: quadratic B-spline through edge midpoints, sharp corners kept
function smoothPath(P){const n=P.length;if(n<4)return 'M'+P.map(f2).join('L')+'Z';const mid=i=>{const a=P[i%n],b=P[(i+1)%n];return[(a[0]+b[0])/2,(a[1]+b[1])/2];};
 let s='M'+f2(mid(n-1));for(let i=0;i<n;i++){const a=P[(i+n-1)%n],b=P[i],c=P[(i+1)%n];const ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-b[0],vy=c[1]-b[1];
  const cos=(ux*vx+uy*vy)/(Math.hypot(ux,uy)*Math.hypot(vx,vy)||1);s+=cos<-0.3?'L'+f2(b)+'L'+f2(mid(i)):'Q'+f2(b)+' '+f2(mid(i));}return s+'Z';}
function f2(q){return (Math.round(q[0]*10)/10)+' '+(Math.round(q[1]*10)/10);}
self.onmessage=e=>{const{id,w,h,data,cfg,name}=e.data;const t0=Date.now();const P=(f,stage)=>self.postMessage({id,progress:f,stage});
 try{const n=w*h,k=cfg.layers;P(0.02,'Blurring');
  if(cfg.remove_bg)removeBg(data,w,h);const hasB=(()=>{for(let i=3;i<data.length;i+=4)if(data[i]<128)return true;return false;})();
  let img=data;if(cfg.smooth_s){P(0.03,'Smoothing');for(let b=0;b<(cfg.smooth_passes||1);b++)img=bilateral(img,w,h,cfg.smooth_s,cfg.smooth_r||25);}for(let b=0;b<cfg.blur;b++)img=boxBlur(img,w,h);P(0.08,'Finding tones');const lab=toLab(img,n);const C=kmeans(lab,n,k,hasB?data:null);
  const order=[...Array(k).keys()].sort((a,b)=>C[b*3]-C[a*3]);let lbl=new Int8Array(n);
  for(let i=0;i<n;i++){if(hasB&&data[i*4+3]<=127){lbl[i]=-1;continue;}let best=0,bd=1e30;for(let r=0;r<k;r++){const c=order[r];const a=lab[i*3]-C[c*3],b=lab[i*3+1]-C[c*3+1],d=lab[i*3+2]-C[c*3+2];const dd=a*a+b*b+d*d;if(dd<bd){bd=dd;best=r;}}lbl[i]=best;}
  P(0.2,'Cleaning specks');lbl=modeFilter(lbl,w,h,k,cfg.mode_r||2);const minA=Math.round((cfg.min_area_in2||0)*cfg.dpi*cfg.dpi);
  if(minA)mergeSmall(lbl,w,h,k,minA);if(cfg.min_width_in){P(0.22,'Removing slivers');removeSlivers(lbl,w,h,k,cfg.min_width_in*cfg.dpi/2);}if(minA)mergeSmall(lbl,w,h,k,minA);
  const sum=new Float64Array(k*3),cnt=new Float64Array(k);for(let i=0;i<n;i++){const t=lbl[i];if(t<0)continue;cnt[t]++;for(let q=0;q<3;q++)sum[t*3+q]+=data[i*4+q];}
  let colors=[...Array(k).keys()].map(t=>hex(cnt[t]?[sum[t*3]/cnt[t],sum[t*3+1]/cnt[t],sum[t*3+2]/cnt[t]]:[128,128,128]));
  if(cfg.palette&&cfg.palette.length>=k){// tones (light->dark) take the chosen paint colors, sorted light->dark
   const pal=cfg.palette.slice(0,k).map(h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]);
   const Lp=pal.map(c=>toLab(new Uint8ClampedArray([c[0],c[1],c[2],255]),1)[0]);colors=[...Array(k).keys()].sort((a,b)=>Lp[b]-Lp[a]).map(i=>hex(pal[i]));}
  const minpx=Math.round((cfg.min_feature_in*cfg.dpi)**2*4),bw=cfg.bridge_in*cfg.dpi;
  const tones=[...Array(k).keys()].slice(hasB?0:1);const layers=[];
  tones.forEach((t,li)=>{P(0.25+0.7*li/tones.length,`Layer ${li+1} of ${tones.length}`);let m=new Uint8Array(n);for(let i=0;i<n;i++)m[i]=lbl[i]===t?1:0;
   m=clean(m,w,h,minpx);const b=bridge(m,w,h,bw,minA);const d=trace(b.m,w,h,cfg.simplify);
   layers.push({svg:svg(d,cfg,h,li+1,tones.length,colors[t],name),mask:b.m,color:colors[t],bridges:b.nb,islands:b.left});});
  self.postMessage({id,done:true,w,h,base:hasB?null:colors[0],colors,layers,ms:Date.now()-t0},layers.map(l=>l.mask.buffer));
 }catch(err){self.postMessage({id,error:String(err&&err.stack||err)});}};
