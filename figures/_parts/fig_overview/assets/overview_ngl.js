"use strict";

const plot = document.querySelector('#plot');
const statusNode = document.querySelector('#status');
const sampleInput = document.querySelector('#sample-select');
const isoInput = document.querySelector('#iso');
const densityColorInput = document.querySelector('#density-color');
const densityHexInput = document.querySelector('#density-hex');
let densityColor = '#4761F2', gradientColor = '#8092F6', densityStyle = 'mesh';
let densityProximity = true;
const isExpanded=view=>views.get(view)?.entries.find(entry=>entry.key==='grid')?.visible===false;
const nglUrl=(view,iso,proximity=densityProximity,expanded=isExpanded(view),style=densityStyle)=>`/api/ngl?sample=${config.sample}&view=${view}&iso=${iso}&proximity=${proximity?1:0}&expanded=${expanded?1:0}&density_style=${style}`;
const stage = new NGL.Stage(plot, {cameraType:'perspective', quality:'high', sampleLevel:1, backgroundColor:'white'});
stage.setParameters({cameraType:'perspective', clipNear:0, clipFar:100, fogNear:100, fogFar:100,
  lightIntensity:1, ambientIntensity:.55, hoverTimeout:0});
let config, currentView, busy=false;
const views = new Map();
const thresholds = {overview:1, gradient:.35};
// Broad, matte shading preserves depth without glossy molecular highlights.
const matteMaterial = {roughness:1, metalness:0};
const elementScheme = NGL.ColormakerRegistry.addScheme(function () {
  this.atomColor = atom => parseInt((config.ligand_elements[atom.element.toUpperCase()] || config.palette.ligand).slice(1),16);
}, 'overview-qual-elements');

function status(text,error=false) { statusNode.textContent=text; statusNode.classList.toggle('error',error); }
function lock(value) { busy=value; document.querySelectorAll('button,input,select').forEach(el=>{el.disabled=value;}); }
async function api(url,options) { const r=await fetch(url,options), body=await r.json(); if(!r.ok)throw new Error(body.error||r.statusText);return body; }
function camera() {
  const controls=stage.viewerControls;
  // A reset can be read before the next render updates the rotation matrix.
  stage.viewer.rotationGroup.updateMatrix();
  return {rotation:controls.rotation.toArray(),position:controls.position.toArray(),distance:controls.getCameraDistance(),
    orientation:controls.getOrientation().toArray(),camera_type:stage.getParameters().cameraType,fov_degrees:stage.getParameters().cameraFov};
}
function restoreCamera(value) {
  const c=stage.viewerControls;
  c.rotation.set(...value.rotation);c.position.set(...value.position);c.distance(value.distance);
  stage.viewer.requestRender();
}
function resetCamera() {
  if(config.ngl_camera) {
    stage.setParameters({cameraType:config.ngl_camera.camera_type,cameraFov:config.ngl_camera.fov_degrees});
    restoreCamera(config.ngl_camera);return;
  }
  const camera=config.camera;
  stage.viewerControls.position.set(0,0,0);
  stage.viewerControls.align(new NGL.Matrix4().lookAt(
    new NGL.Vector3(camera.eye.x,camera.eye.y,camera.eye.z),new NGL.Vector3(0,0,0),
    new NGL.Vector3(camera.up.x,camera.up.y,camera.up.z)));
  const bounds=new NGL.Box3(new NGL.Vector3(-8,-8,-8),new NGL.Vector3(8,8,8));
  stage.viewerControls.distance(stage.getZoomForBox(bounds));
}
function settled() { return new Promise(resolve=>stage.tasks.onZeroOnce(resolve)); }
function decode(encoded,integer=false) {
  if(typeof encoded!=='string')return encoded;
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  return integer ? new Uint32Array(bytes.buffer) : new Float32Array(bytes.buffer);
}
function rgb(hex) { const n=parseInt(hex.slice(1),16);return [(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]; }
function colorArray(count,color) { const out=new Float32Array(count*3),c=rgb(color);for(let i=0;i<count;i++)out.set(c,i*3);return out; }
function maskFaces(boxes) {
  const c=stage.viewerControls,inverse=c.rotation.clone().invert();
  // NGL puts its camera on negative Z. Use its actual position rather than
  // assuming the positive-Z convention used by the angle report.
  const worldCamera=stage.viewer.camera.position.clone().applyQuaternion(inverse);
  const direction=worldCamera.clone().normalize(),eye=worldCamera.sub(c.position);
  const perspective=stage.getParameters().cameraType==='perspective',faces=[];
  for(let i=0;i<boxes.length;i+=6)for(let axis=0;axis<3;axis++)
    faces.push((perspective?eye.getComponent(axis)-(boxes[i+axis]+boxes[i+axis+3])/2:direction.getComponent(axis))>=0?1:0);
  return faces;
}
function maskGeometry(layer) {
  const boxes=decode(layer.maskBoxes),faces=maskFaces(boxes),position=[],index=[],start=[],end=[];
  for(let i=0;i<boxes.length;i+=6) {
    const lo=Array.from(boxes.slice(i,i+3)),hi=Array.from(boxes.slice(i+3,i+6)),front=faces.slice(i/2,i/2+3);
    if(layer.key==='mask-fill') {
      for(let axis=0;axis<3;axis++) {
        const u=(axis+1)%3,v=(axis+2)%3,base=position.length/3;
        for(const [a,b] of [[0,0],[1,0],[1,1],[0,1]]) {
          const p=lo.slice();p[axis]=front[axis]?hi[axis]:lo[axis];p[u]=a?hi[u]:lo[u];p[v]=b?hi[v]:lo[v];position.push(...p);
        }
        index.push(base,base+1,base+2,base,base+2,base+3);
      }
    }else{
      const offset=layer.outlineOffset??0;
      const corners=Array.from({length:8},(_,corner)=>lo.map((low,axis)=>corner&(1<<axis)?hi[axis]+offset:low-offset));
      for(let corner=0;corner<8;corner++)for(let axis=0;axis<3;axis++) {
        const other=corner^(1<<axis);if(corner>=other)continue;
        if([0,1,2].some(a=>a!==axis&&((corner>>a)&1)===front[a])) {start.push(...corners[corner]);end.push(...corners[other]);}
      }
    }
  }
  return {...layer,maskSignature:faces.join(''),position:new Float32Array(position),index:new Uint32Array(index),
    start:new Float32Array(start),end:new Float32Array(end)};
}
function rearGridFaces(layer) {
  const e=layer.extent;return maskFaces([-e,-e,-e,e,e,e]).map(face=>1-face);
}
function rearGridGeometry(layer) {
  const faces=rearGridFaces(layer),e=layer.extent,position=[],index=[],start=[],end=[],borderStart=[],borderEnd=[],edges=new Set();
  for(let axis=0;axis<3;axis++) {
    const u=(axis+1)%3,v=(axis+2)%3,side=faces[axis]?1:-1,base=position.length/3,corners=[];
    for(const [a,b] of [[-1,-1],[1,-1],[1,1],[-1,1]]) {
      const p=[0,0,0];p[axis]=side*e;p[u]=a*e;p[v]=b*e;corners.push(p);
      // Put fill just behind its lines to avoid coplanar flicker.
      const fill=p.slice();fill[axis]+=side*.005;position.push(...fill);
    }
    index.push(base,base+1,base+2,base,base+2,base+3);
    for(let k=0;k<4;k++) {
      const a=corners[k],b=corners[(k+1)%4],key=[a.join(','),b.join(',')].sort().join('|');
      if(!edges.has(key)) {edges.add(key);borderStart.push(...a);borderEnd.push(...b);}
    }
    for(let t=-e+layer.spacing;t<e-1e-6;t+=layer.spacing)for(const varying of [u,v]) {
      const fixed=varying===u?v:u,a=[0,0,0],b=[0,0,0];
      a[axis]=b[axis]=side*e;a[fixed]=b[fixed]=t;a[varying]=-e;b[varying]=e;
      start.push(...a);end.push(...b);
    }
  }
  return {...layer,gridSignature:faces.join(''),position:new Float32Array(position),index:new Uint32Array(index),
    start:new Float32Array(start),end:new Float32Array(end),borderStart:new Float32Array(borderStart),borderEnd:new Float32Array(borderEnd)};
}
function shapeLayer(layer) {
  if(layer.maskBoxes)layer=maskGeometry(layer);
  if(layer.kind==='rear-grid')layer=rearGridGeometry(layer);
  if(layer.key==='density')layer={...layer,color:densityColor};
  if(layer.key==='gradient')layer={...layer,color:gradientColor};
  const shape=new NGL.Shape(layer.key);
  const depthWrite=Boolean(layer.maskBoxes)||(layer.opacity??1)===1;
  // Draw translucent mask faces after the scene, recording their nearest depth.
  // Draw outlines last in the transparent pass, so occluded edges fail depth test.
  const addBuffer=buffer=>{
    if(layer.maskBoxes)buffer.getRenderOrder=()=>layer.key==='mask-fill'?10:11;
    shape.addBuffer(buffer);
  };
  if(layer.kind==='rear-grid') {
    shape.addBuffer(new NGL.MeshBuffer({position:layer.position,color:colorArray(layer.position.length/3,layer.fillColor),index:layer.index},
      {background:true,side:'double'}));
    for(const [start,end,color,width] of [[layer.start,layer.end,layer.color,layer.width],
      [layer.borderStart,layer.borderEnd,layer.borderColor,layer.borderWidth]]) {
      if(start.length)shape.addBuffer(new NGL.WidelineBuffer({position1:start,position2:end,color:colorArray(start.length/3,color)},
        {linewidth:width,opacity:1,depthWrite:true}));
    }
  } else if(layer.kind==='lines') {
    const start=decode(layer.start),end=decode(layer.end);
    if(start.length)addBuffer(new NGL.WidelineBuffer({position1:start,position2:end,color:colorArray(start.length/3,layer.color)},
      {linewidth:layer.width,opacity:layer.opacity,depthWrite,forceTransparent:Boolean(layer.maskBoxes)}));
  } else {
    const position=decode(layer.position),index=decode(layer.index,true);
    if(position.length)addBuffer(new NGL.MeshBuffer({position,color:colorArray(position.length/3,layer.color),index},
      {...matteMaterial,opacity:layer.opacity??1,side:'double',depthWrite}));
  }
  const component=stage.addComponentFromObject(shape);
  const rep=component.addRepresentation('buffer',{...matteMaterial,opacity:layer.opacity??1,side:'double',depthWrite});
  rep.setVisibility(false);
  return {...layer,component,rep};
}
async function loadText(text,ext) { return stage.loadFile(new Blob([text],{type:'text/plain'}),{ext,firstModelOnly:true}); }
async function makeView(payload) {
  const entries=[]; const owned=[];
  try {
    let ligand,pocket,protein,pocketSelection,pocketExpandedSelection,ligandSelection='not hydrogen';
    const atomMask=payload.atom_mask_boxes?decode(payload.atom_mask_boxes):null;
    const unmasked=atom=>{
      if(!atomMask)return true;
      for(let i=0;i<atomMask.length;i+=6)if(atom.x>=atomMask[i]&&atom.x<atomMask[i+3]
        &&atom.y>=atomMask[i+1]&&atom.y<atomMask[i+4]&&atom.z>=atomMask[i+2]&&atom.z<atomMask[i+5])return false;
      return true;
    };
    const selection=indices=>indices.length?'@'+indices.join(','):'NONE';
    if(payload.ligand_sdf) {ligand=await loadText(payload.ligand_sdf,'sdf');owned.push(ligand);}
    if(ligand&&atomMask) {const indices=[];ligand.structure.eachAtom(atom=>{if(unmasked(atom))indices.push(atom.index);});ligandSelection=selection(indices);}
    if(payload.protein_pdb) {protein=await loadText(payload.protein_pdb,'pdb');owned.push(protein);}
    if(payload.pocket_pdb) {
      pocket=await loadText(payload.pocket_pdb,'pdb');owned.push(pocket);
      const indices=[],expandedIndices=[];
      pocket.structure.eachAtom(atom=>{if(!unmasked(atom))return;expandedIndices.push(atom.index);
        if(Math.max(Math.abs(atom.x),Math.abs(atom.y),Math.abs(atom.z))<=8.15)indices.push(atom.index);});
      pocketSelection=selection(indices);pocketExpandedSelection=selection(expandedIndices);
    }
    for(const layer of payload.layers) {
      let rep;
      if(layer.kind==='ligand') {
        rep=ligand.addRepresentation('licorice',{...matteMaterial,sele:ligandSelection,colorScheme:elementScheme,
          radiusType:'size',radiusSize:.24,radiusScale:1,multipleBond:'off',quality:'high',disableImpostor:true,visible:false});
      } else if(layer.kind==='protein') {
        rep=pocket.addRepresentation('surface',{...matteMaterial,sele:pocketSelection,surfaceType:'ms',probeRadius:1.4,
          smooth:2,scaleFactor:2,colorScheme:'uniform',colorValue:parseInt(layer.color.slice(1),16),
          opacity:layer.opacity,opaqueBack:false,side:'double',depthWrite:false,useWorker:true,visible:false});
      } else if(layer.kind==='protein-sticks') {
        rep=pocket.addRepresentation('licorice',{...matteMaterial,sele:pocketSelection,colorScheme:'uniform',
          colorValue:parseInt(layer.color.slice(1),16),radiusType:'size',radiusSize:.16,radiusScale:1,
          multipleBond:'off',quality:'high',disableImpostor:true,visible:false});
      } else if(layer.kind==='protein-cartoon') {
        rep=protein.addRepresentation('cartoon',{...matteMaterial,sele:'protein',colorScheme:'uniform',
          colorValue:parseInt(layer.color.slice(1),16),opacity:layer.opacity,
          quality:'high',visible:false});
      } else { const entry=shapeLayer(layer);owned.push(entry.component);entries.push(entry);continue; }
      entries.push({...layer,rep,pocketSelection:['protein','protein-sticks'].includes(layer.kind)?pocketSelection:null,pocketExpandedSelection});
    }
    await settled();
    return {entries,owned};
  } catch(error) { owned.forEach(component=>stage.removeComponent(component));throw error; }
}
function controls() {
  const entries=views.get(currentView).entries,container=document.querySelector('#layers');container.replaceChildren();
  const order=['ligand','ligand-occupancy','protein-sticks','protein','protein-cartoon','pocket-occupancy',
    'density','gradient','mask-fill','mask-outline','grid'];
  for(const entry of [...entries].sort((a,b)=>order.indexOf(a.key)-order.indexOf(b.key))) {
    if(entry.maskBoxes)continue; // The single Masking toggle owns the cubes.
    if(entry.gridOffOnly&&!isExpanded(currentView))continue;
    const button=document.createElement('button');button.className='layer';button.setAttribute('aria-label',entry.label);
    button.setAttribute('aria-pressed',String(entry.visible));button.style.setProperty('--color',entry.color);
    const dot=document.createElement('span');dot.className='dot';const label=document.createElement('span');label.textContent=entry.label;
    const toggle=document.createElement('span');toggle.className='switch';button.append(dot,label,toggle);
    button.onclick=async()=>{
      if(busy)return;const keep=camera();
      if(entry.key==='grid') {
        lock(true);status('표시 범위를 변경하고 있어요…');
        try {
          const data=await api(nglUrl(currentView,thresholds[currentView],densityProximity,entry.visible));
          replaceGeometry(currentView,data,['density','gradient','pocket-occupancy']);
          entry.visible=!entry.visible;entry.rep.setVisibility(entry.visible);updateDataClip();
          for(const protein of entries.filter(item=>item.pocketSelection))
            protein.rep.setSelection(entry.visible?protein.pocketSelection:protein.pocketExpandedSelection);
          for(const item of entries.filter(item=>item.gridOffOnly))item.rep.setVisibility(!entry.visible&&item.visible);
          await settled();controls();updateExtentLabel();status(entry.visible?'16 Å 큐브 범위로 표시합니다.':'주변 24 Å까지 표시합니다.');
        }catch(error){status(error.message,true);}finally{restoreCamera(keep);lock(false);}
      }else{
        entry.visible=!entry.visible;entry.rep.setVisibility(entry.visible);
        restoreCamera(keep);button.setAttribute('aria-pressed',String(entry.visible));
      }
    };
    container.append(button);
  }
  const legend=document.querySelector('#element-legend');legend.replaceChildren();legend.hidden=currentView==='gradient';
  for(const [name,color] of Object.entries(config.ligand_elements)) {const span=document.createElement('span'),dot=document.createElement('i');
    dot.className='dot';dot.style.setProperty('--color',color);span.append(dot,document.createTextNode(name));legend.append(span);}
}
function updateExtentLabel() {
  document.querySelector('#sample').textContent=`${config.sample.toUpperCase()} · NGL · ${isExpanded(currentView)?24:16} Å`;
}
function updateDataClip() {
  overviewBoxClip.value=isExpanded(currentView)?0:8;
  stage.viewer.requestRender();
}
function updateCameraFaces() {
  const list=views.get(currentView);if(!list)return;
  for(const entry of list.entries.filter(item=>item.maskBoxes||item.kind==='rear-grid')) {
    const unchanged=entry.kind==='rear-grid'?rearGridFaces(entry).join('')===entry.gridSignature:
      maskFaces(decode(entry.maskBoxes)).join('')===entry.maskSignature;
    if(unchanged)continue;
    const next=shapeLayer(entry),old=entry.component;next.rep.setVisibility(entry.visible);
    list.owned=list.owned.filter(component=>component!==old);list.owned.push(next.component);
    Object.assign(entry,next);stage.removeComponent(old);
  }
}
let maskFrame=null;
stage.viewerControls.signals.changed.add(()=>{
  if(maskFrame!==null)return;
  maskFrame=requestAnimationFrame(()=>{maskFrame=null;updateCameraFaces();});
});
function changeDensityColor(value) {
  if(busy||!currentView)return;
  if(!/^#[0-9a-f]{6}$/i.test(value)) {densityHexInput.setCustomValidity('예: #8092F6 형식으로 입력하세요.');densityHexInput.reportValidity();return;}
  densityHexInput.setCustomValidity('');
  const key=currentView==='gradient'?'gradient':'density',color=value.toUpperCase();
  if(key==='gradient')gradientColor=color;else densityColor=color;
  densityColorInput.value=color;densityHexInput.value=color;
  const keep=camera();
  for(const [view,list] of views) {
    const index=list.entries.findIndex(entry=>entry.key===key);
    if(index<0)continue;
    const old=list.entries[index],next=shapeLayer(old);
    next.rep.setVisibility(view===currentView&&next.visible);list.entries[index]=next;
    list.owned=list.owned.filter(component=>component!==old.component);list.owned.push(next.component);
    stage.removeComponent(old.component);
  }
  restoreCamera(keep);controls();status(`Density 색상을 ${color}로 적용했어요.`);
}
densityColorInput.oninput=()=>changeDensityColor(densityColorInput.value);
densityHexInput.oninput=()=>{densityHexInput.setCustomValidity('');if(/^#[0-9a-f]{6}$/i.test(densityHexInput.value))changeDensityColor(densityHexInput.value);};
densityHexInput.onchange=()=>changeDensityColor(densityHexInput.value);
async function switchView(view) {
  if(busy||view===currentView)return;lock(true);status('NGL 화면을 준비하고 있어요…');
  const keep=currentView?camera():null;
  try {
    if(!views.has(view))views.set(view,await makeView(await api(nglUrl(view,thresholds[view]))));
    for(const [name,{entries}] of views)for(const entry of entries)
      entry.rep.setVisibility(name===view&&entry.visible&&(!entry.gridOffOnly||isExpanded(name)));
    await settled();
    currentView=view;updateDataClip();if(keep)restoreCamera(keep);else resetCamera();updateCameraFaces();
    document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===view)));
    document.querySelector('#masking').hidden=view!=='overview';
    document.querySelector('#masking').setAttribute('aria-pressed',String(maskingEnabled()));
    isoInput.max=view==='gradient'?'2':String(config.density_max);isoInput.value=thresholds[view];
    document.querySelector('#iso-value').value=thresholds[view].toFixed(2);
    document.querySelector('#iso-label').textContent=view==='gradient'?'Gradient iso':'Density iso';
    document.querySelector('#density-proximity-control').hidden=view==='gradient';
    document.querySelector('#density-style-control').hidden=view==='gradient';
    document.querySelector('#density-color-label').textContent=view==='gradient'?'Density gradient 색상':'Electron density 색상';
    densityColorInput.value=view==='gradient'?gradientColor:densityColor;densityHexInput.value=densityColorInput.value.toUpperCase();
    document.querySelector('#name').value=`overview-${config.sample}-${view}`;
    controls();updateExtentLabel();status('준비됐어요. 레이어와 시점을 조절한 뒤 저장하세요.');
  } catch(error) {status(error.message,true);}finally{lock(false);}
}
isoInput.oninput=()=>{document.querySelector('#iso-value').value=Number(isoInput.value).toFixed(2);};
function replaceGeometry(view,data,keys) {
  const list=views.get(view);
  for(const key of keys) {
    const index=list.entries.findIndex(entry=>entry.key===key);
    if(index<0)continue;
    const old=list.entries[index],next=shapeLayer(data.layers.find(layer=>layer.key===key));next.visible=old.visible;
    next.rep.setVisibility(view===currentView&&next.visible);
    list.entries[index]=next;list.owned=list.owned.filter(c=>c!==old.component);list.owned.push(next.component);
    stage.removeComponent(old.component);
  }
}
const replaceDensity=(view,data)=>replaceGeometry(view,data,[view==='gradient'?'gradient':'density']);
document.querySelector('#density-style').onchange=async()=>{
  if(busy)return;lock(true);const keep=camera(),input=document.querySelector('#density-style'),next=input.value;
  try {
    const targets=[...views.keys()].filter(view=>view!=='gradient');
    const data=await Promise.all(targets.map(view=>api(nglUrl(view,thresholds[view],densityProximity,isExpanded(view),next))));
    targets.forEach((view,index)=>replaceDensity(view,data[index]));densityStyle=next;
    controls();status(next==='mesh'?'등밀도면 wire mesh를 적용했어요.':'단면 등고선을 적용했어요.');
  }catch(error){input.value=densityStyle;status(error.message,true);}finally{restoreCamera(keep);lock(false);}
};
document.querySelector('#density-proximity').onclick=async()=>{
  if(busy)return;lock(true);const keep=camera(),next=!densityProximity;
  try {
    const targets=[...views.keys()].filter(view=>view!=='gradient');
    const data=await Promise.all(targets.map(view=>api(nglUrl(view,thresholds[view],next))));
    targets.forEach((view,index)=>replaceDensity(view,data[index]));densityProximity=next;
    document.querySelector('#density-proximity').setAttribute('aria-pressed',String(next));
    controls();status(next?'Density를 원자 주변으로 제한했어요.':'원자 주변 제한을 해제했어요.');
  }catch(error){status(error.message,true);}finally{restoreCamera(keep);lock(false);}
};
isoInput.onchange=async()=>{
  if(busy)return;lock(true);const keep=camera();
  try {
    const iso=Number(isoInput.value),data=await api(nglUrl(currentView,iso));
    replaceDensity(currentView,data);thresholds[currentView]=iso;restoreCamera(keep);controls();status('Density threshold를 적용했어요.');
  } catch(error){status(error.message,true);}finally{lock(false);}
};
document.querySelectorAll('[data-view]').forEach(button=>{button.onclick=()=>switchView(button.dataset.view);});
function maskingEnabled() {
  return views.get('overview')?.entries.find(entry=>entry.key==='mask-fill')?.visible??false;
}
document.querySelector('#masking').onclick=()=>{
  if(busy||currentView!=='overview')return;
  const next=!maskingEnabled();
  for(const entry of views.get('overview').entries.filter(entry=>entry.maskBoxes)) {
    entry.visible=next;entry.rep.setVisibility(next);
  }
  updateCameraFaces();
  document.querySelector('#masking').setAttribute('aria-pressed',String(next));
  status(next?'Overview 위에 masking 정육면체를 표시합니다.':'Masking 정육면체를 숨겼어요.');
};
sampleInput.onchange=async()=>{
  if(busy)return;lock(true);status('새 protein–ligand pair를 불러오고 있어요…');
  try {const next=await api(`/api/config?sample=${sampleInput.value}`);stage.removeAllComponents();views.clear();currentView=null;config=next;
    thresholds.overview=1;thresholds.gradient=.35;
    document.querySelector('#sample').textContent=`${config.sample.toUpperCase()} · NGL · 16 Å`;
    lock(false);await switchView('overview');
  }catch(error){sampleInput.value=config.sample;status(error.message,true);lock(false);}
};
document.querySelector('#reset').onclick=()=>{if(!busy)resetCamera();};
document.querySelector('#print-camera').onclick=()=>{
  if(busy||!currentView)return;
  const direction=new NGL.Vector3(0,0,1).applyQuaternion(stage.viewerControls.rotation.clone().invert());
  const angles={azimuth:Math.atan2(direction.y,direction.x)*180/Math.PI,
    elevation:Math.atan2(direction.z,Math.hypot(direction.x,direction.y))*180/Math.PI};
  const snapshot={sample:config.sample,view:currentView,masking:currentView==='overview'&&maskingEnabled(),renderer:'NGL',angles_deg:angles,camera:camera()};
  document.querySelector('#camera-output').value=JSON.stringify(snapshot,null,2);
  document.querySelector('#camera-summary').textContent=`출력 시점: ${config.sample.toUpperCase()} / ${currentView} · 방위각 ${angles.azimuth.toFixed(2)}° · 고도 ${angles.elevation.toFixed(2)}° · NGL camera`;
  document.querySelector('#camera-report').hidden=false;status('아래에 NGL 카메라 설정을 출력했어요.');
};
document.querySelector('#copy-camera').onclick=async()=>{const output=document.querySelector('#camera-output');
  try{await navigator.clipboard.writeText(output.value);status('카메라 설정을 복사했어요.');}
  catch(_){output.focus();output.select();status('Ctrl+C 또는 ⌘C로 복사하세요.');}
};
async function capture() {
  updateCameraFaces();
  const blob=await stage.makeImage({factor:Math.max(1,Math.ceil(1520/stage.viewer.width)),antialias:true,trim:false,transparent:true});
  const bitmap=await createImageBitmap(blob),canvas=document.createElement('canvas');canvas.width=1520;canvas.height=1520;
  canvas.getContext('2d').drawImage(bitmap,0,0,1520,1520);bitmap.close();return canvas.toDataURL('image/png');
}
document.querySelector('#save').onclick=async()=>{
  if(busy||!currentView)return;const name=document.querySelector('#name');if(!name.value||!name.reportValidity())return;
  lock(true);status('현재 이미지를 저장하고 있어요…');
  try {const result=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name.value,
      format:document.querySelector('#format').value,image:await capture()})});
    status('저장 완료: ');for(const file of result.files){const a=document.createElement('a');a.href=file.url;a.textContent=file.path;
      a.target='_blank';a.rel='noopener';statusNode.append(a,document.createElement('br'));}
  }catch(error){status(error.message,true);}finally{lock(false);}
};
new ResizeObserver(()=>stage.handleResize()).observe(plot);
window.overviewNGL={stage,camera};
(async()=>{try{config=await api('/api/config');for(const item of config.samples){const option=document.createElement('option');
  option.value=item.id;option.textContent=item.label;sampleInput.append(option);}sampleInput.value=config.sample;
  document.querySelector('#sample').textContent=`${config.sample.toUpperCase()} · NGL · 16 Å`;await switchView('overview');
}catch(error){status(error.message,true);}})();
