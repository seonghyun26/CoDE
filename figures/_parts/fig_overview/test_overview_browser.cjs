const {chromium}=require('playwright');
const assert=require('node:assert/strict');const fs=require('node:fs');
const path=require('node:path');
const base=process.env.OVERVIEW_URL||'http://127.0.0.1:8767';
(async()=>{const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{}),args:['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
 try{
 const page=await browser.newPage({viewport:{width:1280,height:1200}});page.setDefaultTimeout(120000);const errors=[];
 page.on('pageerror',e=>{errors.push(e.message);console.log('PAGE ERROR',e.message)});
 page.on('console',m=>{if(m.type()==='error'){errors.push(m.text());console.log('CONSOLE ERROR',m.text())}});
 const ready=()=>page.waitForFunction(()=>window.overviewNGL&&!document.querySelector('#save').disabled&&document.querySelector('#layers').children.length>0);
 const camera=()=>page.evaluate(()=>window.overviewNGL.camera());
 const compare=(a,b)=>{for(const key of ['rotation','position','orientation'])a[key].forEach((x,i)=>assert.ok(Math.abs(x-b[key][i])<1e-5,`${key} changed`));assert.ok(Math.abs(a.distance-b.distance)<1e-5);};
 await page.goto(base);await ready();console.log('NGL READY');
 assert.equal(await page.locator('#sample-select').inputValue(),'2p0d');
 const initialCamera=await camera();
 const savedCamera={rotation:[0.0008164604594059425,-0.1458655588337554,-0.6932352125843333,0.7057956589350693],
   position:[-1.497001572625262,-1.4907252683723589,1.0207102414613276],distance:29.99510411104215,
   orientation:[-0.11109114062164176,-29.35929771667633,6.14211421045706,0,29.34500883360854,1.1652662073567899,6.10072797735547,0,-6.210023022998332,6.031588752709689,28.718666783195903,0,-1.497001572625262,-1.4907252683723589,1.0207102414613276,1]};
 compare(savedCamera,initialCamera);assert.equal(initialCamera.fov_degrees,40);assert.equal(initialCamera.camera_type,'perspective');
 assert.equal(await page.locator('#density-style').inputValue(),'mesh');
 assert.equal(await page.locator('#density-color').inputValue(),'#4761f2');
 const densityStyle=()=>page.evaluate(()=>views.get('overview').entries.find(e=>e.key==='density').style);
 assert.equal(await densityStyle(),'mesh');
 await page.locator('#density-style').selectOption('contours');await ready();compare(savedCamera,await camera());assert.equal(await densityStyle(),'contours');
 await page.locator('#density-style').selectOption('mesh');await ready();compare(savedCamera,await camera());assert.equal(await densityStyle(),'mesh');
 assert.equal(await page.locator('#iso').inputValue(),'1');
 assert.equal(await page.getByRole('button',{name:'Protein sticks',exact:true}).getAttribute('aria-pressed'),'true');
 assert.equal(await page.getByRole('button',{name:'Protein surface',exact:true}).getAttribute('aria-pressed'),'false');
 const proteinState=()=>page.evaluate(()=>{
   const entry=views.get('overview').entries.find(e=>e.key==='protein-sticks');
   return {type:entry.rep.repr.type,atoms:entry.rep.repr.structureView.atomCount};
 });
 const cropped=await proteinState();assert.equal(cropped.type,'licorice');
 const cartoonState=()=>page.evaluate(()=>{const r=views.get('overview').entries.find(e=>e.key==='protein-cartoon').rep.repr;
   return {type:r.type,visible:r.visible,buffers:r.bufferList.length};});
 assert.equal((await cartoonState()).visible,false);
 assert.equal(await page.getByRole('button',{name:'Protein cartoon',exact:true}).count(),0);
 const bounds=()=>page.evaluate(()=>Object.fromEntries(views.get(currentView).entries
   .filter(e=>['pocket-occupancy','density','gradient'].includes(e.key))
   .map(e=>[e.key,decode(e.position??e.start).reduce((m,x)=>Math.max(m,Math.abs(x)),0)])));
 const croppedBounds=await bounds();
 const rearGrid=()=>page.evaluate(()=>{
   const grid=views.get(currentView).entries.find(e=>e.key==='grid'),e=grid.extent,planes=new Set();
   let allOnFaces=true;
   for(let i=0;i<grid.start.length;i+=3){
     let onFace=false;
     for(let axis=0;axis<3;axis++)if(grid.start[i+axis]===grid.end[i+axis]&&Math.abs(grid.start[i+axis])===e){
       onFace=true;planes.add(`${axis}:${grid.start[i+axis]}`);
     }
     allOnFaces&&=onFace;
   }
   return {kind:grid.kind,signature:grid.gridSignature,planes:[...planes],allOnFaces,
     faces:grid.index.length/6,borders:grid.borderStart.length/3,visible:grid.rep.repr.visible};
 });
 const initialGrid=await rearGrid();assert.equal(initialGrid.kind,'rear-grid');assert.equal(initialGrid.signature,'111');
 assert.equal(await page.evaluate(()=>overviewBoxClip.value),8);
 assert.equal(initialGrid.faces,3);assert.equal(initialGrid.planes.length,3);assert.equal(initialGrid.borders,9);assert.ok(initialGrid.allOnFaces);
 await page.screenshot({path:'/tmp/overview-ngl-initial.png',fullPage:true});
 for(const name of ['Electron density','Voxel grid'])await page.getByRole('button',{name,exact:true}).click();
 await ready();
 assert.equal((await rearGrid()).visible,false);
 assert.equal(await page.evaluate(()=>overviewBoxClip.value),0);
 await page.waitForFunction(()=>window.overviewNGL.stage.tasks.count===0);
 const expanded=await proteinState();assert.ok(expanded.atoms>cropped.atoms,`Protein did not expand: ${JSON.stringify({cropped,expanded})}`);
 const expandedBounds=await bounds();
 for(const key of ['pocket-occupancy','density'])assert.ok(expandedBounds[key]>8&&croppedBounds[key]<=8,`${key} did not expand`);
 assert.equal(await page.getByRole('button',{name:'Pocket occupancy',exact:true}).getAttribute('aria-pressed'),'false');
 console.log('Protein crop / surroundings',cropped.atoms,expanded.atoms);
 const cartoon=await cartoonState();assert.equal(cartoon.type,'cartoon');assert.equal(cartoon.visible,true);assert.ok(cartoon.buffers>0);
 await page.getByRole('button',{name:'Protein cartoon',exact:true}).click();assert.equal((await cartoonState()).visible,false);compare(savedCamera,await camera());
 await page.getByRole('button',{name:'Protein cartoon',exact:true}).click();assert.equal((await cartoonState()).visible,true);
 await page.locator('#plot').screenshot({path:'/tmp/overview-ngl-qual-style.png'});
 for(const name of ['Electron density','Voxel grid'])await page.getByRole('button',{name,exact:true}).click();
 await ready();assert.deepEqual(await bounds(),croppedBounds);
 assert.equal(await page.evaluate(()=>overviewBoxClip.value),8);
 assert.equal((await cartoonState()).visible,false);
 const box=await page.locator('#plot').boundingBox();await page.mouse.move(box.x+box.width*.4,box.y+box.height*.4);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.5,{steps:15});await page.mouse.up();await page.mouse.wheel(0,80);await page.waitForTimeout(400);
 const before=await camera();await page.getByRole('button',{name:'Protein surface',exact:true}).click();compare(before,await camera());await page.getByRole('button',{name:'Protein surface',exact:true}).click();compare(before,await camera());
 await page.locator('#density-hex').fill('#EFA0C8');compare(before,await camera());
 assert.equal(await page.locator('#density-color').inputValue(),'#efa0c8');
 const densityState=()=>page.evaluate(()=>{const entry=views.get(currentView).entries.find(e=>e.key==='density'||e.key==='gradient');return {color:entry.color,start:entry.start,position:entry.position};});
 const limited=await densityState();
 await page.locator('#density-proximity').click();await ready();compare(before,await camera());
 assert.equal(await page.locator('#density-proximity').getAttribute('aria-pressed'),'false');
 assert.notEqual((await densityState()).start,limited.start,'Proximity toggle should change density geometry');
 for(const view of ['gradient','overview']){await page.locator(`[data-view="${view}"]`).click();await ready();compare(before,await camera());console.log('view',view,'camera preserved');}
 assert.equal(await page.locator('[data-view="masked"]').count(),0);
 await page.evaluate(()=>{window.beforeMask=views.get('overview').entries.filter(e=>!e.maskBoxes&&e.key!=='grid').map(e=>({entry:e,rep:e.rep,visible:e.visible,selection:e.rep.repr.selection?.string}));});
 const sameOverview=async()=>assert.ok(await page.evaluate(()=>currentView==='overview'&&window.beforeMask.every(({entry,rep,visible,selection})=>
   views.get('overview').entries.includes(entry)&&entry.rep===rep&&entry.visible===visible&&entry.rep.repr.selection?.string===selection)), 'Masking changed Overview data or settings');
 await page.locator('#masking').click();compare(before,await camera());await sameOverview();
 assert.equal(await page.locator('#masking').getAttribute('aria-pressed'),'true');
 assert.ok(await page.evaluate(()=>views.get('overview').entries.filter(e=>e.maskBoxes).every(e=>e.visible&&e.rep.repr.visible)));
 for(const name of ['Ligand sticks','Protein sticks']) {
   const button=page.getByRole('button',{name,exact:true});assert.equal(await button.getAttribute('aria-pressed'),'true');
 }
 const maskSignature=()=>page.evaluate(()=>views.get('overview').entries.find(e=>e.key==='mask-fill').maskSignature);
 const originalFaces=await maskSignature();
 const originalGridFaces=(await rearGrid()).signature;
 await page.evaluate(()=>stage.viewerControls.rotate(new NGL.Quaternion().setFromAxisAngle(new NGL.Vector3(0,1,0),Math.PI)));
 await page.waitForFunction(old=>views.get('overview').entries.find(e=>e.key==='mask-fill').maskSignature!==old,originalFaces);
 assert.notEqual((await rearGrid()).signature,originalGridFaces);assert.ok((await rearGrid()).allOnFaces);
 await page.locator('#masking').click();await sameOverview();
 assert.ok(await page.evaluate(()=>views.get('overview').entries.filter(e=>e.maskBoxes).every(e=>!e.visible&&!e.rep.repr.visible)));
 await page.locator('#masking').click();await sameOverview();
 await page.evaluate(value=>restoreCamera(value),before);
 await page.waitForFunction(old=>views.get('overview').entries.find(e=>e.key==='mask-fill').maskSignature===old,originalFaces);
 await page.locator('#plot').screenshot({path:'/tmp/overview-masking-camera.png'});
 await page.locator('[data-view="overview"]').click();await ready();
 assert.equal((await densityState()).color,'#EFA0C8');
 for(const view of ['gradient','overview']) {
   await page.locator(`[data-view="${view}"]`).click();await ready();
   await page.getByRole('button',{name:'Voxel grid',exact:true}).click();await ready();compare(before,await camera());
   const extent=await bounds();for(const value of Object.values(extent))assert.ok(value>8);
 }
 assert.equal(await page.locator('#masking').getAttribute('aria-pressed'),'true');
 assert.ok(await page.evaluate(()=>views.get('overview').entries.filter(e=>e.maskBoxes).every(e=>e.rep.repr.visible)));
 console.log('PASS masking overlay preserves Overview representations, controls, camera and grid state');
 await page.locator('#iso').fill('0.2');await page.locator('#iso').dispatchEvent('change');await ready();compare(before,await camera());
 assert.equal((await densityState()).color,'#EFA0C8');
 assert.ok((await bounds()).density>8,'Threshold update lost expanded density');
 await page.locator('#density-proximity').click();await ready();compare(before,await camera());
 assert.ok((await bounds()).density>8,'Proximity update lost expanded density');
 await page.locator('#density-color').fill('#8092f6');compare(before,await camera());assert.equal((await densityState()).color,'#8092F6');
 await page.locator('#print-camera').click();const report=JSON.parse(await page.locator('#camera-output').inputValue());compare(before,report.camera);assert.equal(report.renderer,'NGL');assert.ok(Number.isFinite(report.angles_deg.azimuth));assert.ok(Number.isFinite(report.angles_deg.elevation));
 await page.locator('#name').fill('overview-ngl-smoke');await page.locator('#format').selectOption('both');
 const savedResponse=page.waitForResponse(r=>r.url().endsWith('/api/save')&&r.request().method()==='POST');await page.locator('#save').click();const saved=await (await savedResponse).json();console.log('SAVE',saved);assert.equal(saved.files.length,2);await ready();compare(before,await camera());
 const paths=saved.files.map(file=>path.resolve(__dirname,'../../..',file.path));
 require('node:child_process').execFileSync(process.env.VOXBIND_PYTHON||require('node:child_process').execFileSync(process.env.CONDA_EXE||'conda',['run','-n','voxbind','python','-c','import sys;print(sys.executable)']).toString().trim(),['-c',
   'import sys; from PIL import Image; from pypdf import PdfReader; im=Image.open(sys.argv[1]); assert im.mode=="RGBA"; alpha=im.getchannel("A"); assert alpha.getextrema()==(0,255); images=PdfReader(sys.argv[2]).pages[0]["/Resources"]["/XObject"].get_object(); masks=[x.get_object()["/SMask"].get_object().get_data() for x in images.values() if "/SMask" in x.get_object()]; assert masks and min(masks[0])==0 and max(masks[0])==255; print("PASS PNG alpha and PDF soft mask")',...paths],{stdio:'inherit'});
 for(const file of saved.files){assert.ok(file.path.startsWith('figures/fig-overview/exports/pre-training/overview-ngl-smoke-'));assert.equal((await page.request.get(base+file.url)).status(),200);fs.unlinkSync(path.resolve(__dirname,'../../..',file.path));}
 await page.locator('#reset').click();compare(savedCamera,await camera());
 assert.equal((await cartoonState()).visible,true);
 await page.getByRole('button',{name:'Protein cartoon',exact:true}).click();
 await page.locator('[data-view="gradient"]').click();await ready();assert.equal((await cartoonState()).visible,false);
 await page.locator('[data-view="overview"]').click();await ready();assert.equal((await cartoonState()).visible,false);
 // Isolate the clipping probe from camera-dependent overlay reconstruction.
 await page.locator('#masking').click();assert.equal(await page.locator('#masking').getAttribute('aria-pressed'),'false');
 const clipPixels=await page.evaluate(async()=>{
   const saved=camera(),oldClip=overviewBoxClip.value,components=stage.compList.slice();
   components.forEach(component=>component.setVisibility(false));
   const probe=shapeLayer({key:'clip-probe',kind:'mesh',color:'#FF0000',opacity:1,
     position:new Float32Array([9,-2,0,11,-2,0,11,2,0,9,2,0]),index:new Uint32Array([0,1,2,0,2,3])});
   try {
     probe.rep.setVisibility(true);restoreCamera({rotation:[0,0,0,1],position:[0,0,0],distance:45});
     const pixels=async extent=>{
       overviewBoxClip.value=extent;
       const bitmap=await createImageBitmap(await stage.makeImage({factor:1,antialias:false,trim:false,transparent:true}));
       const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
       const context=canvas.getContext('2d');context.drawImage(bitmap,0,0);bitmap.close();
       const data=context.getImageData(0,0,canvas.width,canvas.height).data;
       let count=0;for(let i=3;i<data.length;i+=4)if(data[i])count++;return count;
     };
     return {cropped:await pixels(8),expanded:await pixels(0)};
   }finally{stage.removeComponent(probe.component);overviewBoxClip.value=oldClip;components.forEach(component=>component.setVisibility(true));restoreCamera(saved);}
 });
 assert.equal(clipPixels.cropped,0,'Outside geometry leaked through the cube clip');assert.ok(clipPixels.expanded>100,'Grid OFF did not expose outside geometry');console.log('PASS GPU cube clipping',clipPixels);
 assert.deepEqual(errors,[]);console.log('PASS saved default/reset camera, NGL controls, threshold, PNG/PDF');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
