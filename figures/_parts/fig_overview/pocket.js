"use strict";

const $ = (root, key) => root.querySelector(`[data-control="${key}"]`);
async function api(url, options) {
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
const ready = stage => new Promise(resolve => stage.tasks.onZeroOnce(resolve));
const numberColor = hex => parseInt(hex.slice(1), 16);
const option = (value, text) => new Option(text, value);
const blobURL = blob => new Promise((resolve, reject) => {
  const reader = new FileReader(); reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(reader.error); reader.readAsDataURL(blob);
});

class PocketPanel {
  constructor(task, targets) {
    this.task = task;
    this.root = document.querySelector("#panel-template").content.firstElementChild.cloneNode(true);
    this.root.dataset.task = task;
    this.root.querySelector(".eyebrow").textContent = task === "ligand" ? "DOWNSTREAM 02" : "DOWNSTREAM 03";
    this.root.querySelector("h2").textContent = task === "ligand" ? "Pocket-conditioned ligand generation" : "Macrocyclic peptide generation";
    document.querySelector("#panels").append(this.root);
    this.status = this.root.querySelector(".status");
    this.stage = new NGL.Stage(this.root.querySelector(".scene"), {
      cameraType:"orthographic", backgroundColor:"white", quality:"high", sampleLevel:1,
    });
    this.stage.setParameters({clipNear:0, clipFar:100, fogNear:100, fogFar:100,
      lightIntensity:1, ambientIntensity:.55, hoverTimeout:0});
    new ResizeObserver(() => this.stage.handleResize()).observe(this.root.querySelector(".scene"));
    $(this.root,"ligand-color").value = "#9EA3AA";
    this.scheme = NGL.ColormakerRegistry.addScheme(function () {
      this.atomColor = atom => ({H:0xF8F8F6,N:0x8FA9D7,O:0xE59A9A,P:0xE6B485,S:0xE3D382,F:0xADCBA6,CL:0xA3C79E,BR:0xCAA28A,I:0xB8A0D1}[atom.element.toUpperCase()] ?? numberColor($(panel.root,"ligand-color").value));
    }, `pocket-${task}-elements`);
    const panel = this;
    for (const target of targets) $(this.root,"target").add(option(target,target));
    if (targets.includes("target_33")) $(this.root,"target").value = "target_33";
    for (const key of ["target","radius"]) $(this.root,key).addEventListener("change", () => this.run(() => this.load()));
    $(this.root,"molecule").addEventListener("change", () => this.run(() => this.loadMolecule()));
    for (const key of ["surface","sticks","cartoon","ligand","pocket-color","pocket-stick-color","ligand-color","opacity","elements"])
      $(this.root,key).addEventListener("input", () => this.apply());
    $(this.root,"style").addEventListener("change", () => this.run(async () => {this.addMoleculeRepresentation();await ready(this.stage);}));
    for (const button of this.root.querySelectorAll("[data-action]")) button.addEventListener("click", () => this.run(async () => {
      switch (button.dataset.action) {
        case "reset": this.reset(); break;
        case "remember": localStorage.setItem(this.cameraKey(), JSON.stringify(this.camera())); this.message("현재 시점을 기억했습니다."); break;
        case "save": await this.save(); break;
        case "pocket-only": $(this.root,"ligand").checked=false;this.apply();break;
        case "complex": $(this.root,"ligand").checked=true;this.apply();break;
      }
    }));
  }
  query() {return new URLSearchParams({task:this.task,target:$(this.root,"target").value,radius:$(this.root,"radius").value});}
  cameraKey() {return `downstream-pocket-camera-${this.task}-${$(this.root,"target").value}`;}
  camera() {
    const c=this.stage.viewerControls;
    return {rotation:c.rotation.toArray(),position:c.position.toArray(),distance:c.getCameraDistance()};
  }
  restore(value) {
    const c=this.stage.viewerControls;c.rotation.set(...value.rotation);c.position.set(...value.position);c.distance(value.distance);this.stage.viewer.requestRender();
  }
  reset() {
    const c=this.stage.viewerControls, d=this.data.direction;
    c.position.set(0,0,0);
    const up=Math.abs(d[2])>.9?new NGL.Vector3(0,1,0):new NGL.Vector3(0,0,1);
    c.align(new NGL.Matrix4().lookAt(new NGL.Vector3(...d),new NGL.Vector3(0,0,0),up));
    const e=this.data.extent;
    c.distance(this.stage.getZoomForBox(new NGL.Box3(new NGL.Vector3(-e,-e,-e),new NGL.Vector3(e,e,e))));
    this.stage.viewer.requestRender();
  }
  message(text,error=false) {this.status.textContent=text;this.status.classList.toggle("error",error);}
  async run(action) {
    if(this.busy)return;
    this.busy=true;this.root.querySelectorAll("input,select,button").forEach(el=>el.disabled=true);
    try {await action();} catch(e) {this.message(e.message,true);console.error(e);}
    finally {this.busy=false;this.root.querySelectorAll("input,select,button").forEach(el=>el.disabled=false);}
  }
  async loadText(text,ext) {return this.stage.loadFile(new Blob([text],{type:"text/plain"}),{ext,firstModelOnly:true});}
  async load() {
    this.message("구조를 불러오고 있습니다…");
    this.data=await api(`/api/scene?${this.query()}`);
    this.stage.removeAllComponents();
    this.molecule=null;
    this.protein=await this.loadText(this.data.pocket_pdb,"pdb");
    this.surface=this.protein.addRepresentation("surface",{sele:"not hydrogen",surfaceType:"ms",probeRadius:1.4,
      smooth:2,scaleFactor:2,colorScheme:"uniform",colorValue:numberColor($(this.root,"pocket-color").value),
      opacity:Number($(this.root,"opacity").value),opaqueBack:false,side:"double",depthWrite:false,useWorker:true});
    this.sticks=this.protein.addRepresentation("licorice",{sele:"not hydrogen",radiusType:"size",radiusSize:.12,colorScheme:"uniform",multipleBond:"off",quality:"high",visible:true});
    this.cartoon=this.protein.addRepresentation("cartoon",{sele:"protein",colorScheme:"uniform",visible:true});
    const select=$(this.root,"molecule");select.replaceChildren(option("reference",`Crystal reference · ${this.data.reference_atoms} atoms`));
    for(const item of this.data.samples) select.add(option(String(item.index),`Generated #${item.index+1} · ${item.atoms} atoms`));
    select.value=this.data.samples.length?String(this.data.samples[0].index):"reference";
    await this.loadMolecule();await ready(this.stage);this.reset();
    try {const saved=JSON.parse(localStorage.getItem(this.cameraKey()));if(saved)this.restore(saved);}catch{}
    this.apply();
  }
  async loadMolecule() {
    this.message("분자를 불러오고 있습니다…");
    const selected=$(this.root,"molecule").value;
    const sdf=selected==="reference"?this.data.reference_sdf:(await api(`/api/molecule?${this.query()}&index=${selected}`)).sdf;
    if(this.molecule)this.stage.removeComponent(this.molecule);
    this.molecule=await this.loadText(sdf,"sdf");
    this.addMoleculeRepresentation();await ready(this.stage);this.apply();
    this.message(`${this.data.pocket_atoms} pocket atoms · ${this.data.samples.length} generated samples · ${selected==="reference"?"Crystal reference":`Generated #${Number(selected)+1}`}`);
  }
  addMoleculeRepresentation() {
    this.molecule.removeAllRepresentations();
    this.moleculeRep=this.molecule.addRepresentation($(this.root,"style").value,{sele:"not hydrogen",radiusType:"size",radiusSize:.20,
      aspectRatio:1.6,multipleBond:"off",quality:"high",colorScheme:"uniform",colorValue:numberColor($(this.root,"ligand-color").value)});
    this.apply();
  }
  apply() {
    if(!this.surface)return;
    const color=numberColor($(this.root,"pocket-color").value);
    this.surface.setParameters({colorValue:color,opacity:Number($(this.root,"opacity").value)});
    const stickColor=numberColor($(this.root,"pocket-stick-color").value);
    this.sticks.setParameters({colorValue:stickColor});this.cartoon.setParameters({colorValue:stickColor});
    for(const key of ["surface","sticks","cartoon"]) this[key].setVisibility($(this.root,key).checked);
    if(this.moleculeRep){this.moleculeRep.setVisibility($(this.root,"ligand").checked);
      this.moleculeRep.setParameters({colorScheme:$(this.root,"elements").checked?this.scheme:"uniform",colorValue:numberColor($(this.root,"ligand-color").value)});}
  }
  async save() {
    await ready(this.stage);this.message("PNG와 PDF를 저장하고 있습니다…");
    const size=Number($(this.root,"size").value);
    const blob=await this.stage.makeImage({factor:Math.ceil(size/this.stage.viewer.width),antialias:true,trim:false,transparent:true});
    const bitmap=await createImageBitmap(blob), canvas=document.createElement("canvas");canvas.width=size;canvas.height=size;
    canvas.getContext("2d").drawImage(bitmap,0,0,size,size);bitmap.close();
    const resized=await new Promise(resolve=>canvas.toBlob(resolve,"image/png"));
    if(!resized)throw new Error("PNG export failed");
    const settings={};for(const el of this.root.querySelectorAll("[data-control]"))settings[el.dataset.control]=el.type==="checkbox"?el.checked:el.value;
    const role=$(this.root,"ligand").checked?"complex":"pocket";
    const result=await api("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      name:`${this.task}-${this.data.target}-${role}`,image:await blobURL(resized),
      scene:{task:this.task,target:this.data.target,camera:this.camera(),settings,source:this.data.source,center:this.data.center},
    })});
    this.message("");
    for(const file of result.files||[{format:"png",url:result.url}]){
      const link=document.createElement("a");link.href=file.url;link.textContent=`${file.format.toUpperCase()} 다운로드`;link.download=file.url.split("/").pop();
      this.status.append(link,document.createTextNode(" · "));
    }
    this.status.append(document.createTextNode(result.path));
  }
}

window.pocketPanels=[];
if (!window.POCKET_NO_AUTO_START) api("/api/catalog").then(async catalog=>{
  for(const task of ["ligand","mcp"]){
    const panel=new PocketPanel(task,catalog[task]);window.pocketPanels.push(panel);
    await panel.run(()=>panel.load());
  }
}).catch(error=>{document.querySelector("header p").textContent=error.message;console.error(error);});
