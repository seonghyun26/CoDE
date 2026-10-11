"use strict";

let records=[], byID=new Map(), shortlist=new Set(), colors=[];
try {shortlist=new Set(JSON.parse(localStorage.getItem("affinity-shortlist")||"[]"));} catch {}
const panels=[];
const globalStatus=document.querySelector("#global-status");

class AffinityPanel extends PocketPanel {
  constructor(slot, targets) {
    super(`affinity-${slot}`,targets);
    this.slot=slot;
    this.root.querySelector(".eyebrow").textContent=slot==="high"?"CANDIDATE A · HIGHER AFFINITY":"CANDIDATE B · LOWER AFFINITY";
    this.root.querySelector("h2").textContent="Binding affinity";
    this.root.querySelector('[data-control="molecule"]').closest("label").hidden=true;
    const heading=this.root.querySelector(".heading");
    this.metrics=document.createElement("div");this.metrics.className="metric";
    this.detail=document.createElement("div");this.detail.className="record-detail";
    heading.append(this.metrics,this.detail);
    const palette=document.createElement("div");palette.className="ligand-palette";
    const label=document.createElement("div");label.className="palette-label";label.textContent="Ligand / scatter color";
    this.swatches=document.createElement("div");this.swatches.className="swatches";
    for(const entry of colors){
      const button=document.createElement("button");button.className="swatch";button.style.backgroundColor=entry.color;
      button.dataset.color=entry.color.toUpperCase();button.title=`${entry.label} · ${entry.color}`;
      button.setAttribute("aria-label",`Select ${entry.label} color`);
      button.onclick=()=>this.setPaletteColor(entry.color);this.swatches.append(button);
    }
    palette.append(label,this.swatches);this.root.querySelector(".controls").prepend(palette);
    $(this.root,"ligand-color").addEventListener("input",()=>this.setPaletteColor($(this.root,"ligand-color").value));
    let initial=colors.find(c=>c.label===(slot==="high"?"VoxBind":"AR"))?.color||colors[0].color;
    try{
      const saved=localStorage.getItem(`affinity-color-${slot}`);
      const method=localStorage.getItem(`affinity-palette-method-${slot}`);
      const legacy={"#FBB178":"VoxBind","#0FA8C0":"AR","#ED76A3":"Pocket2Mol","#EB4140":"DiffSBDD","#36BA46":"DecompDiff","#9F872B":"FuncBind","#B58BDF":"TargetDiff","#3C5FDF":"CoDE","#80B993":"CoE","#C36C6C":"DeepDTA","#939951":"ProFSA","#8664B0":"Nesso-1","#9AA0A6":"Reference ligand"};
      const entry=colors.find(c=>c.label===(method||legacy[saved?.toUpperCase()]));
      if(entry)initial=entry.color;else if(/^#[0-9a-f]{6}$/i.test(saved))initial=saved;
    }catch{}
    this.setPaletteColor(initial);
    $(this.root,"target").addEventListener("change",markScatterDirty);
    for(const opt of $(this.root,"target").options){const row=byID.get(opt.value);opt.textContent=`${row.pid.toUpperCase()} · GT ${row.gt.toFixed(2)} · Pred ${row.predicted.toFixed(2)}`;}
  }
  setPaletteColor(color) {
    this.color=color.toUpperCase();$(this.root,"ligand-color").value=color;
    this.root.style.setProperty("--case-color",this.color);
    for(const button of this.swatches.querySelectorAll("button"))button.setAttribute("aria-pressed",String(button.dataset.color===this.color));
    try{
      localStorage.setItem(`affinity-color-${this.slot}`,this.color);
      const method=colors.find(c=>c.color.toUpperCase()===this.color)?.label;
      if(method)localStorage.setItem(`affinity-palette-method-${this.slot}`,method);
      else localStorage.removeItem(`affinity-palette-method-${this.slot}`);
    }catch{}
    this.apply();markScatterDirty();
  }
  async load() {
    await super.load();
    const row=this.data.affinity;
    this.root.querySelector("h2").textContent=row.pid.toUpperCase();
    this.metrics.replaceChildren();
    for(const [label,value] of [["Ground Truth pK",row.gt],["Predicted pK",row.predicted],["Absolute error",row.error]]){
      const item=document.createElement("div"),valueNode=document.createElement("strong");
      valueNode.textContent=value.toFixed(2);item.append(valueNode,document.createTextNode(label));this.metrics.append(item);
    }
    const details=[row.protein_class,row.measurement,`${row.atoms} ligand atoms`,row.resolution?`${row.resolution} Å resolution`:"",row.covalent==="True"?"Covalent":""].filter(Boolean);
    this.detail.textContent=details.join(" · ");
    this.message(`${this.data.pocket_atoms} pocket atoms · crystal ligand · ${$(this.root,"radius").value} Å residue crop`);
    try{localStorage.setItem(`affinity-case-${this.slot}`,row.pid);}catch{}
    updatePair();renderTable();markScatterDirty();
  }
  async choose(pid) {
    if(this.busy)return;
    $(this.root,"target").value=pid;
    markScatterDirty();
    await this.run(()=>this.load());
  }
}

function liveSelection() {
  return panels.map(p=>({pid:$(p.root,"target").value,color:p.color})).filter(p=>byID.has(p.pid));
}

function drawLiveScatter() {
  if(!records.length)return;
  const svg=document.querySelector("#scatter-preview"),ns="http://www.w3.org/2000/svg";
  const node=(tag,attributes,parent=svg)=>{const el=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attributes))el.setAttribute(key,value);parent.append(el);return el;};
  const selected=liveSelection(),ids=new Set(selected.map(s=>s.pid));
  const lo=Math.floor(Math.min(...records.flatMap(r=>[r.predicted,r.gt])));
  const hi=Math.ceil(Math.max(...records.flatMap(r=>[r.predicted,r.gt])));
  // Match the exported 3.2-inch figure in point units (72 points/inch).
  const left=41.472,bottom=188.928,side=179.712;
  const x=v=>left+(v-lo)/(hi-lo)*side,y=v=>bottom-(v-lo)/(hi-lo)*side;
  svg.replaceChildren();
  node("title",{}).textContent="Binding affinity scatter, updated live from the selected cases";
  const defs=node("defs",{}),clip=node("clipPath",{id:"scatter-clip"},defs);
  node("rect",{x:left,y:bottom-side,width:side,height:side},clip);
  const dots=node("g",{"clip-path":"url(#scatter-clip)"});
  node("line",{x1:left,y1:bottom,x2:left+side,y2:bottom-side,stroke:"black","stroke-width":2.6,"stroke-opacity":.25},dots);
  for(const row of records){
    if(ids.has(row.pid))continue;
    const dot=node("circle",{cx:x(row.predicted),cy:y(row.gt),r:Math.sqrt(24*1.4)/2,fill:"#3C5FDF","fill-opacity":.45,"data-pid":row.pid},dots);
    node("title",{},dot).textContent=`${row.pid.toUpperCase()} · Pred ${row.predicted.toFixed(2)} · Experimental ${row.gt.toFixed(2)}`;
  }
  for(const item of selected){
    const row=byID.get(item.pid);
    const dot=node("circle",{cx:x(row.predicted),cy:y(row.gt),r:Math.sqrt(40*1.4*2)/2,fill:item.color,stroke:"black","stroke-width":1.2,"data-selected":item.pid},dots);
    node("title",{},dot).textContent=`${row.pid.toUpperCase()} · Pred ${row.predicted.toFixed(2)} · Experimental ${row.gt.toFixed(2)}`;
  }
  node("path",{d:`M ${left} ${bottom-side} V ${bottom} H ${left+side}`,fill:"none",stroke:"black","stroke-width":2,"stroke-linecap":"square"});
  for(let i=0;i<4;i++){
    const offset=side*i/3;
    node("path",{d:`M ${left+offset} ${bottom} v 3 M ${left} ${bottom-offset} h -3`,stroke:"black","stroke-width":1.6});
  }
  const style={"font-family":"DejaVu Sans, sans-serif","font-size":17.6,"text-anchor":"middle",fill:"black"};
  node("text",{...style,x:left+side/2,y:bottom+20}).textContent="Predicted Affinity";
  node("text",{...style,transform:`translate(${left-12} ${bottom-side/2}) rotate(-90)`}).textContent="Experimental Affinity";
  document.querySelector("#scatter-caption").textContent=selected.map(s=>`${s.pid.toUpperCase()} · ${s.color}`).join(" / ");
}

function markScatterDirty() {
  drawLiveScatter();
  document.querySelector("#scatter-status").textContent="Live preview · Update scatter plot saves PNG / PDF / SVG.";
  document.querySelector("#scatter-links").replaceChildren();
}

document.querySelector("#update-scatter").onclick=async()=>{
  const status=document.querySelector("#scatter-status"),button=document.querySelector("#update-scatter");
  if(panels.length!==2||panels.some(p=>p.busy||!p.data)){status.textContent="Wait for both structures to finish loading.";return;}
  const selection=panels.map(p=>({pid:p.data.target,color:p.color}));
  button.disabled=true;status.textContent="Saving scatter plot…";
  try{
    const result=await api("/api/scatter",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({selected:selection})});
    const current=panels.map(p=>({pid:p.data.target,color:p.color}));
    if(JSON.stringify(selection)!==JSON.stringify(current)){markScatterDirty();status.textContent="Saved the previous selection. Update again to use your latest choices.";return;}
    drawLiveScatter();
    const links=document.querySelector("#scatter-links");links.replaceChildren();
    for(const file of result.files){const a=document.createElement("a");a.href=file.url;a.textContent=file.format.toUpperCase();a.download=file.path.split("/").pop();links.append(a);}
    status.textContent=`Saved binding_affinity_scatter_selected · ${selection.map(s=>s.pid.toUpperCase()).join(" / ")}`;
  }catch(e){status.textContent=e.message;}finally{button.disabled=false;}
};

function updatePair() {
  if(panels.length!==2||panels.some(p=>!p.data))return;
  const a=panels[0].data.affinity,b=panels[1].data.affinity,diff=a.gt-b.gt;
  document.querySelector("#delta").textContent=`${a.pid.toUpperCase()} vs ${b.pid.toUpperCase()} · Ground Truth ΔpK = ${diff.toFixed(2)}${diff<0?" · Candidate A currently has lower affinity than B":""}`;
}

function filtered() {
  const query=document.querySelector("#search").value.trim().toLowerCase();
  const min=document.querySelector("#min-gt").value,max=document.querySelector("#max-gt").value;
  const [key,order]=document.querySelector("#sort").value.split("-");
  return records.filter(r=>(!query||`${r.pid} ${r.protein_class} ${r.measurement}`.toLowerCase().includes(query))&&
    (min===""||r.gt>=Number(min))&&(max===""||r.gt<=Number(max))&&
    (!document.querySelector("#favorites-only").checked||shortlist.has(r.pid)))
    .sort((a,b)=>(order==="asc"?1:-1)*(a[key]-b[key])||a.pid.localeCompare(b.pid));
}

function renderTable() {
  const filteredRows=filtered(),body=document.querySelector("#candidates");
  body.replaceChildren();
  document.querySelector("#count").textContent=`${filteredRows.length} / ${records.length} cases · ${shortlist.size} shortlisted`;
  for(const row of filteredRows){
    const tr=document.createElement("tr");tr.dataset.pid=row.pid;
    if(panels.some(p=>p.data?.target===row.pid))tr.className="selected-row";
    const starCell=document.createElement("td"),star=document.createElement("button");
    star.className="favorite";star.textContent=shortlist.has(row.pid)?"★":"☆";
    star.setAttribute("aria-pressed",String(shortlist.has(row.pid)));star.setAttribute("aria-label",`Shortlist ${row.pid.toUpperCase()}`);
    star.onclick=()=>{shortlist.has(row.pid)?shortlist.delete(row.pid):shortlist.add(row.pid);try{localStorage.setItem("affinity-shortlist",JSON.stringify([...shortlist]));}catch{}renderTable();};
    starCell.append(star);tr.append(starCell);
    for(const value of [row.pid.toUpperCase(),row.gt.toFixed(2),row.predicted.toFixed(2)]){const td=document.createElement("td");td.textContent=value;tr.append(td);}
    const actions=document.createElement("td");
    for(const [i,label] of [[0,"A"],[1,"B"]]){
      const button=document.createElement("button");button.textContent=label;button.title=`View ${row.pid.toUpperCase()} in candidate ${label}`;
      button.onclick=()=>panels[i].choose(row.pid);actions.append(button);
    }
    tr.append(actions);tr.title=`${row.protein_class} · ${row.atoms} atoms · absolute error ${row.error.toFixed(2)}`;body.append(tr);
  }
  if(!filteredRows.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=5;td.textContent="No matching cases. Try resetting the filters.";tr.append(td);body.append(tr);}
}

for(const id of ["search","sort","min-gt","max-gt","favorites-only"])
  document.querySelector(`#${id}`).addEventListener("input",renderTable);
document.querySelector("#reset-filters").onclick=()=>{
  for(const id of ["search","min-gt","max-gt"])document.querySelector(`#${id}`).value="";
  document.querySelector("#favorites-only").checked=false;document.querySelector("#sort").value="gt-desc";renderTable();
};
document.querySelector("#extremes").onclick=async()=>{
  const ordered=[...records].sort((a,b)=>b.gt-a.gt||a.pid.localeCompare(b.pid));
  if(!ordered.length)return;
  await panels[0].choose(ordered[0].pid);await panels[1].choose(ordered[ordered.length-1].pid);
};
document.querySelector("#save-selection").onclick=async()=>{
  if(panels.length!==2||panels.some(p=>p.busy||!p.data)){globalStatus.textContent="Wait for both structures to finish loading.";return;}
  const button=document.querySelector("#save-selection");button.disabled=true;
  try{
    const result=await api("/api/selection",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      panels:panels.map(p=>({slot:p.slot,pid:p.data.target,color:p.color,camera:p.camera(),radius:Number($(p.root,"radius").value)})),shortlist:[...shortlist],
    })});
    globalStatus.replaceChildren();const link=document.createElement("a");link.href=result.url;link.download=result.url.split("/").pop();link.textContent="Download selected pair + shortlist";
    globalStatus.append(link,document.createTextNode(` · ${result.path}`));
  }catch(e){globalStatus.textContent=e.message;}finally{button.disabled=false;}
};

window.affinityPanels=panels;
api("/api/catalog").then(async data=>{
  records=data.records;colors=data.palette;byID=new Map(records.map(r=>[r.pid,r]));shortlist=new Set([...shortlist].filter(pid=>byID.has(pid)));
  document.querySelector("#source").textContent=`${data.model} · ${records.length} structures · same data as the overview scatter${data.unavailable.length?` · ${data.unavailable.length} unavailable`:""}`;
  if(!records.length)throw new Error("No structures available");
  const ordered=[...records].sort((a,b)=>b.gt-a.gt||a.pid.localeCompare(b.pid));
  for(const slot of ["high","low"]){const panel=new AffinityPanel(slot,ordered.map(r=>r.pid));panels.push(panel);window.pocketPanels.push(panel);}
  renderTable();
  for(const [index,panel] of panels.entries()){
    let pid=index===0?ordered[0].pid:ordered[ordered.length-1].pid;
    if(byID.has(data.defaults?.[panel.slot]))pid=data.defaults[panel.slot];
    try{const saved=localStorage.getItem(`affinity-case-${panel.slot}`);if(byID.has(saved))pid=saved;}catch{}
    await panel.choose(pid);
  }
}).catch(e=>{globalStatus.textContent=e.message;console.error(e);});
