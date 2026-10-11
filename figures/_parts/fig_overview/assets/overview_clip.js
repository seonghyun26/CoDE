"use strict";

// NGL's built-in radius clip is spherical. This viewer needs an axis-aligned
// 16 Å box in molecular coordinates, including sphere/cylinder surface pixels.
// Geometry representations use Mesh shaders; keep this adapter outside ngl.js.
const overviewBoxClip = {value:8};
const overviewBufferPrototype = Object.getPrototypeOf(NGL.MeshBuffer.prototype);
const overviewOriginalShader = overviewBufferPrototype.getShader;
overviewBufferPrototype.getShader = function(name, type) {
  const shader = overviewOriginalShader.call(this, name, type);
  if(this.parameters.background || !['Mesh.vert','Mesh.frag'].includes(name))return shader;
  // The same uniform object is shared by normal and picking materials, so a
  // grid toggle changes the crop without rebuilding or moving the molecule.
  this.uniforms.overviewClipExtent = overviewBoxClip;
  this.pickingUniforms.overviewClipExtent = overviewBoxClip;
  const declarations = 'varying vec3 vOverviewObjectPosition;\nuniform float overviewClipExtent;\n';
  const code = name==='Mesh.vert' ? 'vOverviewObjectPosition = position;' :
    'if (overviewClipExtent > 0.0 && any(greaterThan(abs(vOverviewObjectPosition), vec3(overviewClipExtent)))) discard;';
  return declarations + shader.replace(/void\s+main\s*\(\s*\)\s*\{/, match=>match+'\n'+code+'\n');
};
