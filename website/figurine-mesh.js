"use strict";

window.createFigurineMeshView = function createFigurineMeshView(mesh, requestDraw) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl", { antialias: true, alpha: true, preserveDrawingBuffer: true });
  if (!gl) return null;

  function decode(encoded, ArrayType) {
    const binary = atob(encoded);
    if (binary.length % 4) throw new Error("Invalid mesh buffer");
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const view = new DataView(bytes.buffer);
    const read = ArrayType === Float32Array ? view.getFloat32 : view.getUint32;
    const result = new ArrayType(view.byteLength / 4);
    for (let i = 0; i < result.length; i++) result[i] = read.call(view, i * 4, true);
    return result;
  }

  function makeShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  try {
    const positions = decode(mesh.positions, Float32Array);
    const vertexCount = positions.length / 3;
    const uvs = decode(mesh.uvs, Float32Array);
    const indices = decode(mesh.indices, Uint32Array);
    if (
      !vertexCount || uvs.length !== 2 * vertexCount ||
      !indices.length || indices.length % 3 || indices.some(index => index >= vertexCount) ||
      !positions.every(Number.isFinite) || !uvs.every(Number.isFinite)
    ) return null;

    const low = [Infinity, Infinity, Infinity];
    const high = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i++) {
      const axis = i % 3;
      low[axis] = Math.min(low[axis], positions[i]);
      high[axis] = Math.max(high[axis], positions[i]);
    }
    const center = high.map((value, axis) => (value + low[axis]) * 0.5);
    const radius = 0.5 * Math.hypot(...high.map((value, axis) => value - low[axis]));
    if (!(radius > 0)) return null;

    const vertexShader = makeShader(gl.VERTEX_SHADER, `
      attribute vec3 aPosition;
      attribute vec2 aUV;
      uniform vec3 uCenter;
      uniform vec4 uOrbit;
      uniform vec2 uScale;
      uniform float uDistance;
      varying vec2 vUV;

      void main() {
        vec3 p = aPosition - uCenter;
        float x = uOrbit.x * p.x - uOrbit.y * p.z;
        float z = uOrbit.y * p.x + uOrbit.x * p.z;
        float y = uOrbit.z * p.y - uOrbit.w * z;
        float depth = uOrbit.w * p.y + uOrbit.z * z;
        gl_Position = vec4(x * uScale.x, -y * uScale.y, depth, uDistance + depth);
        vUV = aUV;
      }
    `);
    const fragmentShader = makeShader(gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform sampler2D uTexture;
      varying vec2 vUV;

      void main() {
        gl_FragColor = texture2D(uTexture, vUV);
      }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    gl.useProgram(program);

    function bindAttribute(name, components, data) {
      const location = gl.getAttribLocation(program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, components, gl.FLOAT, false, 0, 0);
    }

    const indexType = vertexCount <= 65536 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;
    if (indexType === gl.UNSIGNED_INT && !gl.getExtension("OES_element_index_uint")) return null;
    bindAttribute("aPosition", 3, positions);
    bindAttribute("aUV", 2, uvs);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexType === gl.UNSIGNED_SHORT ? Uint16Array.from(indices) : indices, gl.STATIC_DRAW);

    const orbitUniform = gl.getUniformLocation(program, "uOrbit");
    const scaleUniform = gl.getUniformLocation(program, "uScale");
    const distance = 3.5 * radius;
    gl.uniform3fv(gl.getUniformLocation(program, "uCenter"), center);
    gl.uniform1f(gl.getUniformLocation(program, "uDistance"), distance);
    gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);

    let ready = false;
    const image = new Image();
    image.onload = () => {
      if (gl.isContextLost()) return;
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      ready = gl.getError() === gl.NO_ERROR;
      if (ready) requestDraw();
    };
    image.src = `data:image/jpeg;base64,${mesh.texture}`;
    canvas.addEventListener("webglcontextlost", requestDraw);

    return function drawMesh(width, height, yaw, pitch, zoom) {
      if (!ready || gl.isContextLost()) return null;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
      const scale = 0.49 * distance * Math.min(width, height) * zoom / radius;
      gl.uniform4f(orbitUniform, Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch));
      gl.uniform2f(scaleUniform, 2 * scale / width, 2 * scale / height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.drawElements(gl.TRIANGLES, indices.length, indexType, 0);
      return canvas;
    };
  } catch {
    return null;
  }
};
