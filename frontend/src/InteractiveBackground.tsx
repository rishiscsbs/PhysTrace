import { useEffect, useRef } from "react";

/**
 * Full-screen WebGL fluid behind every page: moving the mouse drops silky blue/teal ink into an alabaster field.
 * A compact Navier–Stokes solver after Pavel Dobryakov's WebGL-Fluid-Simulation (MIT), the same core as the
 * tkabalin / webgl-fluid demos, tuned for a light theme: dye is stored as premultiplied ink (rgb·s, s) and
 * composited over BACK_COLOR, instead of the original additive light on black.
 */
const CONFIG = {
  BACK_COLOR: "#F8F9FA",
  COLOR_PALETTE: ["#0055FF", "#14B8A6"],
  DENSITY_DISSIPATION: 0.98, // dye kept per 60 Hz frame: fades in ~2 s so the page never muddies
  VELOCITY_DISSIPATION: 0.985,
  PRESSURE: 0.8,
  PRESSURE_ITERATIONS: 20,
  CURL: 22,
  SPLAT_RADIUS: 0.16,
  SPLAT_FORCE: 5000,
  INK: 0.55, // dye added per splat
  OPACITY: 0.85, // max ink coverage, keeps it silky rather than opaque
  SIM_RESOLUTION: 128,
  DYE_RESOLUTION: 512,
  IDLE_MS: 6000, // after this long without input the dye has faded out and the solver sleeps
};

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255) as [number, number, number];

const VERT = `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv, vL, vR, vT, vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0); vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y); vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;
const HEAD = "precision highp float;\nprecision highp sampler2D;\nvarying vec2 vUv, vL, vR, vT, vB;\n";
const FRAG = {
  splat: `uniform sampler2D uTarget; uniform float aspectRatio, radius; uniform vec4 color; uniform vec2 point;
    void main () { vec2 p = vUv - point; p.x *= aspectRatio; gl_FragColor = texture2D(uTarget, vUv) + exp(-dot(p, p) / radius) * color; }`,
  advection: `uniform sampler2D uVelocity, uSource; uniform vec2 texelSize; uniform float dt, dissipation;
    void main () { vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize; gl_FragColor = dissipation * texture2D(uSource, coord); }`,
  divergence: `uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).x, R = texture2D(uVelocity, vR).x, T = texture2D(uVelocity, vT).y, B = texture2D(uVelocity, vB).y;
      vec2 C = texture2D(uVelocity, vUv).xy;
      if (vL.x < 0.0) L = -C.x; if (vR.x > 1.0) R = -C.x; if (vT.y > 1.0) T = -C.y; if (vB.y < 0.0) B = -C.y;
      gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
    }`,
  curl: `uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).y, R = texture2D(uVelocity, vR).y, T = texture2D(uVelocity, vT).x, B = texture2D(uVelocity, vB).x;
      gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
    }`,
  vorticity: `uniform sampler2D uVelocity, uCurl; uniform float curl, dt;
    void main () {
      float L = texture2D(uCurl, vL).x, R = texture2D(uCurl, vR).x, T = texture2D(uCurl, vT).x, B = texture2D(uCurl, vB).x, C = texture2D(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001; force *= curl * C; force.y *= -1.0;
      gl_FragColor = vec4(texture2D(uVelocity, vUv).xy + force * dt, 0.0, 1.0);
    }`,
  pressure: `uniform sampler2D uPressure, uDivergence;
    void main () {
      float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x, T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
      gl_FragColor = vec4((L + R + B + T - texture2D(uDivergence, vUv).x) * 0.25, 0.0, 0.0, 1.0);
    }`,
  gradient: `uniform sampler2D uPressure, uVelocity;
    void main () {
      float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x, T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
      gl_FragColor = vec4(texture2D(uVelocity, vUv).xy - vec2(R - L, T - B), 0.0, 1.0);
    }`,
  clear: `uniform sampler2D uTexture; uniform float value; void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`,
  // Premultiplied ink over the page colour.
  display: `uniform sampler2D uTexture; uniform vec3 back; uniform float opacity;
    void main () {
      vec4 d = texture2D(uTexture, vUv);
      float a = clamp(d.a, 0.0, 1.0);
      vec3 ink = clamp(d.rgb / max(d.a, 0.0001), 0.0, 1.0);
      gl_FragColor = vec4(mix(back, ink, a * opacity), 1.0);
    }`,
};

type Target = { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number; attach: (unit: number) => number };
type Double = { read: Target; write: Target; w: number; h: number; swap: () => void };

/** Starts the simulation on a canvas; returns a teardown, or null when WebGL2 float render targets are unavailable. */
function startFluid(canvas: HTMLCanvasElement): (() => void) | null {
  const gl = canvas.getContext("webgl2", { alpha: false, depth: false, stencil: false, antialias: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) return null;

  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const vs = compile(gl.VERTEX_SHADER, VERT);
  const prog = (frag: string) => {
    const p = gl.createProgram()!;
    gl.attachShader(p, vs); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, HEAD + frag));
    gl.bindAttribLocation(p, 0, "aPosition"); gl.linkProgram(p);
    const u: Record<string, WebGLUniformLocation | null> = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const name = gl.getActiveUniform(p, i)!.name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { use: () => { gl.useProgram(p); return u; } };
  };
  const P = Object.fromEntries(Object.entries(FRAG).map(([k, f]) => [k, prog(f)])) as Record<keyof typeof FRAG, ReturnType<typeof prog>>;

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);
  const blit = (t: Target | null) => {
    if (t) { gl.viewport(0, 0, t.w, t.h); gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); }
    else { gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight); gl.bindFramebuffer(gl.FRAMEBUFFER, null); }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  };

  const textures: WebGLTexture[] = [], buffers: WebGLFramebuffer[] = [];
  const target = (w: number, h: number): Target => {
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    textures.push(tex); buffers.push(fb);
    return { tex, fb, w, h, attach: (unit) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); return unit; } };
  };
  const double = (w: number, h: number): Double => {
    let a = target(w, h), b = target(w, h);
    return { get read() { return a; }, get write() { return b; }, w, h, swap() { [a, b] = [b, a]; } };
  };
  // Long side scales with the aspect ratio so cells stay square.
  const res = (r: number) => {
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight, big = Math.round(r * Math.max(aspect, 1 / aspect));
    return aspect > 1 ? [big, r] : [r, big];
  };

  let velocity: Double, dye: Double, pressure: Double, divergence: Target, curl: Target;
  const resize = () => {
    canvas.width = window.innerWidth; canvas.height = window.innerHeight;
    textures.forEach((t) => gl.deleteTexture(t)); buffers.forEach((f) => gl.deleteFramebuffer(f));
    textures.length = buffers.length = 0;
    const [sw, sh] = res(CONFIG.SIM_RESOLUTION), [dw, dh] = res(CONFIG.DYE_RESOLUTION);
    velocity = double(sw, sh); dye = double(dw, dh); pressure = double(sw, sh);
    divergence = target(sw, sh); curl = target(sw, sh);
    dirty = true;
  };

  const aspect = () => canvas.width / canvas.height;
  const splat = (x: number, y: number, dx: number, dy: number, color: [number, number, number]) => {
    let u = P.splat.use();
    gl.uniform1i(u.uTarget, velocity.read.attach(0));
    gl.uniform1f(u.aspectRatio, aspect());
    gl.uniform2f(u.point, x, y);
    gl.uniform4f(u.color, dx, dy, 0, 0);
    gl.uniform1f(u.radius, (CONFIG.SPLAT_RADIUS / 100) * Math.max(1, aspect()));
    blit(velocity.write); velocity.swap();
    u = P.splat.use();
    gl.uniform1i(u.uTarget, dye.read.attach(0));
    const s = CONFIG.INK;
    gl.uniform4f(u.color, color[0] * s, color[1] * s, color[2] * s, s);
    blit(dye.write); dye.swap();
  };

  const step = (dt: number) => {
    const tx = 1 / velocity.w, ty = 1 / velocity.h;
    let u = P.curl.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1i(u.uVelocity, velocity.read.attach(0)); blit(curl);
    u = P.vorticity.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1i(u.uVelocity, velocity.read.attach(0)); gl.uniform1i(u.uCurl, curl.attach(1));
    gl.uniform1f(u.curl, CONFIG.CURL); gl.uniform1f(u.dt, dt); blit(velocity.write); velocity.swap();
    u = P.divergence.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1i(u.uVelocity, velocity.read.attach(0)); blit(divergence);
    u = P.clear.use();
    gl.uniform1i(u.uTexture, pressure.read.attach(0)); gl.uniform1f(u.value, CONFIG.PRESSURE); blit(pressure.write); pressure.swap();
    u = P.pressure.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1i(u.uDivergence, divergence.attach(0));
    for (let i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) { gl.uniform1i(u.uPressure, pressure.read.attach(1)); blit(pressure.write); pressure.swap(); }
    u = P.gradient.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1i(u.uPressure, pressure.read.attach(0)); gl.uniform1i(u.uVelocity, velocity.read.attach(1));
    blit(velocity.write); velocity.swap();
    // Dissipation is specified per 60 Hz frame; scale it to the real frame time.
    u = P.advection.use();
    gl.uniform2f(u.texelSize, tx, ty); gl.uniform1f(u.dt, dt);
    gl.uniform1i(u.uVelocity, velocity.read.attach(0)); gl.uniform1i(u.uSource, velocity.read.attach(0));
    gl.uniform1f(u.dissipation, Math.pow(CONFIG.VELOCITY_DISSIPATION, dt * 60)); blit(velocity.write); velocity.swap();
    gl.uniform1i(u.uVelocity, velocity.read.attach(0)); gl.uniform1i(u.uSource, dye.read.attach(1));
    gl.uniform1f(u.dissipation, Math.pow(CONFIG.DENSITY_DISSIPATION, dt * 60)); blit(dye.write); dye.swap();
  };

  const back = rgb(CONFIG.BACK_COLOR), palette = CONFIG.COLOR_PALETTE.map(rgb);
  const render = () => {
    const u = P.display.use();
    gl.uniform1i(u.uTexture, dye.read.attach(0)); gl.uniform3f(u.back, ...back); gl.uniform1f(u.opacity, CONFIG.OPACITY);
    blit(null);
  };

  // The canvas ignores pointer events so the UI stays clickable; the mouse is read from the window instead.
  const ptr = { x: 0, y: 0, dx: 0, dy: 0, moved: false, seen: false };
  let lastInput = -Infinity, dirty = true;
  const move = (e: PointerEvent) => {
    const x = e.clientX / canvas.width, y = 1 - e.clientY / canvas.height;
    if (ptr.seen) {
      const a = aspect();
      ptr.dx += (x - ptr.x) * (a < 1 ? a : 1);
      ptr.dy += (y - ptr.y) / (a > 1 ? a : 1);
      ptr.moved = true;
    }
    ptr.x = x; ptr.y = y; ptr.seen = true;
    lastInput = performance.now();
  };
  const leave = () => { ptr.seen = false; };

  resize();
  window.addEventListener("pointermove", move, { passive: true });
  document.documentElement.addEventListener("pointerleave", leave);
  window.addEventListener("resize", resize);

  let raf = 0, prev = performance.now();
  const frame = (now: number) => {
    const dt = Math.min((now - prev) / 1000, 1 / 60);
    prev = now;
    if (now - lastInput < CONFIG.IDLE_MS) {
      if (ptr.moved) {
        // Alternate between the two inks every 0.9 s of motion.
        splat(ptr.x, ptr.y, ptr.dx * CONFIG.SPLAT_FORCE, ptr.dy * CONFIG.SPLAT_FORCE, palette[Math.floor(now / 900) % palette.length]);
        ptr.dx = ptr.dy = 0; ptr.moved = false;
      }
      step(dt); render(); dirty = true;
    } else if (dirty) {
      // Idle: the ink has faded, so clear the dye, paint the background once and stop simulating.
      for (const t of [dye.read, dye.write, velocity.read, velocity.write]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      }
      render(); dirty = false;
    }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("pointermove", move);
    document.documentElement.removeEventListener("pointerleave", leave);
    window.removeEventListener("resize", resize);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  };
}

/**
 * Fixed fluid backdrop behind every page, with the 40px Swiss grid drawn over it (an opaque canvas would hide the
 * body's grid). The canvas is created per mount, so StrictMode's double mount never reuses a lost context.
 * Without WebGL2 float targets nothing is drawn and the body's alabaster and grid show as before.
 */
export default function InteractiveBackground() {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, { width: "100%", height: "100%", display: "block" });
    host.current!.appendChild(canvas);
    let stop: (() => void) | null = null;
    try { stop = startFluid(canvas); } catch (e) { console.warn("fluid background disabled:", e); }
    if (!stop) canvas.remove();
    return () => { stop?.(); canvas.remove(); };
  }, []);
  return (
    <>
      <div ref={host} aria-hidden style={{ position: "fixed", inset: 0, zIndex: -10, pointerEvents: "none" }} />
      <div aria-hidden className="pt-gridlines" style={{ position: "fixed", inset: 0, zIndex: -9, pointerEvents: "none" }} />
    </>
  );
}
