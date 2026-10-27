import { memo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Box, ContactShadows, Cylinder, Environment, Float, Html, MeshTransmissionMaterial, PresentationControls, RoundedBox, Sphere, Torus } from "@react-three/drei";
import * as THREE from "three";
import { C, Live, RecEvent } from "./theme";
import { useLive, useRecordedEvents, useSamples } from "./live";
import { Optional } from "./scenes";

const mono = "'JetBrains Mono', monospace";
// drei maps 1 CSS px to distanceFactor / 400 world units: 640 × 400 px at 1.5 → a 2.4 × 1.5 screen.
const SCREEN_W = 640, SCREEN_H = 400, N = 120, LIMIT = 0.05, SPAN = 0.08;
// drei's CSS 3D treats one world unit as one CSS px, so with the camera ~5 units away the browser magnifies by ~125×
// and its sub-pixel layer snapping visibly shifts the screen off the bezel (~28 px). Modelling in metres but
// rendering at 100× world scale brings that magnification near 1×. Everything below is authored at unit scale.
const WORLD = 100;
// The canvas is transparent, so give the transmission pass the page colour to refract instead of black.
const GLASS_BG = new THREE.Color("#F8F9FA");

const Matte = () => <meshStandardMaterial color="#ffffff" roughness={0.2} />;
const Dark = () => <meshStandardMaterial color="#0F172A" metalness={0.8} roughness={0.3} />;

/** The live UI mapped onto the kiosk screen: twin residual trace, verdict and the newest recorder entries. */
function KioskScreen({ d, residuals, events }: { d: Live; residuals: number[]; events: RecEvent[] }) {
  const live = d.source === "LIVE";
  const r = d.level - d.twin;
  const color = d.alarm ? C.crit : Math.abs(r) > LIMIT ? C.warn : C.ok;
  const W = SCREEN_W - 48, H = 112;
  const X = (i: number) => (i / (N - 1)) * W;
  const Y = (v: number) => H / 2 - (Math.max(-SPAN, Math.min(SPAN, v)) / SPAN) * (H / 2 - 4);
  const trace = residuals.map((v, i) => `${i ? "L" : "M"}${X(i + N - residuals.length).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const log = live
    ? events.slice(-2).reverse().map((e) => ({ id: e.event_id, sev: e.severity, text: `${e.category} › ${e.message}` }))
    : [{ id: "off", sev: "INFO", text: "BACKEND OFFLINE · NO EVENTS RECORDED" }];
  const sevColor = (s: string) => ({ CRITICAL: C.crit, WARNING: C.warn, NOTICE: C.twin }[s] ?? C.mute);
  return (
    <div style={{ width: SCREEN_W, height: SCREEN_H, boxSizing: "border-box", padding: 24, background: "#fff", color: C.ink,
      display: "grid", gridTemplateRows: "auto auto 1fr auto", gridTemplateColumns: "minmax(0, 1fr)", gap: 10, overflow: "hidden", fontFamily: "Inter, sans-serif", userSelect: "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", font: `800 17px ${mono}`, letterSpacing: "0.08em" }}>
        <span>TK-101 · CONTROL KIOSK</span>
        <span style={{ color: live ? C.ok : C.warn, fontSize: 15 }}>● {live ? `LIVE t=${d.t.toFixed(1)}s` : "SIMULATED"}</span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", borderTop: `2px solid ${C.ink}`, paddingTop: 10 }}>
        <div>
          <div style={{ font: `700 14px ${mono}`, color: C.mute, letterSpacing: "0.08em" }}>TWIN RESIDUAL r = y − ŷ</div>
          <div style={{ font: `800 46px/1 ${mono}`, color, marginTop: 4 }}>{r >= 0 ? "+" : "−"}{Math.abs(r).toFixed(3)}<span style={{ fontSize: 20, color: C.mute }}> m</span></div>
        </div>
        <div style={{ font: "900 24px Inter", color: d.alarm ? C.crit : C.ok, textAlign: "right", maxWidth: 260 }}>
          {d.alarm ? (d.leading ?? d.verdict).replaceAll("_", " ") : "NOMINAL"}
        </div>
      </div>
      <svg width={W} height={H} style={{ display: "block", alignSelf: "center" }}>
        <rect width={W} height={H} fill="#F8F9FA" />
        {[LIMIT, -LIMIT].map((v) => <line key={v} x1={0} x2={W} y1={Y(v)} y2={Y(v)} stroke={C.crit} strokeWidth={2} strokeDasharray="8 6" />)}
        <line x1={0} x2={W} y1={Y(0)} y2={Y(0)} stroke={C.line} strokeWidth={2} />
        <path d={trace} fill="none" stroke={C.twin} strokeWidth={4} strokeLinejoin="round" />
        <text x={W - 6} y={Y(LIMIT) - 6} textAnchor="end" fill={C.crit} style={{ font: `700 13px ${mono}` }}>±{LIMIT} m ALARM</text>
      </svg>
      <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 8 }}>
        <div style={{ font: `700 13px ${mono}`, color: C.orange, letterSpacing: "0.08em" }}>FLIGHT RECORDER</div>
        {log.length === 0 && <div style={{ font: `500 15px ${mono}`, color: C.mute, marginTop: 4 }}>no events recorded for this run yet</div>}
        {log.map((e) => (
          <div key={e.id} style={{ font: `500 15px/1.5 ${mono}`, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            <b style={{ color: sevColor(e.sev) }}>{e.sev}</b> {e.text}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Sleek white kiosk: body, dark bezel, live DOM screen on its front face, neck and foot. */
function Kiosk({ children }: { children: React.ReactNode }) {
  return (
    <group position={[0, 0.25, 0]}>
      <Box args={[2.8, 1.9, 1]}><Matte /></Box>
      <Box args={[2.56, 1.66, 0.02]} position={[0, 0, 0.5]}><meshStandardMaterial color="#0B0D10" roughness={0.4} /></Box>
      <Html transform wrapperClass="htmlScreen" distanceFactor={1.5} position={[0, 0, 0.51]} zIndexRange={[10, 0]} pointerEvents="none">
        {children}
      </Html>
      <Box args={[0.5, 0.75, 0.4]} position={[0, -1.3, -0.1]}><Matte /></Box>
      <Box args={[1.7, 0.08, 1.1]} position={[0, -1.68, 0]}><Matte /></Box>
    </group>
  );
}

function Microchip() {
  const ref = useRef<THREE.Group>(null!);
  useFrame((_, dt) => { ref.current.rotation.y += dt * 0.6; });
  const pins = [-0.24, -0.08, 0.08, 0.24];
  return (
    <group ref={ref} rotation={[0.95, 0, 0.15]}>
      <RoundedBox args={[0.72, 0.12, 0.72]} radius={0.03}><Matte /></RoundedBox>
      <Box args={[0.34, 0.03, 0.34]} position={[0, 0.07, 0]}><Dark /></Box>
      {pins.flatMap((p) => [
        <Box key={`a${p}`} args={[0.06, 0.03, 0.14]} position={[p, -0.02, 0.42]}><Dark /></Box>,
        <Box key={`b${p}`} args={[0.06, 0.03, 0.14]} position={[p, -0.02, -0.42]}><Dark /></Box>,
        <Box key={`c${p}`} args={[0.14, 0.03, 0.06]} position={[0.42, -0.02, p]}><Dark /></Box>,
        <Box key={`d${p}`} args={[0.14, 0.03, 0.06]} position={[-0.42, -0.02, p]}><Dark /></Box>,
      ])}
    </group>
  );
}

/** Sensor node: matte base, glass dome over a teal core. */
function SensorNode() {
  return (
    <group rotation={[0.25, 0, -0.15]}>
      <Cylinder args={[0.3, 0.34, 0.2, 48]}><Matte /></Cylinder>
      <Sphere args={[0.12, 32, 32]} position={[0, 0.18, 0]}>
        <meshStandardMaterial color={C.teal} emissive={C.teal} emissiveIntensity={0.5} />
      </Sphere>
      <Sphere args={[0.27, 48, 48, 0, Math.PI * 2, 0, Math.PI / 2]} position={[0, 0.1, 0]}>
        <MeshTransmissionMaterial resolution={256} samples={4} thickness={0.2} roughness={0.05} transmission={1} ior={1.2}
          chromaticAberration={0.02} background={GLASS_BG} />
      </Sphere>
    </group>
  );
}

function Padlock() {
  return (
    <group rotation={[0.1, -0.4, 0.12]}>
      <RoundedBox args={[0.56, 0.46, 0.24]} radius={0.05}><Matte /></RoundedBox>
      <Torus args={[0.17, 0.045, 16, 48, Math.PI]} position={[0, 0.23, 0]}><Dark /></Torus>
      <Cylinder args={[0.05, 0.05, 0.02, 24]} position={[0, 0.03, 0.125]} rotation={[Math.PI / 2, 0, 0]}><Dark /></Cylinder>
      <Box args={[0.03, 0.1, 0.02]} position={[0, -0.05, 0.125]}><Dark /></Box>
    </group>
  );
}

// Static, so it skips the re-render the scene gets on every live tick.
const FloatingAssets = memo(function FloatingAssets() {
  return (
    <>
      <Float speed={1.6} rotationIntensity={0.3} floatIntensity={0.6} position={[-1.95, -0.55, 0.3]}><Microchip /></Float>
      <Float speed={1.3} rotationIntensity={0.4} floatIntensity={0.5} position={[1.8, 1.25, -0.7]}><SensorNode /></Float>
      <Float speed={1.1} rotationIntensity={0.35} floatIntensity={0.5} position={[1.75, -0.95, 0.5]}><Padlock /></Float>
    </>
  );
});

/** Landing-page centerpiece: a rotatable 3D control kiosk whose screen is live React DOM, ringed by floating assets. */
export default function HeroScene() {
  const d = useLive();
  const residuals = useSamples(d, (x) => x.level - x.twin, N);
  const events = useRecordedEvents(d.source === "LIVE" ? d.runId : null);
  return (
    <Canvas camera={{ position: [0, 0.15 * WORLD, 5.6 * WORLD], fov: 38, near: 0.1 * WORLD, far: 50 * WORLD }} dpr={[1, 2]} style={{ cursor: "grab" }}>
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 5, 4]} intensity={1.2} />
      <Optional><Environment preset="studio" /></Optional>
      <group scale={WORLD}>
        <PresentationControls global rotation={[0.04, -0.22, 0]} polar={[-0.1, 0.1]} azimuth={[-0.2, 0.2]} config={{ mass: 2, tension: 400 }}>
          <Kiosk><KioskScreen d={d} residuals={residuals} events={events} /></Kiosk>
          <FloatingAssets />
        </PresentationControls>
        <ContactShadows position={[0, -1.5, 0]} opacity={0.35} scale={10} blur={2.6} far={3} />
      </group>
    </Canvas>
  );
}
