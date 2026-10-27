import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Html, OrbitControls, RoundedBox } from "@react-three/drei";
import { AnimatePresence, animate, motion, useMotionValue, useTransform } from "framer-motion";
import { Activity, AlertTriangle, Cpu, Droplets, Gauge, Pause, Play, Radio, ShieldCheck, Square, Waves } from "lucide-react";
import * as THREE from "three";
import { C, EASE, Live, SHADOW, reveal } from "./theme";
import { api, useLive } from "./live";
import { InvestigationDrawer } from "./Investigation";
import { Acrylic, Shield, StudioFx } from "./scenes";
import { ResponsePlanner, useSecurity } from "./ResponsePlanner";
import InteractiveBackground from "./InteractiveBackground";
import LandingPage from "./LandingPage";
import WatermarkPage from "./WatermarkPage";
import DigitalTwinPage from "./DigitalTwinPage";
import FlightRecorderPage from "./FlightRecorderPage";
import type { View } from "./PageShell";

const TANK_H = 3.0; // m (overflow rim)
const SCALE = 1.0; // scene units per metre

function Ticker({ value, digits = 3 }: { value: number; digits?: number }) {
  const mv = useMotionValue(value);
  const txt = useTransform(mv, (v) => v.toFixed(digits));
  useEffect(() => { const c = animate(mv, value, { duration: 0.4, ease: "easeOut" }); return c.stop; }, [value, mv]);
  return <motion.span>{txt}</motion.span>;
}

/* ------------------------------------------------------------------ 3D */

function Water({ level }: { level: number }) {
  const ref = useRef<THREE.Mesh>(null!);
  const cur = useRef(level);
  useFrame((_, dt) => {
    cur.current += (level - cur.current) * Math.min(1, dt * 4);
    const h = Math.max(0.01, Math.min(cur.current, TANK_H)) * SCALE;
    ref.current.scale.y = h;
    ref.current.position.y = h / 2;
  });
  return (
    <mesh ref={ref} castShadow>
      <cylinderGeometry args={[0.92, 0.92, 1, 64]} />
      <meshPhysicalMaterial color={C.twin} transmission={0.35} roughness={0.15} thickness={1} transparent opacity={0.85} />
    </mesh>
  );
}

function Plane({ y, color, label, glow = 1 }: { y: number; color: string; label?: string; glow?: number }) {
  const ref = useRef<THREE.Group>(null!);
  useFrame((_, dt) => { ref.current.position.y += (y * SCALE - ref.current.position.y) * Math.min(1, dt * 4); });
  return (
    <group ref={ref}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0, 1.25, 48, 1]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.55 * glow} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.22, 1.3, 64]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      {label && (
        <Html position={[1.45, 0, 0]} center={false} style={{ pointerEvents: "none" }}>
          <div style={{ ...tag, borderLeft: `3px solid ${color}` }}>{label}</div>
        </Html>
      )}
    </group>
  );
}

function Impeller({ speed }: { speed: number }) {
  const ref = useRef<THREE.Group>(null!);
  useFrame((_, dt) => { ref.current.rotation.z += dt * speed * 14; });
  return (
    <group ref={ref}>
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <mesh key={i} rotation={[0, 0, (i * Math.PI) / 3]} position={[0, 0, 0.36]}>
          <boxGeometry args={[0.05, 0.42, 0.04]} />
          <meshStandardMaterial color={C.ink} metalness={0.8} roughness={0.25} />
        </mesh>
      ))}
    </group>
  );
}

function FlowPulses({ from, to, rate, color }: { from: THREE.Vector3; to: THREE.Vector3; rate: number; color: string }) {
  const refs = useRef<THREE.Mesh[]>([]);
  const phase = useRef(0);
  useFrame((_, dt) => {
    phase.current = (phase.current + dt * Math.max(0.05, rate / 14) * 0.6) % 1;
    refs.current.forEach((m, i) => {
      if (!m) return;
      const p = (phase.current + i / 5) % 1;
      m.position.lerpVectors(from, to, p);
    });
  });
  return (
    <>
      {[0, 1, 2, 3, 4].map((i) => (
        <mesh key={i} ref={(el) => { if (el) refs.current[i] = el; }}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color={color} toneMapped={false} />
        </mesh>
      ))}
    </>
  );
}

function Pipe({ from, to }: { from: [number, number, number]; to: [number, number, number] }) {
  const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const len = a.distanceTo(b);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  return (
    <mesh position={mid} quaternion={q} castShadow>
      <cylinderGeometry args={[0.08, 0.08, len, 24]} />
      <meshStandardMaterial color="#CBD5E1" metalness={0.9} roughness={0.2} />
    </mesh>
  );
}

function Plant({ d, quarantine }: { d: Live; quarantine: boolean }) {
  const residual = d.level - d.twin;
  const resColor = Math.abs(residual) > 0.05 ? C.crit : Math.abs(residual) > 0.02 ? C.warn : C.ok;
  const inA = new THREE.Vector3(-3.0, 0.35, 0), inB = new THREE.Vector3(-1.0, 0.35, 0);
  const outA = new THREE.Vector3(1.0, 0.25, 0), outB = new THREE.Vector3(3.1, 0.25, 0);
  return (
    <group position={[0, -1.5, 0]}>
      {/* tank shell */}
      <mesh position={[0, TANK_H / 2, 0]} castShadow>
        <cylinderGeometry args={[1, 1, TANK_H, 64, 1, true]} />
        <Acrylic />
      </mesh>
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <cylinderGeometry args={[1.04, 1.08, 0.06, 64]} />
        <meshStandardMaterial color="#E2E8F0" metalness={0.4} roughness={0.4} />
      </mesh>
      <Water level={d.level} />
      {quarantine && <Shield radius={2.3} position={[0, TANK_H / 2, 0]} />}
      {/* safety limits */}
      <Plane y={2.7} color={C.crit} label="HH 2.70 m" glow={0.4} />
      <Plane y={0.3} color={C.crit} label="LL 0.30 m" glow={0.4} />
      {/* residual planes: observed vs twin */}
      <Plane y={d.level} color={resColor} label={`y  ${d.level.toFixed(3)} m · SENSOR`} />
      <Plane y={d.twin} color={C.twin} label={`ŷ  ${d.twin.toFixed(3)} m · TWIN`} />
      {d.camera != null && <Plane y={d.camera} color={C.warn} label={`cam ${d.camera.toFixed(3)} m · CAMERA`} />}

      {/* pump */}
      <group position={[-3.4, 0.35, 0]}>
        <RoundedBox args={[0.9, 0.9, 0.7]} radius={0.04} castShadow>
          <meshStandardMaterial color="#F1F5F9" metalness={0.3} roughness={0.35} />
        </RoundedBox>
        <Impeller speed={d.pump} />
        <Html position={[0, 0.85, 0]} center style={{ pointerEvents: "none" }}>
          <div style={tag}><b>P-101</b> {(d.pump * 100).toFixed(1)}% · {d.qin.toFixed(2)} L/s</div>
        </Html>
      </group>
      <Pipe from={[-2.95, 0.35, 0]} to={[-0.98, 0.35, 0]} />
      <FlowPulses from={inA} to={inB} rate={d.qin} color={C.twin} />

      {/* valve */}
      <Pipe from={[0.98, 0.25, 0]} to={[3.4, 0.25, 0]} />
      <group position={[2.2, 0.25, 0]}>
        <mesh rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 0.4, 32]} />
          <meshStandardMaterial color={C.ink} metalness={0.7} roughness={0.3} />
        </mesh>
        <mesh position={[0, 0.25 + (d.valve / 100) * 0.3, 0]}>
          <boxGeometry args={[0.06, 0.5, 0.06]} />
          <meshStandardMaterial color={C.warn} />
        </mesh>
        <Html position={[0, 0.95, 0]} center style={{ pointerEvents: "none" }}>
          <div style={tag}><b>FV-101</b> {d.valve.toFixed(1)}% · {d.qout.toFixed(2)} L/s</div>
        </Html>
      </group>
      <FlowPulses from={outA} to={outB} rate={d.qout} color={C.ok} />

      {/* level transmitter */}
      <mesh position={[0, TANK_H + 0.12, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.2, 0.24, 32]} />
        <meshStandardMaterial color={d.alarm ? C.crit : C.ok} emissive={d.alarm ? C.crit : C.ok} emissiveIntensity={0.4} />
      </mesh>
      <Html position={[0, TANK_H + 0.6, 0]} center style={{ pointerEvents: "none" }}>
        <div style={{ ...tag, borderLeft: `3px solid ${resColor}` }}>
          <b>TK-101 / LT-101</b> r = y − ŷ = {residual >= 0 ? "+" : ""}{residual.toFixed(3)} m
        </div>
      </Html>
    </group>
  );
}

const tag: React.CSSProperties = {
  background: "#fff", color: C.ink, font: "500 11px 'JetBrains Mono', monospace", padding: "6px 9px",
  whiteSpace: "nowrap", border: `1px solid ${C.line}`, boxShadow: SHADOW, borderRadius: 2,
};

/* ------------------------------------------------------------------ UI */

function Card({ idx, title, icon, children, accent = C.ink, delay = 0 }: {
  idx: string; title: string; icon: React.ReactNode; children: React.ReactNode; accent?: string; delay?: number;
}) {
  return (
    <motion.div {...reveal(delay)}
      style={{ background: "#fff", border: `1px solid ${C.line}`, borderTop: `3px solid ${accent}`, padding: 16,
        boxShadow: SHADOW, borderRadius: 2 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <span style={{ font: "900 12px Inter", letterSpacing: "0.1em", color: C.mute }}>{idx} / {title}</span>
        <span style={{ color: accent }}>{icon}</span>
      </div>
      {children}
    </motion.div>
  );
}

const big: React.CSSProperties = { font: "800 34px 'JetBrains Mono', monospace", letterSpacing: "-0.03em", color: C.ink };
const unit: React.CSSProperties = { font: "600 13px 'JetBrains Mono', monospace", color: C.mute, marginLeft: 6 };
const small: React.CSSProperties = { font: "400 11px 'JetBrains Mono', monospace", color: C.mute, marginTop: 6 };

export function Dashboard({ onBack }: { onBack?: () => void }) {
  const d = useLive();
  const live = d.source === "LIVE";
  const [drawer, setDrawer] = useState(true);
  const [simBusy, setSimBusy] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const sec = useSecurity(d);
  const q = sec.securityStatus === "QUARANTINE_ACTIVE";
  const residual = d.level - d.twin;
  // Under quarantine a protective response overrides normal control: "normal" accents turn amber.
  const status = d.alarm ? (d.verdict === "INCONCLUSIVE" ? C.warn : C.crit) : q ? C.warn : C.ok;
  const sensor = q ? C.warn : undefined;
  const controlSimulation = (path: string) => async () => {
    setSimBusy(true);
    setSimError(null);
    try { await api(path); }
    catch (error) { setSimError((error as Error).message); }
    finally { setSimBusy(false); }
  };
  return (
    <div style={{ height: "100vh", display: "grid", gridTemplateColumns: "repeat(12, 1fr)", background: C.bg,
      fontFamily: "Inter, Helvetica Neue, sans-serif", color: C.ink, overflow: "hidden" }}>
      {/* LEFT 30% */}
      <aside style={{ gridColumn: "span 4", borderRight: `1px solid ${C.line}`, boxShadow: q ? `inset 0 0 0 4px ${C.warn}` : "none", display: "flex", flexDirection: "column",
        overflowY: "auto", backgroundImage: `linear-gradient(${C.line} 1px, transparent 1px)`, backgroundSize: "100% 48px" }}>
        <header style={{ padding: "28px 24px 20px", borderBottom: `1px solid ${C.line}`, background: C.bg }}>
          {onBack && (
            <button onClick={onBack} style={{ all: "unset", cursor: "pointer", font: "600 11px 'JetBrains Mono'", color: C.mute,
              marginBottom: 14, display: "inline-block" }}>← OVERVIEW</button>
          )}
          <div style={{ font: "900 52px/0.9 Inter", letterSpacing: "-0.02em" }}>PhysTrace</div>
          <div style={{ font: "600 12px 'JetBrains Mono'", color: C.mute, marginTop: 10 }}>
            CYBER-PHYSICAL INVESTIGATION · TK-101 WATER PLANT
          </div>
          <div style={{ marginTop: 14, display: "flex", gap: 8, alignItems: "center" }}>
            <motion.span animate={{ opacity: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: 1.6 }}
              style={{ width: 9, height: 9, background: live ? C.ok : C.warn, display: "inline-block" }} />
            <span style={{ font: "600 11px 'JetBrains Mono'", color: live ? C.ok : C.warn }}>
              {live ? `LIVE BACKEND · t = ${d.t.toFixed(1)} s` : "OFFLINE · SIMULATED FALLBACK SIGNAL (NOT PLANT DATA)"}
            </span>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            {d.simStatus === "RUNNING" ? (
              <>
                <button disabled={simBusy} onClick={controlSimulation("/api/sim/pause")} style={{ display: "inline-flex", alignItems: "center", gap: 6,
                  border: `1px solid ${C.warn}`, background: "#fff", color: C.warn, padding: "7px 10px", cursor: simBusy ? "wait" : "pointer",
                  font: "700 10px 'JetBrains Mono'" }}><Pause size={13} /> PAUSE</button>
                <button disabled={simBusy} onClick={controlSimulation("/api/sim/stop")} style={{ display: "inline-flex", alignItems: "center", gap: 6,
                  border: `1px solid ${C.crit}`, background: "#fff", color: C.crit, padding: "7px 10px", cursor: simBusy ? "wait" : "pointer",
                  font: "700 10px 'JetBrains Mono'" }}><Square size={12} /> STOP</button>
              </>
            ) : (
              <button disabled={simBusy} onClick={controlSimulation("/api/sim/start")} style={{ display: "inline-flex", alignItems: "center", gap: 6,
                border: `1px solid ${C.ok}`, background: "#fff", color: C.ok, padding: "7px 10px", cursor: simBusy ? "wait" : "pointer",
                font: "700 10px 'JetBrains Mono'" }}><Play size={13} /> {d.simStatus === "PAUSED" ? "RESUME" : "START SIMULATION"}</button>
            )}
          </div>
          {simError && <div role="alert" style={{ color: C.crit, font: "500 10px 'JetBrains Mono'", marginTop: 8 }}>{simError}</div>}
        </header>
        <div style={{ padding: 20, display: "grid", gap: 16 }}>
          <Card idx="01" title="TANK LEVEL · LT-101" icon={<Droplets size={18} />} accent={sensor ?? C.twin} delay={0.05}>
            <div><span style={big}><Ticker value={d.level} /></span><span style={unit}>m</span></div>
            <div style={{ height: 6, background: C.line, marginTop: 10 }}>
              <motion.div animate={{ width: `${(d.level / TANK_H) * 100}%` }} style={{ height: 6, background: C.twin }} />
            </div>
            <div style={small}>{live ? "SENSOR OBSERVATION" : "SYNTHETIC FALLBACK"} · rim 3.00 m · HH 2.70 m</div>
          </Card>
          <Card idx="02" title="TWIN RESIDUAL" icon={<Waves size={18} />} accent={Math.abs(residual) > 0.05 ? C.crit : sensor ?? C.ok} delay={0.1}>
            <div><span style={big}>{residual >= 0 ? "+" : ""}<Ticker value={residual} /></span><span style={unit}>m</span></div>
            <div style={small}>r = y − ŷ · ŷ = {d.twin.toFixed(3)} m (DIGITAL-TWIN PREDICTION)</div>
          </Card>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <Card idx="03" title="PUMP SC-101" icon={<Cpu size={18} />} accent={sensor} delay={0.15}>
              <div><span style={{ ...big, fontSize: 26 }}><Ticker value={d.pump * 100} digits={1} /></span><span style={unit}>%</span></div>
              <div style={small}>Q_in <Ticker value={d.qin} digits={2} /> L/s</div>
            </Card>
            <Card idx="04" title="VALVE ZT-101" icon={<Gauge size={18} />} accent={sensor} delay={0.2}>
              <div><span style={{ ...big, fontSize: 26 }}><Ticker value={d.valve} digits={1} /></span><span style={unit}>%</span></div>
              <div style={small}>Q_out <Ticker value={d.qout} digits={2} /> L/s</div>
            </Card>
          </div>
          <Card idx="05" title="ANOMALY STATUS" icon={d.alarm ? <AlertTriangle size={18} /> : <ShieldCheck size={18} />} accent={status} delay={0.25}>
            <AnimatePresence mode="wait">
              <motion.div key={d.verdict + d.leading} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                <div style={{ font: "900 22px Inter", letterSpacing: "-0.02em", color: status }}>
                  {d.alarm ? (d.leading ?? d.verdict).replaceAll("_", " ") : "NOMINAL"}
                </div>
                <div style={small}>
                  {d.alarm ? `${d.verdict} · component ${d.component ?? "—"} · EVIDENCE SCORE (not a probability)` : "No active anomaly alarm"}
                </div>
              </motion.div>
            </AnimatePresence>
          </Card>
          <Card idx="06" title="VISUAL WITNESS" icon={<Radio size={18} />} accent={C.warn} delay={0.3}>
            <div style={{ font: "700 16px 'JetBrains Mono'" }}>{d.camera != null ? `${d.camera.toFixed(3)} m` : "UNAVAILABLE"}</div>
            <div style={small}>CAMERA VISUAL ESTIMATE · no estimate is shown when quality fails</div>
          </Card>
          <ResponsePlanner sec={sec} />
        </div>
      </aside>

      {/* RIGHT 70% */}
      <main style={{ gridColumn: "span 8", position: "relative" }}>
        <div style={{ position: "absolute", inset: 0, backgroundImage: `linear-gradient(${C.line} 1px, transparent 1px), linear-gradient(90deg, ${C.line} 1px, transparent 1px)`,
          backgroundSize: "calc(100% / 8) 64px", pointerEvents: "none", opacity: 0.7 }} />
        <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
          style={{ position: "absolute", top: 20, left: 24, zIndex: 2, background: "#fff", border: `1px solid ${C.line}`,
            padding: "12px 16px", boxShadow: SHADOW }}>
          <div style={{ font: "900 13px Inter", letterSpacing: "0.1em" }}>02 / LIVE 3D PLANT</div>
          <div style={{ ...small, display: "flex", gap: 14 }}>
            <span style={{ color: C.ok }}>■ sensor y</span><span style={{ color: C.twin }}>■ twin ŷ</span>
            <span style={{ color: C.warn }}>■ camera</span><span style={{ color: C.crit }}>■ HH / LL</span>
          </div>
        </motion.div>
        <div style={{ position: "absolute", top: 28, right: 24, zIndex: 2, display: "flex", gap: 8, alignItems: "center",
          font: "600 11px 'JetBrains Mono'", color: C.mute }}>
          <Activity size={14} /> {live ? "DRIVEN BY /ws/live" : "BACKEND UNREACHABLE — ANIMATION USES SYNTHETIC SIGNAL"}
        </div>
        <InvestigationDrawer d={d} open={drawer} onToggle={() => setDrawer((o) => !o)} />
        <Canvas shadows camera={{ position: [6, 3.2, 7], fov: 38 }} dpr={[1, 2]}
          style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: drawer ? 300 : 44, width: "auto", height: "auto" }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 4]} intensity={1.2} castShadow shadow-mapSize={[1024, 1024]} />
          <Plant d={d} quarantine={q} />
          <ContactShadows position={[0, -1.5, 0]} opacity={0.45} scale={14} blur={2.4} far={4} />
          <OrbitControls autoRotate autoRotateSpeed={0.5} enablePan={false} minDistance={5} maxDistance={16} target={[0, 0, 0]} />
          <StudioFx />
        </Canvas>
      </main>
    </div>
  );
}

/* -------------------------------------------------------------- router */

export default function App() {
  const [currentView, setCurrentView] = useState<View>("landing");
  useEffect(() => { window.scrollTo(0, 0); }, [currentView]);
  const page = {
    landing: <LandingPage setCurrentView={setCurrentView} />,
    dashboard: <Dashboard onBack={() => setCurrentView("landing")} />,
    watermark: <WatermarkPage setCurrentView={setCurrentView} />,
    twin: <DigitalTwinPage setCurrentView={setCurrentView} />,
    recorder: <FlightRecorderPage setCurrentView={setCurrentView} />,
  }[currentView];
  return (
    <>
    {/* Outside AnimatePresence so the wave keeps running across page transitions. */}
    <InteractiveBackground />
    <AnimatePresence mode="wait">
      {/* Exit propagates to every reveal() card inside, so the outgoing page slides up and fades before the next enters. */}
      <motion.div key={currentView} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3, ease: EASE }}>
        {page}
      </motion.div>
    </AnimatePresence>
    </>
  );
}
