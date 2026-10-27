import { Component, ReactNode, Suspense, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Environment, Html, Icosahedron, Line, MeshTransmissionMaterial, OrbitControls, RoundedBox, Sphere } from "@react-three/drei";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { motion } from "framer-motion";
import * as THREE from "three";
import { BLUEPRINT, C, SHADOW, reveal } from "./theme";
import { eyebrow, mono, panel } from "./PageShell";

const tag: React.CSSProperties = {
  background: "#fff", color: C.ink, font: `600 11px ${mono}`, padding: "5px 8px", whiteSpace: "nowrap",
  border: `1px solid ${C.line}`, boxShadow: SHADOW, borderRadius: 2,
};

/** Swallows a failed environment-map download (the presets come from a CDN; venue Wi-Fi may block it) so the canvas keeps rendering. */
export class Optional extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? null : <Suspense fallback={null}>{this.props.children}</Suspense>; }
}

/** Studio lighting plus a light bloom so emissive (teal / crimson) parts glow. Shared by every canvas. */
export function StudioFx() {
  return (
    <>
      <Optional><Environment preset="studio" environmentIntensity={0.8} /></Optional>
      <EffectComposer multisampling={4}>
        <Bloom luminanceThreshold={0.8} mipmapBlur intensity={0.5} />
      </EffectComposer>
    </>
  );
}

// The transmission pass renders the scene offscreen; the canvas itself is transparent, so give it the page white to refract.
const GLASS_BG = new THREE.Color("#ffffff");

/** Refractive acrylic for tank shells. */
export function Acrylic() {
  return <MeshTransmissionMaterial backside resolution={1024} thickness={0.2} roughness={0.05} transmission={1} ior={1.5}
    chromaticAberration={0.04} background={GLASS_BG} />;
}

// Over-bright so the amber wireframe crosses the bloom threshold and glows.
const SHIELD_COLOR = new THREE.Color(C.warn).multiplyScalar(2.2);

/** Quarantine shield: a faceted wireframe icosahedron that eases in on activation and slowly turns. */
export function Shield({ radius, position = [0, 0, 0] }: { radius: number; position?: [number, number, number] }) {
  const g = useRef<THREE.Group>(null!);
  const mat = useRef<THREE.MeshBasicMaterial>(null!);
  const grow = useRef(0);
  useFrame(({ clock }, dt) => {
    grow.current = Math.min(1, grow.current + dt * 2);
    g.current.scale.setScalar(1 - Math.pow(1 - grow.current, 3));
    g.current.rotation.y += dt * 0.15;
    mat.current.opacity = 0.3 * (0.8 + 0.2 * Math.sin(clock.elapsedTime * 2));
  });
  return (
    <group ref={g} position={position} scale={0}>
      <Icosahedron args={[radius, 2]}>
        <meshBasicMaterial ref={mat} color={SHIELD_COLOR} wireframe transparent opacity={0.3} toneMapped={false} depthWrite={false} />
      </Icosahedron>
    </group>
  );
}

/** White Swiss panel holding a responsive canvas with the shared lighting rig. */
export function SceneFrame({ title, note, accent, height = 340, camera, children }: {
  title: string; note?: string; accent: string; height?: number; camera: [number, number, number]; children: React.ReactNode;
}) {
  return (
    <motion.div {...reveal()} style={{ ...panel, padding: 0, borderTop: `3px solid ${accent}`, overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", padding: "12px 20px", borderBottom: `1px solid ${C.line}` }}>
        <span style={eyebrow}>{title}</span>
        {note && <span style={{ ...eyebrow, color: C.mute, fontWeight: 500 }}>{note}</span>}
      </div>
      <div style={{ position: "relative", height: `clamp(240px, 40vw, ${height}px)`,
        ...BLUEPRINT }}>
        <Canvas shadows camera={{ position: camera, fov: 36 }} dpr={[1, 2]} style={{ position: "absolute", inset: 0 }}>
          <ambientLight intensity={0.45} />
          <directionalLight position={[5, 8, 4]} intensity={1.1} castShadow shadow-mapSize={[1024, 1024]} />
          {children}
          <ContactShadows position={[0, -1.2, 0]} opacity={0.42} scale={14} blur={2.4} far={4} />
          <OrbitControls enablePan={false} enableZoom={false} autoRotate autoRotateSpeed={0.4} />
          <StudioFx />
        </Canvas>
      </div>
    </motion.div>
  );
}

/* --------------------------------------------------------- 1. pump motor */

/** amp: vibration amplitude in scene units (0..0.05), from the live watermark ε. */
export function PumpScene({ amp, speed, ringColor }: { amp: number; speed: number; ringColor: string }) {
  const motor = useRef<THREE.Group>(null!);
  const rotor = useRef<THREE.Group>(null!);
  const ring = useRef<THREE.Mesh>(null!);
  const ringMat = useRef<THREE.MeshStandardMaterial>(null!);
  const a = useRef(amp);
  useFrame(({ clock }, dt) => {
    const t = clock.elapsedTime;
    a.current += (amp - a.current) * Math.min(1, dt * 6);
    motor.current.position.x = Math.sin(t * 20) * a.current;
    motor.current.position.z = Math.cos(t * 23) * a.current;
    rotor.current.rotation.x += dt * (2 + speed * 18);
    const pulse = 0.5 + 0.5 * Math.sin(t * 3);
    ring.current.scale.setScalar(1 + pulse * 0.08);
    ringMat.current.emissiveIntensity = 0.6 + pulse * 1.6;
    ringMat.current.opacity = 0.55 + pulse * 0.4;
  });
  const steel = <meshStandardMaterial color="#CBD5E1" metalness={0.95} roughness={0.22} />;
  return (
    <group position={[0, -0.4, 0]}>
      <RoundedBox args={[4.2, 0.16, 1.8]} radius={0.03} position={[0.3, -0.75, 0]} receiveShadow castShadow>
        <meshStandardMaterial color="#F1F5F9" metalness={0.2} roughness={0.5} />
      </RoundedBox>
      <group ref={motor}>
        {/* motor body + cooling fins */}
        <mesh rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.55, 0.55, 1.7, 48]} />{steel}
        </mesh>
        {[-0.6, -0.36, -0.12, 0.12, 0.36, 0.6].map((x) => (
          <mesh key={x} position={[x, 0, 0]} rotation={[0, Math.PI / 2, 0]} castShadow>
            <torusGeometry args={[0.56, 0.035, 12, 48]} />
            <meshStandardMaterial color="#94A3B8" metalness={0.9} roughness={0.3} />
          </mesh>
        ))}
        <mesh position={[-0.9, 0, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.42, 0.5, 0.12, 48]} />
          <meshStandardMaterial color={C.ink} metalness={0.7} roughness={0.35} />
        </mesh>
        <mesh position={[0, -0.6, 0]} castShadow><boxGeometry args={[1.2, 0.18, 0.9]} /><meshStandardMaterial color={C.ink} metalness={0.6} roughness={0.4} /></mesh>
        {/* shaft, rotor and volute */}
        <group ref={rotor} position={[1.1, 0, 0]}>
          <mesh rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.07, 0.07, 0.5, 16]} />{steel}</mesh>
          {[0, 1, 2, 3].map((i) => (
            <mesh key={i} rotation={[(i * Math.PI) / 4, 0, 0]} position={[0.15, 0, 0]}>
              <boxGeometry args={[0.04, 0.5, 0.06]} /><meshStandardMaterial color={C.teal} metalness={0.6} roughness={0.3} />
            </mesh>
          ))}
        </group>
        <mesh position={[1.45, 0, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.45, 0.45, 0.4, 48]} />{steel}
        </mesh>
        {/* watermark ring */}
        <mesh ref={ring} rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[0.85, 0.035, 16, 96]} />
          <meshStandardMaterial ref={ringMat} color={ringColor} emissive={ringColor} transparent toneMapped={false} />
        </mesh>
      </group>
      {/* discharge pipe (fixed to the skid; the motor vibrates against it) */}
      <mesh position={[2.4, 0, 0]} castShadow><boxGeometry args={[1.5, 0.3, 0.3]} />{steel}</mesh>
      <mesh position={[3.0, 0.7, 0]} castShadow><boxGeometry args={[0.3, 1.7, 0.3]} />{steel}</mesh>
      <Html position={[0, 1.15, 0]} center style={{ pointerEvents: "none" }}>
        <div style={{ ...tag, borderLeft: `3px solid ${ringColor}` }}><b>P-101</b> motor · ε injection ring</div>
      </Html>
      <Html position={[3.0, 1.75, 0]} center style={{ pointerEvents: "none" }}>
        <div style={tag}>discharge → TK-101</div>
      </Html>
    </group>
  );
}

/* ------------------------------------------------------ 2. sensor vs twin */

const TANK_H = 3.0, VIS = 0.8; // m rim; scene units per metre

function Water({ level, wire, color }: { level: number; wire?: boolean; color: string }) {
  const ref = useRef<THREE.Mesh>(null!);
  const cur = useRef(level);
  useFrame((_, dt) => {
    cur.current += (level - cur.current) * Math.min(1, dt * 4);
    const h = Math.max(0.01, Math.min(cur.current, TANK_H)) * VIS;
    ref.current.scale.y = h;
    ref.current.position.y = h / 2;
  });
  return (
    <mesh ref={ref} castShadow={!wire}>
      <cylinderGeometry args={[0.72, 0.72, 1, 48, wire ? 8 : 1]} />
      {wire
        ? <meshBasicMaterial color={color} wireframe transparent opacity={0.9} toneMapped={false} />
        : <meshPhysicalMaterial color={color} transmission={0.3} roughness={0.15} thickness={1} transparent opacity={0.88} />}
    </mesh>
  );
}

function LevelRing({ level, color }: { level: number; color: string }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, dt) => { ref.current.position.y += (Math.min(level, TANK_H) * VIS - ref.current.position.y) * Math.min(1, dt * 4); });
  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.8, 0.86, 64]} />
      <meshBasicMaterial color={color} toneMapped={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function TanksScene({ y, yhat, quarantine = false }: { y: number; yhat: number; quarantine?: boolean }) {
  const holo = useRef<THREE.MeshBasicMaterial>(null!);
  useFrame(({ clock }) => { holo.current.opacity = 0.45 + 0.25 * Math.sin(clock.elapsedTime * 2.2); });
  const r = y - yhat;
  const gapColor = Math.abs(r) > 0.05 ? C.crit : Math.abs(r) > 0.02 ? C.warn : C.ok;
  const H = TANK_H * VIS;
  return (
    <group position={[0, -1.2, 0]}>
      {/* sensor tank: glass */}
      <group position={[-1.3, 0, 0]}>
        <mesh position={[0, H / 2, 0]} castShadow>
          <cylinderGeometry args={[0.8, 0.8, H, 64, 1, true]} />
          <Acrylic />
        </mesh>
        <mesh position={[0, 0.02, 0]} receiveShadow><cylinderGeometry args={[0.84, 0.88, 0.05, 64]} /><meshStandardMaterial color="#E2E8F0" metalness={0.4} roughness={0.4} /></mesh>
        <Water level={y} color={C.twin} />
        {/* twin prediction marked on the sensor tank: the distance to the water surface is the residual */}
        <LevelRing level={yhat} color={C.twin} />
        <Html position={[0, H + 0.35, 0]} center style={{ pointerEvents: "none" }}>
          <div style={{ ...tag, borderLeft: `3px solid ${quarantine ? C.warn : C.ok}` }}><b>SENSOR</b> y = {y.toFixed(3)} m</div>
        </Html>
      </group>
      {/* twin tank: holographic wireframe */}
      <group position={[1.3, 0, 0]}>
        <mesh position={[0, H / 2, 0]}>
          <cylinderGeometry args={[0.8, 0.8, H, 32, 10, true]} />
          <meshBasicMaterial ref={holo} color={C.twin} wireframe transparent toneMapped={false} />
        </mesh>
        <Water level={yhat} wire color="#38BDF8" />
        <Html position={[0, H + 0.35, 0]} center style={{ pointerEvents: "none" }}>
          <div style={{ ...tag, borderLeft: `3px solid ${C.twin}` }}><b>TWIN</b> ŷ = {yhat.toFixed(3)} m</div>
        </Html>
      </group>
      <Html position={[0, Math.min(Math.max(y, yhat), TANK_H) * VIS, 0]} center style={{ pointerEvents: "none" }}>
        <div style={{ ...tag, color: "#fff", background: gapColor, border: "none" }}>|r| = {Math.abs(r).toFixed(3)} m</div>
      </Html>
      {quarantine && <Shield radius={1.55} position={[-1.3, H / 2, 0]} />}
    </group>
  );
}

/* --------------------------------------------------- 3. forensic network */

type NodeDef = { id: string; pos: [number, number, number]; color: string };
const NODES: NodeDef[] = [
  { id: "LT-101", pos: [-2.4, 0, -1.2], color: C.ok }, { id: "FT-101", pos: [-2.4, 0, 0], color: C.ok },
  { id: "FT-102", pos: [-2.4, 0, 1.2], color: C.ok }, { id: "ZT-101", pos: [-0.8, 0, 1.6], color: C.ok },
  { id: "PLC-101", pos: [0, 0.4, 0], color: C.twin }, { id: "P-101", pos: [1.6, 0, -1.2], color: C.teal },
  { id: "FV-101", pos: [1.6, 0, 1.2], color: C.teal }, { id: "CAM-01", pos: [-0.8, 0, -1.8], color: C.warn },
  { id: "ENGINE", pos: [2.6, 0.6, 0], color: C.orange }, { id: "REMOTE-HMI", pos: [0.6, 1.2, -2.2], color: C.mute },
];
const EDGES: [string, string][] = [
  ["LT-101", "PLC-101"], ["FT-101", "PLC-101"], ["FT-102", "PLC-101"], ["ZT-101", "PLC-101"],
  ["PLC-101", "P-101"], ["PLC-101", "FV-101"], ["PLC-101", "ENGINE"], ["CAM-01", "ENGINE"],
  ["P-101", "ENGINE"], ["FV-101", "ENGINE"], ["REMOTE-HMI", "PLC-101"],
];

const FIELDBUS = new Set(["LT-101", "FT-101", "FT-102", "ZT-101"]);

/** How each simulated response policy changes a link: cut (no traffic), rerouted/limited (amber), or unchanged. */
function linkState(policy: string, a: string, b: string): "cut" | "amber" | null {
  if (policy === "ISOLATE_REMOTE_COMMUNICATION_PATH") {
    if (a === "REMOTE-HMI") return "cut"; // remote setpoint writes blocked
    if (FIELDBUS.has(a)) return "amber"; // PLC reads the field devices over hardwired local inputs
  }
  if (policy === "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE") {
    if (a === "LT-101") return "cut"; // LT-101 taken out of the control loop
    if ((a === "CAM-01" && b === "ENGINE") || (a === "PLC-101" && b === "ENGINE")) return "amber"; // substituted estimate path
  }
  if (policy === "BOUNDED_COMMAND_LIMITING" && a === "PLC-101" && b === "P-101") return "amber"; // clamped pump command
  return null;
}
const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

function Packet({ from, to, offset, frozen, color }: { from: THREE.Vector3; to: THREE.Vector3; offset: number; frozen: boolean; color: string }) {
  const ref = useRef<THREE.Mesh>(null!);
  const p = useRef(offset);
  useFrame((_, dt) => {
    if (!frozen) p.current = (p.current + dt * 0.45) % 1;
    ref.current.position.lerpVectors(from, to, p.current);
  });
  return <mesh ref={ref}><sphereGeometry args={[0.055, 12, 12]} /><meshBasicMaterial color={color} toneMapped={false} /></mesh>;
}

function NetNode({ n, bad }: { n: NodeDef; bad: boolean }) {
  const mat = useRef<THREE.MeshStandardMaterial>(null!);
  useFrame(({ clock }) => {
    // A compromised node freezes: no breathing glow.
    mat.current.emissiveIntensity = bad ? 1.4 : 0.6 + 0.5 * Math.sin(clock.elapsedTime * 2.5 + n.pos[0] * 1.7 + n.pos[2]);
  });
  const color = bad ? C.crit : n.color;
  return (
    <group position={n.pos}>
      <Sphere args={[bad ? 0.24 : 0.18, 32, 32]} castShadow>
        <meshStandardMaterial ref={mat} color={color} emissive={color} metalness={0.3} roughness={0.25} toneMapped={false} />
      </Sphere>
      <Html position={[0, 0.42, 0]} center style={{ pointerEvents: "none" }}>
        <div style={{ ...tag, padding: "2px 6px", fontSize: 10, ...(bad ? { background: C.crit, color: "#fff", border: "none" } : {}) }}>
          {n.id}{bad ? " · COMPROMISED" : ""}
        </div>
      </Html>
    </group>
  );
}

export function NetworkScene({ compromised, policy = "NONE" }: { compromised: string | null; policy?: string }) {
  const edges = useMemo(() => EDGES.map(([a, b]) => ({ a, b, from: new THREE.Vector3(...byId[a].pos), to: new THREE.Vector3(...byId[b].pos) })), []);
  return (
    <group position={[0, -0.6, 0]}>
      {edges.map((e) => {
        const ls = linkState(policy, e.a, e.b);
        const hit = e.a === compromised || e.b === compromised;
        // A cut link carries no traffic; otherwise a compromised link freezes red, a rerouted/limited one runs amber.
        const color = ls === "cut" || hit ? C.crit : ls === "amber" ? C.warn : "#94A3B8";
        return (
          <group key={`${e.a}-${e.b}`}>
            <Line points={[e.from, e.to]} color={color} lineWidth={ls || hit ? 2 : 1} transparent opacity={ls || hit ? 0.9 : 0.6}
              dashed={ls === "cut"} dashSize={0.12} gapSize={0.1} />
            {ls !== "cut" && [0, 0.5].map((o) => <Packet key={o} from={e.from} to={e.to} offset={o} frozen={hit}
              color={hit ? C.crit : ls === "amber" ? C.warn : C.twin} />)}
          </group>
        );
      })}
      {NODES.map((n) => <NetNode key={n.id} n={n} bad={n.id === compromised} />)}
      {policy !== "NONE" && <Shield radius={0.75} position={byId["PLC-101"].pos} />}
    </group>
  );
}
