import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Camera, ChevronDown, ChevronUp, FileClock, Fingerprint, Waves } from "lucide-react";
import { API_URL, C, Live, SHADOW } from "./theme";

const HIST_N = 120;
const RES_THRESH = 0.05; // m — same threshold the 3D scene uses to colour the residual plane red

const mono = "'JetBrains Mono', monospace";
const label: React.CSSProperties = { font: `400 10px ${mono}`, color: C.mute };
export const chip = (color: string): React.CSSProperties => ({
  display: "inline-block", font: `700 10px ${mono}`, letterSpacing: "0.04em", color: "#fff", background: color,
  padding: "3px 7px", borderRadius: 2, whiteSpace: "nowrap",
});

export function fmt(t: number) {
  const s = Math.max(0, Math.floor(t));
  const p = (n: number) => String(n).padStart(2, "0");
  return `[${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}]`;
}

type Sample = { t: number; y: number; yhat: number; eps: number | null };

function useHistory(d: Live): Sample[] {
  const [hist, setHist] = useState<Sample[]>([]);
  const src = useRef(d.source);
  useEffect(() => {
    // A source switch (fallback ↔ live) starts a fresh window so the two are never drawn on one trace.
    const reset = src.current !== d.source;
    src.current = d.source;
    setHist((h) => [...(reset ? [] : h), { t: d.t, y: d.level, yhat: d.twin, eps: d.wmEps }].slice(-HIST_N));
  }, [d.t, d.source]); // eslint-disable-line react-hooks/exhaustive-deps
  return hist;
}

export function Sparkline({ series, height = 56 }: { series: { values: (number | null)[]; color: string; dash?: string }[]; height?: number }) {
  const all = series.flatMap((s) => s.values).filter((v): v is number => v != null);
  if (all.length < 2) return <div style={{ height, ...label, display: "grid", placeItems: "center" }}>COLLECTING…</div>;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-6) { lo -= 1e-3; hi += 1e-3; }
  const W = 200, pad = 3;
  const pts = (vals: (number | null)[]) => vals.map((v, i) => v == null ? null
    : `${(i / Math.max(1, HIST_N - 1)) * W},${pad + (1 - (v - lo) / (hi - lo)) * (height - 2 * pad)}`).filter(Boolean).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      {series.map((s, i) => (
        <polyline key={i} points={pts(s.values)} fill="none" stroke={s.color} strokeWidth={1.5}
          strokeDasharray={s.dash} vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}

function ModCard({ title, icon, accent, tag, children }: {
  title: string; icon: React.ReactNode; accent: string; tag?: string; children: React.ReactNode;
}) {
  return (
    <div style={{ background: "#fff", border: `1px solid ${C.line}`, borderTop: `3px solid ${accent}`, padding: 12,
      boxShadow: SHADOW, borderRadius: 2, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <span style={{ font: "900 11px Inter", letterSpacing: "0.1em", color: C.ink, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{title}</span>
        <span style={{ color: accent, display: "flex" }}>{icon}</span>
      </div>
      {tag && <div style={{ ...label, color: C.warn, marginTop: -4 }}>{tag}</div>}
      {children}
    </div>
  );
}

/* ------------------------------------------------------------ A. residual */

function ResidualCard({ d, hist }: { d: Live; hist: Sample[] }) {
  const r = Math.abs(d.level - d.twin);
  const color = r > RES_THRESH ? C.crit : r > 0.02 ? C.warn : C.ok;
  return (
    <ModCard title="DIGITAL TWIN RESIDUAL" icon={<Waves size={16} />} accent={C.twin}
      tag={d.source === "LIVE" ? undefined : "SYNTHETIC FALLBACK · NOT PLANT DATA"}>
      <div style={{ font: `500 11px ${mono}`, color: C.mute }}>| r | = | y − ŷ |</div>
      <div style={{ font: `800 26px ${mono}`, letterSpacing: "-0.03em", color }}>
        {r.toFixed(4)}<span style={{ font: `600 12px ${mono}`, color: C.mute, marginLeft: 6 }}>m</span>
      </div>
      <Sparkline series={[
        { values: hist.map((h) => h.y), color: C.ok },
        { values: hist.map((h) => h.yhat), color: C.twin, dash: "4 3" },
      ]} />
      <div style={{ ...label, display: "flex", gap: 10 }}>
        <span style={{ color: C.ok }}>— sensor y</span><span style={{ color: C.twin }}>- - twin ŷ</span>
        <span style={{ marginLeft: "auto" }}>thr {RES_THRESH.toFixed(3)} m</span>
      </div>
    </ModCard>
  );
}

/* ----------------------------------------------------------- B. watermark */

export function wmBadge(d: Live): { text: string; color: string } {
  if (d.source !== "LIVE") return { text: "DEMO · NO BACKEND", color: C.mute };
  if (d.wmStatus !== "OK") return { text: (d.wmStatus ?? "NO DATA").replaceAll("_", " "), color: C.mute };
  const s = d.wmLT;
  switch (s?.status) {
    case "CORRELATED":
      return s.detectability === "DETECTABLE"
        ? { text: "DETECTED: HIGH CONFIDENCE", color: C.teal }
        : { text: "DETECTED: DEGRADED SNR", color: C.warn };
    case "WEAK": return { text: "WEAK CORRELATION", color: C.warn };
    case "ABSENT": return { text: "ABSENT: SENSOR NOT TRACKING ACTUATOR", color: C.crit };
    case "UNDETECTABLE": return { text: "UNDETECTABLE: LOW SNR", color: C.mute };
    default: return { text: (s?.status ?? "NO DATA").replaceAll("_", " "), color: C.mute };
  }
}

function SineWave() {
  // Illustrative only: drawn when no live perturbation samples are available.
  const W = 200, H = 56;
  const path = Array.from({ length: 81 }, (_, i) => {
    const x = (i / 80) * W * 2;
    return `${i ? "L" : "M"}${x.toFixed(1)},${(H / 2 + Math.sin((i / 80) * Math.PI * 8) * (H / 2 - 6)).toFixed(1)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block", overflow: "hidden" }}>
      <line x1={0} x2={W} y1={H / 2} y2={H / 2} stroke={C.line} />
      <motion.path d={path} fill="none" stroke={C.teal} strokeWidth={1.5}
        animate={{ x: [0, -W / 2] }} transition={{ repeat: Infinity, duration: 2.4, ease: "linear" }} />
    </svg>
  );
}

function corrLine(id: string, s: Live["wmLT"]) {
  if (!s) return `${id} —`;
  const c = s.correlation == null ? "—" : s.correlation.toFixed(2);
  const e = s.expected_correlation == null ? "—" : s.expected_correlation.toFixed(2);
  return `${id} ρ ${c} / exp ${e} · ${s.status}`;
}

function WatermarkCard({ d, hist }: { d: Live; hist: Sample[] }) {
  const b = wmBadge(d);
  const eps = hist.map((h) => h.eps);
  const haveEps = d.source === "LIVE" && eps.filter((v) => v != null).length > 1;
  return (
    <ModCard title="ACTUATOR WATERMARK" icon={<Fingerprint size={16} />} accent={C.teal}
      tag={haveEps ? undefined : "ILLUSTRATIVE WAVEFORM"}>
      <div style={{ background: "#F0FDFA", border: `1px solid ${C.line}` }}>
        {haveEps ? <Sparkline series={[{ values: eps, color: C.teal }]} /> : <SineWave />}
      </div>
      <div><span style={chip(b.color)}>{b.text}</span></div>
      <div style={label}>{corrLine("LT-101", d.wmLT)}</div>
      <div style={label}>{corrLine("FT-101", d.wmFT)}</div>
    </ModCard>
  );
}

/* ------------------------------------------------------ C. flight recorder */

type Ev = { t: number; text: string; color: string };

const DEMO_SCRIPT: [number, string, string][] = [
  [5, "BASELINE · TWIN TRACKING NOMINAL", C.ok],
  [15, "SCENARIO: SENSOR SPOOF INJECTED ON LT-101", C.crit],
  [16, "TWIN RESIDUAL THRESHOLD BREACHED", C.warn],
  [17, "ANOMALY ALARM RAISED", C.crit],
  [18, "WATERMARK CHECK: LT-101 ABSENT · FT-101 CORRELATED", C.teal],
  [19, "CAMERA ESTIMATE DISAGREES WITH LT-101", C.warn],
  [21, "VERDICT: LEADING HYPOTHESIS SENSOR SPOOFING (LT-101)", C.orange],
];
const DEMO_LOOP_S = 30;

function useTimeline(d: Live): Ev[] {
  const [events, setEvents] = useState<Ev[]>([]);
  const prev = useRef<Live | null>(null);
  const push = (e: Ev) => setEvents((xs) => [...xs, e].slice(-80));

  // Live: derive events from state transitions in the stream.
  useEffect(() => {
    const p = prev.current;
    prev.current = d;
    if (d.source !== "LIVE") return;
    if (!p || p.source !== "LIVE") {
      setEvents([{ t: d.t, text: "LIVE STREAM CONNECTED · /ws/live", color: C.ok }]);
      return;
    }
    const r = Math.abs(d.level - d.twin), pr = Math.abs(p.level - p.twin);
    if (r > RES_THRESH && pr <= RES_THRESH) push({ t: d.t, text: `TWIN RESIDUAL THRESHOLD BREACHED |r| = ${r.toFixed(3)} m`, color: C.warn });
    if (r <= RES_THRESH && pr > RES_THRESH) push({ t: d.t, text: "TWIN RESIDUAL BACK WITHIN THRESHOLD", color: C.ok });
    if (d.alarm && !p.alarm) push({ t: d.t, text: "ANOMALY ALARM RAISED", color: C.crit });
    if (!d.alarm && p.alarm) push({ t: d.t, text: "ANOMALY ALARM CLEARED", color: C.ok });
    for (const [id, a, b] of [["LT-101", d.wmLT, p.wmLT], ["FT-101", d.wmFT, p.wmFT]] as const) {
      if (a?.status && a.status !== b?.status)
        push({ t: d.t, text: `WATERMARK ${id}: ${b?.status ?? "—"} → ${a.status}`, color: a.status === "ABSENT" ? C.crit : C.teal });
    }
    if (d.verdict !== p.verdict || d.leading !== p.leading) {
      const what = d.leading ? `${d.leading.replaceAll("_", " ")}${d.component ? ` (${d.component})` : ""}` : d.verdict.replaceAll("_", " ");
      push({ t: d.t, text: `VERDICT: ${what}`, color: d.leading ? C.orange : C.mute });
    }
    if (d.twinFrozen && d.twinFrozen !== p.twinFrozen) push({ t: d.t, text: `TWIN CORRECTION FROZEN · ${d.twinFrozen}`, color: C.warn });
    if (d.cam?.estStatus && d.cam.estStatus !== p.cam?.estStatus) push({ t: d.t, text: `CAMERA ESTIMATE ${d.cam.estStatus}`, color: C.warn });
  }, [d]);

  // Offline: replay a scripted, clearly labelled demo sequence.
  const live = d.source === "LIVE";
  useEffect(() => {
    if (live) return;
    setEvents([]);
    let i = 0, loop = 0;
    const id = window.setInterval(() => {
      const [t, text, color] = DEMO_SCRIPT[i];
      push({ t: t + loop * DEMO_LOOP_S, text, color });
      if (++i >= DEMO_SCRIPT.length) { i = 0; loop++; }
    }, 1400);
    return () => window.clearInterval(id);
  }, [live]);

  return events;
}

function TimelineCard({ d }: { d: Live }) {
  const events = useTimeline(d);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight; }, [events]);
  return (
    <ModCard title="FORENSIC TIMELINE" icon={<FileClock size={16} />} accent={C.orange}
      tag={d.source === "LIVE" ? undefined : "DEMO SCRIPT · NOT PLANT DATA"}>
      <div ref={box} style={{ height: 150, overflowY: "auto", background: "#0B0D10", padding: "8px 10px",
        font: `500 10.5px/1.6 ${mono}`, color: "#CBD5E1", scrollBehavior: "smooth" }}>
        {events.length === 0 && <div style={{ color: C.mute }}>awaiting events…</div>}
        {events.map((e, i) => (
          <div key={i} style={{ whiteSpace: "nowrap" }}>
            <span style={{ color: C.mute }}>{fmt(e.t)}</span> <span style={{ color: e.color }}>{e.text}</span>
          </div>
        ))}
      </div>
    </ModCard>
  );
}

/* ------------------------------------------------------- D. visual witness */

function CameraCard({ d }: { d: Live }) {
  const feed = d.source === "LIVE" && d.cam != null && d.cam.mode !== "NONE";
  const [n, setN] = useState(0);
  const [broken, setBroken] = useState(false);
  useEffect(() => {
    if (!feed) return;
    const id = window.setInterval(() => setN((x) => x + 1), 500);
    return () => window.clearInterval(id);
  }, [feed]);
  useEffect(() => setBroken(false), [feed]);

  // Illustrative preview geometry (only used when there is no backend frame).
  const TANK_H = 3.0, y = 20 + (1 - Math.min(d.level, TANK_H) / TANK_H) * 150;
  const showFrame = feed && !broken;
  return (
    <ModCard title="CAMERA ESTIMATE" icon={<Camera size={16} />} accent={C.warn}
      tag={showFrame ? (d.cam!.simulated ? "SIMULATED_VISUAL · SYNTHETIC RENDER" : undefined) : "ILLUSTRATIVE PREVIEW · NOT A CAMERA FEED"}>
      <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#0B0D10", overflow: "hidden" }}>
        {showFrame ? (
          <img src={`${API_URL}/api/camera/frame.jpg?annotated=true&n=${n}`} onError={() => setBroken(true)} alt="camera frame"
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }} />
        ) : (
          <svg viewBox="0 0 320 180" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
            <rect x={110} y={20} width={100} height={150} fill="none" stroke="#475569" strokeWidth={2} />
            <motion.rect x={112} width={96} fill={C.twin} opacity={0.55} animate={{ y, height: 169 - y }} transition={{ duration: 0.4 }} />
            <motion.g animate={{ y: y - 10 }} transition={{ duration: 0.4 }}>
              <rect x={100} y={0} width={120} height={20} fill="none" stroke={C.warn} strokeWidth={1.5} strokeDasharray="4 3" />
              <text x={224} y={14} fill={C.warn} style={{ font: `700 9px ${mono}` }}>WATERLINE</text>
            </motion.g>
          </svg>
        )}
        {showFrame && (
          <span style={{ ...chip(d.cam!.calibrated ? C.ok : C.warn), position: "absolute", top: 6, left: 6 }}>
            {d.cam!.calibrated ? "WATERLINE CALIBRATED" : "UNCALIBRATED"}
          </span>
        )}
      </div>
      <div style={{ font: `700 13px ${mono}`, color: C.ink }}>
        {d.camera != null ? `${d.camera.toFixed(3)} m` : feed ? (d.cam!.estStatus ?? "NO ESTIMATE") : "NO CAMERA SOURCE"}
      </div>
      <div style={label}>{feed ? `${d.cam!.mode} · ${d.cam!.state}` : broken ? "FRAME UNAVAILABLE" : "start a camera source from the API to see frames"}</div>
    </ModCard>
  );
}

/* ---------------------------------------------------------------- drawer */

export function InvestigationDrawer({ d, open, onToggle }: { d: Live; open: boolean; onToggle: () => void }) {
  const hist = useHistory(d);
  return (
    <motion.div initial={false} animate={{ height: open ? 300 : 44 }} transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
      style={{ position: "absolute", left: 0, right: 0, bottom: 0, zIndex: 3, background: "rgba(248,249,250,0.96)",
        borderTop: `1px solid ${C.line}`, boxShadow: "0 -18px 40px -20px rgba(11,13,16,0.25)", overflow: "hidden" }}>
      <button onClick={onToggle} style={{ all: "unset", cursor: "pointer", height: 44, padding: "0 24px", width: "100%", boxSizing: "border-box",
        display: "flex", alignItems: "center", justifyContent: "space-between", font: "900 12px Inter", letterSpacing: "0.1em", color: C.ink }}>
        <span>03 / INVESTIGATION MODULES</span>
        {open ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
      </button>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 14, padding: "0 24px 16px", alignItems: "start" }}>
        <ResidualCard d={d} hist={hist} />
        <WatermarkCard d={d} hist={hist} />
        <TimelineCard d={d} />
        <CameraCard d={d} />
      </div>
    </motion.div>
  );
}
