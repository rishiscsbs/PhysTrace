import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, FlaskConical } from "lucide-react";
import { C, Live, SHADOW } from "./theme";
import { api } from "./live";
import { ResponsePlanner, Security } from "./ResponsePlanner";

export type View = "landing" | "dashboard" | "watermark" | "twin" | "recorder";

export const mono = "'JetBrains Mono', monospace";
export const shadow = SHADOW;
export const panel: React.CSSProperties = { background: "#fff", border: `1px solid ${C.line}`, boxShadow: shadow, padding: 20, minWidth: 0 };
export const eyebrow: React.CSSProperties = { font: `700 11px ${mono}`, letterSpacing: "0.1em", color: C.mute };

/** Simulation-only attack lab: injects real scenarios into the backend so the page reacts to genuine detection. */
function AttackLab({ d }: { d: Live }) {
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const live = d.source === "LIVE";
  const run = (label: string, path: string, body?: unknown) => async () => {
    setBusy(true);
    try { await api(path, body); setMsg(`${label} · OK (t = ${d.t.toFixed(1)} s)`); }
    catch (e) { setMsg(`${label} · FAILED: ${(e as Error).message}`); }
    finally { setBusy(false); }
  };
  const btn = (color: string): React.CSSProperties => ({
    all: "unset", cursor: busy || !live ? "not-allowed" : "pointer", opacity: busy || !live ? 0.45 : 1,
    font: `700 11px ${mono}`, padding: "8px 12px", border: `1px solid ${color}`, color, background: "#fff", whiteSpace: "nowrap",
  });
  const off = busy || !live;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", padding: "12px 0", borderBottom: `1px solid ${C.line}` }}>
      <span style={{ ...eyebrow, display: "flex", gap: 6, alignItems: "center" }}><FlaskConical size={14} /> ATTACK LAB · SIMULATION ONLY</span>
      {d.simStatus !== "RUNNING" && <button disabled={off} style={btn(C.ok)} onClick={run("START SIM", "/api/sim/start")}>▶ START SIMULATION</button>}
      <button disabled={off} style={btn(C.crit)} onClick={run("LT-101 REPLAY", "/api/scenarios", { type: "SENSOR_REPLAY", target: "LT-101", start_in_s: 2 })}>INJECT LT-101 REPLAY</button>
      <button disabled={off} style={btn(C.crit)} onClick={run("LT-101 BIAS", "/api/scenarios", { type: "SENSOR_BIAS", target: "LT-101", start_in_s: 2 })}>INJECT LT-101 BIAS</button>
      <button disabled={off} style={btn(C.mute)} onClick={run("CLEAR", "/api/scenarios/clear")}>CLEAR SCENARIOS</button>
      <span style={{ font: `500 11px ${mono}`, color: msg?.includes("FAILED") ? C.crit : C.mute }}>
        {!live ? "backend offline: attack lab disabled" : msg}
      </span>
    </div>
  );
}

export function PageShell({ idx, title, accent, d, sec, setCurrentView, children }: {
  idx: string; title: string; accent: string; d: Live; sec?: Security; setCurrentView: (v: View) => void; children: React.ReactNode;
}) {
  const live = d.source === "LIVE";
  const q = sec?.securityStatus === "QUARANTINE_ACTIVE";
  // Status indicator: green on normal control, amber once a protective response overrides it.
  const statusColor = q ? C.warn : live ? C.ok : C.warn;
  return (
    <div style={{ minHeight: "100vh", background: "transparent", color: C.ink, fontFamily: "Inter, Helvetica Neue, sans-serif",
      boxShadow: q ? `inset 0 0 0 6px ${C.warn}` : "none", transition: "box-shadow 0.3s" }}>
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 16px 48px" }}>
        <header style={{ padding: "28px 0 20px", borderBottom: `1px solid ${C.ink}` }}>
          <motion.button whileHover={{ x: -3 }} onClick={() => setCurrentView("landing")}
            style={{ all: "unset", cursor: "pointer", font: `700 12px ${mono}`, color: C.ink, display: "inline-flex", gap: 8, alignItems: "center",
              border: `1px solid ${C.line}`, padding: "8px 12px", background: "#fff", boxShadow: SHADOW }}>
            <ArrowLeft size={14} /> Back to Investigation Engine
          </motion.button>
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", alignItems: "end", gap: 16, marginTop: 24 }}>
            <div>
              <div style={{ ...eyebrow, color: accent }}>{idx} / MODULE</div>
              <h1 style={{ font: "900 clamp(36px, 6vw, 72px)/0.92 Inter", letterSpacing: "-0.02em", margin: "10px 0 0" }}>{title}</h1>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", font: `600 11px ${mono}`, color: statusColor, textAlign: "right" }}>
              <motion.span animate={{ opacity: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: q ? 0.8 : 1.6 }}
                style={{ width: 9, height: 9, background: statusColor, display: "inline-block", flex: "none" }} />
              {live ? `LIVE · ${d.simStatus ?? ""} · t = ${d.t.toFixed(1)} s` : "OFFLINE · DEMO DATA (NOT PLANT DATA)"}
              {q && " · QUARANTINE ACTIVE"}
            </div>
          </div>
          <div style={{ height: 4, background: q ? C.warn : accent, marginTop: 20, width: q ? "100%" : 96, transition: "all 0.4s" }} />
        </header>
        <AttackLab d={d} />
        {sec && <ResponsePlanner sec={sec} />}
        <div style={{ marginTop: 24 }}>{children}</div>
      </div>
    </div>
  );
}

/** Line chart with y-axis labels, optional horizontal reference lines and a fixed sample capacity. */
export function LineChart({ series, n, height = 220, refs = [], unit = "", fixed }: {
  series: { values: (number | null)[]; color: string; label: string; dash?: string; width?: number }[];
  n: number; height?: number; refs?: { y: number; color: string; label: string }[]; unit?: string; fixed?: [number, number];
}) {
  const W = 800, L = 56, R = 8, T = 10, B = 22;
  const vals = series.flatMap((s) => s.values).filter((v): v is number => v != null).concat(refs.map((r) => r.y));
  let [lo, hi] = fixed ?? (vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1]);
  if (!fixed) { const pad = (hi - lo) * 0.12 || 1e-3; lo -= pad; hi += pad; }
  const X = (i: number) => L + (i / Math.max(1, n - 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (height - T - B);
  const path = (vs: (number | null)[]) => {
    let out = "", pen = false;
    vs.forEach((v, i) => { if (v == null) { pen = false; return; } out += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; });
    return out;
  };
  const ticks = [lo, (lo + hi) / 2, hi];
  const digits = hi - lo < 0.1 ? 3 : 2;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
        {ticks.map((v, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke={C.line} />
            <text x={L - 6} y={Y(v) + 3} textAnchor="end" fill={C.mute} style={{ font: `500 10px ${mono}` }}>{v.toFixed(digits)}</text>
          </g>
        ))}
        <line x1={L} x2={L} y1={T} y2={height - B} stroke={C.ink} />
        {refs.map((r, i) => (
          <g key={`r${i}`}>
            <line x1={L} x2={W - R} y1={Y(r.y)} y2={Y(r.y)} stroke={r.color} strokeDasharray="5 4" />
            <text x={W - R - 4} y={Y(r.y) - 4} textAnchor="end" fill={r.color} style={{ font: `700 10px ${mono}` }}>{r.label}</text>
          </g>
        ))}
        {series.map((s, i) => (
          <path key={i} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash}
            strokeLinejoin="round" />
        ))}
        <text x={L} y={height - 6} fill={C.mute} style={{ font: `500 10px ${mono}` }}>← {n} samples</text>
        <text x={W - R} y={height - 6} textAnchor="end" fill={C.mute} style={{ font: `500 10px ${mono}` }}>now{unit && ` · ${unit}`}</text>
      </svg>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", font: `600 11px ${mono}`, marginTop: 6 }}>
        {series.map((s) => <span key={s.label} style={{ color: s.color }}>{s.dash ? "- -" : "——"} {s.label}</span>)}
      </div>
    </div>
  );
}
