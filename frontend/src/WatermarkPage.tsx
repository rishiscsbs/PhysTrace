import { motion } from "framer-motion";
import { C, Live, WmSignal, reveal } from "./theme";
import { useLive, useSamples } from "./live";
import { LineChart, PageShell, View, eyebrow, mono, panel } from "./PageShell";
import { chip } from "./Investigation";
import { PumpScene, SceneFrame } from "./scenes";
import { useSecurity } from "./ResponsePlanner";

const N = 240;

function integrity(d: Live): { text: string; color: string; note: string } {
  if (d.source !== "LIVE") return { text: "DEMO · NO BACKEND", color: C.mute, note: "Waveforms below are illustrative; no integrity claim is made." };
  if (d.wmStatus === "WARMING_UP") return { text: "WATERMARK: WARMING UP", color: C.mute, note: "Needs 60 s of history before correlation is meaningful." };
  if (d.wmStatus !== "OK") return { text: `WATERMARK: ${(d.wmStatus ?? "NO DATA").replaceAll("_", " ")}`, color: C.mute, note: "Start the simulation to run the analyser." };
  const sigs = [["LT-101", d.wmLT], ["FT-101", d.wmFT]] as const;
  const absent = sigs.filter(([, s]) => s?.status === "ABSENT").map(([id]) => id);
  if (absent.length) return { text: `WATERMARK INTEGRITY: BROKEN · ${absent.join(", ")} ABSENT`, color: C.crit,
    note: "The sensor no longer echoes the secret actuator perturbation: its reading is not tracking the physical plant." };
  if (sigs.every(([, s]) => s?.status === "CORRELATED" && s.detectability === "DETECTABLE"))
    return { text: "WATERMARK INTEGRITY: VERIFIED", color: C.teal, note: "Every watched sensor echoes the injected perturbation with the expected strength." };
  return { text: "WATERMARK INTEGRITY: DEGRADED", color: C.warn, note: "Correlation is weak or the signal-to-noise ratio is too low for a firm call." };
}

function SensorRow({ id, s }: { id: string; s: WmSignal | null }) {
  const c = s?.correlation ?? 0, e = s?.expected_correlation ?? 0;
  const color = s?.status === "CORRELATED" ? C.teal : s?.status === "ABSENT" ? C.crit : s?.status === "WEAK" ? C.warn : C.mute;
  return (
    <div style={{ padding: "14px 0", borderTop: `1px solid ${C.line}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ font: "800 18px Inter" }}>{id}</span>
        <span style={{ font: `700 11px ${mono}`, color }}>{s?.status ?? "—"}</span>
      </div>
      <div style={{ position: "relative", height: 10, background: C.line, marginTop: 10 }}>
        <motion.div animate={{ width: `${Math.max(0, Math.min(1, c)) * 100}%` }} style={{ height: 10, background: color }} />
        {e > 0 && <div title="expected" style={{ position: "absolute", top: -4, bottom: -4, left: `${Math.min(1, e) * 100}%`, width: 2, background: C.ink }} />}
      </div>
      <div style={{ font: `500 11px ${mono}`, color: C.mute, marginTop: 8 }}>
        ρ {s?.correlation?.toFixed(3) ?? "—"} · expected {s?.expected_correlation?.toFixed(3) ?? "—"} · {s?.detectability ?? "—"}
      </div>
    </div>
  );
}

export default function WatermarkPage({ setCurrentView }: { setCurrentView: (v: View) => void }) {
  const d = useLive();
  const live = d.source === "LIVE";
  const sec = useSecurity(d);
  // Offline: an illustrative commanded signal and sine perturbation; live: the controller command and injected ε.
  const samples = useSamples(d, (x) => live
    ? { u: x.uCmd, eps: x.wmEps }
    : { u: 0.61 + 0.05 * Math.sin(x.t / 4), eps: 0.02 * Math.sin(x.t * 2.4) }, N);
  const b = integrity(d);
  return (
    <PageShell idx="01" title="Physical Watermarking" accent={C.teal} d={d} sec={sec} setCurrentView={setCurrentView}>
      <motion.div key={b.text} initial={{ scale: 0.97, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
        style={{ ...panel, borderLeft: `8px solid ${b.color}`, display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ ...chip(b.color), font: "900 clamp(14px, 2vw, 22px) Inter", padding: "12px 18px", letterSpacing: "0.02em", whiteSpace: "normal" }}>{b.text}</span>
        <span style={{ font: "500 14px/1.5 Inter", color: C.mute, maxWidth: 520 }}>{b.note}</span>
      </motion.div>

      <div style={{ marginTop: 24 }}>
        <SceneFrame title="PUMP MOTOR P-101 · WATERMARK INJECTION" accent={C.teal} camera={[4.2, 2.4, 5.2]}
          note={live ? "vibration follows live |ε|, exaggerated for visibility" : "ILLUSTRATIVE · NOT PLANT DATA"}>
          <PumpScene amp={live ? 0.05 * Math.min(1, Math.abs(d.wmEps ?? 0) / 0.02) : 0.05} speed={d.pump}
            ringColor={b.color === C.mute ? C.teal : b.color} />
        </SceneFrame>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2.2fr) minmax(260px, 1fr)", gap: 24, marginTop: 24 }} className="pt-split">
        <motion.div {...reveal(0.1)} style={{ ...panel, borderTop: `3px solid ${C.teal}` }}>
          <div style={{ ...eyebrow, marginBottom: 4 }}>COMMANDED SIGNAL · u_cmd (pump command, 0..1)</div>
          {!live && <div style={{ ...eyebrow, color: C.warn }}>ILLUSTRATIVE WAVEFORM · NOT PLANT DATA</div>}
          <LineChart n={N} height={180} series={[{ values: samples.map((s) => s.u), color: C.twin, label: "Commanded Signal" }]} />
          <div style={{ ...eyebrow, margin: "22px 0 4px" }}>WATERMARK SIGNAL · ε (secret perturbation added at the actuator)</div>
          <LineChart n={N} height={150} series={[{ values: samples.map((s) => s.eps), color: C.teal, label: "Watermark Signal", width: 2.5 }]}
            refs={[{ y: 0, color: C.mute, label: "0" }]} />
        </motion.div>
        <motion.div {...reveal(0.15)} style={{ ...panel, borderTop: `3px solid ${C.ink}` }}>
          <div style={eyebrow}>SENSOR RESPONSE · does each sensor echo ε?</div>
          <SensorRow id="LT-101" s={d.wmLT} />
          <SensorRow id="FT-101" s={d.wmFT} />
          <div style={{ font: "400 13px/1.55 Inter", color: C.mute, marginTop: 12 }}>
            Bar: measured correlation ρ between ε (passed through the pump and tank physics) and the sensor. Tick: the
            correlation physics says to expect at this noise level. A replayed or forged sensor cannot echo an ε it never saw.
          </div>
        </motion.div>
      </div>
      <style>{`@media (max-width: 900px) { .pt-split { grid-template-columns: 1fr !important; } }`}</style>
    </PageShell>
  );
}
