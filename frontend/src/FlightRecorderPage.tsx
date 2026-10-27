import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { motion } from "framer-motion";
import { API_URL, C, RecEvent, reveal } from "./theme";
import { useLive, useRecordedEvents } from "./live";
import { PageShell, View, eyebrow, mono, panel } from "./PageShell";
import { fmt } from "./Investigation";
import { NetworkScene, SceneFrame } from "./scenes";
import { policyTitle, useSecurity } from "./ResponsePlanner";

const SEV: Record<string, string> = { INFO: C.mute, NOTICE: C.twin, WARNING: C.warn, CRITICAL: C.crit };
const SEVERITIES = ["INFO", "NOTICE", "WARNING", "CRITICAL"];

const DEMO: [string, string, string][] = [
  ["INFO", "POLL", "SENSOR POLLING NOMINAL"],
  ["INFO", "TWIN", "TWIN TRACKING · |r| WITHIN THRESHOLD"],
  ["INFO", "POLL", "SENSOR POLLING NOMINAL"],
  ["CRITICAL", "ALARM", "ANOMALY DETECTED: RESIDUAL BREACH ON LT-101"],
  ["NOTICE", "COUNTERFACTUAL", "INITIATING COUNTERFACTUAL TWIN"],
  ["WARNING", "WATERMARK", "WATERMARK LT-101 ABSENT · FT-101 CORRELATED"],
  ["NOTICE", "CAMERA", "CAMERA ESTIMATE DISAGREES WITH LT-101"],
  ["CRITICAL", "VERDICT", "LEADING HYPOTHESIS: SENSOR SPOOFING (LT-101)"],
];

/** Offline only: a clearly labelled demo log stamped with wall-clock time. */
function useDemoLog(enabled: boolean): RecEvent[] {
  const [ev, setEv] = useState<RecEvent[]>([]);
  useEffect(() => {
    if (!enabled) { setEv([]); return; }
    // Restart from an empty log: the counter restarts too, so keeping old entries would duplicate their ids.
    setEv([]);
    let i = 0;
    const add = () => {
      const [severity, category, message] = DEMO[i++ % DEMO.length];
      const now = new Date();
      const t = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
      setEv((xs) => [...xs, { event_id: `demo-${i}`, run_id: "DEMO", timestamp: t, severity, category, component_id: null, message }].slice(-200));
    };
    add();
    const id = window.setInterval(add, 2500);
    return () => window.clearInterval(id);
  }, [enabled]);
  return ev;
}

export default function FlightRecorderPage({ setCurrentView }: { setCurrentView: (v: View) => void }) {
  const d = useLive();
  const live = d.source === "LIVE";
  const sec = useSecurity(d);
  const recorded = useRecordedEvents(live ? d.runId : null);
  const demo = useDemoLog(!live);
  const events = live ? recorded : demo;
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [follow, setFollow] = useState(true);
  const box = useRef<HTMLDivElement>(null);
  const shown = events.filter((e) => !hidden.has(e.severity));
  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight; }, [shown.length, follow]);

  const counts = SEVERITIES.map((s) => [s, events.filter((e) => e.severity === s).length] as const);
  const toggle = (s: string) => setHidden((h) => { const n = new Set(h); n.has(s) ? n.delete(s) : n.add(s); return n; });
  // Live: the component the engine's verdict implicates while the alarm is active. Demo: LT-101 once the script reaches the attack.
  const lastDemo = demo[demo.length - 1];
  const demoIdx = lastDemo ? (parseInt(lastDemo.event_id.slice(5), 10) - 1) % DEMO.length : 0;
  const compromised = live ? (d.alarm ? d.component : null) : demoIdx >= 3 ? "LT-101" : null;

  return (
    <PageShell idx="03" title="Attack Flight Recorder" accent={C.orange} d={d} sec={sec} setCurrentView={setCurrentView}>
      <div style={{ marginBottom: 24 }}>
        <SceneFrame title="FORENSIC DATA STREAM · FIELDBUS TOPOLOGY" accent={C.orange} camera={[0, 5, 6]}
          note={[compromised ? `${compromised} IMPLICATED${live ? ` · ${d.leading?.replaceAll("_", " ") ?? d.verdict}` : " · DEMO"}` : live ? "no component implicated" : "DEMO · NOT PLANT DATA",
            sec.policy !== "NONE" && `SHIELD · ${policyTitle(sec.policy)}`].filter(Boolean).join(" · ")}>
          <NetworkScene compromised={compromised} policy={sec.policy} />
        </SceneFrame>
      </div>
      <motion.div {...reveal(0.1)} style={{ ...panel, borderTop: `3px solid ${C.orange}`, padding: 0 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", justifyContent: "space-between",
          padding: "14px 20px", borderBottom: `1px solid ${C.line}` }}>
          <div style={eyebrow}>
            FORENSIC TIMELINE · {live ? `RUN ${d.runId ?? "—"} · RECORDED EVENTS` : "DEMO LOG · NOT PLANT DATA"}
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            {counts.map(([s, n]) => (
              <button key={s} onClick={() => toggle(s)} style={{ all: "unset", cursor: "pointer", font: `700 10px ${mono}`, padding: "5px 9px",
                border: `1px solid ${SEV[s]}`, color: hidden.has(s) ? SEV[s] : "#fff", background: hidden.has(s) ? "#fff" : SEV[s] }}>
                {s} {n}
              </button>
            ))}
            <button onClick={() => setFollow((f) => !f)} style={{ all: "unset", cursor: "pointer", font: `700 10px ${mono}`, padding: "5px 9px",
              border: `1px solid ${C.ink}`, color: follow ? "#fff" : C.ink, background: follow ? C.ink : "#fff" }}>
              {follow ? "FOLLOWING ↓" : "PAUSED"}
            </button>
            {live && d.runId && (
              <a href={`${API_URL}/api/export/run/${d.runId}.json`} target="_blank" rel="noreferrer"
                style={{ font: `700 10px ${mono}`, padding: "5px 9px", border: `1px solid ${C.ink}`, color: C.ink, textDecoration: "none",
                  display: "inline-flex", gap: 4, alignItems: "center" }}><Download size={12} /> EXPORT JSON</a>
            )}
          </div>
        </div>
        <div ref={box} style={{ height: "calc(100vh - 420px)", minHeight: 360, overflowY: "auto", padding: "14px 20px",
          font: `500 13px/1.75 ${mono}`, background: "#FCFCFD", scrollBehavior: "smooth" }}>
          {shown.length === 0 && <div style={{ color: C.mute }}>{live ? "no events recorded for this run yet…" : "awaiting events…"}</div>}
          {shown.map((e) => (
            <div key={e.event_id} style={{ display: "grid", gridTemplateColumns: "92px 88px minmax(0, 1fr)", gap: 12,
              borderBottom: `1px solid ${C.line}`, padding: "2px 0" }}>
              <span style={{ color: C.mute }}>{fmt(e.timestamp)}</span>
              <span style={{ color: SEV[e.severity] ?? C.ink, fontWeight: 700 }}>{e.severity}</span>
              <span style={{ color: C.ink, overflowWrap: "anywhere" }}>
                <span style={{ color: C.mute }}>{e.category}{e.component_id ? ` · ${e.component_id}` : ""} › </span>{e.message}
              </span>
            </div>
          ))}
          {!live && <div style={{ color: C.orange }}>▌</div>}
        </div>
        <div style={{ padding: "10px 20px", borderTop: `1px solid ${C.line}`, font: `500 11px ${mono}`, color: C.mute }}>
          {live ? "Timestamps are simulation run time. History is read from the run database, then new events stream from /ws/live. Gaps are never filled in."
            : "Backend offline: this scripted log shows the format only and is stamped with wall-clock time."}
        </div>
      </motion.div>
    </PageShell>
  );
}
