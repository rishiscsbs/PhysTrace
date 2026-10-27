import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Camera, Gauge, RotateCcw, ShieldAlert, ShieldCheck, Unplug } from "lucide-react";
import { C, Live, SHADOW, reveal } from "./theme";
import { api } from "./live";

export type SecurityStatus = "NOMINAL" | "QUARANTINE_ACTIVE";

const mono = "'JetBrains Mono', monospace";

/** The three operator protocols, each mapped to a real response policy of the simulator's independent safety layer. */
const PROTOCOLS = [
  { policy: "ISOLATE_REMOTE_COMMUNICATION_PATH", title: "ISOLATE PLC NETWORK", kind: "Simulated path isolation", icon: Unplug,
    what: "PLC switches to hardwired local inputs; remote setpoint writes are blocked." },
  { policy: "BOUNDED_COMMAND_LIMITING", title: "ENGAGE HARDWARE SAFE MODE", kind: "Bounded command limiting", icon: Gauge,
    what: "Safety layer clamps the pump command to the verified envelope [0.45, 0.68] with a rate limit." },
  { policy: "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE", title: "SUBSTITUTE CAMERA ESTIMATE", kind: "Validated estimate substitution", icon: Camera,
    what: "Controller uses the camera waterline (if quality is OK) or the twin estimate instead of LT-101." },
] as const;

export const policyTitle = (p: string) => PROTOCOLS.find((x) => x.policy === p)?.title ?? p.replaceAll("_", " ");

export type Security = {
  securityStatus: SecurityStatus; policy: string; live: boolean; since: number | null; fallbackSrc: string | null;
  busy: boolean; msg: string | null; apply: (policy: string) => void;
};

/**
 * Live: applies the policy through /api/responses/apply and follows the policy the backend reports, so the
 * status reflects what the safety layer is actually doing. Offline: a local, clearly labelled visual demo.
 */
export function useSecurity(d: Live): Security {
  const live = d.source === "LIVE";
  const [securityStatus, setSecurityStatus] = useState<SecurityStatus>("NOMINAL");
  const [demoPolicy, setDemoPolicy] = useState("NONE");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const policy = live ? d.response ?? "NONE" : demoPolicy;
  // Set optimistically on click; reconciled here with the policy the backend (or demo) reports.
  useEffect(() => { setSecurityStatus(policy !== "NONE" ? "QUARANTINE_ACTIVE" : "NOMINAL"); }, [policy]);

  const apply = async (p: string) => {
    const label = p === "NONE" ? "RELEASE" : policyTitle(p);
    setSecurityStatus(p !== "NONE" ? "QUARANTINE_ACTIVE" : "NOMINAL");
    if (!live) { setDemoPolicy(p); setMsg(`${label} · DEMO ONLY, no backend command sent`); return; }
    setBusy(true);
    try {
      const r = await api("/api/responses/apply", { policy: p });
      setMsg(`${label} · applied at t = ${Number(r.activated_at_s).toFixed(1)} s${r.fallback_source ? ` · source ${r.fallback_source}` : ""}`);
    } catch (e) {
      setSecurityStatus(policy !== "NONE" ? "QUARANTINE_ACTIVE" : "NOMINAL");
      setMsg(`${label} · FAILED: ${(e as Error).message}`);
    } finally { setBusy(false); }
  };
  return { securityStatus, policy, live, since: live ? d.responseAt : null, fallbackSrc: live ? d.fallbackSrc : null, busy, msg, apply };
}

export function ResponsePlanner({ sec }: { sec: Security }) {
  const q = sec.securityStatus === "QUARANTINE_ACTIVE";
  const accent = q ? C.warn : C.orange;
  return (
    <motion.section {...reveal(0.05)} aria-label="Response planner"
      style={{ background: "#fff", border: `1px solid ${q ? C.warn : C.line}`, borderTop: `3px solid ${accent}`, boxShadow: SHADOW,
        padding: 20, marginTop: 16, transition: "border-color 0.3s" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ font: `700 11px ${mono}`, letterSpacing: "0.1em", color: accent }}>SAFETY-CONSTRAINED RESPONSE</div>
          <h2 style={{ font: "900 20px Inter", letterSpacing: "0.1em", margin: "4px 0 0" }}>RESPONSE PLANNER: OT SECURITY</h2>
        </div>
        <motion.div key={sec.securityStatus} initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
          style={{ display: "flex", gap: 8, alignItems: "center", font: `800 12px ${mono}`, padding: "8px 12px",
            color: "#fff", background: q ? C.warn : C.ok }}>
          {q ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
          {q ? `QUARANTINE ACTIVE · ${policyTitle(sec.policy)}` : "NOMINAL · NORMAL CONTROL"}
        </motion.div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12, marginTop: 16 }}>
        {PROTOCOLS.map(({ policy, title, kind, icon: Icon, what }) => {
          const on = sec.policy === policy;
          return (
            <motion.button key={policy} disabled={sec.busy} onClick={() => sec.apply(policy)} aria-pressed={on}
              whileHover={sec.busy ? undefined : { y: -2 }} whileTap={sec.busy ? undefined : { scale: 0.98 }}
              style={{ all: "unset", boxSizing: "border-box", cursor: sec.busy ? "wait" : "pointer", padding: 16, minHeight: 132,
                display: "flex", flexDirection: "column", gap: 8, border: `2px solid ${C.warn}`,
                background: on ? C.warn : "#fff", color: on ? "#fff" : C.ink, transition: "background 0.2s, color 0.2s" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <Icon size={22} color={on ? "#fff" : C.warn} />
                {on && <span style={{ font: `800 10px ${mono}`, border: "1px solid #fff", padding: "2px 6px" }}>ACTIVE</span>}
              </div>
              <div style={{ font: "900 16px/1.15 Inter" }}>{title}</div>
              <div style={{ font: `700 10px ${mono}`, color: on ? "#fff" : C.warn }}>{kind.toUpperCase()}</div>
              <div style={{ font: "400 12px/1.45 Inter", color: on ? "#fff" : C.mute }}>{what}</div>
            </motion.button>
          );
        })}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", marginTop: 14 }}>
        <button disabled={!q || sec.busy} onClick={() => sec.apply("NONE")}
          style={{ all: "unset", cursor: !q || sec.busy ? "not-allowed" : "pointer", opacity: !q || sec.busy ? 0.4 : 1,
            display: "inline-flex", gap: 6, alignItems: "center", font: `700 11px ${mono}`, padding: "8px 12px",
            border: `1px solid ${C.ink}`, color: C.ink }}>
          <RotateCcw size={13} /> RELEASE · RESTORE NORMAL CONTROL
        </button>
        <span style={{ font: `500 11px ${mono}`, color: sec.msg?.includes("FAILED") ? C.crit : C.mute }}>
          {sec.msg ?? (q && sec.since != null ? `active since t = ${sec.since.toFixed(1)} s` : "")}
        </span>
      </div>
      <div style={{ font: `500 11px/1.5 ${mono}`, color: C.mute, marginTop: 10 }}>
        {sec.live
          ? "Simulated interventions, applied by the simulator's independent safety layer. One policy is active at a time; choosing another replaces it. No real plant is commanded."
          : "BACKEND OFFLINE · DEMO: buttons change the visuals only; no response is applied."}
      </div>
    </motion.section>
  );
}
