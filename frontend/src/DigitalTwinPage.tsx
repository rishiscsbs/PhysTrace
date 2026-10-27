import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { API_URL, C, Live, reveal } from "./theme";
import { useLive, useSamples } from "./live";
import { LineChart, PageShell, View, eyebrow, mono, panel } from "./PageShell";
import { chip } from "./Investigation";
import { SceneFrame, TanksScene } from "./scenes";
import { useSecurity } from "./ResponsePlanner";

const N = 300;
const THR = 0.05; // m — residual threshold used across the UI

/** Swiss-style corner-bracket targeting box. */
function Target({ top, label, color }: { top: string; label: string; color: string }) {
  const corner = (pos: React.CSSProperties): React.CSSProperties => ({ position: "absolute", width: 18, height: 18, borderColor: color, borderStyle: "solid", ...pos });
  return (
    <div style={{ position: "absolute", left: "18%", right: "18%", top, height: "16%", transform: "translateY(-50%)", transition: "top 0.4s" }}>
      <div style={corner({ left: 0, top: 0, borderWidth: "3px 0 0 3px" })} />
      <div style={corner({ right: 0, top: 0, borderWidth: "3px 3px 0 0" })} />
      <div style={corner({ left: 0, bottom: 0, borderWidth: "0 0 3px 3px" })} />
      <div style={corner({ right: 0, bottom: 0, borderWidth: "0 3px 3px 0" })} />
      <div style={{ position: "absolute", left: 0, right: 0, top: "50%", borderTop: `1px dashed ${color}` }} />
      <span style={{ ...chip(color), position: "absolute", left: 0, top: -26 }}>{label}</span>
    </div>
  );
}

function Schematic({ level }: { level: number }) {
  const y = 30 + (1 - Math.min(level, 3) / 3) * 200;
  return (
    <svg viewBox="0 0 480 270" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
      <defs>
        <pattern id="pt-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#1E293B" /></pattern>
      </defs>
      <rect width="480" height="270" fill="url(#pt-grid)" />
      <rect x="170" y="30" width="140" height="200" fill="none" stroke="#94A3B8" strokeWidth="2" />
      <rect x="172" y={y} width="136" height={229 - y} fill={C.twin} opacity="0.35" style={{ transition: "all 0.4s" }} />
      <line x1="60" x2="170" y1="210" y2="210" stroke="#94A3B8" strokeWidth="2" /><line x1="310" x2="420" y1="220" y2="220" stroke="#94A3B8" strokeWidth="2" />
      <text x="60" y="202" fill="#94A3B8" style={{ font: `600 10px ${mono}` }}>P-101 →</text>
      <text x="360" y="212" fill="#94A3B8" style={{ font: `600 10px ${mono}` }}>→ FV-101</text>
      <text x="170" y="250" fill="#94A3B8" style={{ font: `600 10px ${mono}` }}>TK-101 · SCHEMATIC</text>
    </svg>
  );
}

type Src = "backend" | "webcam";

function Witness({ d, quarantine }: { d: Live; quarantine: boolean }) {
  const backendOk = d.source === "LIVE" && d.cam != null && d.cam.mode !== "NONE";
  const [src, setSrc] = useState<Src>(backendOk ? "backend" : "webcam");
  const [camErr, setCamErr] = useState<string | null>(null);
  const [frameErr, setFrameErr] = useState(false);
  const [n, setN] = useState(0);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (src !== "webcam") return;
    let stream: MediaStream | null = null, cancelled = false;
    setCamErr(null);
    if (!navigator.mediaDevices?.getUserMedia) { setCamErr("getUserMedia not supported"); return; }
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((s) => { if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; } stream = s; if (video.current) video.current.srcObject = s; })
      .catch((e) => setCamErr(e?.name ?? "camera unavailable"));
    return () => { cancelled = true; stream?.getTracks().forEach((t) => t.stop()); };
  }, [src]);

  useEffect(() => {
    if (src !== "backend" || !backendOk) return;
    setFrameErr(false);
    const id = window.setInterval(() => setN((x) => x + 1), 500);
    return () => window.clearInterval(id);
  }, [src, backendOk]);

  const levelTop = `${15 + (1 - Math.min(d.level, 3) / 3) * 70}%`;
  const tab = (s: Src, label: string) => (
    <button onClick={() => setSrc(s)} style={{ all: "unset", cursor: "pointer", font: `700 11px ${mono}`, padding: "7px 12px",
      background: src === s ? C.ink : "#fff", color: src === s ? "#fff" : C.ink, border: `1px solid ${C.ink}` }}>{label}</button>
  );

  let body: React.ReactNode, tag: string;
  if (src === "backend") {
    if (backendOk && !frameErr) {
      body = <img src={`${API_URL}/api/camera/frame.jpg?annotated=true&n=${n}`} onError={() => setFrameErr(true)} alt="OpenCV frame"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" }} />;
      tag = d.cam!.simulated ? "SIMULATED_VISUAL · synthetic render analysed by the OpenCV pipeline" : `${d.cam!.mode} · OpenCV pipeline`;
    } else {
      body = <Schematic level={d.level} />;
      tag = "NO BACKEND CAMERA SOURCE · schematic placeholder";
    }
  } else if (camErr) {
    body = <Schematic level={d.level} />;
    tag = `WEBCAM UNAVAILABLE (${camErr}) · schematic placeholder`;
  } else {
    body = <video ref={video} autoPlay playsInline muted style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />;
    tag = "LOCAL WEBCAM · raw preview, not analysed by the backend";
  }
  const calibrated = src === "backend" && backendOk && !frameErr && d.cam!.calibrated;
  const witnesses: [string, number | null, string][] = [
    ["SENSOR y", d.level, quarantine ? C.warn : C.ok], ["TWIN ŷ", d.twin, C.twin], ["CAMERA", d.camera, C.warn],
  ];

  return (
    <motion.div {...reveal(0.15)} style={{ ...panel, borderTop: `3px solid ${C.warn}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <div style={eyebrow}>VIRTUAL WITNESS · CAMERA</div>
        <div style={{ display: "flex" }}>{tab("backend", "BACKEND CV FEED")}{tab("webcam", "LOCAL WEBCAM")}</div>
      </div>
      <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#0B0D10", overflow: "hidden", marginTop: 14 }}>
        {body}
        <Target top={levelTop} color={calibrated ? C.ok : C.warn}
          label={calibrated ? "CALIBRATED WATERLINE" : "WATERLINE TARGET · UNCALIBRATED"} />
      </div>
      <div style={{ font: `500 11px ${mono}`, color: C.mute, marginTop: 8 }}>{tag}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", marginTop: 16, borderTop: `1px solid ${C.line}` }}>
        {witnesses.map(([k, v, c]) => (
          <div key={k} style={{ padding: "12px 0", borderRight: `1px solid ${C.line}`, paddingLeft: 12 }}>
            <div style={{ font: `700 10px ${mono}`, color: c }}>{k}</div>
            <div style={{ font: `800 22px ${mono}`, letterSpacing: "-0.03em" }}>{v == null ? "—" : v.toFixed(3)}<span style={{ font: `600 11px ${mono}`, color: C.mute }}> m</span></div>
          </div>
        ))}
      </div>
      <div style={{ font: "400 13px/1.5 Inter", color: C.mute, marginTop: 10 }}>
        Three independent views of one water level. When the sensor disagrees with both the twin and the camera, the sensor is the outlier.
      </div>
    </motion.div>
  );
}

export default function DigitalTwinPage({ setCurrentView }: { setCurrentView: (v: View) => void }) {
  const d = useLive();
  const live = d.source === "LIVE";
  const sec = useSecurity(d);
  const q = sec.securityStatus === "QUARANTINE_ACTIVE";
  const s = useSamples(d, (x) => ({ y: x.level, yhat: x.twin, r: x.level - x.twin }), N);
  const r = d.level - d.twin, ar = Math.abs(r);
  const color = ar > THR ? C.crit : ar > 0.02 ? C.warn : C.ok;
  return (
    <PageShell idx="02" title="Virtual Witness / Digital Twin" accent={C.twin} d={d} sec={sec} setCurrentView={setCurrentView}>
      <div style={{ marginBottom: 24 }}>
        <SceneFrame title="SENSOR vs. DIGITAL TWIN · TK-101" accent={C.twin} camera={[0, 1.6, 7]} height={380}
          note={live ? "true scale · blue ring on the glass tank marks the twin's ŷ" : "SYNTHETIC FALLBACK · NOT PLANT DATA"}>
          <TanksScene y={d.level} yhat={d.twin} quarantine={q} />
        </SceneFrame>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 24 }} className="pt-split">
        <motion.div {...reveal(0.1)} style={{ ...panel, borderTop: `3px solid ${C.twin}` }}>
          <div style={eyebrow}>DIGITAL TWIN RESIDUAL{live ? "" : " · SYNTHETIC FALLBACK, NOT PLANT DATA"}</div>
          <div style={{ border: `1px solid ${C.ink}`, padding: "20px 22px", marginTop: 14, background: "#FAFBFC" }}>
            <div style={{ font: `700 clamp(22px, 3vw, 34px) ${mono}`, letterSpacing: "-0.02em" }}>| r | = | y − ŷ |</div>
            <div style={{ font: `500 clamp(14px, 1.6vw, 18px) ${mono}`, color: C.mute, marginTop: 10 }}>
              = | <span style={{ color: C.ok }}>{d.level.toFixed(4)}</span> − <span style={{ color: C.twin }}>{d.twin.toFixed(4)}</span> |
            </div>
            <div style={{ font: `900 clamp(36px, 5vw, 56px) ${mono}`, letterSpacing: "-0.04em", color, marginTop: 6 }}>
              = {ar.toFixed(4)}<span style={{ font: `600 16px ${mono}`, color: C.mute }}> m</span>
            </div>
            <div style={{ font: `700 11px ${mono}`, color, marginTop: 4 }}>
              {ar > THR ? `ABOVE THRESHOLD ${THR.toFixed(3)} m` : `WITHIN THRESHOLD ${THR.toFixed(3)} m`}
              {live && d.alarm ? ` · ALARM · ${d.leading?.replaceAll("_", " ") ?? d.verdict}` : ""}
            </div>
          </div>
          <div style={{ ...eyebrow, margin: "22px 0 4px" }}>RESIDUAL GAP r = y − ŷ</div>
          <LineChart n={N} height={200} unit="m" series={[{ values: s.map((x) => x.r), color, label: "residual r" }]}
            refs={[{ y: THR, color: C.crit, label: `+${THR}` }, { y: -THR, color: C.crit, label: `−${THR}` }, { y: 0, color: C.mute, label: "0" }]} />
          <div style={{ ...eyebrow, margin: "22px 0 4px" }}>SENSOR vs TWIN</div>
          <LineChart n={N} height={150} unit="m" series={[
            { values: s.map((x) => x.y), color: C.ok, label: "sensor y" },
            { values: s.map((x) => x.yhat), color: C.twin, label: "twin ŷ", dash: "6 4" },
          ]} />
        </motion.div>
        <Witness d={d} quarantine={q} />
      </div>
      <style>{`@media (max-width: 900px) { .pt-split { grid-template-columns: 1fr !important; } }`}</style>
    </PageShell>
  );
}
