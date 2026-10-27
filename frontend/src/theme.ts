import type React from "react";

export const WS_URL = "ws://127.0.0.1:8765/ws/live";
export const API_URL = "http://127.0.0.1:8765";

export const C = {
  bg: "#F8F9FA", line: "#E2E8F0", ink: "#0B0D10", mute: "#64748B",
  ok: "#059669", twin: "#0052FF", warn: "#D97706", crit: "#E11D48",
  teal: "#0D9488", orange: "#EA580C",
};

export type WmSignal = {
  status: string; correlation: number | null; expected_correlation: number | null; detectability?: string;
};

export type CameraInfo = {
  mode: string; state: string; calibrated: boolean; simulated: boolean; estStatus: string | null;
};

export type Live = {
  source: "LIVE" | "SIMULATED_FALLBACK";
  t: number;
  level: number; // LT-101 observed, m
  twin: number; // digital-twin prediction, m
  camera: number | null; // camera estimate, m
  pump: number; // SC-101 speed, 0..1
  qin: number; // FT-101 L/s
  qout: number; // FT-102 L/s
  valve: number; // ZT-101 %
  alarm: boolean;
  verdict: string;
  leading: string | null;
  component: string | null;
  wmStatus: string | null; // watermark analyser status (OK / WARMING_UP / DISABLED)
  wmLT: WmSignal | null;
  wmFT: WmSignal | null;
  wmEps: number | null; // injected actuator perturbation this step
  cam: CameraInfo | null;
  twinFrozen: string | null;
  uCmd: number | null; // controller pump command before the watermark is added, 0..1
  runId: string | null;
  simStatus: string | null; // RUNNING / PAUSED / STOPPED
  response: string | null; // active simulated response policy ("NONE" when normal control)
  responseAt: number | null; // sim time the policy was applied, s
  fallbackSrc: string | null; // CAMERA / TWIN while FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE is active
};

export type RecEvent = {
  event_id: string; run_id: string; timestamp: number; severity: string; category: string;
  component_id: string | null; message: string; provenance?: string;
};

/** Deep, soft elevation so cards float over the grid. */
export const SHADOW = "0 20px 50px -12px rgba(0,0,0,0.1)";
/** Lifted further for hover / focus. */
export const SHADOW_LIFT = "0 28px 64px -12px rgba(0,0,0,0.16)";

/** Architectural blueprint background: 1px hairlines every 40px. */
export const BLUEPRINT: React.CSSProperties = {
  backgroundColor: "#fff",
  backgroundImage: "linear-gradient(#F1F5F9 1px, transparent 1px), linear-gradient(90deg, #F1F5F9 1px, transparent 1px)",
  backgroundSize: "40px 40px",
};

/** Expo-out curve shared by every entrance. */
export const EASE = [0.22, 1, 0.36, 1] as const;

/** Blurred-to-sharp slide-up used by every card and text block; pass increasing delays to stagger siblings. */
export const reveal = (delay = 0) => ({
  initial: { opacity: 0, y: 20, filter: "blur(10px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)" },
  exit: { opacity: 0, y: -20 },
  transition: { duration: 0.5, ease: EASE, delay, staggerChildren: 0.1 },
});
