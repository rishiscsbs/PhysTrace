import { useEffect, useRef, useState } from "react";
import { API_URL, Live, RecEvent, WS_URL } from "./theme";

const FALLBACK_START = performance.now();

function simulated(prev: Live | null): Live {
  const s = (performance.now() - FALLBACK_START) / 1000;
  const level = 1.5 + 0.18 * Math.sin(s / 6) + 0.03 * Math.sin(s * 1.7);
  return {
    source: "SIMULATED_FALLBACK", t: s, level, twin: 1.5 + 0.18 * Math.sin((s - 0.6) / 6),
    camera: null, pump: 0.61 + 0.05 * Math.sin(s / 4), qin: 14 + 1.4 * Math.sin(s / 4), qout: 14 + 1.2 * Math.sin(s / 5),
    valve: 50 + 8 * Math.sin(s / 9), alarm: false, verdict: prev?.verdict === "NOMINAL" ? "NOMINAL" : "NO BACKEND",
    leading: null, component: null, wmStatus: null, wmLT: null, wmFT: null, wmEps: null, cam: null, twinFrozen: null,
    uCmd: null, runId: null, simStatus: null, response: null, responseAt: null, fallbackSrc: null,
  };
}

export function useLive(): Live {
  const [data, setData] = useState<Live>(() => simulated(null));
  const lastMsg = useRef(0);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.type !== "state") return;
        lastMsg.current = performance.now();
        const o = m.observed ?? {}, tw = m.twin ?? {};
        const est = m.camera?.estimate;
        const wm = m.watermark ?? {}, cam = m.camera;
        setData({
          source: "LIVE", t: m.t,
          level: o["LT-101"] ?? 0, twin: tw["LT-101"] ?? 0,
          camera: est?.status === "OK" ? est.level_m : null,
          pump: o["SC-101"] ?? 0, qin: o["FT-101"] ?? 0, qout: o["FT-102"] ?? 0, valve: o["ZT-101"] ?? 0,
          alarm: !!m.alarm?.active, verdict: m.verdict?.state ?? "NOMINAL",
          leading: m.verdict?.leading ?? null, component: m.verdict?.component ?? null,
          wmStatus: wm.status ?? null, wmLT: wm.signals?.["LT-101"] ?? null, wmFT: wm.signals?.["FT-101"] ?? null,
          wmEps: m.commands?.wm_eps ?? null,
          cam: cam ? { mode: cam.mode, state: cam.state, calibrated: !!cam.calibrated, simulated: !!cam.simulated,
            estStatus: est?.status ?? null } : null,
          twinFrozen: m.twin_state?.correction_frozen ?? null,
          uCmd: m.commands?.u_cmd ?? null, runId: m.run_id ?? null, simStatus: m.status ?? null,
          response: m.response?.policy ?? null, responseAt: m.response?.activated_at_s ?? null,
          fallbackSrc: m.response?.fallback_source ?? null,
        });
      };
      ws.onclose = () => { retry = window.setTimeout(connect, 2000); };
      ws.onerror = () => ws?.close();
    };
    connect();
    // Fallback: if no live frame for 1.5 s, drive the UI from a clearly labelled synthetic signal.
    const tick = window.setInterval(() => {
      if (performance.now() - lastMsg.current > 1500) setData((p) => simulated(p));
    }, 100);
    return () => { window.clearInterval(tick); window.clearTimeout(retry); ws?.close(); };
  }, []);
  return data;
}

/** The run's recorded event log (history from the database, then live pushes from /ws/live). */
export function useRecordedEvents(runId: string | null): RecEvent[] {
  const [events, setEvents] = useState<RecEvent[]>([]);
  useEffect(() => {
    setEvents([]);
    if (!runId) return;
    let alive = true;
    const seen = new Set<string>();
    const add = (xs: RecEvent[]) => {
      const fresh = xs.filter((e) => e.run_id === runId && !seen.has(e.event_id));
      fresh.forEach((e) => seen.add(e.event_id));
      if (fresh.length) setEvents((cur) => [...cur, ...fresh].sort((a, b) => a.timestamp - b.timestamp).slice(-500));
    };
    fetch(`${API_URL}/api/runs/${runId}/timeline?t0=0&every=1000000`)
      .then((r) => r.json()).then((j) => { if (alive) add(j.events ?? []); }).catch(() => {});
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.type === "event") add([m.event]); };
    return () => { alive = false; ws.close(); };
  }, [runId]);
  return events;
}

/** Rolling window of per-frame samples; restarts when the source switches between live and fallback. */
export function useSamples<T>(d: Live, pick: (d: Live) => T, n: number): T[] {
  const [buf, setBuf] = useState<T[]>([]);
  const src = useRef(d.source);
  const pickRef = useRef(pick);
  pickRef.current = pick;
  useEffect(() => {
    const reset = src.current !== d.source;
    src.current = d.source;
    setBuf((b) => [...(reset ? [] : b), pickRef.current(d)].slice(-n));
  }, [d.t, d.source]); // eslint-disable-line react-hooks/exhaustive-deps
  return buf;
}

export async function api(path: string, body?: unknown) {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}
