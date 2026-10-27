import { useRef, useState } from "react";
import { motion } from "framer-motion";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { ArrowDown, ArrowRight, Camera, Cpu, FileClock, Fingerprint, Move3d } from "lucide-react";
import { C, EASE, SHADOW, SHADOW_LIFT, reveal } from "./theme";
import type { View } from "./PageShell";
import HeroScene from "./HeroScene";
import CustomCursor from "./CustomCursor";
import InteractiveGrid from "./InteractiveGrid";

gsap.registerPlugin(ScrollTrigger, useGSAP);

const mono = "'JetBrains Mono', monospace";
const shadow = SHADOW;

// Fixed geometry so the SVG connectors line up with the grid rows exactly.
const NODE_H = 128, GAP = 24, STACK_H = NODE_H * 3 + GAP * 2, ARROW_W = 120;
const rowMid = (i: number) => i * (NODE_H + GAP) + NODE_H / 2;
// Wide screens get the pinned scroll sequence; narrow ones get a plain stacked layout (see the media query below).
const WIDE = "(min-width: 861px)";

const MODULES: { id: string; view: View; title: string; accent: string; icon: React.ReactNode; body: string }[] = [
  { id: "wm", view: "watermark", title: "Physical Watermarking", accent: C.teal, icon: <Fingerprint size={18} />,
    body: "Injects a small, secret perturbation into the pump command and checks that sensors respond as physics predicts. A spoofed sensor can't echo a signal it never saw." },
  { id: "twin", view: "twin", title: "Virtual Witness / Digital Twin", accent: C.twin, icon: <Camera size={18} />,
    body: "A physics model predicts ŷ from commands alone; the residual r = y − ŷ flags divergence. An independent camera estimate gives a second, out-of-band witness." },
  { id: "afr", view: "recorder", title: "Attack Flight Recorder", accent: C.orange, icon: <FileClock size={18} />,
    body: "Records every observation, alarm, hypothesis and frame on a forensic timeline so an investigation can be replayed and exported after the fact." },
];

/** Card with a conic-gradient border that spins while hovered; the face stays stark white. */
function ShinyCard({ m, i, on, onHover, onOpen }: {
  m: (typeof MODULES)[number]; i: number; on: boolean; onHover: (h: boolean) => void; onOpen: () => void;
}) {
  return (
    <motion.button onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)} onFocus={() => onHover(true)}
      onBlur={() => onHover(false)} onClick={onOpen} aria-label={`Open ${m.title}`}
      animate={{ y: on ? -3 : 0, scale: on ? 1.025 : 1 }} whileTap={{ scale: 0.99 }} transition={{ duration: 0.25, ease: EASE }}
      style={{ all: "unset", cursor: "pointer", boxSizing: "border-box", position: "relative", overflow: "hidden", padding: 2,
        background: C.line, display: "grid", boxShadow: on ? SHADOW_LIFT : shadow }}>
      {/* Oversized square so its rotation always covers the card's corners; only the 2px padding ring shows. */}
      <motion.div aria-hidden animate={{ rotate: 360, opacity: on ? 1 : 0 }}
        transition={{ rotate: { duration: 3, ease: "linear", repeat: Infinity }, opacity: { duration: 0.3 } }}
        style={{ position: "absolute", left: "50%", top: "50%", width: "150%", aspectRatio: "1", x: "-50%", y: "-50%",
          background: `conic-gradient(from 0deg, transparent 0deg, ${C.teal} 70deg, ${C.twin} 140deg, ${C.orange} 210deg, transparent 280deg)` }} />
      <div style={{ position: "relative", background: "#fff", borderLeft: `6px solid ${m.accent}`, padding: "18px 20px",
        display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", justifyContent: "space-between", color: m.accent }}>
          <span style={{ font: `700 11px ${mono}`, letterSpacing: "0.1em" }}>0{i + 1} / MODULE</span>
          {m.icon}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 12 }}>
          <div style={{ font: "800 22px/1.1 Inter", letterSpacing: "-0.02em", color: C.ink }}>{m.title}</div>
          <span style={{ font: `700 11px ${mono}`, color: on ? "#fff" : m.accent, background: on ? m.accent : "transparent",
            border: `1px solid ${m.accent}`, padding: "4px 8px", whiteSpace: "nowrap", display: "inline-flex", gap: 4, alignItems: "center",
            transition: "all 0.2s" }}>OPEN <ArrowRight size={12} /></span>
        </div>
      </div>
    </motion.button>
  );
}

export default function LandingPage({ setCurrentView }: { setCurrentView: (v: View) => void }) {
  const [hover, setHover] = useState<string | null>(null);
  // Index of the card whose connector the scroll has most recently drawn (-1: none yet). Narrow screens show all.
  const [step, setStep] = useState(() => (window.matchMedia(WIDE).matches ? -1 : MODULES.length - 1));
  const section = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const active = MODULES.find((m) => m.id === hover) ?? MODULES[Math.max(0, step)];

  // Lagunitas-style sequence: the stage pins for two screen-heights (section ≈ 300vh with the pin spacing) while the
  // cards rise from below one by one and each connector draws from the engine to its card, all scrubbed to scroll.
  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add(WIDE, () => {
      setStep(-1);
      const q = gsap.utils.selector(stage);
      const cards = q(".pt-slide"), lines = q(".pt-line"), tips = q(".pt-tip");
      const tl = gsap.timeline({
        defaults: { ease: "none" },
        scrollTrigger: {
          trigger: stage.current, start: "top top", end: "+=200%", pin: true, scrub: true, invalidateOnRefresh: true,
          // Each card owns one unit of the timeline: slide [0, 0.6], draw [0.6, 1]. Its connector is "active" from landing on.
          onUpdate: (self) => setStep(Math.min(MODULES.length - 1, Math.floor(self.progress * MODULES.length + 0.4) - 1)),
        },
      });
      MODULES.forEach((_, i) => {
        tl.fromTo(cards[i], { y: () => window.innerHeight, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.6, ease: "power2.out" }, i)
          .fromTo(lines[i], { strokeDashoffset: 1 }, { strokeDashoffset: 0, duration: 0.35 }, i + 0.6)
          .fromTo(tips[i], { scale: 0 }, { scale: 1, duration: 0.05, transformOrigin: "50% 50%" }, i + 0.95);
      });
      return () => setStep(MODULES.length - 1);
    });
  }, { scope: section });

  return (
    <div style={{ minHeight: "100vh", background: "transparent", color: C.ink, fontFamily: "Inter, Helvetica Neue, sans-serif" }}>
      <InteractiveGrid />
      <CustomCursor />
      <style>{`
        @keyframes pt-flow { to { stroke-dashoffset: -20; } }
        .pt-flow { stroke-dasharray: 3 7; animation: pt-flow 0.9s linear infinite; }
        @media (max-width: 860px) {
          .pt-chart, .pt-hero { grid-template-columns: 1fr !important; }
          .pt-arrows, .pt-scrollhint { display: none !important; }
          .pt-engine { height: auto !important; min-height: 140px; }
          .pt-stage { height: auto !important; padding: 24px 0; }
        }
      `}</style>

      <div style={{ maxWidth: 1180, margin: "0 auto", padding: "0 16px" }}>
        <header className="pt-hero" style={{ display: "grid", gridTemplateColumns: "minmax(0, 0.85fr) minmax(0, 1.15fr)", alignItems: "center",
          gap: 24, padding: "40px 0 28px", borderBottom: `1px solid ${C.ink}` }}>
          <div>
            <motion.div {...reveal(0)} style={{ font: `600 12px ${mono}`, color: C.mute, letterSpacing: "0.1em" }}>
              CYBER-PHYSICAL INVESTIGATION · TK-101 WATER PLANT
            </motion.div>
            <motion.h1 {...reveal(0.1)} style={{ font: "900 clamp(48px, 9vw, 112px)/0.88 Inter", letterSpacing: "-0.02em", margin: "14px 0 0" }}>
              PhysTrace
            </motion.h1>
            <motion.p {...reveal(0.2)} style={{ font: "500 20px/1.45 Inter", maxWidth: 560, margin: "28px 0 0", letterSpacing: "-0.01em" }}>
              Tells a sensor attack from a sensor fault by checking every reading against physics: an actuator watermark,
              a digital twin, and an independent camera witness, all recorded for forensic replay.
            </motion.p>
            <motion.div {...reveal(0.3)} style={{ font: `600 11px ${mono}`, color: C.mute, letterSpacing: "0.1em", marginTop: 28,
              display: "flex", gap: 20, flexWrap: "wrap" }}>
              <span>01 ENGINE</span><span>02 EVIDENCE</span><span>03 VERDICT</span>
              <span className="pt-scrollhint" style={{ color: C.ink, display: "inline-flex", gap: 6, alignItems: "center" }}>
                <motion.span animate={{ y: [0, 4, 0] }} transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }} style={{ display: "inline-flex" }}>
                  <ArrowDown size={13} />
                </motion.span>
                SCROLL TO ASSEMBLE
              </span>
            </motion.div>
          </div>
          {/* No blur in this reveal: a filter on an ancestor flattens the kiosk screen's CSS 3D transform and misplaces it. */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5, ease: EASE, delay: 0.15 }} style={{ position: "relative", height: 480 }} aria-label="Interactive 3D control kiosk showing live plant data">
            <HeroScene />
            <div style={{ position: "absolute", right: 0, bottom: 0, display: "flex", gap: 6, alignItems: "center", pointerEvents: "none",
              font: `600 10px ${mono}`, letterSpacing: "0.1em", color: C.mute }}>
              <Move3d size={13} /> DRAG TO ROTATE
            </div>
          </motion.div>
        </header>

        {/* The stage is the section's only child: GSAP wraps it in a pin-spacer, and React never inserts siblings next to it. */}
        <section ref={section} aria-label="System architecture">
          <div ref={stage} className="pt-stage" style={{ height: "100vh", display: "flex", flexDirection: "column", justifyContent: "center", gap: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", font: `700 11px ${mono}`, letterSpacing: "0.1em" }}>
              <span style={{ color: C.mute }}>SYSTEM ARCHITECTURE · ENGINE → EVIDENCE MODULES</span>
              <span style={{ color: step >= 0 ? MODULES[step].accent : C.mute }}>{String(step + 1).padStart(2, "0")} / 03</span>
            </div>

            <div className="pt-chart" style={{ display: "grid", gridTemplateColumns: `minmax(220px, 1fr) ${ARROW_W}px minmax(280px, 1.4fr)`,
              alignItems: "stretch" }}>
              <div className="pt-engine"
                style={{ height: STACK_H, background: C.twin, color: "#fff", padding: 24, boxShadow: shadow, display: "flex",
                  flexDirection: "column", justifyContent: "space-between", border: `1px solid ${C.twin}` }}>
                <Cpu size={28} />
                <div>
                  <div style={{ font: `600 11px ${mono}`, opacity: 0.8, letterSpacing: "0.1em" }}>00 / CORE</div>
                  <div style={{ font: "900 30px/1.02 Inter", letterSpacing: "-0.02em", marginTop: 8 }}>PhysTrace<br />Investigation<br />Engine</div>
                </div>
              </div>

              <svg className="pt-arrows" width={ARROW_W} height={STACK_H} viewBox={`0 0 ${ARROW_W} ${STACK_H}`} style={{ display: "block", overflow: "visible" }}>
                {MODULES.map((m, i) => {
                  const on = (hover ?? MODULES[step]?.id) === m.id;
                  const y0 = STACK_H / 2, y1 = rowMid(i), mx = ARROW_W / 2;
                  const d = `M0,${y0} H${mx} V${y1} H${ARROW_W - 4}`;
                  return (
                    <g key={m.id} opacity={on ? 1 : 0.45} style={{ transition: "opacity 0.3s" }}>
                      {/* pathLength=1 lets GSAP draw the line by scrubbing the dash offset from 1 to 0. */}
                      <path className="pt-line" d={d} pathLength={1} strokeDasharray={1} fill="none" stroke={m.accent} strokeWidth={on ? 3 : 1.5} />
                      {on && i <= step && <path d={d} fill="none" stroke="#fff" strokeWidth={1.5} className="pt-flow" />}
                      <circle className="pt-tip" cx={ARROW_W - 4} cy={y1} r={5} fill={m.accent} />
                    </g>
                  );
                })}
              </svg>

              <div style={{ display: "grid", gridTemplateRows: `repeat(3, ${NODE_H}px)`, gap: GAP }}>
                {MODULES.map((m, i) => (
                  // GSAP moves this wrapper; Framer Motion owns the hover transform on the card inside, so they never fight.
                  <div key={m.id} className="pt-slide" style={{ display: "grid" }}>
                    <ShinyCard m={m} i={i} on={hover === m.id} onHover={(h) => setHover(h ? m.id : null)} onOpen={() => setCurrentView(m.view)} />
                  </div>
                ))}
              </div>
            </div>

            <div style={{ minHeight: 64, paddingTop: 16, borderTop: `1px solid ${C.line}` }}>
              <motion.div key={step < 0 && !hover ? "none" : active.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} style={{ maxWidth: 720 }}>
                {step < 0 && !hover ? (
                  <div style={{ font: `600 11px ${mono}`, color: C.mute, letterSpacing: "0.1em" }}>KEEP SCROLLING · EACH EVIDENCE MODULE CONNECTS TO THE ENGINE IN TURN</div>
                ) : (
                  <>
                    <div style={{ font: `700 11px ${mono}`, color: active.accent, letterSpacing: "0.1em" }}>{active.title.toUpperCase()}</div>
                    <div style={{ font: "400 15px/1.55 Inter", color: C.ink, marginTop: 6 }}>{active.body}</div>
                  </>
                )}
              </motion.div>
            </div>
          </div>
        </section>

        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 24, alignItems: "center",
          margin: "40px 0 64px", paddingTop: 24, borderTop: `1px solid ${C.ink}` }}>
          <div style={{ font: `600 12px/1.6 ${mono}`, color: C.mute, letterSpacing: "0.1em" }}>
            ALL THREE WITNESSES FEED ONE VERDICT · OPEN THE LIVE DASHBOARD
          </div>
          <motion.button onClick={() => setCurrentView("dashboard")} whileHover={{ y: -2 }} whileTap={{ y: 0 }}
            style={{ all: "unset", cursor: "pointer", background: C.ink, color: "#fff", padding: "18px 26px", boxShadow: shadow,
              font: "800 15px Inter", letterSpacing: "0.02em", display: "flex", gap: 12, alignItems: "center" }}>
            Launch Dashboard <ArrowRight size={18} />
          </motion.button>
        </div>
      </div>
    </div>
  );
}
