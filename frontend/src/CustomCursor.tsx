import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { motion, useMotionValue, useSpring } from "framer-motion";

const FINE = typeof window !== "undefined" && window.matchMedia("(pointer: fine)").matches;

/**
 * 15px dot that trails the mouse on a spring. It is white with mix-blend-mode: difference, which renders it black
 * on the white page and inverted over dark surfaces (a black dot under "difference" would be invisible).
 * Portalled to <body> so no transformed ancestor can capture its position: fixed. Touch devices keep their default.
 */
export default function CustomCursor() {
  const x = useMotionValue(-100), y = useMotionValue(-100);
  const sx = useSpring(x, { stiffness: 400, damping: 28, mass: 0.3 });
  const sy = useSpring(y, { stiffness: 400, damping: 28, mass: 0.3 });
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (!FINE) return;
    const root = document.documentElement;
    root.classList.add("pt-nocursor");
    const move = (e: PointerEvent) => { x.set(e.clientX); y.set(e.clientY); setShown(true); };
    const leave = () => setShown(false);
    window.addEventListener("pointermove", move, { passive: true });
    root.addEventListener("pointerleave", leave);
    return () => {
      root.classList.remove("pt-nocursor");
      window.removeEventListener("pointermove", move);
      root.removeEventListener("pointerleave", leave);
    };
  }, [x, y]);

  if (!FINE) return null;
  return createPortal(
    <motion.div aria-hidden style={{ position: "fixed", left: 0, top: 0, width: 15, height: 15, borderRadius: "50%", background: "#fff",
      mixBlendMode: "difference", pointerEvents: "none", zIndex: 9999, x: sx, y: sy, translateX: "-50%", translateY: "-50%",
      opacity: shown ? 1 : 0, transition: "opacity 0.2s" }} />,
    document.body,
  );
}
