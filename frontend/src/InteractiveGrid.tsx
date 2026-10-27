import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const CELL = 40; // matches the body's 40px grid lines, so lit cells sit exactly inside them

const fit = () => ({ cols: Math.ceil(window.innerWidth / CELL), rows: Math.ceil(window.innerHeight / CELL) });

/**
 * Screen-filling layer of borderless cells. The cell under the pointer lights to #E2E8F0 and fades back over 500ms,
 * leaving a trail. The page content sits on top of this layer, so a CSS :hover on the cells would never fire;
 * the hovered cell is instead found from the pointer position and given the .on class (styles in index.css).
 */
export default function InteractiveGrid() {
  const ref = useRef<HTMLDivElement>(null);
  const [dim, setDim] = useState(fit);

  useEffect(() => {
    const resize = () => setDim(fit());
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    let last: Element | null = null;
    const light = (cell: Element | null) => {
      if (cell === last) return;
      last?.classList.remove("on");
      cell?.classList.add("on");
      last = cell;
    };
    const move = (e: PointerEvent) => light(ref.current?.children[Math.floor(e.clientY / CELL) * dim.cols + Math.floor(e.clientX / CELL)] ?? null);
    const leave = () => light(null);
    window.addEventListener("pointermove", move, { passive: true });
    document.documentElement.addEventListener("pointerleave", leave);
    return () => {
      window.removeEventListener("pointermove", move);
      document.documentElement.removeEventListener("pointerleave", leave);
      light(null);
    };
  }, [dim.cols]);

  return createPortal(
    <div ref={ref} aria-hidden style={{ position: "fixed", inset: 0, zIndex: -5, pointerEvents: "none", overflow: "hidden",
      display: "grid", gridTemplateColumns: `repeat(${dim.cols}, ${CELL}px)`, gridAutoRows: `${CELL}px` }}>
      {Array.from({ length: dim.cols * dim.rows }, (_, i) => <div key={i} className="pt-cell" />)}
    </div>,
    document.body,
  );
}
