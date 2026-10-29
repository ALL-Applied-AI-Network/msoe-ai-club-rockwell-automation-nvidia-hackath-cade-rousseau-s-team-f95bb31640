import type { ReactNode } from "react";

import type { RGB } from "@/lib/api";

// Illustrative tank dye colors for composition bars (the dataset's labels: clear, orange, blue).
export const TANK_COLORS: Record<string, string> = { T100: "#d9e2e8", T200: "#e07a2c", T400: "#2d6fb3" };

export const rgbCss = (rgb: RGB) => `rgb(${rgb.map(Math.round).join(" ")})`;

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded border border-line bg-white p-4 ${className}`}>
      {title && <h3 className="mb-3 text-xs font-semibold tracking-wide text-muted uppercase">{title}</h3>}
      {children}
    </section>
  );
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded border border-line bg-white px-4 py-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
    </div>
  );
}

export function Swatch({ rgb, label, size = "lg" }: { rgb: RGB | null; label?: string; size?: "sm" | "lg" }) {
  const box = size === "lg" ? "h-20 w-20" : "h-6 w-6";
  return (
    <div className="flex items-center gap-3">
      <div
        className={`${box} shrink-0 rounded border border-line ${rgb ? "" : "bg-[repeating-linear-gradient(45deg,#eef2f6_0_6px,#fff_6px_12px)]"}`}
        style={rgb ? { background: rgbCss(rgb) } : undefined}
        title={rgb ? `RGB ${rgb.map(Math.round).join(", ")}` : "not available"}
      />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}

/** mL per tank as a stacked bar, in dispense order. */
export function Composition({ ml }: { ml: Record<string, number> }) {
  const total = Object.values(ml).reduce((a, b) => a + b, 0);
  const order = ["T200", "T400", "T100"].filter((t) => t in ml);
  return (
    <div className="w-full">
      <div className="flex h-3 overflow-hidden rounded border border-line">
        {order.map((t) => (
          <div key={t} style={{ width: `${(ml[t] / total) * 100}%`, background: TANK_COLORS[t] }} />
        ))}
      </div>
      <div className="mt-1 text-xs text-muted">{order.map((t) => `${t} ${ml[t]} mL`).join(" · ")}</div>
    </div>
  );
}

const BANNERS = {
  PASS: "border-pass bg-pass-soft text-pass-ink",
  FAIL: "border-flag bg-flag-soft text-flag-ink",
  PENDING: "border-line bg-white text-muted",
};

export function Banner({ kind, children }: { kind: keyof typeof BANNERS; children: ReactNode }) {
  return <div className={`rounded border-l-[6px] px-4 py-3 font-semibold ${BANNERS[kind]}`}>{children}</div>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}
