import type { ReactNode } from "react";

import type { RGB } from "@/lib/api";

// Illustrative tank dye colors for composition bars (the dataset's labels: clear, orange, blue).
export const TANK_COLORS: Record<string, string> = { T100: "#d4dde4", T200: "#e07a2c", T400: "#2d6fb3" };

export const rgbCss = (rgb: RGB) => `rgb(${rgb.map(Math.round).join(" ")})`;

/**
 * The camera sees these liquids as very dark (Recipe 7's clean blue is RGB 1, 31, 46), so raw
 * swatches all look black. Display colors get a gamma lift that keeps hue and lightness order;
 * every check and number still uses the raw camera values.
 */
export const displayRgb = (rgb: RGB): RGB => rgb.map((c) => 255 * Math.pow(Math.max(0, c) / 255, 0.45)) as RGB;

export const DISPLAY_NOTE = "Colors are brightened for display. Checks use the raw camera values.";

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-surface ${className}`}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
          <div>
            {title && <h2 className="text-[15px] font-semibold">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

const VIAL_SIZES = {
  xs: "h-7 w-3.5 border",
  sm: "h-11 w-5 border-2",
  lg: "h-28 w-12 border-[3px]",
};

/** A vial drawn as a tube, filled with the display version of a camera color. */
export function Vial({
  rgb,
  size = "sm",
  level = 0.8,
  title,
}: {
  rgb: RGB | null;
  size?: keyof typeof VIAL_SIZES;
  level?: number;
  title?: string;
}) {
  return (
    <div
      className={`relative shrink-0 overflow-hidden rounded-t-[3px] rounded-b-full border-slate-300 bg-white ${VIAL_SIZES[size]}`}
      title={title}
    >
      {rgb ? (
        <div className="absolute inset-x-0 bottom-0" style={{ height: `${level * 100}%`, background: rgbCss(displayRgb(rgb)) }} />
      ) : (
        <div className="absolute inset-x-0 bottom-0 h-4/5 bg-[repeating-linear-gradient(45deg,#eef0f2_0_4px,#fff_4px_8px)]" />
      )}
    </div>
  );
}

const STATUS = {
  PASS: { label: "Matches recipe", dot: "bg-pass", text: "text-pass", soft: "bg-pass-soft" },
  FAIL: { label: "Off color", dot: "bg-fail", text: "text-fail", soft: "bg-fail-soft" },
  PENDING: { label: "Not checked", dot: "bg-faint", text: "text-muted", soft: "bg-bg" },
} as const;

export type Status = keyof typeof STATUS;

export function StatusPill({ status }: { status: Status }) {
  const s = STATUS[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${s.soft} ${s.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}

export const statusStyle = (status: Status) => STATUS[status];

/** A value against a limit, e.g. color difference 11.0 with the off-color line at 10. */
export function Meter({ value, limit, max }: { value: number; limit: number; max: number }) {
  const at = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  const over = value >= limit;
  return (
    <div>
      <div className="relative h-2 rounded-full bg-bg">
        <div className={`absolute inset-y-0 left-0 rounded-full ${over ? "bg-fail" : "bg-pass"}`} style={{ width: at(value) }} />
        <div className="absolute -inset-y-1 w-0.5 bg-ink" style={{ left: at(limit) }} />
      </div>
      <div className="relative mt-1.5 h-4 text-xs text-muted">
        <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: at(limit) }}>
          limit {limit}
        </span>
      </div>
    </div>
  );
}

/** mL per tank as a stacked bar, in dispense order. */
export function Composition({ ml, compact = false }: { ml: Record<string, number>; compact?: boolean }) {
  const total = Object.values(ml).reduce((a, b) => a + b, 0);
  const order = ["T200", "T400", "T100"].filter((t) => t in ml);
  return (
    <div className="w-full">
      <div className={`flex overflow-hidden rounded-full bg-bg ${compact ? "h-1.5" : "h-2"}`}>
        {order.map((t) => (
          <div key={t} style={{ width: `${(ml[t] / total) * 100}%`, background: TANK_COLORS[t] }} />
        ))}
      </div>
      {!compact && <div className="mt-1 text-xs text-muted tabular-nums">{order.map((t) => `${t} ${ml[t]} mL`).join(" · ")}</div>}
    </div>
  );
}

export const describeRecipe = (ml: Record<string, number>) =>
  ["T200", "T400", "T100"].filter((t) => t in ml).map((t) => `${t} ${ml[t]} mL`).join(" + ");

export function Banner({ kind, children }: { kind: "PASS" | "FAIL" | "PENDING"; children: ReactNode }) {
  const tone = { PASS: "border-pass bg-pass-soft", FAIL: "border-fail bg-fail-soft", PENDING: "border-line bg-surface" }[kind];
  return <div className={`rounded-lg border-l-4 px-4 py-3 text-sm ${tone}`}>{children}</div>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}

const ICONS = {
  up: "M12 19V5m0 0-6 6m6-6 6 6",
  down: "M12 5v14m0 0 6-6m-6 6-6-6",
  remove: "M6 6l12 12M18 6 6 18",
  left: "M15 18l-6-6 6-6",
  right: "M9 6l6 6-6 6",
};

export function Icon({ name, className = "h-4 w-4" }: { name: keyof typeof ICONS; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
      <path d={ICONS[name]} />
    </svg>
  );
}

export const button = {
  primary: "inline-flex items-center justify-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white hover:bg-accent/90 disabled:opacity-40",
  secondary: "inline-flex items-center justify-center gap-2 rounded-md border border-line bg-surface px-3 py-2 text-sm font-medium hover:bg-bg disabled:opacity-40",
  icon: "rounded p-1.5 text-muted hover:bg-bg hover:text-ink disabled:opacity-25",
};
