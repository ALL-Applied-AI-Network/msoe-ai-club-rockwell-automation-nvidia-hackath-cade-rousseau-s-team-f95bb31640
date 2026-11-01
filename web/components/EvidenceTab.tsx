"use client";

import Image from "next/image";
import { useState } from "react";

import { type Frame, imageUrl, type RGB, type Tanks, useApi } from "@/lib/api";

import { Banner, button, Icon, Note, Panel, rgbCss } from "./ui";

const VIDEOS = [1, 2, 3, 4, 5];

/** A tank drawn with its camera color at its measured level. */
function TankGauge({ name, rgb, level }: { name: string; rgb: RGB | null; level: number | null }) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-16 w-10 overflow-hidden rounded-md border-2 border-slate-300 bg-bg">
        {rgb && <div className="absolute inset-x-0 bottom-0" style={{ height: `${level ?? 60}%`, background: rgbCss(rgb) }} />}
      </div>
      <div>
        <div className="text-sm font-medium">{name}</div>
        <div className="text-xs text-muted">
          {name === "T300" ? "out of service" : level === null ? "level not visible" : `${level.toFixed(0)}% full`}
        </div>
      </div>
    </div>
  );
}

export default function EvidenceTab() {
  const [video, setVideo] = useState(5);
  const [index, setIndex] = useState(0);
  const [tanksAt, setTanksAt] = useState<number | null>(null);   // frame whose tanks were requested
  const { data: frames, error } = useApi<Frame[]>(`/api/videos/${video}/frames`);
  const frame = frames?.[Math.min(index, frames.length - 1)];
  const tanks = useApi<Tanks>(frame && tanksAt === frame.index ? `/api/videos/${video}/frames/${frame.index}/tanks` : null);

  if (error) return <Banner kind="FAIL">Could not load video {video}: {error}</Banner>;

  return (
    <div className="space-y-5">
      <Panel>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex rounded-md border border-line p-0.5" role="group" aria-label="Video">
            {VIDEOS.map((v) => (
              <button
                key={v}
                onClick={() => {
                  setVideo(v);
                  setIndex(0);
                }}
                className={`rounded px-3 py-1.5 text-sm ${video === v ? "bg-ink text-white" : "text-muted hover:text-ink"}`}
              >
                Video {v}
              </button>
            ))}
          </div>
          {frames && frame && (
            <div className="flex min-w-72 flex-1 items-center gap-2">
              <button className={button.icon} disabled={frame.index === 0} onClick={() => setIndex(frame.index - 1)} aria-label="Previous frame"><Icon name="left" /></button>
              <input type="range" className="flex-1 accent-accent" min={0} max={frames.length - 1} value={frame.index} onChange={(e) => setIndex(Number(e.target.value))} aria-label="Frame" />
              <button className={button.icon} disabled={frame.index === frames.length - 1} onClick={() => setIndex(frame.index + 1)} aria-label="Next frame"><Icon name="right" /></button>
              <span className="w-24 text-right text-sm text-muted tabular-nums">{frame.index + 1} / {frames.length}</span>
            </div>
          )}
        </div>
        {frame && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            <span className="tabular-nums">{new Date(frame.timestamp).toLocaleString()}</span>
            <span className="rounded-md bg-bg px-2 py-0.5 text-xs text-muted">
              {frame.in_fill === null ? "no labels for this video" : frame.in_fill ? `label: filling vial ${frame.vial}, Recipe ${frame.recipe_id}` : "label: between fills"}
            </span>
            {frame.in_fill !== null && <span className="text-xs text-faint">Still labels run about one vial (~50 s) behind the camera.</span>}
          </div>
        )}
      </Panel>

      {!frame && <Note>Loading frames…</Note>}
      {frame && (
        <>
          <div className="grid gap-5 md:grid-cols-2">
            <Panel title="Fill station">
              <Image src={imageUrl("vial_fill", frame.vial_fill)} alt="Vial fill camera" width={640} height={480} unoptimized className="h-auto w-full rounded-md" />
            </Panel>
            <Panel title="Tanks">
              <Image src={imageUrl("tank_level", frame.tank_level)} alt="Tank camera" width={640} height={480} unoptimized className="h-auto w-full rounded-md" />
            </Panel>
          </div>
          <Panel
            title="Tank reading"
            subtitle="Color and level of each tank, measured from this tank-camera frame."
            action={
              tanksAt !== frame.index && (
                <button className={button.primary} onClick={() => setTanksAt(frame.index)}>Read tanks</button>
              )
            }
          >
            {tanksAt !== frame.index && <Note>Press Read tanks to measure this frame.</Note>}
            {tanks.loading && <Note>Reading tanks…</Note>}
            {tanks.error && <Banner kind="FAIL">{tanks.error}</Banner>}
            {tanks.data && (
              <div className="grid grid-cols-2 gap-5 md:grid-cols-4">
                {Object.entries(tanks.data.fill_percent).map(([tank, fill]) => (
                  <TankGauge key={tank} name={tank} rgb={tanks.data!.colors[tank] ?? null} level={fill} />
                ))}
              </div>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
