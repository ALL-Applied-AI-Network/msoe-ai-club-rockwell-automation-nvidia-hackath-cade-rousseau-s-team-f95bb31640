"use client";

import Image from "next/image";
import { useState } from "react";

import { type Frame, imageUrl, type Tanks, useApi } from "@/lib/api";

import { Banner, Card, Note, Swatch } from "./ui";

const VIDEOS = [1, 2, 3, 4, 5];

export default function EvidenceTab() {
  const [video, setVideo] = useState(5);
  const [index, setIndex] = useState(0);
  const [tanksAt, setTanksAt] = useState<number | null>(null);   // frame whose tanks were requested
  const { data: frames, error } = useApi<Frame[]>(`/api/videos/${video}/frames`);
  const frame = frames?.[Math.min(index, frames.length - 1)];
  const tanks = useApi<Tanks>(frame && tanksAt === frame.index ? `/api/videos/${video}/frames/${frame.index}/tanks` : null);

  if (error) return <Banner kind="FAIL">Could not load video {video}: {error}</Banner>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded border border-line bg-white px-3 py-1.5"
          value={video}
          onChange={(e) => {
            setVideo(Number(e.target.value));
            setIndex(0);
          }}
        >
          {VIDEOS.map((v) => (
            <option key={v} value={v}>
              Video {v}{v === 5 ? " (labeled)" : ""}
            </option>
          ))}
        </select>
        {frames && (
          <>
            <input type="range" className="min-w-64 flex-1 accent-accent" min={0} max={frames.length - 1} value={frame?.index ?? 0} onChange={(e) => setIndex(Number(e.target.value))} />
            <span className="w-28 text-sm text-muted">frame {(frame?.index ?? 0) + 1} / {frames.length}</span>
          </>
        )}
      </div>

      {!frame && <Note>Loading frames…</Note>}
      {frame && (
        <>
          <p className="text-sm">
            {new Date(frame.timestamp).toLocaleString()} ·{" "}
            {frame.in_fill === null ? "unlabeled" : frame.in_fill ? `labeled: filling vial ${frame.vial} (Recipe ${frame.recipe_id})` : "labeled: between fills"}
          </p>
          {frame.in_fill !== null && (
            <Note>The still-image labels run about one vial behind the camera (~50 s), so the label may name the next vial.</Note>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Vial fill camera">
              <Image src={imageUrl("vial_fill", frame.vial_fill)} alt="Vial fill camera" width={640} height={480} unoptimized className="h-auto w-full rounded" />
            </Card>
            <Card title="Tank camera">
              <Image src={imageUrl("tank_level", frame.tank_level)} alt="Tank camera" width={640} height={480} unoptimized className="h-auto w-full rounded" />
            </Card>
          </div>
          <Card title="Tank reading">
            {tanksAt !== frame.index && (
              <button className="rounded bg-accent px-4 py-1.5 text-sm font-semibold text-white" onClick={() => setTanksAt(frame.index)}>
                Read tanks in this frame
              </button>
            )}
            {tanks.loading && <Note>Reading tanks…</Note>}
            {tanks.error && <Banner kind="FAIL">{tanks.error}</Banner>}
            {tanks.data && (
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                {Object.entries(tanks.data.fill_percent).map(([tank, fill]) => (
                  <div key={tank} className="space-y-1">
                    <div className="font-medium">{tank}</div>
                    <Swatch rgb={tanks.data!.colors[tank] ?? null} size="sm" label={tanks.data!.colors[tank] ? "camera color" : "not read (out of service)"} />
                    <div className="text-sm text-muted">{fill === null ? "level: no liquid line" : `level: ${fill.toFixed(0)}%`}</div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
