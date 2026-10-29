"use client";

import Image from "next/image";
import { useState } from "react";

import { type Inspection, useApi } from "@/lib/api";

import { Banner, Card, Metric, Note, Swatch } from "./ui";

const VIALS = Array.from({ length: 30 }, (_, i) => i + 1);   // labeled vials of video 5

export default function InspectionTab() {
  const [vial, setVial] = useState(7);
  const { data: r, error, loading } = useApi<Inspection>(`/api/vials/${vial}/inspection`);
  const changeover = r && r.prev_recipe_id !== null && r.prev_recipe_id !== r.recipe_id;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <button className="rounded border border-line bg-white px-3 py-1.5 disabled:opacity-40" disabled={vial <= 1} onClick={() => setVial(vial - 1)}>
          ‹ Prev
        </button>
        <select className="rounded border border-line bg-white px-3 py-1.5" value={vial} onChange={(e) => setVial(Number(e.target.value))}>
          {VIALS.map((v) => (
            <option key={v} value={v}>
              Vial {v}
            </option>
          ))}
        </select>
        <button className="rounded border border-line bg-white px-3 py-1.5 disabled:opacity-40" disabled={vial >= 30} onClick={() => setVial(vial + 1)}>
          Next ›
        </button>
        <span className="ml-2 text-sm text-muted">Video 5, the labeled run</span>
      </div>

      {error && <Banner kind="FAIL">Could not load vial {vial}: {error}</Banner>}
      {loading && <Note>Loading vial {vial}…</Note>}

      {r && (
        <>
          {r.status === "PASS" && <Banner kind="PASS">✓ Inspection passed</Banner>}
          {r.status === "FAIL" && <Banner kind="FAIL">⚠ Inspection flagged — review required</Banner>}
          {r.status === "PENDING" && (
            <Banner kind="PENDING">Model not connected yet — showing labels and reference colors only</Banner>
          )}

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric label="Current recipe" value={`Recipe ${r.recipe_id}`} />
            <Metric label="Previous vial's recipe" value={r.prev_recipe_id ? `Recipe ${r.prev_recipe_id}` : "—"} />
            <Metric label="Vial" value={<>{r.vial}{changeover && <span className="ml-2 text-sm font-normal text-flag-ink">changeover</span>}</>} />
            <Metric label="Inspection" value={r.status} />
          </div>

          <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
            <Card title="Vial camera">
              <Image src={r.image_url} alt={`Vial ${r.vial} at the fill station`} width={640} height={480} unoptimized className="h-auto w-full rounded" />
            </Card>
            <div className="space-y-4">
              <Card title="Color">
                <div className="space-y-3">
                  <Swatch rgb={r.expected_rgb} label={r.expected_rgb ? `Expected: Recipe ${r.recipe_id} clean color` : "Expected: no reference yet"} />
                  <Swatch rgb={r.observed_rgb} label={r.observed_rgb ? "Observed by the camera" : "Observed: not measured yet"} />
                  {r.delta_e !== null && <p className="text-sm">Color difference ΔE {r.delta_e.toFixed(1)} (off-color at 10)</p>}
                </div>
              </Card>
              <Card title="Fill">
                <p>{r.fill_ml !== null ? `${r.fill_ml.toFixed(0)} mL measured` : "Not measured yet"} / {r.target_ml} mL target</p>
              </Card>
              <Card title="Likely cause and next step">
                {r.cause ? <p>{r.cause}</p> : <Note>No explanation from the model yet.</Note>}
                {r.next_recipe_id !== null && <p className="mt-2 text-sm">Suggested next recipe: {r.next_recipe_id}</p>}
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
