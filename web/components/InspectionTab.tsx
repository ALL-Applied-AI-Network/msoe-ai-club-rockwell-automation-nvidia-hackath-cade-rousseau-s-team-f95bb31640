"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import { type Inspection, type Recipe, useApi } from "@/lib/api";

import { Banner, button, DISPLAY_NOTE, describeRecipe, Icon, Meter, Note, Panel, StatusPill, statusStyle, Vial } from "./ui";

const COLOR_LIMIT = 10;   // contracts.COLOR_WARN_DE

export default function InspectionTab() {
  const { data: vials, error } = useApi<Inspection[]>("/api/vials");
  const { data: recipes } = useApi<Recipe[]>("/api/recipes");
  const [selected, setSelected] = useState(7);

  // Left/right arrow keys step through vials.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowLeft") setSelected((v) => Math.max(1, v - 1));
      if (e.key === "ArrowRight") setSelected((v) => Math.min(vials?.length ?? 30, v + 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [vials?.length]);

  if (error) return <Banner kind="FAIL">Could not load the vials: {error}</Banner>;
  if (!vials) return <Note>Checking all 30 vials…</Note>;

  const batches: Inspection[][] = [];
  for (const v of vials) {
    const last = batches.at(-1);
    if (last && last[0].recipe_id === v.recipe_id) last.push(v);
    else batches.push([v]);
  }
  const offColor = vials.filter((v) => v.status === "FAIL").length;
  const r = vials.find((v) => v.vial === selected) ?? vials[0];

  return (
    <div className="space-y-5">
      <Panel
        title="Run 5 · 30 vials"
        subtitle={
          <>
            <span className="font-medium text-fail">{offColor} off color</span>, each the first vial after a recipe change. Click a vial, or use ← →.
          </>
        }
      >
        <div className="grid grid-cols-5 gap-x-2 gap-y-4 lg:grid-cols-10">
          {batches.map((batch) => (
            <div key={batch[0].vial} className="min-w-0">
              <div className="mb-1.5 text-center text-xs font-medium text-muted">Recipe {batch[0].recipe_id}</div>
              <div className="flex justify-center gap-0.5">
                {batch.map((v) => (
                  <button
                    key={v.vial}
                    onClick={() => setSelected(v.vial)}
                    aria-label={`Vial ${v.vial}, ${statusStyle(v.status).label}`}
                    aria-pressed={v.vial === r.vial}
                    className={`flex flex-col items-center gap-1 rounded-md px-1 pb-1 pt-1.5 ${
                      v.vial === r.vial ? "bg-accent-soft ring-2 ring-accent" : "hover:bg-bg"
                    }`}
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${v.status === "FAIL" ? "bg-fail" : "bg-transparent"}`} />
                    <Vial rgb={v.observed_rgb} size="sm" title={`Vial ${v.vial}`} />
                    <span className={`text-[11px] tabular-nums ${v.status === "FAIL" ? "font-semibold text-fail" : "text-muted"}`}>{v.vial}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <Panel title={`Vial ${r.vial} at the fill station`} subtitle="Video 5, at the end of this vial's fill">
          <Image src={r.image_url} alt={`Vial ${r.vial} at the fill station`} width={640} height={480} unoptimized className="h-auto w-full rounded-md" />
        </Panel>

        <Panel
          title={
            <span className="flex items-center gap-3">
              Vial {r.vial} <StatusPill status={r.status} />
            </span>
          }
          action={
            <div className="flex gap-1">
              <button className={button.icon} disabled={r.vial <= 1} onClick={() => setSelected(r.vial - 1)} aria-label="Previous vial">
                <Icon name="left" />
              </button>
              <button className={button.icon} disabled={r.vial >= vials.length} onClick={() => setSelected(r.vial + 1)} aria-label="Next vial">
                <Icon name="right" />
              </button>
            </div>
          }
        >
          <div className="space-y-5">
            <div>
              <div className="font-medium">Recipe {r.recipe_id}</div>
              <div className="text-sm text-muted">
                {recipes && describeRecipe(recipes.find((x) => x.recipe_id === r.recipe_id)?.ml ?? {})}
              </div>
              {r.prev_recipe_id !== null && r.prev_recipe_id !== r.recipe_id && (
                <div className="mt-2 inline-block rounded-md bg-warn-soft px-2 py-1 text-xs font-medium text-warn">
                  First vial after Recipe {r.prev_recipe_id}
                </div>
              )}
            </div>

            <div className="flex items-end gap-8">
              <figure className="flex flex-col items-center gap-2">
                <Vial rgb={r.expected_rgb} size="lg" />
                <figcaption className="text-xs text-muted">Should look like</figcaption>
              </figure>
              <figure className="flex flex-col items-center gap-2">
                <Vial rgb={r.observed_rgb} size="lg" />
                <figcaption className="text-xs text-muted">This vial</figcaption>
              </figure>
              {r.delta_e !== null && (
                <div className="flex-1 pb-6">
                  <div className="mb-2 flex items-baseline justify-between">
                    <span className="text-sm">Color difference</span>
                    <span className={`text-xl font-semibold tabular-nums ${r.status === "FAIL" ? "text-fail" : "text-pass"}`}>{r.delta_e.toFixed(1)}</span>
                  </div>
                  <Meter value={r.delta_e} limit={COLOR_LIMIT} max={25} />
                </div>
              )}
            </div>

            {r.cause ? (
              <div className="rounded-md border border-fail/30 bg-fail-soft p-3.5">
                <div className="mb-1 text-sm font-semibold text-fail">Likely cause</div>
                <p className="text-sm leading-relaxed">{r.cause}</p>
                <p className="mt-2 text-sm font-medium">Set this vial aside for review.</p>
              </div>
            ) : (
              r.status === "PASS" && <p className="text-sm text-pass">Within the color limit for Recipe {r.recipe_id}. No action needed.</p>
            )}

            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-line pt-4 text-sm">
              <dt className="text-muted">Target fill</dt>
              <dd className="tabular-nums">{r.target_ml} mL</dd>
              <dt className="text-muted">Measured fill</dt>
              <dd>{r.fill_ml !== null ? `${r.fill_ml.toFixed(0)} mL` : <span className="text-muted">not visible on this camera</span>}</dd>
              {r.next_recipe_id !== null && (
                <>
                  <dt className="text-muted">Lowest-risk next recipe</dt>
                  <dd>Recipe {r.next_recipe_id}</dd>
                </>
              )}
            </dl>
            <p className="text-xs text-faint">{DISPLAY_NOTE}</p>
          </div>
        </Panel>
      </div>
    </div>
  );
}
