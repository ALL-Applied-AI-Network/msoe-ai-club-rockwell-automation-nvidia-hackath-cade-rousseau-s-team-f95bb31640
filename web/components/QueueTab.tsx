"use client";

import { useState } from "react";

import { postSchedule, type Recipe, type Schedule, type Transition, useApi } from "@/lib/api";

import { Banner, button, Composition, DISPLAY_NOTE, Icon, Note, Panel, Vial } from "./ui";

const MAX_QUEUE = 12;   // matches api/main.py: the exact optimizer is exponential in queue length

function Chip({ recipe, dim = false }: { recipe: Recipe; dim?: boolean }) {
  return (
    <span className={`inline-flex shrink-0 items-center gap-1.5 rounded-md border border-line bg-surface px-2 py-1 text-sm tabular-nums ${dim ? "opacity-60" : ""}`}>
      <Vial rgb={recipe.clean_rgb} size="xs" />
      {recipe.recipe_id}
    </span>
  );
}

/** A run order as recipe chips, with each changeover colored by its predicted risk. */
function Sequence({ order, transitions, byId, maxCost }: { order: number[]; transitions: Transition[]; byId: Record<number, Recipe>; maxCost: number }) {
  return (
    <div className="flex flex-wrap items-center gap-y-2">
      {order.map((id, i) => {
        const t = transitions[i - 1];
        const share = t?.cost != null && maxCost > 0 ? t.cost / maxCost : 0;
        const tone = share > 0.66 ? "bg-fail" : share > 0.33 ? "bg-warn" : "bg-slate-300";
        return (
          <span key={i} className="flex items-center">
            {i > 0 && <span className={`mx-1 h-1 w-5 rounded-full ${tone}`} title={`${t?.prev_recipe_id} → ${t?.recipe_id}: risk ${t?.cost?.toFixed(2)}`} />}
            <Chip recipe={byId[id]} />
          </span>
        );
      })}
    </div>
  );
}

const pct = (x: number | null) => (x === null ? "—" : `${Math.abs(x * 100).toFixed(1)}%`);

export default function QueueTab() {
  const { data: recipes, error } = useApi<Recipe[]>("/api/recipes");
  const [queue, setQueue] = useState<number[] | null>(null);
  const [plan, setPlan] = useState<{ schedule?: Schedule; error?: string; busy?: boolean }>({});

  if (error) return <Banner kind="FAIL">Could not load recipes: {error}</Banner>;
  if (!recipes) return <Note>Loading recipes…</Note>;

  const byId = Object.fromEntries(recipes.map((r) => [r.recipe_id, r]));
  const order = queue ?? recipes.map((r) => r.recipe_id);   // default: the order video 5 ran
  const update = (next: number[]) => {
    setQueue(next);
    setPlan({});
  };
  const move = (i: number, step: -1 | 1) => {
    const next = [...order];
    [next[i], next[i + step]] = [next[i + step], next[i]];
    update(next);
  };
  const optimize = () => {
    setPlan({ busy: true });
    postSchedule(order).then(
      (schedule) => setPlan({ schedule }),
      (err: Error) => setPlan({ error: err.message }),
    );
  };

  const s = plan.schedule;
  const maxCost = s ? Math.max(0, ...[...s.transitions, ...s.suggested_transitions].map((t) => t.cost ?? 0)) : 0;
  const better = (s?.transition_sum_reduction ?? 0) >= 0;

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1.2fr]">
      <Panel title="Queue" subtitle="Batches run top to bottom. The first one stays first.">
        <ol className="divide-y divide-line">
          {order.map((id, i) => (
            <li key={`${id}-${i}`} className="group flex items-center gap-3 py-1.5">
              <span className="w-5 text-right text-sm text-faint tabular-nums">{i + 1}</span>
              <Vial rgb={byId[id].clean_rgb} size="xs" />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium">Recipe {id}</div>
                <Composition ml={byId[id].ml} compact />
              </div>
              <div className="flex opacity-60 group-hover:opacity-100">
                <button className={button.icon} disabled={i === 0} onClick={() => move(i, -1)} aria-label="Move up"><Icon name="up" /></button>
                <button className={button.icon} disabled={i === order.length - 1} onClick={() => move(i, 1)} aria-label="Move down"><Icon name="down" /></button>
                <button className={button.icon} onClick={() => update(order.filter((_, j) => j !== i))} aria-label="Remove"><Icon name="remove" /></button>
              </div>
            </li>
          ))}
        </ol>

        <div className="mt-4 border-t border-line pt-4">
          <div className="mb-2 text-xs font-medium text-muted">
            {order.length >= MAX_QUEUE ? `Queue is full (${MAX_QUEUE} batches max)` : "Add a batch"}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {recipes.map((r) => (
              <button key={r.recipe_id} disabled={order.length >= MAX_QUEUE} onClick={() => update([...order, r.recipe_id])} className="rounded-md hover:ring-2 hover:ring-accent/40 disabled:opacity-40">
                <Chip recipe={r} />
              </button>
            ))}
          </div>
        </div>

        <button className={`${button.primary} mt-5 w-full py-2.5`} disabled={order.length < 2 || plan.busy} onClick={optimize}>
          {plan.busy ? "Searching every order…" : "Find a lower-carryover order"}
        </button>
      </Panel>

      <Panel title="Suggested order" subtitle="Ranks changeovers by how much leftover liquid is predicted to discolor the next batch.">
        {plan.error && <Banner kind="FAIL">{plan.error}</Banner>}
        {!s && !plan.error && (
          <div className="py-10 text-center">
            <p className="text-sm text-muted">Arrange your queue, then press <span className="font-medium text-ink">Find a lower-carryover order</span>.</p>
            <p className="mt-1 text-sm text-muted">Every possible order is compared; the best one is shown here next to yours.</p>
          </div>
        )}
        {s && s.suggested && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
              <div>
                <div className={`text-4xl font-semibold tracking-tight tabular-nums ${better ? "text-pass" : "text-fail"}`}>{pct(s.transition_sum_reduction)}</div>
                <div className="text-sm text-muted">{better ? "less" : "more"} predicted carryover across all changeovers</div>
              </div>
              {s.submitted_worst_transition !== null && s.suggested_worst_transition !== null && (
                <div>
                  <div className="text-xl font-semibold tabular-nums">{pct(s.worst_transition_reduction)}</div>
                  <div className="text-sm text-muted">{(s.worst_transition_reduction ?? 0) >= 0 ? "lower" : "higher"} risk on the worst changeover</div>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-2 text-xs font-medium text-muted">Your order</div>
                <Sequence order={s.submitted ?? order} transitions={s.transitions} byId={byId} maxCost={maxCost} />
              </div>
              <div>
                <div className="mb-2 text-xs font-medium text-accent">Suggested order</div>
                <Sequence order={s.suggested} transitions={s.suggested_transitions} byId={byId} maxCost={maxCost} />
              </div>
              <div className="flex gap-4 text-xs text-muted">
                <span className="flex items-center gap-1.5"><span className="h-1 w-4 rounded-full bg-slate-300" />low risk</span>
                <span className="flex items-center gap-1.5"><span className="h-1 w-4 rounded-full bg-warn" />medium</span>
                <span className="flex items-center gap-1.5"><span className="h-1 w-4 rounded-full bg-fail" />high</span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button className={button.secondary} onClick={() => update(s.suggested!)}>Use suggested order</button>
            </div>

            <details className="rounded-md border border-line bg-bg/60 p-3 text-sm">
              <summary className="font-medium">How this is scored</summary>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">
                <li>Each changeover is scored by the predicted color damage to the next batch&apos;s first vials, averaged over three ways leftover liquid can behave.</li>
                <li>
                  Total risk {s.submitted_transition_sum?.toFixed(2)} → {s.suggested_transition_sum?.toFixed(2)}; with the worst-changeover penalty {s.submitted_cost?.toFixed(2)} →{" "}
                  {s.suggested_cost?.toFixed(2)} ({pct(s.relative_reduction)} {(s.relative_reduction ?? 0) >= 0 ? "lower" : "higher"}).
                </li>
                {s.warnings.map((w) => <li key={w}>{w}</li>)}
                <li>{DISPLAY_NOTE}</li>
              </ul>
            </details>
          </div>
        )}
      </Panel>
    </div>
  );
}
