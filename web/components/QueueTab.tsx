"use client";

import { useState } from "react";

import { postSchedule, type Recipe, type Schedule, useApi } from "@/lib/api";

import { Banner, Card, Composition, Note, Swatch } from "./ui";

export default function QueueTab() {
  const { data: recipes, error } = useApi<Recipe[]>("/api/recipes");
  const [queue, setQueue] = useState<number[] | null>(null);
  const [plan, setPlan] = useState<{ schedule?: Schedule; error?: string }>({});

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
  const evaluate = () =>
    postSchedule(order).then(
      (schedule) => setPlan({ schedule }),
      (err: Error) => setPlan({ error: err.message }),
    );

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Run order">
        <ol className="space-y-2">
          {order.map((id, i) => (
            <li key={`${id}-${i}`} className="flex items-center gap-3">
              <span className="w-5 text-right text-sm text-muted">{i + 1}</span>
              <Swatch rgb={byId[id].clean_rgb} size="sm" />
              <span className="w-20 font-medium">Recipe {id}</span>
              <div className="flex-1">
                <Composition ml={byId[id].ml} />
              </div>
              <button className="px-1 disabled:opacity-30" disabled={i === 0} onClick={() => move(i, -1)} aria-label="Move up">↑</button>
              <button className="px-1 disabled:opacity-30" disabled={i === order.length - 1} onClick={() => move(i, 1)} aria-label="Move down">↓</button>
              <button className="px-1 text-muted" onClick={() => update(order.filter((_, j) => j !== i))} aria-label="Remove">✕</button>
            </li>
          ))}
        </ol>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <select
            className="rounded border border-line bg-white px-2 py-1.5 text-sm"
            value=""
            onChange={(e) => e.target.value && update([...order, Number(e.target.value)])}
          >
            <option value="">+ Add recipe…</option>
            {recipes.map((r) => (
              <option key={r.recipe_id} value={r.recipe_id}>
                Recipe {r.recipe_id}
              </option>
            ))}
          </select>
          <button className="rounded bg-accent px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-40" disabled={order.length < 2} onClick={evaluate}>
            Predict carryover
          </button>
        </div>
      </Card>

      <Card title="Changeovers">
        {plan.error && <Banner kind="FAIL">{plan.error}</Banner>}
        {!plan.schedule && !plan.error && <Note>Press “Predict carryover” to see the expected first vial after each changeover.</Note>}
        {plan.schedule && (
          <div className="space-y-3">
            {plan.schedule.source === "pending" && (
              <Banner kind="PENDING">Mixing model not connected yet — changeovers listed without predictions</Banner>
            )}
            <table className="w-full text-sm">
              <thead className="text-left text-muted">
                <tr>
                  <th className="py-1 font-normal">Changeover</th>
                  <th className="font-normal">Predicted first vial</th>
                  <th className="font-normal">ΔE</th>
                </tr>
              </thead>
              <tbody>
                {plan.schedule.transitions.map((t, i) => (
                  <tr key={i} className={`border-t border-line ${t.warn ? "bg-flag-soft" : ""}`}>
                    <td className="py-1.5">{t.prev_recipe_id} → {t.recipe_id}</td>
                    <td><Swatch rgb={t.first_vial_rgb} size="sm" /></td>
                    <td>{t.delta_e !== null ? t.delta_e.toFixed(1) : "—"}{t.warn && " ⚠"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-sm">
              Suggested order: {plan.schedule.suggested ? plan.schedule.suggested.join(" → ") : <span className="text-muted">scheduler not connected yet</span>}
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
