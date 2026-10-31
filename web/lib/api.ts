// Types mirror api/schemas.py. Colors are sRGB 0-255 triples, as in contracts.py.
import { useEffect, useState } from "react";

export type RGB = [number, number, number];
export type Source = "model" | "pending";

export interface Recipe {
  recipe_id: number;
  ml: Record<string, number>;
  total_ml: number;
  clean_rgb: RGB | null;
}

export interface Frame {
  index: number;
  timestamp: string;
  vial_fill: string;
  tank_level: string;
  in_fill: boolean | null;
  vial: number | null;
  recipe_id: number | null;
}

export interface Tanks {
  colors: Record<string, RGB>;
  fill_percent: Record<string, number | null>;
}

export interface Inspection {
  vial: number;
  recipe_id: number;
  prev_recipe_id: number | null;
  target_ml: number;
  image_url: string;
  expected_rgb: RGB | null;
  source: Source;
  status: "PASS" | "FAIL" | "PENDING";
  observed_rgb: RGB | null;
  delta_e: number | null;
  fill_ml: number | null;
  cause: string | null;
  next_recipe_id: number | null;
}

export interface Transition {
  prev_batch_id: string | null;
  batch_id: string | null;
  prev_recipe_id: number;
  recipe_id: number;
  first_vial_rgb: RGB | null;
  delta_e: number | null;
  warn: boolean | null;
  cost: number | null;
  uncertainty: number | null;
}

export interface Schedule {
  source: Source;
  transitions: Transition[];
  submitted: number[] | null;
  suggested: number[] | null;
  submitted_batch_ids: string[] | null;
  suggested_batch_ids: string[] | null;
  fixed_start_recipe_id: number | null;
  submitted_cost: number | null;
  suggested_cost: number | null;
  submitted_transition_sum: number | null;
  suggested_transition_sum: number | null;
  submitted_worst_transition: number | null;
  suggested_worst_transition: number | null;
  relative_reduction: number | null;
  transition_sum_reduction: number | null;
  worst_transition_reduction: number | null;
  worst_transition_weight: number | null;
  method: string | null;
  warnings: string[];
}

export const imageUrl = (camera: "vial_fill" | "tank_level", filename: string) =>
  `/api/images/${camera}/${filename}`;

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const postSchedule = (recipeIds: number[]) =>
  request<Schedule>("/api/schedule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recipe_ids: recipeIds }),
  });

/** GET url and track its result; a null url fetches nothing. Stale responses are dropped. */
export function useApi<T>(url: string | null) {
  const [state, setState] = useState<{ url: string | null; data?: T; error?: string }>({ url: null });

  useEffect(() => {
    if (!url) return;
    let current = true;
    request<T>(url).then(
      (data) => current && setState({ url, data }),
      (err: Error) => current && setState({ url, error: err.message }),
    );
    return () => {
      current = false;
    };
  }, [url]);

  const fresh = state.url === url;
  return {
    data: fresh ? state.data : undefined,
    error: fresh ? state.error : undefined,
    loading: url !== null && !fresh,
  };
}
