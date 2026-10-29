"use client";

import { useEffect, useRef, useState } from "react";
import { Card, Swatch } from "@/components/ui";
import type { RGB } from "@/lib/api";

const TANKS = ["T_100", "T_200", "T_300", "T_400"] as const;
type Tank = typeof TANKS[number];
type Region = [number, number, number, number];
type Regions = Partial<Record<Tank, Region>>;
interface Baseline {
  id?: string;
  confirmed: boolean;
  image_url?: string;
  source_name: string;
  calibration: string;
  tanks: Record<Tank, { rgb: RGB; lab: RGB; region: Region; sample_pixels: number; warnings: string[] }>;
}
const STORAGE_KEY = "changeover-guard:baseline-id";
const button = "rounded border border-line px-3 py-2 text-sm disabled:opacity-40";

async function readResponse(res: Response): Promise<Baseline> {
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Image request failed. Check the file and liquid regions.");
  return body;
}

export default function InitializationTab() {
  const [file, setFile] = useState<{ name: string; base64: string; url: string }>();
  const [baseline, setBaseline] = useState<Baseline>();
  const [regions, setRegions] = useState<Regions>({});
  const [selected, setSelected] = useState<Tank>("T_100");
  const [reviewed, setReviewed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const generation = useRef(0);

  function accept(result: Baseline) {
    setBaseline(result);
    setRegions(Object.fromEntries(TANKS.map(tank => [tank, result.tanks[tank].region])));
  }

  useEffect(() => {
    const version = generation.current;
    let active = true;
    try {
      const id = localStorage.getItem(STORAGE_KEY);
      if (id) fetch(`/api/baselines/${id}`).then(readResponse).then(result => {
        if (active && generation.current === version) accept(result);
      }).catch((err: Error) => {
        if (active && generation.current === version) setError(err.message);
      });
    } catch {
      // Upload and download still work when browser storage is unavailable.
    }
    return () => { active = false; };
  }, []);

  async function selectFile(next?: File) {
    if (!next) return;
    const version = ++generation.current;
    setBaseline(undefined);
    setFile(undefined);
    setRegions({});
    setReviewed(false);
    setError("");
    setNotice("");
    setSelected("T_100");
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* Storage is optional. */ }
    if (next.size > 8 * 1024 * 1024 || !["image/png", "image/jpeg"].includes(next.type)) {
      setError("Select a PNG or JPEG image smaller than 8 MiB.");
      return;
    }
    setBusy(true);
    try {
      const url = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(new Error("Could not read this image."));
        reader.readAsDataURL(next);
      });
      if (generation.current !== version) return;
      const uploaded = { name: next.name, base64: url.split(",")[1], url };
      setFile(uploaded);
      const result = await fetch("/api/baselines/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_name: uploaded.name, image_base64: uploaded.base64 }),
      }).then(readResponse);
      if (generation.current === version) accept(result);
    } catch (err) {
      if (generation.current === version) setError((err as Error).message);
    } finally {
      if (generation.current === version) setBusy(false);
    }
  }

  async function analyzeOrSave(save: boolean) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const result = await fetch(save ? "/api/baselines" : "/api/baselines/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_name: file.name, image_base64: file.base64, regions, ...(save ? { confirmed: true } : {}) }),
      }).then(readResponse);
      accept(result);
      if (save && result.id) {
        try { localStorage.setItem(STORAGE_KEY, result.id); }
        catch { setNotice("Baseline saved on the server. Browser storage is unavailable; download the JSON for your records."); }
      }
    } catch (err) {
      setError((err as Error).message);
    } finally { setBusy(false); }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(baseline, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `tank-baseline-${baseline?.id ?? "preview"}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const imageUrl = file?.url ?? baseline?.image_url;
  return (
    <div className="space-y-5">
      <Card title="Part 1 · Initialize tank colors">
        <p className="mb-3 text-sm">Select a tank top-view image with the same camera angle as the reference: four tanks across the lower half. The initial labels run left to right: T_100, T_200, T_300, T_400.</p>
        <label className="block text-sm font-medium">
          Tank Top View
          <input type="file" accept="image/png,image/jpeg" disabled={busy} onChange={event => { void selectFile(event.target.files?.[0]); event.target.value = ""; }} className="mt-2 block w-full text-sm" />
        </label>
        <p className="mt-3 text-sm text-muted">Use consistent lighting and exposure. Values are camera-relative CIELAB estimates (D65, 2° observer, assumed sRGB); a color chart is needed for absolute calibration.</p>
      </Card>

      {error && <p role="alert" className="rounded border border-flag bg-flag-soft p-3 text-sm">{error}</p>}
      {notice && <p role="status" className="text-sm">{notice}</p>}
      {busy && <p role="status">Measuring tank colors…</p>}

      {imageUrl && <Card title={baseline?.confirmed ? "Saved tank baseline" : "Review liquid regions"}>
        {!baseline?.confirmed && <>
          <p className="mb-3 text-sm">Check that each box covers liquid only. To adjust a box, select its tank and click the center of a clean liquid patch in the image. Use the tank buttons and position controls for keyboard adjustments.</p>
          <div className="mb-3 flex flex-wrap gap-2">
            {TANKS.map(tank => <button key={tank} disabled={busy} aria-pressed={selected === tank} onClick={() => setSelected(tank)} className={`${button} ${selected === tank ? "border-accent bg-slate-100" : ""}`}>{tank}</button>)}
          </div>
        </>}
        <div className="relative overflow-hidden rounded border border-line" onClick={event => {
          if (!file || busy || baseline?.confirmed) return;
          const box = event.currentTarget.getBoundingClientRect();
          const x = Math.max(0, Math.min(.92, (event.clientX - box.left) / box.width - .04));
          const y = Math.max(0, Math.min(.92, (event.clientY - box.top) / box.height - .04));
          setRegions({ ...regions, [selected]: [x, y, x + .08, y + .08] });
          setBaseline(undefined); setReviewed(false); setError("");
        }}>
          {/* A local image preview must retain its native aspect ratio for ROI coordinates. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt="Uploaded tank top view with labeled liquid sampling regions" className="block h-auto w-full" />
          {TANKS.map(tank => regions[tank] && <div key={tank} className="pointer-events-none absolute border-2 border-cyan-300" style={{ left: `${regions[tank]![0] * 100}%`, top: `${regions[tank]![1] * 100}%`, width: `${(regions[tank]![2] - regions[tank]![0]) * 100}%`, height: `${(regions[tank]![3] - regions[tank]![1]) * 100}%` }}>
            <span className="absolute bottom-full left-0 bg-slate-900 px-1 text-xs text-white">{tank}</span>
          </div>)}
        </div>
        {!baseline?.confirmed && <>
          <fieldset disabled={busy} className="mt-3 flex flex-wrap gap-3">
            <legend className="mb-2 text-sm">{selected} sampling box (% of image)</legend>
            {["Left", "Top", "Right", "Bottom"].map((label, index) => <label key={label} className="text-xs">{label}
              <input type="number" min="0" max="100" step="0.1" value={regions[selected] ? Number((regions[selected]![index] * 100).toFixed(1)) : ""} className="ml-2 w-20 rounded border border-line p-1" onChange={event => {
                const box: Region = [...(regions[selected] ?? [0, 0, .08, .08])];
                box[index] = Number(event.target.value) / 100;
                setRegions({ ...regions, [selected]: box }); setBaseline(undefined); setReviewed(false);
              }} />
            </label>)}
          </fieldset>
          <button className={`${button} mt-3`} disabled={busy || TANKS.some(tank => !regions[tank])} onClick={() => void analyzeOrSave(false)}>Measure selected regions</button>
        </>}
      </Card>}

      {baseline && <Card title="CIELAB baseline · L*, a*, b*">
        <div className="grid gap-4 sm:grid-cols-2">
          {TANKS.map(tank => <div key={tank} className="rounded border border-line p-3">
            <Swatch rgb={baseline.tanks[tank].rgb} size="sm" label={`${tank} = (${baseline.tanks[tank].lab.map(value => value.toFixed(2)).join(", ")})`} />
            <p className="mt-2 text-xs text-muted">{baseline.tanks[tank].sample_pixels} sampled pixels{tank === "T_300" ? " · Recorded; out of service for recipes" : ""}</p>
            {baseline.tanks[tank].warnings.map(warning => <p key={warning} className="mt-2 text-sm text-flag-ink">{warning}</p>)}
          </div>)}
        </div>
        {baseline.confirmed ? <div className="mt-4 flex items-center gap-3"><p role="status" className="text-sm">Baseline saved. These four values are the inputs to the proposed recipe model.</p><button className={button} onClick={download}>Download baseline JSON</button></div> : <div className="mt-4 space-y-3">
          <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={reviewed} onChange={event => setReviewed(event.target.checked)} />I verified the tank IDs and that all four boxes sample the liquid.</label>
          <button className={button} disabled={!reviewed || busy} onClick={() => void analyzeOrSave(true)}>Save baseline</button>
        </div>}
      </Card>}

      <Card title="Part 2 · Proposed recipe color model">
        <div className="space-y-3 text-sm">
          <p><strong>Calibrated optical-density mixing + recipe carryover.</strong> Use each tank’s baseline and its volume in mL to predict the clean recipe color. Learn the tank-to-vial color mapping from measured mixtures; directly averaging Lab colors would not describe dye absorption.</p>
          <p>Track the <strong>previous recipe run</strong>, its composition, total volume processed, and any flush. A new vial in the same run does not create a new previous recipe. Estimate residual liquid and dye adsorption, then dilute and wash them out as the next recipe flows.</p>
          <p>The <strong>impurity index (0–100)</strong> represents predicted color impact relative to the clean recipe. Keep the retained-liquid fraction separate: the same residue volume can cause a much greater color shift from dark to light than from light to dark.</p>
          <p className="text-muted">Model proposal only; predictions require mixture and transition calibration data. The saved baseline is ready for that calibration. Full equations, training plan, and evaluation criteria are in MODEL_PROPOSAL.md.</p>
        </div>
      </Card>
    </div>
  );
}
