"use client";

import { useEffect, useRef, useState } from "react";
import { Banner, button as buttons, Panel, rgbCss } from "@/components/ui";
import type { RGB } from "@/lib/api";

const TANKS = ["T100", "T200", "T300", "T400"] as const;
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
const button = buttons.secondary;

async function readResponse(res: Response): Promise<Baseline> {
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Image request failed. Check the file and liquid regions.");
  return body;
}

export default function InitializationTab() {
  const [file, setFile] = useState<{ name: string; base64: string; url: string }>();
  const [baseline, setBaseline] = useState<Baseline>();
  const [regions, setRegions] = useState<Regions>({});
  const [selected, setSelected] = useState<Tank>("T100");
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
    setSelected("T100");
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
  const step = (n: number, label: string) => (
    <span className="flex items-center gap-2.5">
      <span className="grid h-6 w-6 place-items-center rounded-full bg-ink text-xs text-white">{n}</span>
      {label}
    </span>
  );
  return (
    <div className="space-y-5">
      <Panel
        title={step(1, "Photo of the tanks")}
        subtitle="A top-down photo from the usual angle: all four tanks across the lower half, T100 to T400 from left to right."
      >
        <div className="flex flex-wrap items-center gap-3">
          <label className={`${buttons.primary} cursor-pointer ${busy ? "pointer-events-none opacity-40" : ""}`}>
            {file ? "Choose a different photo" : "Choose photo"}
            <input type="file" accept="image/png,image/jpeg" disabled={busy} onChange={event => { void selectFile(event.target.files?.[0]); event.target.value = ""; }} className="sr-only" />
          </label>
          <span className="text-sm text-muted">{file ? file.name : baseline ? baseline.source_name : "PNG or JPEG, up to 8 MB"}</span>
        </div>
        <p className="mt-3 text-xs text-faint">Use the same lighting and exposure each shift. Colors are relative to this camera; absolute values would need a color chart.</p>
      </Panel>

      {error && <Banner kind="FAIL"><span role="alert">{error}</span></Banner>}
      {notice && <p role="status" className="text-sm">{notice}</p>}
      {busy && <p role="status" className="text-sm text-muted">Measuring tank colors…</p>}

      {imageUrl && <Panel
        title={baseline?.confirmed ? "Saved tank photo" : step(2, "Check the sampling boxes")}
        subtitle={baseline?.confirmed ? undefined : "Each box should cover liquid only. Pick a tank, then click the middle of a clean patch of its liquid."}
      >
        {!baseline?.confirmed && (
          <div className="mb-3 flex rounded-md border border-line p-0.5" role="group" aria-label="Tank to place">
            {TANKS.map(tank => <button key={tank} disabled={busy} aria-pressed={selected === tank} onClick={() => setSelected(tank)} className={`rounded px-3 py-1.5 text-sm ${selected === tank ? "bg-ink text-white" : "text-muted hover:text-ink"}`}>{tank}</button>)}
          </div>
        )}
        <div className={`relative overflow-hidden rounded-md border border-line ${baseline?.confirmed ? "" : "cursor-crosshair"}`} onClick={event => {
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
          {TANKS.map(tank => regions[tank] && <div key={tank} className={`pointer-events-none absolute border-2 ${selected === tank && !baseline?.confirmed ? "border-yellow-300" : "border-white"}`} style={{ left: `${regions[tank]![0] * 100}%`, top: `${regions[tank]![1] * 100}%`, width: `${(regions[tank]![2] - regions[tank]![0]) * 100}%`, height: `${(regions[tank]![3] - regions[tank]![1]) * 100}%` }}>
            <span className="absolute bottom-full left-0 rounded-t bg-ink px-1.5 text-xs text-white">{tank}</span>
          </div>)}
        </div>
        {!baseline?.confirmed && <>
          <details className="mt-3 text-sm">
            <summary className="text-muted">Adjust the {selected} box by numbers</summary>
            <fieldset disabled={busy} className="mt-2 flex flex-wrap gap-3">
              <legend className="sr-only">{selected} sampling box (% of image)</legend>
              {["Left", "Top", "Right", "Bottom"].map((label, index) => <label key={label} className="text-xs text-muted">{label} %
                <input type="number" min="0" max="100" step="0.1" value={regions[selected] ? Number((regions[selected]![index] * 100).toFixed(1)) : ""} className="ml-2 w-20 rounded border border-line p-1 text-ink" onChange={event => {
                  const box: Region = [...(regions[selected] ?? [0, 0, .08, .08])];
                  box[index] = Number(event.target.value) / 100;
                  setRegions({ ...regions, [selected]: box }); setBaseline(undefined); setReviewed(false);
                }} />
              </label>)}
            </fieldset>
          </details>
          <button className={`${button} mt-3`} disabled={busy || TANKS.some(tank => !regions[tank])} onClick={() => void analyzeOrSave(false)}>Measure these boxes</button>
        </>}
      </Panel>}

      {baseline && <Panel title={baseline.confirmed ? "Tank colors for this shift" : step(3, "Confirm tank colors")}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {TANKS.map(tank => <div key={tank} className="rounded-md border border-line p-3">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 shrink-0 rounded-md border border-line" style={{ background: rgbCss(baseline.tanks[tank].rgb) }} />
              <div>
                <div className="font-medium">{tank}</div>
                <div className="text-xs text-muted">{tank === "T300" ? "out of service" : `${baseline.tanks[tank].sample_pixels} pixels sampled`}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-faint tabular-nums">L*a*b* {baseline.tanks[tank].lab.map(value => value.toFixed(1)).join(", ")}</div>
            {baseline.tanks[tank].warnings.map(warning => <p key={warning} className="mt-2 text-xs text-warn">{warning}</p>)}
          </div>)}
        </div>
        {baseline.confirmed ? <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-md bg-pass-soft px-4 py-3">
          <p role="status" className="text-sm font-medium text-pass">Saved. These are the tank colors for this shift.</p>
          <button className={button} onClick={download}>Download as JSON</button>
        </div> : <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-line pt-4">
          <label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-0.5" checked={reviewed} onChange={event => setReviewed(event.target.checked)} />The tank labels are right and every box samples liquid.</label>
          <button className={buttons.primary} disabled={!reviewed || busy} onClick={() => void analyzeOrSave(true)}>Save tank colors</button>
        </div>}
      </Panel>}

      <details className="rounded-lg border border-line bg-surface px-5 py-3.5">
        <summary className="text-sm font-medium">How these colors are used</summary>
        <div className="mt-3 space-y-3 text-sm text-muted">
          <p><strong className="text-ink">Calibrated optical-density mixing + recipe carryover.</strong> Use each tank&rsquo;s baseline and its volume in mL to predict the clean recipe color. Learn the tank-to-vial color mapping from measured mixtures; directly averaging Lab colors would not describe dye absorption.</p>
          <p>Track the <strong className="text-ink">previous recipe run</strong>, its composition, total volume processed, and any flush. A new vial in the same run does not create a new previous recipe. Estimate residual liquid and dye adsorption, then dilute and wash them out as the next recipe flows.</p>
          <p>The <strong className="text-ink">impurity index (0&ndash;100)</strong> represents predicted color impact relative to the clean recipe. Keep the retained-liquid fraction separate: the same residue volume can cause a much greater color shift from dark to light than from light to dark.</p>
          <p className="text-faint">Model proposal only; predictions require mixture and transition calibration data. Full equations, training plan, and evaluation criteria are in MODEL_PROPOSAL.md.</p>
        </div>
      </details>
    </div>
  );
}
