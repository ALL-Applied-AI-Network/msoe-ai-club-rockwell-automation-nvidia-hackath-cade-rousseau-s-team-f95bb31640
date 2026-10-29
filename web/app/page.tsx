"use client";

import { useSyncExternalStore } from "react";

import EvidenceTab from "@/components/EvidenceTab";
import InspectionTab from "@/components/InspectionTab";
import QueueTab from "@/components/QueueTab";
import InitializationTab from "@/components/InitializationTab";

const TABS = { initialization: "Tank Initialization", inspection: "Vial Inspection", queue: "Recipe Queue", evidence: "Camera Evidence" } as const;
type Tab = keyof typeof TABS;

// The open tab lives in the URL hash (#queue, #evidence), so tabs can be bookmarked and Back works.
const onHashChange = (callback: () => void) => {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
};

export default function Home() {
  const hash = useSyncExternalStore(onHashChange, () => window.location.hash.slice(1), () => "");
  const tab: Tab = hash in TABS ? (hash as Tab) : "initialization";

  return (
    <main className="mx-auto max-w-6xl space-y-5 px-4 py-6">
      <header>
        <h1 className="text-2xl font-semibold">Changeover Guard | Fluid Mixing Station</h1>
        <p className="mt-1 text-xs tracking-wide text-muted uppercase">Operator assistant · Read-only: no machine commands</p>
      </header>

      <nav className="flex flex-wrap gap-1 border-b border-line">
        {Object.entries(TABS).map(([key, label]) => (
          <button
            key={key}
            onClick={() => (window.location.hash = key)}
            className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium ${tab === key ? "border-accent text-accent" : "border-transparent text-muted hover:text-ink"}`}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "initialization" && <InitializationTab />}
      {tab === "inspection" && <InspectionTab />}
      {tab === "queue" && <QueueTab />}
      {tab === "evidence" && <EvidenceTab />}
    </main>
  );
}
