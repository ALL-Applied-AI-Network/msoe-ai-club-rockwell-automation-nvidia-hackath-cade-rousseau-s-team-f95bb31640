"use client";

import { useSyncExternalStore } from "react";

import EvidenceTab from "@/components/EvidenceTab";
import InitializationTab from "@/components/InitializationTab";
import InspectionTab from "@/components/InspectionTab";
import QueueTab from "@/components/QueueTab";

// In the order of a shift: set up tanks, plan the queue, inspect vials; cameras for review.
const TABS = {
  initialization: { label: "Tank setup", step: 1 },
  queue: { label: "Plan queue", step: 2 },
  inspection: { label: "Inspect vials", step: 3 },
  evidence: { label: "Cameras", step: null },
} as const;
type Tab = keyof typeof TABS;

// The open tab lives in the URL hash (#queue, #evidence), so tabs can be bookmarked and Back works.
const onHashChange = (callback: () => void) => {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
};

export default function Home() {
  const hash = useSyncExternalStore(onHashChange, () => window.location.hash.slice(1), () => "");
  const tab: Tab = hash in TABS ? (hash as Tab) : "inspection";

  return (
    <div className="min-h-full">
      <header className="bg-ink text-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 pt-4">
          <div className="flex items-center gap-3">
            <svg viewBox="0 0 24 24" className="h-7 w-7" aria-hidden>
              <path d="M9 2h6v2h-1v13a2 2 0 0 1-4 0V4H9z" fill="none" stroke="currentColor" strokeWidth="1.6" />
              <path d="M10 11h4v6a2 2 0 0 1-4 0z" fill="#5fa8e8" />
            </svg>
            <div>
              <div className="font-semibold tracking-tight">Changeover Guard</div>
              <div className="text-xs text-white/55">CSI fluid mixing · Station 3</div>
            </div>
          </div>
          <span className="rounded-full border border-white/20 px-3 py-1 text-xs text-white/75">Read-only · advises, never controls the line</span>
        </div>
        <nav className="mx-auto mt-3 flex max-w-6xl gap-1 overflow-x-auto px-3">
          {Object.entries(TABS).map(([key, { label, step }]) => (
            <button
              key={key}
              onClick={() => (window.location.hash = key)}
              className={`flex items-center gap-2 whitespace-nowrap border-b-2 px-3 py-2.5 text-sm font-medium ${
                tab === key ? "border-white text-white" : "border-transparent text-white/60 hover:text-white"
              }`}
            >
              {step && (
                <span className={`grid h-5 w-5 place-items-center rounded-full text-[11px] ${tab === key ? "bg-white text-ink" : "bg-white/15"}`}>{step}</span>
              )}
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-6">
        {tab === "initialization" && <InitializationTab />}
        {tab === "queue" && <QueueTab />}
        {tab === "inspection" && <InspectionTab />}
        {tab === "evidence" && <EvidenceTab />}
      </main>
    </div>
  );
}
