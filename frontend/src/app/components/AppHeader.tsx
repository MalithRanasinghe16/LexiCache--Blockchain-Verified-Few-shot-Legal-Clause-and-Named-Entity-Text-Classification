export default function AppHeader() {
  return (
    <div className="border-b border-(--line) bg-(--paper)/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-6 py-6 lg:flex-row lg:items-end lg:justify-between lg:px-10">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-(--muted)">
            Legal Analysis Workspace
          </p>
          <h1 className="mt-1 text-4xl font-semibold leading-tight text-foreground lg:text-5xl">
            LexiCache
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-(--muted) lg:text-base">
            Review, teach, and verify legal clauses in a document-first interface.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-full border border-(--line) bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-(--brand)">
            AI Analysis
          </span>
          <span className="rounded-full border border-(--line) bg-(--brand-soft) px-3 py-1 text-xs font-semibold uppercase tracking-wide text-(--brand)">
            Blockchain Proof
          </span>
        </div>
      </div>
    </div>
  );
}
