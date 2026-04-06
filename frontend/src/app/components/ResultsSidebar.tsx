import { Filter } from "lucide-react";
import {
  AnalysisResult,
  ClauseResult,
  VerificationAttempt,
  VerificationState,
} from "../types";
import FilterPanel from "./FilterPanel";
import ClauseList from "./ClauseList";
import ColorLegend from "./ColorLegend";
import DocumentSearchPopover from "./DocumentSearchPopover";

type Props = {
  result: AnalysisResult;
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  searchTerm: string;
  totalSearchMatches: number;
  currentSearchMatchIndex: number;
  showFilters: boolean;
  activeClause: ClauseResult | null;
  verification: VerificationState | null;
  history: VerificationAttempt[];
  showHistory: boolean;
  isVerifying: boolean;
  reminderDismissed: boolean;
  onToggleFilters: () => void;
  onVerify: () => void;
  onToggleHistory: () => void;
  onDismissReminder: () => void;
  onToggleType: (type: string) => void;
  onConfidenceChange: (val: number) => void;
  onSelectAll: () => void;
  onClauseClick: (clause: ClauseResult) => void;
  onColorChange: (type: string, color: string) => void;
  onRegenerateColors: () => void;
  onSearchChange: (val: string) => void;
  onSearchNext: () => void;
  onSearchPrev: () => void;
  onSearchClear: () => void;
};

export default function ResultsSidebar({
  result,
  colorMap,
  selectedClauseTypes,
  minConfidence,
  searchTerm,
  totalSearchMatches,
  currentSearchMatchIndex,
  showFilters,
  activeClause,
  verification,
  history,
  showHistory,
  isVerifying,
  reminderDismissed,
  onToggleFilters,
  onVerify,
  onToggleHistory,
  onDismissReminder,
  onToggleType,
  onConfidenceChange,
  onSelectAll,
  onClauseClick,
  onColorChange,
  onRegenerateColors,
  onSearchChange,
  onSearchNext,
  onSearchPrev,
  onSearchClear,
}: Props) {
  const formatVerifiedAt = (value: string): string => {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const allClauses = result.result ?? [];
  const cacheMatchType = result.cache_match_type;
  const cachedAt = result.cached_at;
  const changedFields = result.changed_fields ?? [];

  const clauseTypes = Array.from(new Set(allClauses.map((c) => c.clause_type)));

  const filteredClauses = allClauses.filter((clause) => {
    if (clause.clause_type === "Unknown clause") {
      return selectedClauseTypes.has(clause.clause_type);
    }
    return (
      selectedClauseTypes.has(clause.clause_type) &&
      clause.confidence >= minConfidence / 100
    );
  });

  const unknownCount = allClauses.filter(
    (c) => c.clause_type === "Unknown clause",
  ).length;

  const showVerify = verification?.show_verify_button ?? unknownCount > 0;

  const formatCachedAt = (value?: string) => {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <aside className="lg:w-[36%] max-h-screen overflow-y-auto bg-paper px-5 py-6 lg:px-6 lg:py-7">
      <div className="space-y-6">
        {/* Cache banner */}
        {cacheMatchType === "exact" && (
          <section className="rounded-2xl border border-[#c9dff5] bg-[#eef5fd] p-4 text-[#1a4a7a]">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#2563a8]">
              Loaded from Cache
            </p>
            <p className="mt-1 text-sm">
              This document was already analysed
              {cachedAt ? ` on ${formatCachedAt(cachedAt)}` : ""}. Results are
              returned instantly without re-running the model.
            </p>
          </section>
        )}

        {cacheMatchType === "template_variant" && (
          <section className="rounded-2xl border border-[#ebcfad] bg-[#fff4e7] p-4 text-[#6f4112]">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#8a5a22]">
              Updated Document Detected
            </p>
            <p className="mt-1 text-sm">
              This document has the same contract structure as one analysed
              {cachedAt ? ` on ${formatCachedAt(cachedAt)}` : " previously"},
              but some fields were updated
              {changedFields.length > 0
                ? `: ${changedFields.map((f) => f.toLowerCase()).join(", ")}.`
                : "."}
            </p>
            <p className="mt-1 text-xs text-[#855526]">
              Clause types and confidence scores are from the cached analysis.
              Verifying will record these changes on the blockchain.
            </p>
          </section>
        )}

        {showVerify && !reminderDismissed && (
          <section className="rounded-2xl border border-[#ebcfad] bg-[#fff4e7] p-4 text-[#6f4112] shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#8a5a22]">
              Verification Required
            </p>
            <p className="mt-1 text-sm font-medium">
              Confirm this analysis to create permanent blockchain proof.
            </p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={onVerify}
                disabled={isVerifying || !(verification?.can_verify ?? false)}
                className="rounded-lg bg-[#b8722a] px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-[#986023] disabled:opacity-50"
              >
                {isVerifying ? "Verifying..." : "Verify"}
              </button>
              <button
                onClick={onDismissReminder}
                className="rounded-lg border border-[#dfbf96] bg-white px-3 py-2 text-xs font-semibold uppercase tracking-wide text-[#704618] transition hover:bg-[#f8ebdc]"
              >
                Dismiss
              </button>
            </div>
            {verification && (
              <p className="mt-2 text-xs text-[#855526]">
                {verification.message}
              </p>
            )}
          </section>
        )}

        <section className="rounded-2xl border border-line bg-white p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
              Verification History
            </h3>
            <button
              onClick={onToggleHistory}
              className="rounded-lg border border-line bg-panel/50 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-foreground transition hover:border-brand hover:text-brand"
            >
              {showHistory ? "Hide History" : "Show History"}
            </button>
          </div>

          {showHistory && (
            <div className="mt-3 max-h-44 space-y-2 overflow-y-auto">
              {history.length === 0 ? (
                <p className="text-sm text-muted">
                  No verification attempts yet.
                </p>
              ) : (
                history.map((item) => {
                  const unknown = Math.max(0, Number(item.unknown_count || 0));
                  const total = Math.max(0, Number(item.clause_count || 0));
                  const known = Math.max(0, total - unknown);

                  return (
                    <div
                      key={`${item.attempt}-${item.tx_hash}`}
                      className="rounded-xl border border-line bg-paper p-3"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-foreground">
                          Attempt {item.attempt}
                        </p>
                        <span className="rounded-full border border-line bg-white px-2 py-0.5 text-xs font-semibold text-muted">
                          {total} clauses
                        </span>
                      </div>

                      <div className="mt-1 flex items-center gap-2 text-xs">
                        <span className="rounded-full bg-[#e6f6f3] px-2 py-0.5 font-semibold text-[#1a6157]">
                          {known} Known
                        </span>
                        <span className="rounded-full bg-[#fce8d6] px-2 py-0.5 font-semibold text-[#9b5a17]">
                          {unknown} Unknown
                        </span>
                      </div>

                      <div className="mt-2 space-y-1 border-t border-line/70 pt-2">
                        <p className="text-xs text-muted">
                          <span className="font-semibold text-foreground">
                            Verified:
                          </span>{" "}
                          {formatVerifiedAt(item.verified_at)}
                        </p>
                        <p className="text-xs text-muted">
                          <span className="font-semibold text-foreground">
                            Location:
                          </span>{" "}
                          {item.geo_summary || "Not captured"}
                        </p>
                        <p className="break-all text-[11px] text-muted">
                          <span className="font-semibold text-foreground">
                            Location hash:
                          </span>{" "}
                          {item.geo_hash || "N/A"}
                        </p>
                      </div>

                      {item.changed_fields &&
                        item.changed_fields.length > 0 && (
                          <p className="mt-1 text-xs text-muted">
                            <span className="font-semibold text-foreground">
                              Changes recorded:
                            </span>{" "}
                            {item.changed_fields
                              .map((f) => f.toLowerCase())
                              .join(", ")}
                          </p>
                        )}

                      <a
                        href={item.blockchain_link}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-2 inline-block text-xs font-semibold text-brand hover:underline"
                      >
                        View blockchain record
                      </a>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </section>

        <DocumentSearchPopover
          searchTerm={searchTerm}
          totalMatches={totalSearchMatches}
          currentMatchIndex={currentSearchMatchIndex}
          onChange={onSearchChange}
          onNext={onSearchNext}
          onPrev={onSearchPrev}
          onClear={onSearchClear}
        />

        <section>
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-foreground">
                Detected Clauses
              </h3>
              {unknownCount > 0 && (
                <span className="rounded-full bg-[#fce8d6] px-2 py-1 text-xs font-semibold text-[#9b5a17]">
                  {unknownCount} Unknown
                </span>
              )}
            </div>
            <button
              onClick={onToggleFilters}
              className="flex items-center gap-2 rounded-lg border border-line bg-panel/50 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-foreground transition hover:border-brand hover:text-brand"
            >
              <Filter className="h-4 w-4" />
              {showFilters ? "Hide" : "Show"} Filters
            </button>
          </div>

          {showFilters && (
            <FilterPanel
              clauseTypes={clauseTypes}
              selectedClauseTypes={selectedClauseTypes}
              minConfidence={minConfidence}
              colorMap={colorMap}
              allClauses={allClauses}
              filteredCount={filteredClauses.length}
              totalCount={allClauses.length}
              onToggleType={onToggleType}
              onConfidenceChange={onConfidenceChange}
              onSelectAll={onSelectAll}
            />
          )}

          <ClauseList
            clauses={filteredClauses}
            colorMap={colorMap}
            activeClause={activeClause}
            onClauseClick={onClauseClick}
            onResetFilters={() => {
              onSelectAll();
              onConfidenceChange(0);
            }}
          />
        </section>

        <ColorLegend
          clauseTypes={clauseTypes}
          colorMap={colorMap}
          onColorChange={onColorChange}
          onRegenerateColors={onRegenerateColors}
        />
      </div>
    </aside>
  );
}
