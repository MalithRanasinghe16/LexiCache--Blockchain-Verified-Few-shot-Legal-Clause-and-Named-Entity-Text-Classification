import { Filter } from "lucide-react";
import { ClauseResult, VerificationAttempt, VerificationState } from "../types";
import FilterPanel from "./FilterPanel";
import ClauseList from "./ClauseList";
import ColorLegend from "./ColorLegend";
import DocumentSearchPopover from "./DocumentSearchPopover";

type Props = {
  result: { result: ClauseResult[] };
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
  const allClauses = result.result ?? [];

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

  return (
    <aside className="lg:w-[36%] max-h-screen overflow-y-auto bg-paper px-5 py-6 lg:px-6 lg:py-7">
      <div className="space-y-6">
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
                history.map((item) => (
                  <div
                    key={`${item.attempt}-${item.tx_hash}`}
                    className="rounded-xl border border-line bg-paper p-3"
                  >
                    <p className="text-sm font-semibold text-foreground">
                      Attempt {item.attempt}: {item.clause_count} clauses
                    </p>
                    <p className="text-xs text-muted">{item.verified_at}</p>
                    <a
                      href={item.blockchain_link}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs font-semibold text-brand hover:underline"
                    >
                      View blockchain record
                    </a>
                  </div>
                ))
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
