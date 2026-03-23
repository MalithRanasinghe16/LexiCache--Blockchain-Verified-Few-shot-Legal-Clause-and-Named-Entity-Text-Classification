import { Filter } from "lucide-react";
import { ClauseResult, VerificationAttempt, VerificationState } from "../types";
import FilterPanel from "./FilterPanel";
import ClauseList from "./ClauseList";
import ColorLegend from "./ColorLegend";
import SearchBar from "./SearchBar";

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
    if (clause.clause_type === "Unknown clause")
      return selectedClauseTypes.has(clause.clause_type);
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
    <div className="lg:w-2/5 p-8 space-y-6 max-h-screen overflow-y-auto">
      {showVerify && !reminderDismissed && (
        <div className="p-4 rounded-xl border border-amber-200 bg-amber-50 text-amber-900">
          <p className="text-sm font-medium">
            Verify this document to create blockchain proof.
          </p>
          <div className="mt-3 flex gap-2">
            <button
              onClick={onVerify}
              disabled={isVerifying || !(verification?.can_verify ?? false)}
              className="px-3 py-2 text-sm rounded-lg bg-amber-600 text-white hover:bg-amber-700 disabled:opacity-50"
            >
              {isVerifying ? "Verifying..." : "Verify"}
            </button>
            <button
              onClick={onDismissReminder}
              className="px-3 py-2 text-sm rounded-lg bg-white border border-amber-300 hover:bg-amber-100"
            >
              Dismiss
            </button>
          </div>
          {verification && (
            <p className="text-xs mt-2 text-amber-800">
              {verification.message}
            </p>
          )}
        </div>
      )}

      <div className="rounded-xl border border-gray-200 p-4 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-black">Verification History</h3>
          <button
            onClick={onToggleHistory}
            className="px-3 py-1.5 text-sm bg-black text-white rounded-lg hover:bg-gray-800"
          >
            {showHistory ? "Hide History" : "Show History"}
          </button>
        </div>

        {showHistory && (
          <div className="mt-3 space-y-2 max-h-44 overflow-y-auto">
            {history.length === 0 ? (
              <p className="text-sm text-gray-600">
                No verification attempts yet.
              </p>
            ) : (
              history.map((item) => (
                <div
                  key={`${item.attempt}-${item.tx_hash}`}
                  className="rounded-lg border border-gray-200 bg-white p-3"
                >
                  <p className="text-sm font-medium text-black">
                    Attempt {item.attempt}: {item.clause_count} clauses
                  </p>
                  <p className="text-xs text-gray-600">{item.verified_at}</p>
                  <a
                    href={item.blockchain_link}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    View blockchain record
                  </a>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Detected Clauses Header */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-lg text-black">
              Detected Clauses
            </h3>
            {unknownCount > 0 && (
              <span className="px-2 py-1 text-xs font-semibold bg-orange-100 text-orange-700 rounded-full">
                {unknownCount} Unknown
              </span>
            )}
          </div>
          <button
            onClick={onToggleFilters}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition"
          >
            <Filter className="w-4 h-4" />
            {showFilters ? "Hide" : "Show"} Filters
          </button>
        </div>

        {/* Filter Panel */}
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

        {/* Clause List */}
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
      </div>

      {/* Color Legend */}
      <ColorLegend
        clauseTypes={clauseTypes}
        colorMap={colorMap}
        onColorChange={onColorChange}
        onRegenerateColors={onRegenerateColors}
      />

      {/* Search Bar */}
      <SearchBar
        searchTerm={searchTerm}
        totalMatches={totalSearchMatches}
        currentMatchIndex={currentSearchMatchIndex}
        onChange={onSearchChange}
        onNext={onSearchNext}
        onPrev={onSearchPrev}
        onClear={onSearchClear}
      />
    </div>
  );
}
