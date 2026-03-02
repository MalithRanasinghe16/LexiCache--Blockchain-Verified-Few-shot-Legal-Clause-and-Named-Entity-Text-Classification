import { Filter } from "lucide-react";
import { ClauseResult } from "../types";
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
  highlightedText: string;
  showFilters: boolean;
  activeClause: ClauseResult | null;
  onToggleFilters: () => void;
  onToggleType: (type: string) => void;
  onConfidenceChange: (val: number) => void;
  onSelectAll: () => void;
  onClauseClick: (clause: ClauseResult) => void;
  onColorChange: (type: string, color: string) => void;
  onRegenerateColors: () => void;
  onSearchChange: (val: string) => void;
  onSearch: () => void;
};

export default function ResultsSidebar({
  result,
  colorMap,
  selectedClauseTypes,
  minConfidence,
  searchTerm,
  highlightedText,
  showFilters,
  activeClause,
  onToggleFilters,
  onToggleType,
  onConfidenceChange,
  onSelectAll,
  onClauseClick,
  onColorChange,
  onRegenerateColors,
  onSearchChange,
  onSearch,
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

  return (
    <div className="lg:w-2/5 p-8 space-y-6 max-h-screen overflow-y-auto">
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
        highlightedText={highlightedText}
        onChange={onSearchChange}
        onSearch={onSearch}
      />
    </div>
  );
}
