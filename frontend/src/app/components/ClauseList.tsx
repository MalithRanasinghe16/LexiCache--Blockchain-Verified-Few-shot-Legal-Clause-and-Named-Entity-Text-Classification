import { Filter } from "lucide-react";
import { ClauseResult } from "../types";

type Props = {
  clauses: ClauseResult[];
  colorMap: Record<string, string>;
  activeClause: ClauseResult | null;
  onClauseClick: (clause: ClauseResult) => void;
  onResetFilters: () => void;
};

export default function ClauseList({
  clauses,
  colorMap,
  activeClause,
  onClauseClick,
  onResetFilters,
}: Props) {
  if (clauses.length === 0) {
    return (
      <div className="text-center text-black py-8">
        <Filter className="w-12 h-12 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No clauses match your filters</p>
        <button
          onClick={onResetFilters}
          className="text-sm text-blue-600 hover:text-blue-800 mt-2"
        >
          Reset Filters
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3 max-h-96 overflow-y-auto text-black">
      {clauses.map((clause) => {
        const isActive = activeClause?.span === clause.span;
        const isUnknown = clause.clause_type === "Unknown clause";
        const stableKey = `${clause.clause_type}__${clause.start_idx ?? clause.span.slice(0, 40)}`;

        return (
          <div
            key={stableKey}
            className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
              isActive
                ? "border-amber-400 bg-amber-50 shadow-md ring-2 ring-amber-300"
                : isUnknown
                  ? "border-orange-300 bg-orange-50 hover:bg-orange-100"
                  : "border-gray-200 bg-gray-50 hover:bg-gray-100"
            }`}
            onClick={() => onClauseClick(clause)}
          >
            <div className="flex items-center gap-2 mb-2">
              <div
                className="w-4 h-4 rounded-full shadow-sm flex-shrink-0"
                style={{
                  backgroundColor: isUnknown
                    ? "#F97316"
                    : colorMap[clause.clause_type] || "#6b7280",
                }}
              />
              <span
                className={`font-medium ${isActive ? "text-amber-800" : ""}`}
              >
                {clause.clause_type}
              </span>

              {/* Active badge */}
              {isActive && (
                <span className="ml-auto flex items-center gap-1 text-xs font-semibold text-amber-700 bg-amber-200 px-2 py-0.5 rounded-full">
                  📍 Showing
                </span>
              )}

              {/* Unknown → teach prompt */}
              {!isActive && isUnknown && (
                <span className="ml-auto text-xs text-orange-600 font-semibold">
                  Click to teach →
                </span>
              )}

              {/* Confidence */}
              {!isActive && !isUnknown && (
                <span className="ml-auto text-sm text-black">
                  {(clause.confidence * 100).toFixed(1)}%
                </span>
              )}
            </div>

            <p className="text-sm text-black line-clamp-2">{clause.span}</p>

            {/* Active: show jump-to hint */}
            {isActive && (
              <p className="text-xs text-amber-600 mt-1 flex items-center gap-1">
                ↑ Scrolled to location in document
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
