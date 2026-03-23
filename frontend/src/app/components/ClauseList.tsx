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
  const isSameClause = (a: ClauseResult | null, b: ClauseResult): boolean => {
    if (!a) return false;
    if (
      Number.isInteger(a.display_start_idx) &&
      Number.isInteger(a.display_end_idx) &&
      Number.isInteger(b.display_start_idx) &&
      Number.isInteger(b.display_end_idx)
    ) {
      return (
        a.display_start_idx === b.display_start_idx &&
        a.display_end_idx === b.display_end_idx &&
        a.clause_type === b.clause_type
      );
    }
    if (
      Number.isInteger(a.start_idx) &&
      Number.isInteger(a.end_idx) &&
      Number.isInteger(b.start_idx) &&
      Number.isInteger(b.end_idx)
    ) {
      return (
        a.start_idx === b.start_idx &&
        a.end_idx === b.end_idx &&
        a.clause_type === b.clause_type
      );
    }
    return a.span === b.span && a.clause_type === b.clause_type;
  };

  if (clauses.length === 0) {
    return (
      <div className="rounded-2xl border border-line bg-white py-8 text-center text-muted">
        <Filter className="mx-auto mb-2 h-12 w-12 opacity-40" />
        <p className="text-sm">No clauses match your filters</p>
        <button
          onClick={onResetFilters}
          className="mt-2 text-sm font-semibold text-brand hover:underline"
        >
          Reset Filters
        </button>
      </div>
    );
  }

  return (
    <div className="max-h-96 space-y-3 overflow-y-auto">
      {clauses.map((clause) => {
        const isActive = isSameClause(activeClause, clause);
        const isUnknown = clause.clause_type === "Unknown clause";
        const isStaged = Boolean(clause.is_staged);
        const stableKey = `${clause.clause_type}__${clause.start_idx ?? clause.span.slice(0, 40)}`;

        return (
          <div
            key={stableKey}
            className={`group cursor-pointer rounded-xl border-l-4 p-4 transition-all ${
              isActive
                ? "border-l-[#b8722a] border-t border-r border-b border-[#e4c59f] bg-[#fff6eb] shadow-sm"
                : isUnknown
                  ? "border-l-[#de7a1f] border-t border-r border-b border-[#edd2b1] bg-[#fff9f1] hover:bg-[#fff3e2]"
                  : "border-l-brand border-t border-r border-b border-line bg-white hover:bg-panel/25"
            }`}
            onClick={() => onClauseClick(clause)}
          >
            <div className="flex items-center gap-2 mb-2">
              <div
                className="h-3.5 w-3.5 shrink-0 rounded-full shadow-sm"
                style={{
                  backgroundColor: isUnknown
                    ? "#F97316"
                    : colorMap[clause.clause_type] || "#6b7280",
                }}
              />
              <span
                className={`text-sm font-semibold ${isActive ? "text-[#8a4f16]" : "text-foreground"}`}
              >
                {clause.clause_type}
              </span>

              {/* Active badge */}
              {isActive && (
                <span className="ml-auto rounded-full bg-[#f8dfc0] px-2 py-0.5 text-xs font-semibold text-[#8a4f16]">
                  Showing
                </span>
              )}

              {!isActive && isStaged && (
                <span className="ml-auto rounded-full bg-[#dff7e3] px-2 py-0.5 text-xs font-semibold text-[#1e7a33]">
                  Pending verify
                </span>
              )}

              {/* Unknown → teach prompt */}
              {!isActive && isUnknown && !isStaged && (
                <span className="ml-auto text-xs font-semibold text-[#b55f16]">
                  Click to teach →
                </span>
              )}

              {/* Confidence */}
              {!isActive && !isUnknown && !isStaged && (
                <span className="ml-auto text-xs font-semibold text-muted">
                  {(clause.confidence * 100).toFixed(1)}%
                </span>
              )}
            </div>

            <p className="line-clamp-2 text-sm leading-relaxed text-muted">
              {clause.span}
            </p>

            {/* Active: show jump-to hint */}
            {isActive && (
              <p className="mt-1 flex items-center gap-1 text-xs text-[#9a5d21]">
                ↑ Scrolled to location in document
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
