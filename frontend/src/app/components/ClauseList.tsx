import { Filter } from "lucide-react";
import { ClauseResult } from "../types";

type Props = {
  clauses: ClauseResult[];
  colorMap: Record<string, string>;
  onClauseClick: (clause: ClauseResult) => void;
  onResetFilters: () => void;
};

export default function ClauseList({
  clauses,
  colorMap,
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
      {clauses.map((clause, idx) => (
        <div
          key={idx}
          className={`p-4 rounded-xl border transition cursor-pointer ${
            clause.clause_type === "Unknown clause"
              ? "border-orange-300 bg-orange-50 hover:bg-orange-100"
              : "border-gray-200 bg-gray-50 hover:bg-gray-100"
          }`}
          onClick={() => onClauseClick(clause)}
        >
          <div className="flex items-center gap-2 mb-2">
            <div
              className="w-4 h-4 rounded-full shadow-sm"
              style={{
                backgroundColor:
                  clause.clause_type === "Unknown clause"
                    ? "#F97316"
                    : colorMap[clause.clause_type] || "#6b7280",
              }}
            />
            <span className="font-medium">{clause.clause_type}</span>
            {clause.clause_type === "Unknown clause" ? (
              <span className="ml-auto text-xs text-orange-600 font-semibold">
                Click to teach →
              </span>
            ) : (
              <span className="ml-auto text-sm text-black">
                {(clause.confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-sm text-black line-clamp-2">{clause.span}</p>
        </div>
      ))}
    </div>
  );
}
