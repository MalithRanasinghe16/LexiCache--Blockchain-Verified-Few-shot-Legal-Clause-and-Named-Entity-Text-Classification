import { ClauseResult } from "../types";

type Props = {
  clauseTypes: string[];
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  colorMap: Record<string, string>;
  allClauses: ClauseResult[];
  filteredCount: number;
  totalCount: number;
  onToggleType: (type: string) => void;
  onConfidenceChange: (value: number) => void;
  onSelectAll: () => void;
};

export default function FilterPanel({
  clauseTypes,
  selectedClauseTypes,
  minConfidence,
  colorMap,
  allClauses,
  filteredCount,
  totalCount,
  onToggleType,
  onConfidenceChange,
  onSelectAll,
}: Props) {
  return (
    <div className="mb-4 space-y-4 rounded-2xl border border-line bg-white p-4">
      {/* Clause Type Checkboxes */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold text-foreground">
            Clause Types
          </label>
          <button
            onClick={onSelectAll}
            className="text-xs font-semibold text-brand hover:underline"
          >
            {selectedClauseTypes.size === clauseTypes.length
              ? "Deselect All"
              : "Select All"}
          </button>
        </div>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {clauseTypes.map((type) => (
            <label
              key={type}
              className="flex cursor-pointer items-center gap-2 rounded-lg p-2 transition hover:bg-panel/45"
            >
              <input
                type="checkbox"
                checked={selectedClauseTypes.has(type)}
                onChange={() => onToggleType(type)}
                className="h-4 w-4 rounded border-line text-brand focus:ring-2 focus:ring-brand"
              />
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: colorMap[type] }}
              />
              <span className="flex-1 text-sm text-foreground">{type}</span>
              <span className="text-xs text-muted">
                ({allClauses.filter((c) => c.clause_type === type).length})
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Confidence Slider */}
      <div>
        <label className="mb-2 block text-sm font-semibold text-foreground">
          Minimum Confidence: {minConfidence}%
        </label>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={minConfidence}
          onChange={(e) => onConfidenceChange(Number(e.target.value))}
          className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[#e6dbca]"
        />
        <div className="mt-1 flex justify-between text-xs text-muted">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Summary */}
      <div className="border-t border-line pt-2 text-xs text-muted">
        Showing {filteredCount} of {totalCount} clauses
      </div>
    </div>
  );
}
