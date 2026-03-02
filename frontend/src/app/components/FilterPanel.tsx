import { Filter } from "lucide-react";
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
    <div className="mb-4 p-4 bg-gray-50 rounded-xl border border-gray-200 space-y-4 text-black">
      {/* Clause Type Checkboxes */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-black">Clause Types</label>
          <button
            onClick={onSelectAll}
            className="text-xs text-blue-600 hover:text-blue-800"
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
              className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 p-2 rounded"
            >
              <input
                type="checkbox"
                checked={selectedClauseTypes.has(type)}
                onChange={() => onToggleType(type)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: colorMap[type] }}
              />
              <span className="text-sm flex-1">{type}</span>
              <span className="text-xs text-black">
                ({allClauses.filter((c) => c.clause_type === type).length})
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Confidence Slider */}
      <div>
        <label className="text-sm font-medium text-black block mb-2">
          Minimum Confidence: {minConfidence}%
        </label>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={minConfidence}
          onChange={(e) => onConfidenceChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-black mt-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Summary */}
      <div className="text-xs text-black pt-2 border-t border-gray-200">
        Showing {filteredCount} of {totalCount} clauses
      </div>
    </div>
  );
}
