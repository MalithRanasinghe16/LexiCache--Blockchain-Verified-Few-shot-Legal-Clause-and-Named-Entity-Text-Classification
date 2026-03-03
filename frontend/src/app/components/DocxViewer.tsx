import { FileText } from "lucide-react";
import { ClauseResult } from "../types";
import { useCallback } from "react";

type Props = {
  file: File | null;
  documentText: string;
  result: { result: ClauseResult[] } | null;
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  highlightedText: string;
};

export default function DocxViewer({
  file,
  documentText,
  result,
  colorMap,
  selectedClauseTypes,
  minConfidence,
  highlightedText,
}: Props) {
  const colorWithOpacity = useCallback(
    (color: string, opacity: number): string => {
      if (color.startsWith("hsl")) {
        const match = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
        if (match) {
          const [, h, s, l] = match;
          return `hsla(${h}, ${s}%, ${l}%, ${opacity})`;
        }
      }
      if (color.startsWith("#")) {
        const hex = color.replace("#", "");
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
      }
      return color;
    },
    [],
  );

  const getFilteredClauses = useCallback((): ClauseResult[] => {
    if (!result?.result) return [];
    return result.result.filter((clause) => {
      if (clause.clause_type === "Unknown clause")
        return selectedClauseTypes.has(clause.clause_type);
      return (
        selectedClauseTypes.has(clause.clause_type) &&
        clause.confidence >= minConfidence / 100
      );
    });
  }, [result, selectedClauseTypes, minConfidence]);

  const renderHighlightedText = () => {
    if (!documentText || !result?.result) return <span>{documentText}</span>;

    const segments: { text: string; color?: string; clauseType?: string }[] =
      [];
    let lastIndex = 0;

    const clausePositions: {
      start: number;
      end: number;
      color: string;
      clauseType: string;
    }[] = [];

    getFilteredClauses().forEach((clause) => {
      const normalizedDocText = documentText.toLowerCase();
      const normalizedSpan = clause.span.toLowerCase();
      const searchLength = Math.min(normalizedSpan.length, 200);
      const searchText = normalizedSpan.substring(0, searchLength);
      const index = normalizedDocText.indexOf(searchText);
      if (index !== -1) {
        const clauseColor =
          clause.clause_type === "Unknown clause"
            ? "#F97316"
            : colorMap[clause.clause_type] || "#6b7280";
        clausePositions.push({
          start: index,
          end: index + searchText.length,
          color: clauseColor,
          clauseType: clause.clause_type,
        });
      }
    });

    if (highlightedText) {
      const normalizedDocText = documentText.toLowerCase();
      const normalizedHighlight = highlightedText.toLowerCase();
      const searchLength = Math.min(normalizedHighlight.length, 200);
      const searchText = normalizedHighlight.substring(0, searchLength);
      const searchIndex = normalizedDocText.indexOf(searchText);
      if (searchIndex !== -1) {
        const alreadyHighlighted = clausePositions.some(
          (pos) => searchIndex >= pos.start && searchIndex < pos.end,
        );
        if (!alreadyHighlighted) {
          clausePositions.push({
            start: searchIndex,
            end: searchIndex + searchText.length,
            color: "#fbbf24",
            clauseType: "Selected Clause",
          });
        }
      }
    }

    clausePositions.sort((a, b) => a.start - b.start);

    // Deduplicate overlapping ranges — keep the first one (highest priority)
    const dedupedPositions: typeof clausePositions = [];
    let lastEnd = 0;
    for (const pos of clausePositions) {
      if (pos.start >= lastEnd) {
        dedupedPositions.push(pos);
        lastEnd = pos.end;
      } else if (pos.end > lastEnd) {
        // Partial overlap — clip the start to lastEnd
        dedupedPositions.push({ ...pos, start: lastEnd });
        lastEnd = pos.end;
      }
      // else fully contained in previous — skip
    }

    dedupedPositions.forEach((pos) => {
      if (lastIndex < pos.start) {
        segments.push({ text: documentText.substring(lastIndex, pos.start) });
      }
      segments.push({
        text: documentText.substring(pos.start, pos.end),
        color: pos.color,
        clauseType: pos.clauseType,
      });
      lastIndex = pos.end;
    });

    if (lastIndex < documentText.length) {
      segments.push({ text: documentText.substring(lastIndex) });
    }

    return (
      <div className="whitespace-pre-wrap font-mono text-sm leading-relaxed p-6">
        {segments.map((segment, idx) =>
          segment.color ? (
            <mark
              key={idx}
              style={{
                backgroundColor: colorWithOpacity(segment.color, 0.3),
                borderBottom: `2px solid ${segment.color}`,
                padding: "2px 0",
              }}
              title={segment.clauseType}
            >
              {segment.text}
            </mark>
          ) : (
            <span key={idx}>{segment.text}</span>
          ),
        )}
      </div>
    );
  };

  return (
    <div className="bg-white p-8 shadow-sm">
      <div className="mb-4 text-xs text-black flex items-center gap-2">
        <FileText className="w-4 h-4" />
        <span>
          {file?.name} ({file?.name.split(".").pop()?.toUpperCase()})
        </span>
      </div>
      {renderHighlightedText()}
    </div>
  );
}
