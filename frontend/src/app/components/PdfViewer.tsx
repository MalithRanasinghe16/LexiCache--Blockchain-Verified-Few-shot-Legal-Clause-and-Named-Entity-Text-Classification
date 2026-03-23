"use client";

import { useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import {
  AnalysisResult,
  ClauseResult,
  PageTextContent,
  SearchMatch,
  TextItem,
} from "../types";

const Document = dynamic(
  () => import("react-pdf").then((mod) => mod.Document),
  { ssr: false },
);
const Page = dynamic(() => import("react-pdf").then((mod) => mod.Page), {
  ssr: false,
});

if (typeof window !== "undefined") {
  import("react-pdf/dist/Page/AnnotationLayer.css");
  import("react-pdf/dist/Page/TextLayer.css");
}

type Props = {
  file: File | null;
  numPages: number | null;
  pageWidth: number;
  pageHeights: number[];
  pageTextContents: PageTextContent[];
  result: AnalysisResult | null;
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  highlightedText: string;
  searchMatches: SearchMatch[];
  activeClause: ClauseResult | null;
  activeSearchPageIndex: number | null;
  activeSearchCharOffset: number | null;
  isClient: boolean;
  onDocumentLoadSuccess: (data: { numPages: number }) => void;
};

export default function PdfViewer({
  file,
  numPages,
  pageWidth,
  pageHeights,
  pageTextContents,
  result,
  colorMap,
  selectedClauseTypes,
  minConfidence,
  highlightedText,
  searchMatches,
  activeClause,
  activeSearchPageIndex,
  activeSearchCharOffset,
  isClient,
  onDocumentLoadSuccess,
}: Props) {
  const LINE_GROUP_TOLERANCE = 6;
  const canvasRefs = useRef<HTMLCanvasElement[]>([]);
  // Refs to each page container div for scrolling
  const pageContainerRefs = useRef<(HTMLDivElement | null)[]>([]);

  const isSameClause = useCallback(
    (a: ClauseResult, b: ClauseResult): boolean => {
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
    },
    [],
  );

  const getClauseSearchText = useCallback(
    (clause: ClauseResult): string => {
      const sourceText = result?.extracted_text;
      const dispStart = clause.display_start_idx;
      const dispEnd = clause.display_end_idx;
      const start = clause.start_idx;
      const end = clause.end_idx;

      if (
        typeof sourceText === "string" &&
        Number.isInteger(dispStart) &&
        Number.isInteger(dispEnd) &&
        (dispStart as number) >= 0 &&
        (dispEnd as number) > (dispStart as number) &&
        (dispEnd as number) <= sourceText.length
      ) {
        const displaySlice = sourceText.slice(
          dispStart as number,
          dispEnd as number,
        );
        return displaySlice.replace(/\s+/g, " ").trim().slice(0, 320);
      }

      if (
        typeof sourceText === "string" &&
        Number.isInteger(start) &&
        Number.isInteger(end) &&
        (start as number) >= 0 &&
        (end as number) > (start as number) &&
        (end as number) <= sourceText.length
      ) {
        const exactSlice = sourceText.slice(start as number, end as number);
        return exactSlice.replace(/\s+/g, " ").trim().slice(0, 260);
      }

      return clause.span_display || clause.span_exact || clause.span;
    },
    [result],
  );

  const getClauseRange = useCallback(
    (clause: ClauseResult): { start: number; end: number } | null => {
      const s = Number.isInteger(clause.display_start_idx)
        ? (clause.display_start_idx as number)
        : Number.isInteger(clause.start_idx)
          ? (clause.start_idx as number)
          : null;
      const e = Number.isInteger(clause.display_end_idx)
        ? (clause.display_end_idx as number)
        : Number.isInteger(clause.end_idx)
          ? (clause.end_idx as number)
          : null;

      if (s === null || e === null) return null;
      if (e <= s) return null;
      return { start: s, end: e };
    },
    [],
  );

  const getPageInfo = useCallback(
    (pageNumber: number) => {
      const pages = result?.page_texts;
      if (!pages || pages.length === 0) return null;
      return pages.find((p) => p.page === pageNumber) || null;
    },
    [result],
  );

  const clauseIntersectsPage = useCallback(
    (clause: ClauseResult, pageNumber: number): boolean => {
      const range = getClauseRange(clause);
      const pageInfo = getPageInfo(pageNumber);
      if (range && pageInfo) {
        return (
          range.start < pageInfo.end_char && range.end > pageInfo.start_char
        );
      }
      if (clause.page_number !== undefined) {
        return clause.page_number === pageNumber;
      }
      return true;
    },
    [getClauseRange, getPageInfo],
  );

  const getClauseSearchCandidates = useCallback(
    (clause: ClauseResult, pageNumber?: number): string[] => {
      const sourceText = result?.extracted_text;
      const candidates: string[] = [];

      if (typeof sourceText === "string" && typeof pageNumber === "number") {
        const pageInfo = getPageInfo(pageNumber);
        const range = getClauseRange(clause);
        if (pageInfo && range) {
          const localStart = Math.max(range.start, pageInfo.start_char);
          const localEnd = Math.min(range.end, pageInfo.end_char);
          if (localEnd > localStart) {
            candidates.push(
              sourceText
                .slice(localStart, localEnd)
                .replace(/\s+/g, " ")
                .trim()
                .slice(0, 320),
            );
          }
        }
      }

      if (
        typeof sourceText === "string" &&
        Number.isInteger(clause.display_start_idx) &&
        Number.isInteger(clause.display_end_idx)
      ) {
        const ds = clause.display_start_idx as number;
        const de = clause.display_end_idx as number;
        if (ds >= 0 && de > ds && de <= sourceText.length) {
          candidates.push(
            sourceText.slice(ds, de).replace(/\s+/g, " ").trim().slice(0, 320),
          );
        }
      }

      if (
        typeof sourceText === "string" &&
        Number.isInteger(clause.start_idx) &&
        Number.isInteger(clause.end_idx)
      ) {
        const s = clause.start_idx as number;
        const e = clause.end_idx as number;
        if (s >= 0 && e > s && e <= sourceText.length) {
          candidates.push(
            sourceText.slice(s, e).replace(/\s+/g, " ").trim().slice(0, 260),
          );
        }
      }

      candidates.push(clause.span_display || "");
      candidates.push(clause.span_exact || "");
      candidates.push(clause.span || "");

      const unique: string[] = [];
      const seen = new Set<string>();
      for (const c of candidates) {
        const v = c.trim();
        if (!v) continue;
        const key = v.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        unique.push(v);
      }
      return unique;
    },
    [result, getClauseRange, getPageInfo],
  );

  const isClauseOnPage = useCallback(
    (clause: ClauseResult, pageNumber: number): boolean => {
      return (
        clause.page_number === undefined || clause.page_number === pageNumber
      );
    },
    [],
  );

  // Convert any color to rgba with given opacity
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

  // ── Helpers for text normalization and line grouping ───────────────────
  const normalizeForMatch = useCallback(
    (text: string): string =>
      text
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim(),
    [],
  );

  type LineInfo = {
    y: number;
    minX: number;
    maxX: number;
    avgHeight: number;
    count: number;
    indices: number[];
  };

  const buildPageIndex = useCallback(
    (
      pageContent: PageTextContent,
    ): {
      pageText: string;
      charToItemIdx: (number | undefined)[];
      allLines: LineInfo[];
      itemToLineIdx: Map<number, number>;
    } => {
      let pageText = "";
      const charToItemIdx: (number | undefined)[] = [];

      pageContent.items.forEach((item: TextItem, itemIdx: number) => {
        const normalizedItem = item.str
          .toLowerCase()
          .replace(/[^\w\s]/g, " ")
          .replace(/\s+/g, " ");
        for (let i = 0; i < normalizedItem.length; i++) {
          charToItemIdx.push(itemIdx);
          pageText += normalizedItem[i];
        }
        if (itemIdx < pageContent.items.length - 1) {
          charToItemIdx.push(itemIdx);
          pageText += " ";
        }
      });

      // Build line groups from all positioned items
      type PositionedItem = {
        item: TextItem;
        x: number;
        y: number;
        idx: number;
      };
      const allPositioned: PositionedItem[] = [];
      pageContent.items.forEach((item: TextItem, idx: number) => {
        const x = item.transform[4];
        const y = pageContent.viewport.height - item.transform[5] - item.height;
        if (x >= 0 && y >= 0 && item.width > 0 && item.height > 0) {
          allPositioned.push({ item, x, y, idx });
        }
      });

      const allLines: LineInfo[] = [];
      const itemToLineIdx = new Map<number, number>();

      allPositioned
        .sort((a, b) => a.y - b.y || a.x - b.x)
        .forEach((current) => {
          let lineIndex = -1;
          for (let i = 0; i < allLines.length; i++) {
            if (Math.abs(allLines[i].y - current.y) <= LINE_GROUP_TOLERANCE) {
              lineIndex = i;
              break;
            }
          }
          if (lineIndex === -1) {
            allLines.push({
              y: current.y,
              minX: current.x,
              maxX: current.x + current.item.width,
              avgHeight: current.item.height,
              count: 1,
              indices: [current.idx],
            });
            lineIndex = allLines.length - 1;
          } else {
            const line = allLines[lineIndex];
            line.minX = Math.min(line.minX, current.x);
            line.maxX = Math.max(line.maxX, current.x + current.item.width);
            line.avgHeight =
              (line.avgHeight * line.count + current.item.height) /
              (line.count + 1);
            line.count += 1;
            line.indices.push(current.idx);
          }
          itemToLineIdx.set(current.idx, lineIndex);
        });

      return { pageText, charToItemIdx, allLines, itemToLineIdx };
    },
    [],
  );

  // Helper: produce line-level position rectangles from matched item indices
  const lineRectsFromItems = useCallback(
    (
      matchedItemIndices: Set<number>,
      pageContent: PageTextContent,
      allLines: LineInfo[],
      itemToLineIdx: Map<number, number>,
      extendToFullClause: boolean,
    ): { x: number; y: number; width: number; height: number }[] => {
      const positions: {
        x: number;
        y: number;
        width: number;
        height: number;
      }[] = [];

      const matchedLineIndices = new Set<number>();
      matchedItemIndices.forEach((idx) => {
        const lineIdx = itemToLineIdx.get(idx);
        if (lineIdx !== undefined) matchedLineIndices.add(lineIdx);
      });

      if (matchedLineIndices.size === 0) return positions;

      let lineIndicesToDraw: Set<number>;

      if (extendToFullClause && matchedLineIndices.size > 0) {
        // Extend to all lines between the first and last matched line
        const sorted = Array.from(matchedLineIndices).sort((a, b) => a - b);
        const minLine = sorted[0];
        const maxLine = sorted[sorted.length - 1];
        lineIndicesToDraw = new Set<number>();
        for (let i = minLine; i <= maxLine; i++) {
          lineIndicesToDraw.add(i);
        }
      } else {
        lineIndicesToDraw = matchedLineIndices;
      }

      lineIndicesToDraw.forEach((lineIndex) => {
        const line = allLines[lineIndex];
        if (!line) return;
        positions.push({
          x: Math.max(0, line.minX - 4),
          y: Math.max(0, line.y),
          width: Math.min(line.maxX - line.minX + 8, pageWidth - line.minX),
          height: Math.max(line.avgHeight, 14),
        });
      });

      return positions;
    },
    [pageWidth],
  );

  // Find text positions for a given span in a page
  const findTextPositions = useCallback(
    (
      span: string,
      pageContent: PageTextContent,
      isSearch = false,
    ): { x: number; y: number; width: number; height: number }[] => {
      if (!span || !pageContent || pageContent.items.length === 0) return [];

      const normalizedSpan = normalizeForMatch(span);
      // Use 500 chars for clause highlights (up from 200), 40 for search
      const searchLength = isSearch
        ? Math.min(normalizedSpan.length, 40)
        : Math.min(normalizedSpan.length, 500);
      const searchText = normalizedSpan.substring(0, searchLength);

      const { pageText, charToItemIdx, allLines, itemToLineIdx } =
        buildPageIndex(pageContent);

      // Strategy 1: Exact substring match (try progressively shorter)
      let searchIdx = pageText.indexOf(searchText);
      if (searchIdx === -1 && searchText.length > 120)
        searchIdx = pageText.indexOf(searchText.substring(0, 120));
      if (searchIdx === -1 && searchText.length > 80)
        searchIdx = pageText.indexOf(searchText.substring(0, 80));
      if (searchIdx === -1 && searchText.length > 50)
        searchIdx = pageText.indexOf(searchText.substring(0, 50));
      if (searchIdx === -1 && searchText.length > 30)
        searchIdx = pageText.indexOf(searchText.substring(0, 30));

      // Strategy 2: Progressive word matching — handles whitespace/punctuation mismatches
      if (searchIdx === -1) {
        const words = normalizedSpan.split(" ").filter((w) => w.length > 2);

        // Try sequences of 5, 4, 3 consecutive significant words
        for (
          let windowSize = Math.min(5, words.length);
          windowSize >= 3 && searchIdx === -1;
          windowSize--
        ) {
          for (
            let start = 0;
            start <= words.length - windowSize && searchIdx === -1;
            start++
          ) {
            const phrase = words.slice(start, start + windowSize).join(" ");
            searchIdx = pageText.indexOf(phrase);
          }
        }
      }

      // Strategy 3: Find the full clause extent using the WHOLE normalized span text
      // This catches the rest of the clause beyond the initial search window
      if (searchIdx !== -1) {
        // Try to extend the match to cover as much of the full normalizedSpan as possible
        let bestMatchEnd = Math.min(
          searchIdx + searchText.length,
          pageText.length,
        );

        // If we have more span text, see if the full span exists starting at searchIdx
        if (normalizedSpan.length > searchLength) {
          // Try to match progressively more of the full span
          const fullMatch = pageText.indexOf(normalizedSpan, searchIdx);
          if (fullMatch === searchIdx) {
            bestMatchEnd = Math.min(
              searchIdx + normalizedSpan.length,
              pageText.length,
            );
          } else {
            // Try a longer portion than searchLength
            for (
              let tryLen = normalizedSpan.length;
              tryLen > searchLength;
              tryLen = Math.floor(tryLen * 0.7)
            ) {
              const tryText = normalizedSpan.substring(0, tryLen);
              if (pageText.indexOf(tryText, searchIdx) === searchIdx) {
                bestMatchEnd = Math.min(searchIdx + tryLen, pageText.length);
                break;
              }
            }
          }
        }

        const matchedItemIndices = new Set<number>();
        for (let i = searchIdx; i < bestMatchEnd; i++) {
          const itemIdx = charToItemIdx[i];
          if (itemIdx !== undefined) matchedItemIndices.add(itemIdx);
        }

        if (matchedItemIndices.size > 0) {
          return lineRectsFromItems(
            matchedItemIndices,
            pageContent,
            allLines,
            itemToLineIdx,
            !isSearch, // extend to full clause for non-search highlights
          );
        }
      }

      return [];
    },
    [normalizeForMatch, buildPageIndex, lineRectsFromItems],
  );

  const findClausePositions = useCallback(
    (
      clause: ClauseResult,
      pageContent: PageTextContent,
      pageNumber: number,
    ) => {
      // Strategy A: Text-based matching (try each candidate)
      const candidates = getClauseSearchCandidates(clause, pageNumber);
      for (const candidate of candidates) {
        const positions = findTextPositions(candidate, pageContent);
        if (positions.length > 0) {
          return positions;
        }
      }

      // Strategy B: Character-offset-based fallback
      // Use backend-provided char offsets + page boundaries to estimate
      // which lines in the page contain the clause.
      const range = getClauseRange(clause);
      const pageInfo = getPageInfo(pageNumber);
      if (range && pageInfo && pageInfo.end_char > pageInfo.start_char) {
        const { pageText, charToItemIdx, allLines, itemToLineIdx } =
          buildPageIndex(pageContent);

        // Map clause char range to proportional position within the page
        const pageLen = pageInfo.end_char - pageInfo.start_char;
        const clauseStartInPage = Math.max(
          0,
          range.start - pageInfo.start_char,
        );
        const clauseEndInPage = Math.min(
          pageLen,
          range.end - pageInfo.start_char,
        );
        if (clauseEndInPage > clauseStartInPage && pageText.length > 0) {
          // Map char offsets to approximate positions in normalized page text
          const ratio = pageText.length / pageLen;
          const normStart = Math.floor(clauseStartInPage * ratio);
          const normEnd = Math.min(
            Math.ceil(clauseEndInPage * ratio),
            pageText.length,
          );

          const matchedItemIndices = new Set<number>();
          for (let i = normStart; i < normEnd; i++) {
            const itemIdx = charToItemIdx[i];
            if (itemIdx !== undefined) matchedItemIndices.add(itemIdx);
          }

          if (matchedItemIndices.size > 0) {
            return lineRectsFromItems(
              matchedItemIndices,
              pageContent,
              allLines,
              itemToLineIdx,
              true,
            );
          }
        }
      }

      return [] as { x: number; y: number; width: number; height: number }[];
    },
    [
      findTextPositions,
      getClauseSearchCandidates,
      getClauseRange,
      getPageInfo,
      buildPageIndex,
      lineRectsFromItems,
    ],
  );

  const getSearchMatchRects = useCallback(
    (pageContent: PageTextContent, charOffset: number, length: number) => {
      const { charToItemIdx, allLines, itemToLineIdx } =
        buildPageIndex(pageContent);
      const matchedItemIndices = new Set<number>();
      for (let i = charOffset; i < charOffset + length; i++) {
        const itemIdx = charToItemIdx[i];
        if (itemIdx !== undefined) matchedItemIndices.add(itemIdx);
      }
      return lineRectsFromItems(
        matchedItemIndices,
        pageContent,
        allLines,
        itemToLineIdx,
        false,
      );
    },
    [buildPageIndex, lineRectsFromItems],
  );

  // ── Helper for exact-position scrolling ──────────────────────────────────────────────
  const scrollToExactPosition = useCallback(
    (
      pageIndex: number,
      pos: { x: number; y: number; width: number; height: number } | undefined,
      delayMs = 150,
    ) => {
      const pageEl = pageContainerRefs.current[pageIndex];
      if (!pageEl) return;

      setTimeout(() => {
        if (pos) {
          const tempScrollTarget = document.createElement("div");
          tempScrollTarget.style.position = "absolute";
          tempScrollTarget.style.left = `${pos.x}px`;
          tempScrollTarget.style.top = `${pos.y}px`;
          tempScrollTarget.style.width = `${pos.width}px`;
          tempScrollTarget.style.height = `${pos.height}px`;
          tempScrollTarget.style.visibility = "hidden";
          tempScrollTarget.style.pointerEvents = "none";
          pageEl.appendChild(tempScrollTarget);

          tempScrollTarget.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });

          setTimeout(() => {
            if (pageEl.contains(tempScrollTarget)) {
              pageEl.removeChild(tempScrollTarget);
            }
          }, 1000);
        } else {
          pageEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, delayMs);
    },
    [],
  );

  // ── Auto-scroll to the page containing the active clause ───────────────
  useEffect(() => {
    if (!activeClause || pageTextContents.length === 0) {
      return;
    }

    if (
      typeof activeClause.page_number === "number" &&
      activeClause.page_number >= 1 &&
      activeClause.page_number <= pageTextContents.length
    ) {
      const targetIndex = activeClause.page_number - 1;
      const mappedPositions = findClausePositions(
        activeClause,
        pageTextContents[targetIndex],
        targetIndex + 1,
      );
      if (mappedPositions.length > 0) {
        scrollToExactPosition(targetIndex, mappedPositions[0], 250);
        return;
      }
    }

    // Find the first page where the active clause text appears
    for (let i = 0; i < pageTextContents.length; i++) {
      const positions = findClausePositions(
        activeClause,
        pageTextContents[i],
        i + 1,
      );
      if (positions.length > 0) {
        scrollToExactPosition(i, positions[0], 250);
        break;
      }
    }
  }, [
    activeClause,
    pageTextContents,
    findClausePositions,
    scrollToExactPosition,
  ]);

  // ── Auto-scroll to the exact position of the active search match ───────
  useEffect(() => {
    if (activeSearchPageIndex !== null && activeSearchCharOffset !== null) {
      const pageContent = pageTextContents[activeSearchPageIndex];
      if (!pageContent) return;

      const positions = getSearchMatchRects(
        pageContent,
        activeSearchCharOffset,
        highlightedText.length,
      );

      if (positions.length > 0) {
        scrollToExactPosition(activeSearchPageIndex, positions[0], 100);
      } else {
        scrollToExactPosition(activeSearchPageIndex, undefined, 100);
      }
    }
  }, [
    activeSearchPageIndex,
    activeSearchCharOffset,
    pageTextContents,
    getSearchMatchRects,
    highlightedText.length,
    scrollToExactPosition,
  ]);

  // ── Draw clause + active + search highlights on canvas ────────────────
  const drawHighlights = useCallback(() => {
    if (!result?.result || pageTextContents.length === 0) return;

    console.log("Drawing highlights...", {
      totalClauses: result.result.length,
      activeClause: activeClause?.clause_type,
      highlightedText: highlightedText?.substring(0, 30),
    });

    setTimeout(() => {
      pageTextContents.forEach((pageContent, pageIndex) => {
        const canvas = canvasRefs.current[pageIndex];
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = pageWidth;
        canvas.height = pageHeights[pageIndex] || 1100;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Pass 1: Draw all regular clause highlights (coloured, 30% opacity)
        result.result.forEach((clause: ClauseResult) => {
          if (!clauseIntersectsPage(clause, pageIndex + 1)) return;
          if (!selectedClauseTypes.has(clause.clause_type)) return;
          if (
            clause.clause_type !== "Unknown clause" &&
            clause.confidence < minConfidence / 100
          )
            return;
          // Skip active clause — drawn separately below with stronger style
          if (activeClause && isSameClause(clause, activeClause)) return;

          const color =
            clause.clause_type === "Unknown clause"
              ? "#F97316"
              : colorMap[clause.clause_type] || "#6b7280";
          const positions = findClausePositions(
            clause,
            pageContent,
            pageIndex + 1,
          );
          positions.forEach((pos) => {
            ctx.fillStyle = colorWithOpacity(color, 0.3);
            ctx.fillRect(pos.x, pos.y, pos.width, pos.height);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(pos.x, pos.y, pos.width, pos.height);
          });
        });

        // Pass 2: Draw the ACTIVE (selected) clause with a strong yellow highlight
        if (activeClause) {
          const positions = findClausePositions(
            activeClause,
            pageContent,
            pageIndex + 1,
          );
          positions.forEach((pos) => {
            // Bright yellow fill
            ctx.fillStyle = "rgba(253, 224, 71, 0.65)";
            ctx.fillRect(pos.x - 2, pos.y - 2, pos.width + 4, pos.height + 4);
            // Bold amber border
            ctx.strokeStyle = "#d97706";
            ctx.lineWidth = 2.5;
            ctx.strokeRect(pos.x - 2, pos.y - 2, pos.width + 4, pos.height + 4);
          });
        }

        // Pass 3: Draw search results
        if (highlightedText && searchMatches.length > 0) {
          const pageMatches = searchMatches.filter(
            (m) => m.pageIndex === pageIndex,
          );
          pageMatches.forEach((match) => {
            const positions = getSearchMatchRects(
              pageContent,
              match.charOffset,
              highlightedText.length,
            );
            const isActive =
              activeSearchPageIndex === pageIndex &&
              activeSearchCharOffset === match.charOffset;

            positions.forEach((pos) => {
              if (isActive) {
                ctx.fillStyle = "rgba(255, 152, 0, 0.6)"; // Stronger orange
                ctx.fillRect(
                  pos.x - 1,
                  pos.y - 1,
                  pos.width + 2,
                  pos.height + 2,
                );
                ctx.strokeStyle = "#ea580c"; // Tailwind orange-600
                ctx.lineWidth = 2;
                ctx.strokeRect(
                  pos.x - 1,
                  pos.y - 1,
                  pos.width + 2,
                  pos.height + 2,
                );
              } else {
                ctx.fillStyle = "rgba(251, 191, 36, 0.4)"; // Yellow
                ctx.fillRect(
                  pos.x - 1,
                  pos.y - 1,
                  pos.width + 2,
                  pos.height + 2,
                );
                ctx.strokeStyle = "#f59e0b";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(
                  pos.x - 1,
                  pos.y - 1,
                  pos.width + 2,
                  pos.height + 2,
                );
              }
            });
          });
        }
      });
    }, 300);
  }, [
    result,
    colorMap,
    pageTextContents,
    pageWidth,
    pageHeights,
    highlightedText,
    searchMatches,
    activeClause,
    activeSearchPageIndex,
    activeSearchCharOffset,
    selectedClauseTypes,
    minConfidence,
    colorWithOpacity,
    findClausePositions,
    clauseIntersectsPage,
    isSameClause,
    getSearchMatchRects,
  ]);

  useEffect(() => {
    drawHighlights();
  }, [drawHighlights]);

  if (!isClient) {
    return (
      <div className="flex items-center justify-center gap-3 p-12 text-center">
        <Loader2 className="h-6 w-6 animate-spin text-brand" />
        <span className="text-muted">Initializing PDF viewer...</span>
      </div>
    );
  }

  return (
    <Document
      file={file}
      onLoadSuccess={onDocumentLoadSuccess}
      loading={
        <div className="flex items-center justify-center gap-3 p-12 text-center text-muted">
          <Loader2 className="h-6 w-6 animate-spin" />
          Loading PDF...
        </div>
      }
      error={
        <div className="p-12 text-center text-red-600">
          Failed to load PDF. The file may be corrupted or unsupported.
        </div>
      }
    >
      {Array.from(new Array(numPages), (_, index) => (
        <div
          key={index}
          className="relative mb-6"
          ref={(el) => {
            pageContainerRefs.current[index] = el;
          }}
        >
          <Page
            pageNumber={index + 1}
            width={pageWidth}
            className="mx-auto rounded-sm shadow-[0_12px_28px_rgba(58,46,32,0.15)]"
          />
          <canvas
            ref={(el) => {
              if (el) canvasRefs.current[index] = el;
            }}
            className="absolute top-0 left-0 pointer-events-none"
            width={pageWidth}
            height={pageHeights[index] || 1100}
            style={{ width: pageWidth, height: pageHeights[index] || 1100 }}
          />
          <div className="py-2 text-center text-xs uppercase tracking-[0.11em] text-muted">
            Page {index + 1} of {numPages}
          </div>
        </div>
      ))}
    </Document>
  );
}
