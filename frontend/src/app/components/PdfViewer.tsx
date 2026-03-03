"use client";

import { useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { ClauseResult, PageTextContent, TextItem } from "../types";

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
  result: { result: ClauseResult[] } | null;
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  highlightedText: string;
  activeClause: ClauseResult | null;
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
  activeClause,
  isClient,
  onDocumentLoadSuccess,
}: Props) {
  const canvasRefs = useRef<HTMLCanvasElement[]>([]);
  // Refs to each page container div for scrolling
  const pageContainerRefs = useRef<(HTMLDivElement | null)[]>([]);

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

  // Find text positions for a given span in a page
  const findTextPositions = useCallback(
    (
      span: string,
      pageContent: PageTextContent,
      isSearch = false,
    ): { x: number; y: number; width: number; height: number }[] => {
      const positions: {
        x: number;
        y: number;
        width: number;
        height: number;
      }[] = [];
      if (!span || !pageContent || pageContent.items.length === 0)
        return positions;

      const normalizedSpan = span
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      const searchLength = isSearch
        ? Math.min(normalizedSpan.length, 40)
        : Math.min(normalizedSpan.length, 200);
      const searchText = normalizedSpan.substring(0, searchLength);

      let pageText = "";
      const charToItemMap: {
        char: number;
        itemIdx: number;
        localPos: number;
      }[] = [];

      pageContent.items.forEach((item: TextItem, itemIdx: number) => {
        const normalizedItem = item.str
          .toLowerCase()
          .replace(/[^\w\s]/g, " ")
          .replace(/\s+/g, " ");
        for (let i = 0; i < normalizedItem.length; i++) {
          charToItemMap.push({ char: pageText.length, itemIdx, localPos: i });
          pageText += normalizedItem[i];
        }
        if (itemIdx < pageContent.items.length - 1) {
          charToItemMap.push({
            char: pageText.length,
            itemIdx,
            localPos: normalizedItem.length,
          });
          pageText += " ";
        }
      });

      let searchIdx = pageText.indexOf(searchText);
      if (searchIdx === -1 && searchText.length > 80)
        searchIdx = pageText.indexOf(searchText.substring(0, 80));
      if (searchIdx === -1 && searchText.length > 50)
        searchIdx = pageText.indexOf(searchText.substring(0, 50));
      if (searchIdx === -1) {
        const words = searchText
          .split(" ")
          .filter((w) => w.length > 3)
          .slice(0, 6);
        if (words.length >= 3) searchIdx = pageText.indexOf(words.join(" "));
      }

      if (searchIdx !== -1) {
        const matchEnd = Math.min(
          searchIdx + searchText.length,
          pageText.length,
        );
        const matchedItemIndices = new Set<number>();
        for (let i = searchIdx; i < matchEnd; i++) {
          const mapping = charToItemMap.find((m) => m.char === i);
          if (mapping) matchedItemIndices.add(mapping.itemIdx);
        }

        const matchedItems: {
          item: TextItem;
          x: number;
          y: number;
          idx: number;
        }[] = [];
        matchedItemIndices.forEach((idx) => {
          const item = pageContent.items[idx];
          if (item) {
            const x = item.transform[4];
            const y =
              pageContent.viewport.height - item.transform[5] - item.height;
            if (x >= 0 && y >= 0 && item.width > 0 && item.height > 0)
              matchedItems.push({ item, x, y, idx });
          }
        });

        if (matchedItems.length > 0) {
          const lines: (typeof matchedItems)[] = [];
          matchedItems
            .sort((a, b) => a.y - b.y || a.x - b.x)
            .forEach((current) => {
              let addedToLine = false;
              for (const line of lines) {
                if (Math.abs(line[0].y - current.y) < 3) {
                  line.push(current);
                  addedToLine = true;
                  break;
                }
              }
              if (!addedToLine) lines.push([current]);
            });

          lines.forEach((line) => {
            line.sort((a, b) => a.x - b.x);
            const minX = line[0].x;
            const maxX =
              line[line.length - 1].x + line[line.length - 1].item.width;
            const avgHeight =
              line.reduce((s, i) => s + i.item.height, 0) / line.length;
            const avgY = line.reduce((s, l) => s + l.y, 0) / line.length;
            positions.push({
              x: Math.max(0, minX - 1),
              y: Math.max(0, avgY),
              width: Math.min(maxX - minX + 2, pageWidth - minX),
              height: Math.max(avgHeight, 14),
            });
          });
        }
      }

      return positions;
    },
    [pageWidth],
  );

  // ── Auto-scroll to the page containing the active clause ───────────────
  useEffect(() => {
    if (!activeClause || pageTextContents.length === 0) {
      console.log("PDF Scroll: Skipping (no active clause or page content)");
      return;
    }

    console.log("PDF Scroll: Looking for active clause on pages...", {
      clause: activeClause.span.substring(0, 50) + "...",
      numPages: pageTextContents.length,
    });

    // Find the first page where the active clause text appears
    for (let i = 0; i < pageTextContents.length; i++) {
      const positions = findTextPositions(
        activeClause.span,
        pageTextContents[i],
      );
      if (positions.length > 0) {
        console.log(
          `Found clause on page ${i + 1}, scrolling to it...`,
          positions,
        );
        // Delay slightly so canvas highlights are drawn first
        setTimeout(() => {
          pageContainerRefs.current[i]?.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
          console.log(`Scrolled to page ${i + 1}`);
        }, 350);
        break;
      }
    }
  }, [activeClause, pageTextContents, findTextPositions]);

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
          if (!selectedClauseTypes.has(clause.clause_type)) return;
          if (
            clause.clause_type !== "Unknown clause" &&
            clause.confidence < minConfidence / 100
          )
            return;
          // Skip active clause — drawn separately below with stronger style
          if (activeClause && clause.span === activeClause.span) return;

          const color =
            clause.clause_type === "Unknown clause"
              ? "#F97316"
              : colorMap[clause.clause_type] || "#6b7280";
          const positions = findTextPositions(clause.span, pageContent);
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
          const positions = findTextPositions(activeClause.span, pageContent);
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

        // Pass 3: Search highlight (if from search bar, not clause click)
        if (
          highlightedText &&
          (!activeClause || highlightedText !== activeClause.span)
        ) {
          const positions = findTextPositions(
            highlightedText,
            pageContent,
            true,
          );
          positions.forEach((pos) => {
            ctx.fillStyle = "rgba(251, 191, 36, 0.4)";
            ctx.fillRect(pos.x - 1, pos.y - 1, pos.width + 2, pos.height + 2);
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(pos.x - 1, pos.y - 1, pos.width + 2, pos.height + 2);
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
    findTextPositions,
    highlightedText,
    activeClause,
    selectedClauseTypes,
    minConfidence,
    colorWithOpacity,
  ]);

  useEffect(() => {
    drawHighlights();
  }, [drawHighlights]);

  if (!isClient) {
    return (
      <div className="p-12 text-center flex items-center justify-center gap-3">
        <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
        <span className="text-gray-600">Initializing PDF viewer...</span>
      </div>
    );
  }

  return (
    <Document
      file={file}
      onLoadSuccess={onDocumentLoadSuccess}
      loading={
        <div className="p-12 text-center flex items-center justify-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin" />
          Loading PDF...
        </div>
      }
      error={
        <div className="p-12 text-center text-red-500">
          Failed to load PDF. The file may be corrupted or unsupported.
        </div>
      }
    >
      {Array.from(new Array(numPages), (_, index) => (
        <div
          key={index}
          className="relative mb-4"
          ref={(el) => {
            pageContainerRefs.current[index] = el;
          }}
        >
          <Page
            pageNumber={index + 1}
            width={pageWidth}
            className="shadow-sm"
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
          <div className="text-center text-xs text-black py-2">
            Page {index + 1} of {numPages}
          </div>
        </div>
      ))}
    </Document>
  );
}
