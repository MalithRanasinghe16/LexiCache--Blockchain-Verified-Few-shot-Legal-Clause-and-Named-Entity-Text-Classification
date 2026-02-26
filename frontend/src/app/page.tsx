"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import {
  Palette,
  Upload,
  Search,
  Eye,
  FileText,
  CheckCircle,
  AlertCircle,
  Loader2,
  Filter,
  X,
} from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

type ClauseResult = {
  clause_type: string;
  span: string;
  confidence: number;
};

type TextItem = {
  str: string;
  transform: number[];
  width: number;
  height: number;
};

type PageTextContent = {
  pageIndex: number;
  items: TextItem[];
  viewport: { width: number; height: number; scale: number };
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [highlightedText, setHighlightedText] = useState("");
  const [pageWidth, setPageWidth] = useState(780);
  const [pdfDocument, setPdfDocument] = useState<any>(null);
  const [pageTextContents, setPageTextContents] = useState<PageTextContent[]>(
    [],
  );
  const [pageHeights, setPageHeights] = useState<number[]>([]);
  const [documentText, setDocumentText] = useState<string>("");
  const [fileType, setFileType] = useState<string>("pdf");

  // Filter states
  const [selectedClauseTypes, setSelectedClauseTypes] = useState<Set<string>>(
    new Set(),
  );
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [showFilters, setShowFilters] = useState(false);

  // Color map for clauses - will be populated with random colors
  const [colorMap, setColorMap] = useState<Record<string, string>>({});

  // Generate a random visible color on white background
  const generateRandomColor = useCallback(() => {
    // Generate colors that are visible on white background (darker, saturated colors)
    const hue = Math.floor(Math.random() * 360);
    const saturation = 60 + Math.floor(Math.random() * 30); // 60-90%
    const lightness = 35 + Math.floor(Math.random() * 20); // 35-55% (dark enough to see on white)
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }, []);

  // Initialize colors for clause types when results are loaded
  useEffect(() => {
    if (result?.result && result.result.length > 0) {
      const newColorMap: Record<string, string> = { ...colorMap };
      const types = new Set<string>();
      result.result.forEach((clause: ClauseResult) => {
        types.add(clause.clause_type);
        if (!newColorMap[clause.clause_type]) {
          newColorMap[clause.clause_type] = generateRandomColor();
        }
      });
      setColorMap(newColorMap);
      // Initialize filter to show all clauses
      if (selectedClauseTypes.size === 0) {
        setSelectedClauseTypes(new Set(types));
      }
    }
  }, [result, generateRandomColor]);

  // Convert any color format to rgba with opacity
  const colorWithOpacity = useCallback((color: string, opacity: number) => {
    // If it's an HSL color
    if (color.startsWith("hsl")) {
      // Extract hue, saturation, lightness
      const match = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
      if (match) {
        const [, h, s, l] = match;
        return `hsla(${h}, ${s}%, ${l}%, ${opacity})`;
      }
    }

    // If it's a hex color
    if (color.startsWith("#")) {
      const hex = color.replace("#", "");
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }

    // If it's already rgba or other format, return as is
    return color;
  }, []);

  const canvasRefs = useRef<HTMLCanvasElement[]>([]);

  // Handle window resize for responsive PDF width
  useEffect(() => {
    const handleResize = () => {
      setPageWidth(Math.min(780, window.innerWidth - 120));
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Extract text content with positions from all pages
  const extractTextContent = useCallback(
    async (pdf: any) => {
      const contents: PageTextContent[] = [];
      const heights: number[] = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1 });
        const scale = pageWidth / viewport.width;
        const scaledViewport = page.getViewport({ scale });

        heights.push(scaledViewport.height);

        const textContent = await page.getTextContent();
        const items: TextItem[] = textContent.items.map((item: any) => ({
          str: item.str,
          transform: item.transform.map(
            (t: number, idx: number) =>
              idx === 4 || idx === 5 ? t * scale : t * scale, // Scale positions
          ),
          width: item.width * scale,
          height: item.height * scale,
        }));

        contents.push({
          pageIndex: i - 1,
          items,
          viewport: {
            width: scaledViewport.width,
            height: scaledViewport.height,
            scale,
          },
        });
      }

      setPageTextContents(contents);
      setPageHeights(heights);
    },
    [pageWidth],
  );

  // When PDF loads, extract text
  useEffect(() => {
    if (pdfDocument) {
      extractTextContent(pdfDocument);
    }
  }, [pdfDocument, extractTextContent]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setHighlightedText("");
    setPdfDocument(null);
    setPageTextContents([]);
    setDocumentText("");
    setFileType(
      selectedFile?.name.toLowerCase().endsWith(".pdf") ? "pdf" : "docx",
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("http://localhost:8000/upload-file", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(
            errorData.detail || `Upload failed with status ${res.status}`,
          );
        }
        const data = await res.json();
        // Normalize result to always be an array
        if (data.result && !Array.isArray(data.result)) {
          data.result = [data.result];
        }
        setResult(data);
        setDocumentText(data.extracted_text || "");
        setFileType(data.file_type || "pdf");
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  // Find text positions for a given clause span - IMPROVED for Word-like accuracy
  const findTextPositions = useCallback(
    (span: string, pageContent: PageTextContent, isSearch: boolean = false) => {
      const positions: {
        x: number;
        y: number;
        width: number;
        height: number;
      }[] = [];

      if (!span || !pageContent || pageContent.items.length === 0) {
        return positions;
      }

      // Normalize the span for matching - preserve more structure
      const normalizedSpan = span
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      // For search, use shorter text; for clauses, use more text for better accuracy
      const searchLength = isSearch
        ? Math.min(normalizedSpan.length, 40)
        : Math.min(normalizedSpan.length, 200);

      const searchText = normalizedSpan.substring(0, searchLength);

      // Build page text with better tracking of positions
      let pageText = "";
      const charToItemMap: {
        char: number;
        itemIdx: number;
        localPos: number;
      }[] = [];

      pageContent.items.forEach((item, itemIdx) => {
        const itemStr = item.str;
        const normalizedItem = itemStr
          .toLowerCase()
          .replace(/[^\w\s]/g, " ")
          .replace(/\s+/g, " ");

        for (let i = 0; i < normalizedItem.length; i++) {
          charToItemMap.push({
            char: pageText.length,
            itemIdx,
            localPos: i,
          });
          pageText += normalizedItem[i];
        }
        // Add space between items
        if (itemIdx < pageContent.items.length - 1) {
          charToItemMap.push({
            char: pageText.length,
            itemIdx,
            localPos: normalizedItem.length,
          });
          pageText += " ";
        }
      });

      // Find match with multiple strategies
      let searchIdx = pageText.indexOf(searchText);

      // Fallback strategy 1: Try first sentence or 80 chars
      if (searchIdx === -1 && searchText.length > 80) {
        const shorterSearch = searchText.substring(0, 80);
        searchIdx = pageText.indexOf(shorterSearch);
      }

      // Fallback strategy 2: Try first 50 chars
      if (searchIdx === -1 && searchText.length > 50) {
        const shorterSearch = searchText.substring(0, 50);
        searchIdx = pageText.indexOf(shorterSearch);
      }

      // Fallback strategy 3: Try significant words
      if (searchIdx === -1) {
        const words = searchText
          .split(" ")
          .filter((w) => w.length > 3)
          .slice(0, 6);
        if (words.length >= 3) {
          const keywordSearch = words.join(" ");
          searchIdx = pageText.indexOf(keywordSearch);
        }
      }

      if (searchIdx !== -1) {
        const matchEnd = Math.min(
          searchIdx + searchText.length,
          pageText.length,
        );

        // Get all text items that contribute to this match
        const matchedItemIndices = new Set<number>();
        for (let i = searchIdx; i < matchEnd; i++) {
          const mapping = charToItemMap.find((m) => m.char === i);
          if (mapping) {
            matchedItemIndices.add(mapping.itemIdx);
          }
        }

        // Build position data for matched items
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
            // PDF Y coordinates: 0 at bottom, transform[5] is baseline
            // Convert to top-down and adjust for text height to get top of text
            const y =
              pageContent.viewport.height - item.transform[5] - item.height;

            if (x >= 0 && y >= 0 && item.width > 0 && item.height > 0) {
              matchedItems.push({ item, x, y, idx });
            }
          }
        });

        if (matchedItems.length > 0) {
          // Group by lines (Y position within 3px for better accuracy)
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
              if (!addedToLine) {
                lines.push([current]);
              }
            });

          // Create precise highlight boxes for each line
          lines.forEach((line) => {
            // Sort items in line by X position
            line.sort((a, b) => a.x - b.x);

            const minX = line[0].x;
            const maxX =
              line[line.length - 1].x + line[line.length - 1].item.width;
            const avgHeight =
              line.reduce((sum, item) => sum + item.item.height, 0) /
              line.length;
            const avgY = line.reduce((sum, l) => sum + l.y, 0) / line.length;

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

  // Draw highlights on PDF pages - with filtering support
  const drawHighlights = useCallback(() => {
    if (!result?.result || pageTextContents.length === 0) return;

    console.log(
      `Drawing highlights for ${result.result.length} clauses on ${pageTextContents.length} pages`,
    );

    // Wait for canvas refs to be ready
    setTimeout(() => {
      pageTextContents.forEach((pageContent, pageIndex) => {
        const canvas = canvasRefs.current[pageIndex];
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Set canvas size to match page
        canvas.width = pageWidth;
        canvas.height = pageHeights[pageIndex] || 1100;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw highlights for each clause (with filtering)
        result.result.forEach((clause: ClauseResult, idx: number) => {
          // Apply filters
          if (!selectedClauseTypes.has(clause.clause_type)) return;
          if (clause.confidence < minConfidence / 100) return;

          const color = colorMap[clause.clause_type] || "#6b7280";
          const positions = findTextPositions(clause.span, pageContent);

          console.log(
            `Clause ${idx} (${clause.clause_type}): Found ${positions.length} positions on page ${pageIndex + 1}`,
          );

          positions.forEach((pos, posIdx) => {
            console.log(
              `  Position ${posIdx}: x=${pos.x.toFixed(1)}, y=${pos.y.toFixed(1)}, w=${pos.width.toFixed(1)}, h=${pos.height.toFixed(1)}`,
            );

            // Draw highlight background with proper opacity
            ctx.fillStyle = colorWithOpacity(color, 0.3); // 30% opacity
            ctx.fillRect(pos.x, pos.y, pos.width, pos.height);

            // Draw clean border
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(pos.x, pos.y, pos.width, pos.height);
          });
        });

        // Also highlight search term if present
        if (highlightedText) {
          const positions = findTextPositions(
            highlightedText,
            pageContent,
            true,
          );
          positions.forEach((pos) => {
            ctx.fillStyle = "rgba(251, 191, 36, 0.4)"; // Yellow highlight with opacity
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
    selectedClauseTypes,
    minConfidence,
    colorWithOpacity,
  ]);

  // Redraw when result, colors, or text content changes
  useEffect(() => {
    drawHighlights();
  }, [drawHighlights]);

  const handleSearch = () => {
    if (searchTerm.trim()) {
      setHighlightedText(searchTerm);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setNumPages(null);
    setHighlightedText("");
    setSearchTerm("");
    setPdfDocument(null);
    setPageTextContents([]);
    setSelectedClauseTypes(new Set());
    setMinConfidence(0);
    setShowFilters(false);
    setDocumentText("");
    setFileType("pdf");
  };

  // Toggle clause type filter
  const toggleClauseType = (type: string) => {
    const newSet = new Set(selectedClauseTypes);
    if (newSet.has(type)) {
      newSet.delete(type);
    } else {
      newSet.add(type);
    }
    setSelectedClauseTypes(newSet);
  };

  // Get filtered clauses for display
  const getFilteredClauses = (): ClauseResult[] => {
    if (!result?.result) return [];
    return result.result.filter(
      (clause: ClauseResult) =>
        selectedClauseTypes.has(clause.clause_type) &&
        clause.confidence >= minConfidence / 100,
    );
  };

  // Render highlighted text for DOC/DOCX files
  const renderHighlightedText = () => {
    if (!documentText || !result?.result) return documentText;

    // Create an array of text segments with their highlight info
    const segments: { text: string; color?: string; clauseType?: string }[] =
      [];
    let lastIndex = 0;

    // Get all clause positions in the text
    const clausePositions: {
      start: number;
      end: number;
      color: string;
      clauseType: string;
    }[] = [];

    getFilteredClauses().forEach((clause) => {
      const normalizedDocText = documentText.toLowerCase();
      const normalizedSpan = clause.span.toLowerCase();

      // Find the clause in the document
      const searchLength = Math.min(normalizedSpan.length, 200);
      const searchText = normalizedSpan.substring(0, searchLength);
      const index = normalizedDocText.indexOf(searchText);

      if (index !== -1) {
        clausePositions.push({
          start: index,
          end: index + searchText.length,
          color: colorMap[clause.clause_type] || "#6b7280",
          clauseType: clause.clause_type,
        });
      }
    });

    // Also highlight search term if present
    if (highlightedText) {
      const searchIndex = documentText
        .toLowerCase()
        .indexOf(highlightedText.toLowerCase());
      if (searchIndex !== -1) {
        clausePositions.push({
          start: searchIndex,
          end: searchIndex + highlightedText.length,
          color: "#fbbf24",
          clauseType: "Search Result",
        });
      }
    }

    // Sort by start position
    clausePositions.sort((a, b) => a.start - b.start);

    // Build segments
    clausePositions.forEach((pos) => {
      // Add non-highlighted text before this clause
      if (lastIndex < pos.start) {
        segments.push({ text: documentText.substring(lastIndex, pos.start) });
      }

      // Add highlighted text
      segments.push({
        text: documentText.substring(pos.start, pos.end),
        color: pos.color,
        clauseType: pos.clauseType,
      });

      lastIndex = pos.end;
    });

    // Add remaining text
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

  // Get unique clause types from results for dynamic color legend
  const getClauseTypes = (): string[] => {
    if (!result?.result) return [];
    const types = new Set<string>(
      result.result.map((c: ClauseResult) => c.clause_type),
    );
    return Array.from(types);
  };

  // Handle PDF document load
  const onDocumentLoadSuccess = async ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);

    // Load the PDF document for text extraction
    if (file) {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
      setPdfDocument(pdf);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-7xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-blue-700 text-white p-8 text-center">
          <h1 className="text-4xl font-bold">LexiCache</h1>
          <p className="text-xl mt-2">Blockchain Verified Few-shot Legal AI</p>
          <p className="text-sm mt-1 opacity-90">
            IIT × University of Westminster
          </p>
        </div>

        {/* Show upload section when no analysis result */}
        {!result ? (
          <div className="p-8 lg:p-12">
            <div className="max-w-2xl mx-auto">
              {/* Upload Form */}
              <form onSubmit={handleSubmit} className="space-y-6">
                <h2 className="text-2xl font-semibold text-center mb-6">
                  Upload Legal Contract for Analysis
                </h2>

                <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-2xl cursor-pointer bg-gray-50 hover:bg-gray-100 hover:border-blue-400 transition-all duration-200">
                  {file ? (
                    <>
                      <FileText className="w-12 h-12 text-blue-500 mb-3" />
                      <span className="text-lg font-medium text-gray-800">
                        {file.name}
                      </span>
                      <span className="text-sm text-gray-500 mt-1">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                    </>
                  ) : (
                    <>
                      <Upload className="w-12 h-12 text-gray-400 mb-3" />
                      <span className="text-lg font-medium text-gray-600">
                        Click or drag file here
                      </span>
                      <span className="text-sm text-gray-400 mt-1">
                        PDF, DOC, DOCX supported
                      </span>
                    </>
                  )}
                  <input
                    type="file"
                    accept=".pdf,.doc,.docx"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>

                {/* Error Display */}
                {error && (
                  <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading || !file}
                  className={`w-full py-4 px-6 text-white font-semibold rounded-xl transition-all duration-200 shadow-md flex items-center justify-center gap-3 ${
                    loading || !file
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-700 hover:shadow-lg"
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing Document...
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Analyze & Highlight Document
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        ) : (
          /* Show PDF viewer and results after analysis */
          <div className="flex flex-col lg:flex-row">
            {/* PDF Viewer with Highlights */}
            <div className="lg:w-3/5 p-8 border-r border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-semibold flex items-center gap-2">
                  <Eye className="w-6 h-6" /> Analyzed Document
                </h2>
                <button
                  onClick={resetAnalysis}
                  className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition"
                >
                  Upload New Document
                </button>
              </div>

              {/* Success Banner */}
              <div className="flex items-center gap-3 p-4 mb-4 bg-green-50 border border-green-200 rounded-xl text-green-700">
                <CheckCircle className="w-5 h-5 flex-shrink-0" />
                <span>
                  Found <strong>{result.result?.length || 0}</strong> clause(s),
                  showing <strong>{getFilteredClauses().length}</strong>
                </span>
              </div>

              <div className="border border-gray-200 rounded-2xl overflow-hidden bg-gray-50 max-h-[70vh] overflow-y-auto">
                {fileType === "pdf" ? (
                  /* PDF Viewer with Canvas Highlighting */
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
                        Failed to load PDF. The file may be corrupted or
                        unsupported.
                      </div>
                    }
                  >
                    {Array.from(new Array(numPages), (_, index) => (
                      <div key={index} className="relative mb-4">
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
                          style={{
                            width: pageWidth,
                            height: pageHeights[index] || 1100,
                          }}
                        />
                        <div className="text-center text-xs text-gray-400 py-2">
                          Page {index + 1} of {numPages}
                        </div>
                      </div>
                    ))}
                  </Document>
                ) : (
                  /* Text Viewer with HTML Highlighting for DOC/DOCX */
                  <div className="bg-white p-8 shadow-sm">
                    <div className="mb-4 text-xs text-gray-500 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      <span>
                        {file?.name} ({fileType.toUpperCase()})
                      </span>
                    </div>
                    {renderHighlightedText()}
                  </div>
                )}
              </div>
            </div>

            {/* Sidebar with Results */}
            <div className="lg:w-2/5 p-8 space-y-6 max-h-screen overflow-y-auto">
              {/* Detected Clauses with Filters */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-lg">Detected Clauses</h3>
                  <button
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition"
                  >
                    <Filter className="w-4 h-4" />
                    {showFilters ? "Hide" : "Show"} Filters
                  </button>
                </div>

                {/* Filter Panel */}
                {showFilters && (
                  <div className="mb-4 p-4 bg-gray-50 rounded-xl border border-gray-200 space-y-4">
                    {/* Clause Type Filters */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-medium text-gray-700">
                          Clause Types
                        </label>
                        <button
                          onClick={() => {
                            const allTypes = getClauseTypes();
                            if (selectedClauseTypes.size === allTypes.length) {
                              setSelectedClauseTypes(new Set());
                            } else {
                              setSelectedClauseTypes(new Set(allTypes));
                            }
                          }}
                          className="text-xs text-blue-600 hover:text-blue-800"
                        >
                          {selectedClauseTypes.size === getClauseTypes().length
                            ? "Deselect All"
                            : "Select All"}
                        </button>
                      </div>
                      <div className="space-y-2 max-h-40 overflow-y-auto">
                        {getClauseTypes().map((type) => (
                          <label
                            key={type}
                            className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 p-2 rounded"
                          >
                            <input
                              type="checkbox"
                              checked={selectedClauseTypes.has(type)}
                              onChange={() => toggleClauseType(type)}
                              className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                            />
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: colorMap[type] }}
                            />
                            <span className="text-sm flex-1">{type}</span>
                            <span className="text-xs text-gray-500">
                              (
                              {result.result?.filter(
                                (c: ClauseResult) => c.clause_type === type,
                              ).length || 0}
                              )
                            </span>
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* Confidence Filter */}
                    <div>
                      <label className="text-sm font-medium text-gray-700 block mb-2">
                        Minimum Confidence: {minConfidence}%
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={minConfidence}
                        onChange={(e) =>
                          setMinConfidence(Number(e.target.value))
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                      </div>
                    </div>

                    {/* Active Filters Summary */}
                    <div className="text-xs text-gray-600 pt-2 border-t border-gray-200">
                      Showing {getFilteredClauses().length} of{" "}
                      {result.result?.length || 0} clauses
                    </div>
                  </div>
                )}

                {/* Clauses List */}
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {getFilteredClauses().length > 0 ? (
                    getFilteredClauses().map(
                      (clause: ClauseResult, idx: number) => (
                        <div
                          key={idx}
                          className="p-4 rounded-xl border border-gray-200 bg-gray-50 hover:bg-gray-100 transition cursor-pointer"
                          onClick={() => setHighlightedText(clause.span)}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <div
                              className="w-4 h-4 rounded-full shadow-sm"
                              style={{
                                backgroundColor:
                                  colorMap[clause.clause_type] || "#6b7280",
                              }}
                            />
                            <span className="font-medium">
                              {clause.clause_type}
                            </span>
                            <span className="ml-auto text-sm text-gray-500">
                              {(clause.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {clause.span}
                          </p>
                        </div>
                      ),
                    )
                  ) : (
                    <div className="text-center text-gray-500 py-8">
                      <Filter className="w-12 h-12 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No clauses match your filters</p>
                      <button
                        onClick={() => {
                          setSelectedClauseTypes(new Set(getClauseTypes()));
                          setMinConfidence(0);
                        }}
                        className="text-sm text-blue-600 hover:text-blue-800 mt-2"
                      >
                        Reset Filters
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Color Legend */}
              <div>
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Palette className="w-5 h-5" /> Clause Color Legend
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {getClauseTypes().map((clause) => (
                    <div key={clause} className="flex items-center gap-3">
                      <div className="flex gap-1">
                        <div
                          className="w-8 h-8 rounded-lg border shadow-sm cursor-pointer hover:scale-110 transition"
                          style={{
                            backgroundColor: colorMap[clause] || "#6b7280",
                          }}
                          title="Click to change color"
                          onClick={() => {
                            const input = document.createElement("input");
                            input.type = "color";
                            input.value = colorMap[clause] || "#6b7280";
                            input.onchange = (e) => {
                              setColorMap((prev) => ({
                                ...prev,
                                [clause]: (e.target as HTMLInputElement).value,
                              }));
                            };
                            input.click();
                          }}
                        />
                        <button
                          className="w-6 h-8 rounded border bg-gray-50 hover:bg-gray-100 flex items-center justify-center text-xs transition"
                          title="Generate new random color"
                          onClick={() => {
                            setColorMap((prev) => ({
                              ...prev,
                              [clause]: generateRandomColor(),
                            }));
                          }}
                        >
                          🎲
                        </button>
                      </div>
                      <span className="text-sm font-medium truncate">
                        {clause}
                      </span>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => {
                    const newColorMap: Record<string, string> = {};
                    getClauseTypes().forEach((type) => {
                      newColorMap[type] = generateRandomColor();
                    });
                    setColorMap(newColorMap);
                  }}
                  className="mt-3 w-full px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition flex items-center justify-center gap-2"
                >
                  🎨 Regenerate All Colors
                </button>
              </div>

              {/* Search Bar */}
              <div>
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Search className="w-5 h-5" /> Search in Document
                </h3>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    placeholder="Search text..."
                    className="flex-1 p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                  <button
                    onClick={handleSearch}
                    className="px-5 py-3 bg-gray-800 text-white rounded-xl hover:bg-gray-900 transition"
                  >
                    Search
                  </button>
                </div>
                {highlightedText && (
                  <p className="mt-2 text-sm text-gray-500">
                    Highlighting:{" "}
                    <span className="font-medium">
                      "{highlightedText.substring(0, 50)}..."
                    </span>
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
