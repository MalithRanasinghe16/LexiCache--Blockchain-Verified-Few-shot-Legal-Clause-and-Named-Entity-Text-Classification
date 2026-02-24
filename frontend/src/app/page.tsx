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

  // Color map for clauses
  const [colorMap, setColorMap] = useState<Record<string, string>>({
    "Governing Law": "#3b82f6",
    Termination: "#ef4444",
    Confidentiality: "#10b981",
    Indemnification: "#f59e0b",
    "Payment Terms": "#8b5cf6",
    Unknown: "#6b7280",
  });

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
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  // Find text positions for a given clause span - precise matching
  const findTextPositions = useCallback(
    (span: string, pageContent: PageTextContent, isSearch: boolean = false) => {
      const positions: {
        x: number;
        y: number;
        width: number;
        height: number;
      }[] = [];

      // Normalize the span for matching
      const normalizedSpan = span
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      // Try to match more text - first 60 chars or first sentence
      const firstSentence = span.split(/[.!?]/)[0];
      const searchLength = isSearch
        ? Math.min(normalizedSpan.length, 30)
        : Math.min(Math.max(firstSentence.length, 60), 150);

      const searchText = normalizedSpan.substring(0, searchLength);

      // Build concatenated page text with item positions
      let pageText = "";
      const itemMappings: { start: number; end: number; item: TextItem }[] = [];

      pageContent.items.forEach((item) => {
        const start = pageText.length;
        const itemStr = item.str.replace(/[^\w\s]/g, " ");
        pageText += itemStr;
        itemMappings.push({ start, end: pageText.length, item });
        pageText += " ";
      });

      const normalizedPageText = pageText.toLowerCase().replace(/\s+/g, " ");

      // Find the exact position of the clause text
      let searchIdx = normalizedPageText.indexOf(searchText);

      // If exact match not found, try progressive fallback
      if (searchIdx === -1) {
        // Try first 50 chars
        const shorterSearch = searchText.substring(
          0,
          Math.min(50, searchText.length),
        );
        searchIdx = normalizedPageText.indexOf(shorterSearch);
      }

      if (searchIdx === -1) {
        // Try first 3-5 significant words
        const words = searchText.split(" ").filter((w) => w.length > 3);
        if (words.length >= 3) {
          const multiWordSearch = words
            .slice(0, Math.min(5, words.length))
            .join(" ");
          searchIdx = normalizedPageText.indexOf(multiWordSearch);
        }
      }

      if (searchIdx !== -1) {
        // Find all text items that fall within this match
        const matchEnd = searchIdx + searchText.length;
        const matchedItems: { item: TextItem; x: number; y: number }[] = [];

        itemMappings.forEach(({ start, end, item }) => {
          // Check if this item overlaps with our match
          if (start < matchEnd && end > searchIdx) {
            const x = item.transform[4];
            const y = pageContent.viewport.height - item.transform[5];

            if (x >= 0 && y >= 0 && item.width > 0) {
              matchedItems.push({ item, x, y });
            }
          }
        });

        if (matchedItems.length > 0) {
          // Group items by line (Y position within 5px)
          const lines: (typeof matchedItems)[] = [];
          matchedItems.forEach((current) => {
            let addedToLine = false;
            for (const line of lines) {
              if (Math.abs(line[0].y - current.y) < 5) {
                line.push(current);
                addedToLine = true;
                break;
              }
            }
            if (!addedToLine) {
              lines.push([current]);
            }
          });

          // Create a highlight box for each line
          lines.forEach((line) => {
            const minX = Math.min(...line.map((l) => l.x));
            const maxX = Math.max(...line.map((l) => l.x + l.item.width));
            const avgY = line.reduce((sum, l) => sum + l.y, 0) / line.length;

            positions.push({
              x: minX,
              y: avgY - 2,
              width: Math.min(maxX - minX + 10, pageWidth - minX - 10),
              height: 18,
            });
          });
        }
      }

      return positions;
    },
    [pageWidth],
  );

  // Draw highlights on PDF pages
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

        // Draw highlights for each clause
        result.result.forEach((clause: ClauseResult, idx: number) => {
          const color =
            colorMap[clause.clause_type] || colorMap["Unknown"] || "#f59e0b";
          const positions = findTextPositions(clause.span, pageContent);

          console.log(
            `Clause ${idx} (${clause.clause_type}): Found ${positions.length} positions on page ${pageIndex + 1}`,
          );

          positions.forEach((pos, posIdx) => {
            console.log(
              `  Position ${posIdx}: x=${pos.x.toFixed(1)}, y=${pos.y.toFixed(1)}, w=${pos.width.toFixed(1)}, h=${pos.height}`,
            );

            // Draw highlight rectangle with slight shadow
            ctx.fillStyle = color + "50"; // 30% opacity
            ctx.fillRect(pos.x, pos.y, pos.width, pos.height);

            // Draw border
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(pos.x, pos.y, pos.width, pos.height);

            // Add subtle glow effect
            ctx.shadowColor = color;
            ctx.shadowBlur = 4;
            ctx.strokeRect(pos.x, pos.y, pos.width, pos.height);
            ctx.shadowBlur = 0;
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
            ctx.fillStyle = "#fbbf2480"; // Yellow highlight
            ctx.fillRect(pos.x - 2, pos.y - 2, pos.width + 4, pos.height + 4);
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 2;
            ctx.strokeRect(pos.x - 2, pos.y - 2, pos.width + 4, pos.height + 4);
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
                  Found <strong>{result.result?.length || 0}</strong> clause(s)
                  in the document
                </span>
              </div>

              <div className="border border-gray-200 rounded-2xl overflow-hidden bg-gray-50 max-h-[70vh] overflow-y-auto">
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
              </div>
            </div>

            {/* Sidebar with Results */}
            <div className="lg:w-2/5 p-8 space-y-6 max-h-screen overflow-y-auto">
              {/* Detected Clauses */}
              <div>
                <h3 className="font-semibold mb-4 text-lg">Detected Clauses</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {result.result?.map((clause: ClauseResult, idx: number) => (
                    <div
                      key={idx}
                      className="p-4 rounded-xl border border-gray-200 bg-gray-50 hover:bg-gray-100 transition cursor-pointer"
                      onClick={() => setHighlightedText(clause.span)}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className="w-4 h-4 rounded-full"
                          style={{
                            backgroundColor:
                              colorMap[clause.clause_type] ||
                              colorMap["Unknown"],
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
                  ))}
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
                      <div
                        className="w-8 h-8 rounded-lg border shadow-sm cursor-pointer hover:scale-110 transition"
                        style={{
                          backgroundColor:
                            colorMap[clause] || colorMap["Unknown"],
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
                      <span className="text-sm font-medium truncate">
                        {clause}
                      </span>
                    </div>
                  ))}
                </div>
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
