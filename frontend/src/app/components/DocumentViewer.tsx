import { Eye, CheckCircle } from "lucide-react";
import { ClauseResult, PageTextContent } from "../types";
import PdfViewer from "./PdfViewer";
import DocxViewer from "./DocxViewer";

type Props = {
  file: File | null;
  fileType: string;
  result: { result: ClauseResult[] };
  numPages: number | null;
  pageWidth: number;
  pageHeights: number[];
  pageTextContents: PageTextContent[];
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  highlightedText: string;
  activeClause: ClauseResult | null;
  documentText: string;
  isClient: boolean;
  onReset: () => void;
  onDocumentLoadSuccess: (data: { numPages: number }) => void;
};

export default function DocumentViewer({
  file,
  fileType,
  result,
  numPages,
  pageWidth,
  pageHeights,
  pageTextContents,
  colorMap,
  selectedClauseTypes,
  minConfidence,
  highlightedText,
  activeClause,
  documentText,
  isClient,
  onReset,
  onDocumentLoadSuccess,
}: Props) {
  const allClauses = result.result ?? [];
  const unknownCount = allClauses.filter(
    (c) => c.clause_type === "Unknown clause",
  ).length;
  const filteredCount = allClauses.filter((clause) => {
    if (clause.clause_type === "Unknown clause")
      return selectedClauseTypes.has(clause.clause_type);
    return (
      selectedClauseTypes.has(clause.clause_type) &&
      clause.confidence >= minConfidence / 100
    );
  }).length;

  return (
    <div className="lg:w-3/5 p-8 border-r border-gray-200">
      {/* Viewer Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold flex items-center gap-2 text-black">
          <Eye className="w-6 h-6 text-black" /> Analyzed Document
        </h2>
        <button
          onClick={onReset}
          className="px-4 py-2 text-sm bg-black text-white hover:bg-gray-800 rounded-lg transition"
        >
          Upload New Document
        </button>
      </div>

      {/* Success Banner */}
      <div className="flex items-center gap-3 p-4 mb-4 bg-green-50 border border-green-200 rounded-xl text-green-700">
        <CheckCircle className="w-5 h-5 shrink-0" />
        <div>
          <span>
            Found <strong>{allClauses.length}</strong> clause(s), showing{" "}
            <strong>{filteredCount}</strong>
          </span>
          {unknownCount > 0 && (
            <div className="text-xs text-orange-600 mt-1">
              💡 <strong>{unknownCount}</strong> unknown clause
              {unknownCount > 1 ? "s" : ""} – click to teach the system!
            </div>
          )}
        </div>
      </div>

      {/* Document Render Area */}
      <div className="border border-gray-200 rounded-2xl overflow-hidden bg-gray-50 max-h-[70vh] overflow-y-auto">
        {fileType === "pdf" ? (
          <PdfViewer
            file={file}
            numPages={numPages}
            pageWidth={pageWidth}
            pageHeights={pageHeights}
            pageTextContents={pageTextContents}
            result={result}
            colorMap={colorMap}
            selectedClauseTypes={selectedClauseTypes}
            minConfidence={minConfidence}
            highlightedText={highlightedText}
            activeClause={activeClause}
            isClient={isClient}
            onDocumentLoadSuccess={onDocumentLoadSuccess}
          />
        ) : (
          <DocxViewer
            file={file}
            documentText={documentText}
            result={result}
            colorMap={colorMap}
            selectedClauseTypes={selectedClauseTypes}
            minConfidence={minConfidence}
            highlightedText={highlightedText}
          />
        )}
      </div>
    </div>
  );
}
