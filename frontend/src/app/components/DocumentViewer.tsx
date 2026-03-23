import { Eye } from "lucide-react";
import {
  AnalysisResult,
  ClauseResult,
  PageTextContent,
  SearchMatch,
} from "../types";
import PdfViewer from "./PdfViewer";
import DocxViewer from "./DocxViewer";

type Props = {
  file: File | null;
  fileType: string;
  result: AnalysisResult;
  numPages: number | null;
  pageWidth: number;
  pageHeights: number[];
  pageTextContents: PageTextContent[];
  colorMap: Record<string, string>;
  selectedClauseTypes: Set<string>;
  minConfidence: number;
  highlightedText: string;
  searchMatches: SearchMatch[];
  activeClause: ClauseResult | null;
  activeSearchPageIndex: number | null;
  activeSearchCharOffset: number | null;
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
  searchMatches,
  activeClause,
  activeSearchPageIndex,
  activeSearchCharOffset,
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
    if (clause.clause_type === "Unknown clause") {
      return selectedClauseTypes.has(clause.clause_type);
    }
    return (
      selectedClauseTypes.has(clause.clause_type) &&
      clause.confidence >= minConfidence / 100
    );
  }).length;

  return (
    <section className="lg:w-[64%] border-r border-line bg-[#f3eee5]/70">
      <div className="sticky top-0 z-20 border-b border-line bg-paper/95 px-5 py-4 backdrop-blur lg:px-7">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="mr-auto flex items-center gap-2 text-xl font-semibold text-foreground lg:text-2xl">
            <Eye className="h-5 w-5 text-brand" /> Document Review
          </h2>

          <span className="rounded-full border border-line bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-muted">
            {file?.name || "Document"}
          </span>
          <span className="rounded-full border border-line bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-muted">
            {result.page_count ? `${result.page_count} pages` : "Single page"}
          </span>

          <button
            onClick={onReset}
            className="rounded-lg border border-line bg-white px-4 py-2 text-xs font-semibold uppercase tracking-wide text-foreground transition hover:border-brand hover:text-brand"
          >
            Upload New
          </button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted">
          <span className="rounded-full bg-brand-soft px-3 py-1 font-semibold text-brand">
            {allClauses.length} detected
          </span>
          <span className="rounded-full bg-[#ece6d8] px-3 py-1 font-semibold text-[#6e5e47]">
            {filteredCount} visible by filters
          </span>
          {unknownCount > 0 && (
            <span className="rounded-full bg-[#fce8d6] px-3 py-1 font-semibold text-[#9b5a17]">
              {unknownCount} unknown ready to teach
            </span>
          )}
        </div>
      </div>

      <div className="px-5 py-6 lg:px-7">
        <div className="mb-4 rounded-xl border border-line bg-paper px-4 py-3 text-sm text-muted">
          Review highlighted text directly in context. Click a clause in the right
          panel to jump to its location.
        </div>

        <div className="rounded-2xl border border-line bg-[#eae3d7] p-3 shadow-inner lg:p-4">
          <div className="max-h-[72vh] overflow-y-auto rounded-xl border border-[#d7cab8] bg-[#fefdfb] p-3 lg:p-5">
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
                searchMatches={searchMatches}
                activeClause={activeClause}
                activeSearchPageIndex={activeSearchPageIndex}
                activeSearchCharOffset={activeSearchCharOffset}
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
      </div>
    </section>
  );
}