import { Upload, FileText, Search, Loader2, AlertCircle } from "lucide-react";

type Props = {
  file: File | null;
  loading: boolean;
  error: string | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: (e: React.FormEvent) => void;
};

export default function UploadForm({
  file,
  loading,
  error,
  onFileChange,
  onSubmit,
}: Props) {
  return (
    <div className="px-6 py-8 lg:px-10 lg:py-10">
      <div className="mx-auto max-w-5xl rounded-3xl border border-line bg-paper p-5 shadow-[0_24px_60px_rgba(70,58,40,0.09)] lg:p-8">
        <div className="grid gap-6 lg:grid-cols-[1.05fr_1fr] lg:items-stretch">
          <section className="rounded-2xl border border-line bg-white p-6 lg:p-7">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted">
              Start Analysis
            </p>
            <h2 className="mt-2 text-3xl font-semibold leading-tight text-foreground">
              Upload Your Contract
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Drag and drop a PDF, DOC, or DOCX. LexiCache will detect clauses,
              highlight them directly in the document, and prepare verification
              history for audit-ready review.
            </p>

            <div className="mt-6 grid gap-3 text-sm text-muted">
              <div className="rounded-xl border border-line bg-panel/45 px-4 py-3">
                1. Upload your agreement
              </div>
              <div className="rounded-xl border border-line bg-panel/45 px-4 py-3">
                2. Review highlighted clauses in a document workspace
              </div>
              <div className="rounded-xl border border-line bg-panel/45 px-4 py-3">
                3. Teach unknown clauses and verify on blockchain
              </div>
            </div>
          </section>

          <form onSubmit={onSubmit} className="space-y-5 rounded-2xl border border-line bg-white p-6 lg:p-7">
            <label className="flex h-56 w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-panel/35 px-4 text-center transition-all duration-200 hover:border-brand hover:bg-brand-soft/45">
            {file ? (
              <>
                <FileText className="mb-3 h-12 w-12 text-brand" />
                <span className="line-clamp-2 text-base font-semibold text-foreground">
                  {file.name}
                </span>
                <span className="mt-1 text-sm text-muted">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </>
            ) : (
              <>
                <Upload className="mb-3 h-12 w-12 text-brand" />
                <span className="text-lg font-semibold text-foreground">
                  Click or drag file here
                </span>
                <span className="mt-1 text-sm text-muted">
                  PDF, DOC, DOCX supported
                </span>
              </>
            )}
            <input
              type="file"
              accept=".pdf,.doc,.docx"
              className="hidden"
              onChange={onFileChange}
            />
          </label>

          {error && (
            <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
              <AlertCircle className="h-5 w-5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file}
            className={`flex w-full items-center justify-center gap-3 rounded-xl px-6 py-4 text-sm font-semibold uppercase tracking-wide text-white shadow-sm transition-all duration-200 ${
              loading || !file
                ? "cursor-not-allowed bg-gray-400"
                : "bg-brand hover:bg-[#18413d]"
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Analyzing Document...
              </>
            ) : (
              <>
                <Search className="h-5 w-5" />
                Analyze Document
              </>
            )}
          </button>

            <p className="text-center text-xs text-muted">
              Your document is processed for analysis and clause verification flow.
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}
