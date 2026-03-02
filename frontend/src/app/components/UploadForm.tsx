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
    <div className="p-8 lg:p-12">
      <div className="max-w-2xl mx-auto">
        <form onSubmit={onSubmit} className="space-y-6">
          <h2 className="text-2xl font-semibold text-center mb-6">
            Upload Legal Contract for Analysis
          </h2>

          {/* Drop Zone */}
          <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-2xl cursor-pointer bg-gray-50 hover:bg-gray-100 hover:border-blue-400 transition-all duration-200">
            {file ? (
              <>
                <FileText className="w-12 h-12 text-blue-500 mb-3" />
                <span className="text-lg font-medium text-black">
                  {file.name}
                </span>
                <span className="text-sm text-black mt-1">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </>
            ) : (
              <>
                <Upload className="w-12 h-12 text-black mb-3" />
                <span className="text-lg font-medium text-black">
                  Click or drag file here
                </span>
                <span className="text-sm text-black mt-1">
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

          {/* Error Display */}
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
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
                <Loader2 className="w-5 h-5 animate-spin text-black" />
                Analyzing Document...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Analyze &amp; Highlight Document
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
