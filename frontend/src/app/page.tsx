// frontend/src/app/page.tsx
'use client';

import { useState } from 'react';

export default function Home() {
  const [text, setText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (file) {
        // File upload mode
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('http://localhost:8000/upload-file', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'File upload failed');
        }

        const data = await res.json();
        setResult(data);
      } else if (text.trim()) {
        // Text-only mode
        const res = await fetch('http://localhost:8000/predict-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'Prediction failed');
        }

        const data = await res.json();
        setResult(data);
      } else {
        setError('Please enter text or upload a file');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex flex-col items-center justify-center p-6 md:p-12">
      <div className="max-w-5xl w-full bg-white shadow-2xl rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-blue-700 text-white p-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-2">LexiCache</h1>
          <p className="text-lg md:text-xl opacity-90 mb-1">
            Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification
          </p>
          <p className="text-sm md:text-base opacity-80">
            Final Year Project | IIT × University of Westminster
          </p>
          <p className="text-sm mt-3 font-medium italic">
            Tagline: "One Upload = One Immutable Proof"
          </p>
        </div>

        {/* Main Content */}
        <div className="p-8 md:p-12">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Text Input */}
            <div>
              <label htmlFor="text" className="block text-lg font-medium text-gray-800 mb-3">
                Paste legal clause or contract text (optional)
              </label>
              <textarea
                id="text"
                rows={6}
                className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y font-mono text-sm shadow-sm transition-all duration-200"
                placeholder="e.g. This Agreement shall be governed by the laws of the State of Delaware..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>

            {/* File Upload */}
            <div>
              <label htmlFor="file" className="block text-lg font-medium text-gray-800 mb-3">
                Or upload a legal contract (PDF, DOC, DOCX)
              </label>
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="file"
                  className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors duration-200"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center">
                    <svg
                      className="w-10 h-10 mb-4 text-gray-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                    <p className="mb-2 text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">PDF, DOC, DOCX (max 10MB)</p>
                  </div>
                  <input
                    id="file"
                    type="file"
                    accept=".pdf,.doc,.docx"
                    className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </label>
              </div>
              {file && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: <span className="font-medium">{file.name}</span>
                </p>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || (!text.trim() && !file)}
              className={`w-full py-4 px-6 text-white font-semibold text-lg rounded-xl transition-all duration-200 shadow-md ${
                loading || (!text.trim() && !file)
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8h8a8 8 0 01-16 0z"
                    />
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Analyze Contract'
              )}
            </button>
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-8 p-5 bg-red-50 border border-red-200 text-red-700 rounded-xl shadow-sm">
              <p className="font-medium">Error:</p>
              <p>{error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="mt-10 p-8 bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-2xl shadow-lg">
              <h2 className="text-2xl font-bold text-green-800 mb-6">Analysis Result</h2>

              {result.status === 'success' && result.result ? (
                <div className="space-y-6">
                  <div className="bg-white p-6 rounded-xl shadow-sm border border-green-100">
                    <h3 className="text-xl font-semibold text-gray-800 mb-3">Predicted Clause</h3>
                    <p className="text-lg text-gray-900">
                      <strong>Clause Type:</strong> {result.result.clause_type || 'N/A'}
                    </p>
                    <p className="text-lg text-gray-900 mt-2">
                      <strong>Confidence:</strong>{' '}
                      <span className="font-bold text-green-700">
                        {(result.result.confidence * 100).toFixed(1)}%
                      </span>
                    </p>
                  </div>

                  {result.result.span && (
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-green-100">
                      <h3 className="text-xl font-semibold text-gray-800 mb-3">Extracted Span</h3>
                      <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                        {result.result.span}
                      </p>
                    </div>
                  )}

                  {result.extracted_text_preview && (
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-green-100">
                      <h3 className="text-xl font-semibold text-gray-800 mb-3">Text Preview</h3>
                      <p className="text-gray-600 italic">
                        {result.extracted_text_preview}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-gray-700">No detailed result available.</p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-100 p-6 text-center text-sm text-gray-600 border-t">
          <p>Final Year Project | Informatics Institute of Technology × University of Westminster</p>
          <p className="mt-2">© {new Date().getFullYear()} T.M.M.S. Ranasinghe</p>
        </div>
      </div>
    </main>
  );
}