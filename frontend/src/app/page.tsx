// frontend/src/app/page.tsx
'use client';

import { useState } from 'react';

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:8000/predict', {  // ← your Python backend API (we will create this)
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error('Prediction failed');

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">
      <div className="max-w-4xl w-full bg-white shadow-xl rounded-2xl p-10">
        <h1 className="text-4xl font-bold text-center text-gray-900 mb-2">
          LexiCache
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification
        </p>
        <p className="text-center text-sm text-gray-500 mb-10">
          Final Year Project | IIT × University of Westminster
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="text" className="block text-lg font-medium text-gray-700 mb-2">
              Paste legal clause or contract text
            </label>
            <textarea
              id="text"
              rows={8}
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y font-mono text-sm"
              placeholder="e.g. This Agreement shall be governed by the laws of the State of Delaware..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3 px-6 text-white font-semibold rounded-lg transition-colors ${
              loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {loading ? 'Analyzing...' : 'Analyze Clause'}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
            Error: {error}
          </div>
        )}

        {result && (
          <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-lg">
            <h2 className="text-xl font-semibold text-green-800 mb-4">Prediction Result</h2>
            <pre className="text-sm text-gray-800 whitespace-pre-wrap">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </main>
  );
}