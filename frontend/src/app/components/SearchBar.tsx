import { useEffect, useRef } from "react";
import { Search, ChevronUp, ChevronDown, X } from "lucide-react";

type Props = {
  searchTerm: string;
  totalMatches: number;
  currentMatchIndex: number;
  onChange: (value: string) => void;
  onNext: () => void;
  onPrev: () => void;
  onClear: () => void;
};

export default function SearchBar({
  searchTerm,
  totalMatches,
  currentMatchIndex,
  onChange,
  onNext,
  onPrev,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  // Keyboard shortcut: Ctrl+F focuses the search bar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "f") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      // Enter/Shift+Enter for next/prev when input is focused
      if (
        document.activeElement === inputRef.current &&
        e.key === "Enter" &&
        searchTerm.trim()
      ) {
        e.preventDefault();
        if (e.shiftKey) {
          onPrev();
        } else {
          onNext();
        }
      }
      // Escape clears search
      if (document.activeElement === inputRef.current && e.key === "Escape") {
        onClear();
        inputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [searchTerm, onNext, onPrev, onClear]);

  const hasSearch = searchTerm.trim().length > 0;

  return (
    <div>
      <h3 className="font-semibold mb-4 flex items-center gap-2 text-black">
        <Search className="w-5 h-5 text-black" /> Search in Document
      </h3>
      <div className="flex items-center gap-1.5">
        {/* Input */}
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            value={searchTerm}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Search text... (Ctrl+F)"
            className="w-full p-3 pr-20 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-black"
          />
          {/* Match counter inside input */}
          {hasSearch && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500 font-mono select-none pointer-events-none">
              {totalMatches > 0
                ? `${currentMatchIndex + 1} of ${totalMatches}`
                : "No matches"}
            </span>
          )}
        </div>

        {/* Navigation buttons */}
        <button
          onClick={onPrev}
          disabled={!hasSearch || totalMatches === 0}
          title="Previous match (Shift+Enter)"
          className="p-2.5 rounded-lg border border-gray-300 bg-white hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition text-black"
        >
          <ChevronUp className="w-4 h-4" />
        </button>
        <button
          onClick={onNext}
          disabled={!hasSearch || totalMatches === 0}
          title="Next match (Enter)"
          className="p-2.5 rounded-lg border border-gray-300 bg-white hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition text-black"
        >
          <ChevronDown className="w-4 h-4" />
        </button>

        {/* Clear button */}
        <button
          onClick={onClear}
          disabled={!hasSearch}
          title="Clear search (Esc)"
          className="p-2.5 rounded-lg border border-gray-300 bg-white hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition text-black"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
