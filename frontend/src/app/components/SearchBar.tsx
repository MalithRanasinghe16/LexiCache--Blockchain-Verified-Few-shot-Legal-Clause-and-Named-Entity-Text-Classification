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
    <div className="rounded-2xl border border-line bg-white p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
        <Search className="h-4 w-4 text-brand" /> Search in Document
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
            className="w-full rounded-xl border border-line p-3 pr-20 text-sm text-foreground outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/25"
          />
          {/* Match counter inside input */}
          {hasSearch && (
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 select-none font-mono text-xs text-muted">
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
          className="rounded-lg border border-line bg-panel/30 p-2.5 text-foreground transition hover:border-brand hover:text-brand disabled:cursor-not-allowed disabled:opacity-30"
        >
          <ChevronUp className="w-4 h-4" />
        </button>
        <button
          onClick={onNext}
          disabled={!hasSearch || totalMatches === 0}
          title="Next match (Enter)"
          className="rounded-lg border border-line bg-panel/30 p-2.5 text-foreground transition hover:border-brand hover:text-brand disabled:cursor-not-allowed disabled:opacity-30"
        >
          <ChevronDown className="w-4 h-4" />
        </button>

        {/* Clear button */}
        <button
          onClick={onClear}
          disabled={!hasSearch}
          title="Clear search (Esc)"
          className="rounded-lg border border-line bg-panel/30 p-2.5 text-foreground transition hover:border-brand hover:text-brand disabled:cursor-not-allowed disabled:opacity-30"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
