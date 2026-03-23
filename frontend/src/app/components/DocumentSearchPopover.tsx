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

export default function DocumentSearchPopover({
  searchTerm,
  totalMatches,
  currentMatchIndex,
  onChange,
  onNext,
  onPrev,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
        event.preventDefault();
        setTimeout(() => inputRef.current?.focus(), 10);
      }

      if (
        document.activeElement === inputRef.current &&
        event.key === "Enter" &&
        searchTerm.trim()
      ) {
        event.preventDefault();
        if (event.shiftKey) {
          onPrev();
        } else {
          onNext();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [searchTerm, onNext, onPrev]);

  return (
    <div className="pointer-events-auto flex items-center gap-2">
      <div className="relative flex-1">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input
          ref={inputRef}
          type="text"
          value={searchTerm}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Search text"
          className="w-full rounded-lg border border-line bg-white py-2 pl-9 pr-20 text-sm text-foreground outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/25"
        />
        <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 font-mono text-[11px] text-muted">
          {searchTerm.trim()
            ? totalMatches > 0
              ? `${currentMatchIndex + 1} of ${totalMatches}`
              : "0 of 0"
            : ""}
        </span>
      </div>

      <button
        type="button"
        onClick={onPrev}
        disabled={!searchTerm.trim() || totalMatches === 0}
        className="flex h-8 w-8 items-center justify-center rounded-lg border border-line bg-panel/30 text-foreground disabled:opacity-40"
        title="Previous result"
      >
        <ChevronUp className="h-4 w-4" />
      </button>

      <button
        type="button"
        onClick={onNext}
        disabled={!searchTerm.trim() || totalMatches === 0}
        className="flex h-8 w-8 items-center justify-center rounded-lg border border-line bg-panel/30 text-foreground disabled:opacity-40"
        title="Next result"
      >
        <ChevronDown className="h-4 w-4" />
      </button>

      <button
        type="button"
        onClick={() => {
          if (searchTerm.trim()) {
            onClear();
          }
        }}
        className="flex h-8 w-8 items-center justify-center rounded-lg border border-line bg-panel/30 text-foreground"
        title="Clear search"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}