import { Search } from "lucide-react";

type Props = {
  searchTerm: string;
  highlightedText: string;
  onChange: (value: string) => void;
  onSearch: () => void;
};

export default function SearchBar({
  searchTerm,
  highlightedText,
  onChange,
  onSearch,
}: Props) {
  return (
    <div>
      <h3 className="font-semibold mb-4 flex items-center gap-2 text-black">
        <Search className="w-5 h-5 text-black" /> Search in Document
      </h3>
      <div className="flex gap-3 text-black">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSearch()}
          placeholder="Search text..."
          className="flex-1 p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
        />
        <button
          onClick={onSearch}
          className="px-5 py-3 bg-black text-white rounded-xl hover:bg-gray-800 transition"
        >
          Search
        </button>
      </div>
      {highlightedText && (
        <p className="mt-2 text-sm text-black">
          Highlighting:{" "}
          <span className="font-medium">
            &quot;{highlightedText.substring(0, 50)}...&quot;
          </span>
        </p>
      )}
    </div>
  );
}
