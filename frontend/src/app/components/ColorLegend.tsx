import { Palette } from "lucide-react";

type Props = {
  clauseTypes: string[];
  colorMap: Record<string, string>;
  onColorChange: (type: string, color: string) => void;
  onRegenerateColors: () => void;
};

export default function ColorLegend({
  clauseTypes,
  colorMap,
  onColorChange,
  onRegenerateColors,
}: Props) {
  const handleSwatchClick = (clauseType: string) => {
    if (clauseType === "Unknown clause") return;

    const input = document.createElement("input");
    input.type = "color";
    input.value = colorMap[clauseType] || "#6b7280";
    input.onchange = (e) => {
      onColorChange(clauseType, (e.target as HTMLInputElement).value);
    };
    input.click();
  };

  return (
    <div>
      <h3 className="font-semibold mb-4 flex items-center gap-2 text-black">
        <Palette className="w-5 h-5" /> Clause Color Legend
      </h3>

      <div className="grid grid-cols-2 gap-3">
        {clauseTypes.map((clause) => (
          <div key={clause} className="flex items-center gap-3 text-black">
            <div
              className={`w-8 h-8 rounded-lg border shadow-sm transition ${
                clause === "Unknown clause"
                  ? "border-orange-400"
                  : "cursor-pointer hover:scale-110"
              }`}
              style={{
                backgroundColor:
                  clause === "Unknown clause"
                    ? "#F97316"
                    : colorMap[clause] || "#6b7280",
              }}
              title={
                clause === "Unknown clause"
                  ? "Unknown clause (click items to teach)"
                  : "Click to change color"
              }
              onClick={() => handleSwatchClick(clause)}
            />
            <span className="text-sm font-medium truncate text-black flex items-center gap-1">
              {clause}
              {clause === "Unknown clause" && (
                <span className="text-xs text-orange-600">(teach)</span>
              )}
            </span>
          </div>
        ))}
      </div>

      <button
        onClick={onRegenerateColors}
        className="mt-3 w-full px-3 py-2 text-sm bg-black text-white hover:bg-gray-800 rounded-lg transition flex items-center justify-center gap-2"
      >
        Regenerate All Colors
      </button>
    </div>
  );
}
