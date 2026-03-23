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
    <div className="rounded-2xl border border-line bg-white p-4">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
        <Palette className="h-4 w-4 text-brand" /> Clause Color Legend
      </h3>

      <div className="grid grid-cols-2 gap-3">
        {clauseTypes.map((clause) => (
          <div key={clause} className="flex items-center gap-3 text-foreground">
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
            <span className="flex items-center gap-1 truncate text-sm font-medium text-foreground">
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
        className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-line bg-panel/50 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-foreground transition hover:border-brand hover:text-brand"
      >
        Regenerate All Colors
      </button>
    </div>
  );
}
