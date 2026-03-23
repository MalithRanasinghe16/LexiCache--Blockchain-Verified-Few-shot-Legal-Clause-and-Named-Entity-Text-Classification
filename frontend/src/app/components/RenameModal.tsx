import { useEffect } from "react";
import { X, CheckCircle, Loader2 } from "lucide-react";
import { ClauseResult } from "../types";

type Props = {
  isOpen: boolean;
  clause: ClauseResult | null;
  newName: string;
  isRenaming: boolean;
  onNameChange: (value: string) => void;
  onConfirm: () => void;
  onClose: () => void;
};

export default function RenameModal({
  isOpen,
  clause,
  newName,
  isRenaming,
  onNameChange,
  onConfirm,
  onClose,
}: Props) {
  // Close on Escape key — must be before the early return to satisfy hooks rules
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isOpen, onClose]);

  if (!isOpen || !clause) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4 backdrop-blur-sm">
      <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl border border-line bg-paper shadow-2xl">
        <div className="p-6 lg:p-7">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-foreground">
              Teach the System
            </h2>
            <button
              onClick={onClose}
              className="rounded-lg border border-line bg-white p-1.5 text-muted transition hover:text-foreground"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Body */}
          <div className="mb-6">
            <p className="mb-4 text-sm text-muted">
              The system found a clause it doesn&apos;t recognize. Help it learn
              by naming this clause type!
            </p>

            <div className="mb-4 rounded-xl border border-[#ebcfad] bg-[#fff4e7] p-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-[#8a5a22]">
                Unknown Clause Text:
              </p>
              <p className="max-h-40 overflow-y-auto text-sm italic text-[#6f4112]">
                &quot;{clause.span}&quot;
              </p>
            </div>

            <div>
              <label className="mb-2 block text-sm font-semibold text-foreground">
                What kind of clause is this?
              </label>
              <input
                type="text"
                value={newName}
                onChange={(e) => onNameChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newName.trim()) onConfirm();
                }}
                placeholder="e.g., Escrow Provision, Audit Rights, etc."
                className="w-full rounded-xl border border-line bg-white p-3 text-foreground outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/25"
                autoFocus
              />
              <p className="mt-2 text-xs text-muted">
                Examples: &quot;Confidentiality&quot;, &quot;Payment
                Terms&quot;, &quot;Liability Waiver&quot;
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 rounded-xl border border-line bg-white px-4 py-3 text-sm font-semibold uppercase tracking-wide text-foreground transition hover:border-brand hover:text-brand"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={!newName.trim() || isRenaming}
              className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold uppercase tracking-wide transition ${
                !newName.trim() || isRenaming
                  ? "cursor-not-allowed bg-gray-300 text-gray-500"
                    : "bg-brand text-white hover:bg-[#18413d]"
              }`}
            >
              {isRenaming ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Teaching...
                </>
              ) : (
                <>
                  <CheckCircle className="w-5 h-5" />
                  Teach &amp; Re-classify
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
