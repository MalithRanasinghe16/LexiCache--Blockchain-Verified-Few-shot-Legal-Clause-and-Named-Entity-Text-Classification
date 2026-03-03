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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-900">
              Teach the System
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Body */}
          <div className="mb-6">
            <p className="text-gray-600 mb-4">
              The system found a clause it doesn&apos;t recognize. Help it learn
              by naming this clause type!
            </p>

            <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 mb-4">
              <p className="text-sm font-semibold text-orange-800 mb-2">
                Unknown Clause Text:
              </p>
              <p className="text-sm text-gray-700 italic max-h-40 overflow-y-auto">
                &quot;{clause.span}&quot;
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
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
                className="w-full p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-gray-900"
                autoFocus
              />
              <p className="text-xs text-gray-500 mt-2">
                Examples: &quot;Confidentiality&quot;, &quot;Payment
                Terms&quot;, &quot;Liability Waiver&quot;
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-3 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition font-semibold"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={!newName.trim() || isRenaming}
              className={`flex-1 px-4 py-3 rounded-xl font-semibold transition flex items-center justify-center gap-2 ${
                !newName.trim() || isRenaming
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-blue-600 text-white hover:bg-blue-700"
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
