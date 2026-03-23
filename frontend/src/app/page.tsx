"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  AnalysisResult,
  ClauseResult,
  PageTextContent,
  VerificationAttempt,
  VerificationState,
} from "./types";
import AppHeader from "./components/AppHeader";
import UploadForm from "./components/UploadForm";
import DocumentViewer from "./components/DocumentViewer";
import ResultsSidebar from "./components/ResultsSidebar";
import RenameModal from "./components/RenameModal";

export default function Home() {
  // ── File & analysis state ──────────────────────────────────────────────
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string>("pdf");
  const [documentText, setDocumentText] = useState<string>("");
  const [docHash, setDocHash] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>("anonymous");
  const [verification, setVerification] = useState<VerificationState | null>(
    null,
  );
  const [history, setHistory] = useState<VerificationAttempt[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [reminderDismissed, setReminderDismissed] = useState(false);
  const discardSentRef = useRef(false);

  // ── PDF viewer state ───────────────────────────────────────────────────
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageWidth, setPageWidth] = useState(780);
  const [pageHeights, setPageHeights] = useState<number[]>([]);
  const [pageTextContents, setPageTextContents] = useState<PageTextContent[]>(
    [],
  );
  const [pdfDocument, setPdfDocument] = useState<any>(null);
  const [isClient, setIsClient] = useState(false);

  // ── Filter & search state ──────────────────────────────────────────────
  const [selectedClauseTypes, setSelectedClauseTypes] = useState<Set<string>>(
    new Set(),
  );
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [highlightedText, setHighlightedText] = useState("");
  // Active clause: tracks which clause was clicked → drives PDF scroll + highlight
  const [activeClause, setActiveClause] = useState<ClauseResult | null>(null);

  // ── Color map ──────────────────────────────────────────────────────────
  const [colorMap, setColorMap] = useState<Record<string, string>>({});

  // ── Rename/teach modal state ───────────────────────────────────────────
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [selectedUnknownClause, setSelectedUnknownClause] =
    useState<ClauseResult | null>(null);
  const [newClauseTypeName, setNewClauseTypeName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);

  // ── Configure PDF.js worker (client-only) ─────────────────────────────
  useEffect(() => {
    setIsClient(true);

    if (typeof window !== "undefined") {
      const existing = localStorage.getItem("lexicache_user_id");
      if (existing) {
        setUserId(existing);
      } else {
        const generated =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : `user_${Date.now()}`;
        localStorage.setItem("lexicache_user_id", generated);
        setUserId(generated);
      }
    }
  }, []);

  useEffect(() => {
    const configurePdfWorker = async () => {
      if (typeof window !== "undefined") {
        const pdfjs = await import("react-pdf").then((mod) => mod.pdfjs);
        pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;
      }
    };
    configurePdfWorker();
  }, []);

  // ── Responsive PDF width ───────────────────────────────────────────────
  useEffect(() => {
    const handleResize = () =>
      setPageWidth(Math.min(780, window.innerWidth - 120));
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // ── Color helpers ──────────────────────────────────────────────────────
  const generateRandomColor = useCallback((): string => {
    const hue = Math.floor(Math.random() * 360);
    const saturation = 60 + Math.floor(Math.random() * 30);
    const lightness = 35 + Math.floor(Math.random() * 20);
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }, []);

  // Initialize colors when results arrive
  useEffect(() => {
    if (result?.result && result.result.length > 0) {
      const newColorMap: Record<string, string> = { ...colorMap };
      const types = new Set<string>();
      result.result.forEach((clause) => {
        types.add(clause.clause_type);
        if (!newColorMap[clause.clause_type]) {
          newColorMap[clause.clause_type] =
            clause.clause_type === "Unknown clause"
              ? "#F97316"
              : generateRandomColor();
        }
      });
      setColorMap(newColorMap);
      setSelectedClauseTypes(new Set(types));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  // ── PDF text extraction ────────────────────────────────────────────────
  const extractTextContent = useCallback(
    async (pdf: any) => {
      const contents: PageTextContent[] = [];
      const heights: number[] = [];
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1 });
        const scale = pageWidth / viewport.width;
        const scaledViewport = page.getViewport({ scale });
        heights.push(scaledViewport.height);
        const textContent = await page.getTextContent();
        const items = textContent.items.map((item: any) => ({
          str: item.str,
          transform: item.transform.map((t: number) => t * scale),
          width: item.width * scale,
          height: item.height * scale,
        }));
        contents.push({
          pageIndex: i - 1,
          items,
          viewport: {
            width: scaledViewport.width,
            height: scaledViewport.height,
            scale,
          },
        });
      }
      setPageTextContents(contents);
      setPageHeights(heights);
    },
    [pageWidth],
  );

  useEffect(() => {
    if (pdfDocument) extractTextContent(pdfDocument);
  }, [pdfDocument, extractTextContent]);

  // ── Handlers ───────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setHighlightedText("");
    setPdfDocument(null);
    setPageTextContents([]);
    setDocumentText("");
    setFileType(
      selectedFile?.name.toLowerCase().endsWith(".pdf") ? "pdf" : "docx",
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("user_id", userId);
        const res = await fetch("http://localhost:8000/upload-file", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(
            errorData.detail || `Upload failed with status ${res.status}`,
          );
        }
        const data = await res.json();
        if (data.result && !Array.isArray(data.result))
          data.result = [data.result];
        setResult(data);
        setDocHash(data.doc_hash || null);
        setVerification(data.verification || null);
        setHistory(data.history || []);
        setReminderDismissed(false);
        setDocumentText(data.extracted_text || "");
        setFileType(data.file_type || "pdf");
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentLoadSuccess = async ({
    numPages,
  }: {
    numPages: number;
  }) => {
    setNumPages(numPages);
    if (file && typeof window !== "undefined") {
      const pdfjs = await import("react-pdf").then((mod) => mod.pdfjs);
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
      setPdfDocument(pdf);
    }
  };

  const resetAnalysis = () => {
    setActiveClause(null);
    setFile(null);
    setResult(null);
    setError(null);
    setNumPages(null);
    setHighlightedText("");
    setSearchTerm("");
    setPdfDocument(null);
    setPageTextContents([]);
    setSelectedClauseTypes(new Set());
    setMinConfidence(0);
    setShowFilters(false);
    setDocumentText("");
    setFileType("pdf");
    setDocHash(null);
    setVerification(null);
    setHistory([]);
    setShowHistory(false);
    setIsVerifying(false);
    setReminderDismissed(false);
  };

  const handleToggleType = (type: string) => {
    const newSet = new Set(selectedClauseTypes);
    if (newSet.has(type)) newSet.delete(type);
    else newSet.add(type);
    setSelectedClauseTypes(newSet);
  };

  const handleSelectAll = () => {
    const allTypes = Array.from(
      new Set(result?.result?.map((c) => c.clause_type) ?? []),
    );
    if (selectedClauseTypes.size === allTypes.length) {
      setSelectedClauseTypes(new Set());
    } else {
      setSelectedClauseTypes(new Set(allTypes));
    }
  };

  const handleClauseClick = (clause: ClauseResult) => {
    const displaySpan = clause.span_display || clause.span;
    console.log("Clause clicked:", {
      type: clause.clause_type,
      isUnknown: clause.clause_type === "Unknown clause",
      span: displaySpan.substring(0, 50) + "...",
    });

    if (clause.clause_type === "Unknown clause") {
      console.log("Opening rename modal for unknown clause");
      setSelectedUnknownClause(clause);
      setShowRenameModal(true);
    } else {
      console.log("Setting active clause for highlighting and scroll");
      // Set active clause to trigger PDF scroll and highlight
      setActiveClause(clause);
      // Also set highlighted text for DOCX rendering and search indicator
      setHighlightedText(displaySpan);

      // DOCX: scroll to the <mark> element
      if (fileType !== "pdf") {
        console.log("DOCX: Scrolling to mark element");
        setTimeout(() => {
          const marks = document.querySelectorAll("mark");
          marks.forEach((mark) => {
            if (mark.textContent?.includes(displaySpan.substring(0, 50))) {
              mark.scrollIntoView({ behavior: "smooth", block: "center" });
              console.log("Scrolled to mark element");
            }
          });
        }, 100);
      } else {
        console.log("PDF: Active clause set, useEffect should trigger scroll");
      }
    }
  };

  const handleColorChange = (type: string, color: string) => {
    setColorMap((prev) => ({ ...prev, [type]: color }));
  };

  const handleRegenerateColors = () => {
    const newColorMap: Record<string, string> = {};
    const clauseTypes = Array.from(
      new Set(result?.result?.map((c) => c.clause_type) ?? []),
    );
    clauseTypes.forEach((type) => {
      newColorMap[type] =
        type === "Unknown clause" ? "#F97316" : generateRandomColor();
    });
    setColorMap(newColorMap);
  };

  const handleSearch = () => {
    if (searchTerm.trim()) setHighlightedText(searchTerm);
  };

  const handleRenameUnknown = async () => {
    if (!selectedUnknownClause || !newClauseTypeName.trim()) return;
    setIsRenaming(true);
    try {
      const res = await fetch("http://localhost:8000/rename-unknown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contract_text: documentText,
          unknown_span: selectedUnknownClause.span_exact || selectedUnknownClause.span,
          new_type_name: newClauseTypeName.trim(),
          color: colorMap[newClauseTypeName.trim()] || generateRandomColor(),
          doc_hash: docHash,
          user_id: userId,
        }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `Rename failed with status ${res.status}`,
        );
      }
      const data = await res.json();
      if (data.updated_results) {
        setResult({ ...result, result: data.updated_results });
        const newColor =
          colorMap[newClauseTypeName.trim()] || generateRandomColor();
        setColorMap((prev) => ({
          ...prev,
          [newClauseTypeName.trim()]: newColor,
        }));
        // Ensure the newly named type is visible in the sidebar immediately
        setSelectedClauseTypes((prev) => {
          const next = new Set(prev);
          next.add(newClauseTypeName.trim());
          return next;
        });
      }
      if (data.verification) {
        setVerification(data.verification);
        if (data.verification.show_verify_button) {
          setReminderDismissed(false);
        }
      }
      if (Array.isArray(data.history)) {
        setHistory(data.history);
      }
      setShowRenameModal(false);
      setSelectedUnknownClause(null);
      setNewClauseTypeName("");
    } catch (err: any) {
      alert(`Failed to rename clause: ${err.message}`);
    } finally {
      setIsRenaming(false);
    }
  };

  const closeRenameModal = () => {
    setShowRenameModal(false);
    setSelectedUnknownClause(null);
    setNewClauseTypeName("");
  };

  // Warn user on refresh/close whenever this document has NOT been verified yet.
  useEffect(() => {
    const shouldWarn = Boolean(
      result && docHash && verification?.show_verify_button,
    );
    if (!shouldWarn) return;

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [result, docHash, verification?.show_verify_button]);

  // If the user leaves anyway, immediately discard unverified document data.
  // This must also apply after teaching unknown clauses, as long as there is
  // an open verification cycle for the document.
  useEffect(() => {
    const shouldDiscardOnLeave = Boolean(
      result && docHash && verification?.show_verify_button,
    );
    if (!shouldDiscardOnLeave) {
      discardSentRef.current = false;
      return;
    }

    const sendDiscard = () => {
      if (!docHash || discardSentRef.current) return;
      discardSentRef.current = true;

      const payload = JSON.stringify({
        doc_hash: docHash,
        user_id: userId,
      });

      if (navigator.sendBeacon) {
        const blob = new Blob([payload], { type: "application/json" });
        navigator.sendBeacon("http://localhost:8000/discard-document", blob);
      } else {
        fetch("http://localhost:8000/discard-document", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        }).catch(() => {
          // best-effort discard during unload
        });
      }
    };

    const handlePageHide = () => {
      sendDiscard();
    };

    const handleBeforeUnload = () => {
      // Fire discard in addition to browser's native leave confirmation.
      // This runs on actual close/refresh navigation acceptance.
      sendDiscard();
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        // Extra mobile/browser compatibility path for tab close/background.
        sendDiscard();
      }
    };

    window.addEventListener("pagehide", handlePageHide);
    window.addEventListener("beforeunload", handleBeforeUnload);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      window.removeEventListener("beforeunload", handleBeforeUnload);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [result, docHash, verification?.show_verify_button, userId]);

  const handleVerify = async () => {
    if (!docHash || !result?.result) return;
    setIsVerifying(true);
    try {
      const res = await fetch("http://localhost:8000/verify-document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doc_hash: docHash,
          user_id: userId,
          clauses: result.result,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          data.detail || `Verify failed with status ${res.status}`,
        );
      }

      if (data.verification) {
        setVerification(data.verification);
      }
      if (Array.isArray(data.history)) {
        setHistory(data.history);
      }
      setShowHistory(true);
      alert(
        `${data.message || "Document verified! Permanent proof created on blockchain."}\n${data.record?.blockchain_link || ""}`,
      );
    } catch (err: any) {
      alert(err.message || "Verification failed");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleToggleHistory = async () => {
    const next = !showHistory;
    setShowHistory(next);
    if (!next || !docHash) return;

    try {
      const res = await fetch(
        `http://localhost:8000/document-history/${docHash}`,
      );
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data.history)) {
        setHistory(data.history);
      }
    } catch {
      // non-fatal: keep currently loaded history
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-7xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
        {/* ── Header ── */}
        <AppHeader />

        {!result ? (
          /* ── Upload Screen ── */
          <UploadForm
            file={file}
            loading={loading}
            error={error}
            onFileChange={handleFileChange}
            onSubmit={handleSubmit}
          />
        ) : (
          /* ── Analysis Screen ── */
          <div className="flex flex-col lg:flex-row">
            <DocumentViewer
              file={file}
              fileType={fileType}
              result={result}
              numPages={numPages}
              pageWidth={pageWidth}
              pageHeights={pageHeights}
              pageTextContents={pageTextContents}
              colorMap={colorMap}
              selectedClauseTypes={selectedClauseTypes}
              minConfidence={minConfidence}
              highlightedText={highlightedText}
              activeClause={activeClause}
              documentText={documentText}
              isClient={isClient}
              onReset={resetAnalysis}
              onDocumentLoadSuccess={handleDocumentLoadSuccess}
            />
            <ResultsSidebar
              result={result}
              colorMap={colorMap}
              selectedClauseTypes={selectedClauseTypes}
              minConfidence={minConfidence}
              searchTerm={searchTerm}
              highlightedText={highlightedText}
              activeClause={activeClause}
              verification={verification}
              history={history}
              showHistory={showHistory}
              isVerifying={isVerifying}
              reminderDismissed={reminderDismissed}
              showFilters={showFilters}
              onToggleFilters={() => setShowFilters(!showFilters)}
              onVerify={handleVerify}
              onToggleHistory={handleToggleHistory}
              onDismissReminder={() => setReminderDismissed(true)}
              onToggleType={handleToggleType}
              onConfidenceChange={setMinConfidence}
              onSelectAll={handleSelectAll}
              onClauseClick={handleClauseClick}
              onColorChange={handleColorChange}
              onRegenerateColors={handleRegenerateColors}
              onSearchChange={setSearchTerm}
              onSearch={handleSearch}
            />
          </div>
        )}

        {/* ── Rename / Teach Modal ── */}
        <RenameModal
          isOpen={showRenameModal}
          clause={selectedUnknownClause}
          newName={newClauseTypeName}
          isRenaming={isRenaming}
          onNameChange={setNewClauseTypeName}
          onConfirm={handleRenameUnknown}
          onClose={closeRenameModal}
        />
      </div>
    </main>
  );
}
