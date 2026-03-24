// Shared TypeScript types for LexiCache frontend

export type ClauseResult = {
  clause_type: string;
  span: string;
  span_exact?: string;
  span_display?: string;
  confidence: number;
  is_unknown?: boolean;
  needs_review?: boolean;
  source?: string;
  start_idx?: number;
  end_idx?: number;
  display_start_idx?: number;
  display_end_idx?: number;
  page_number?: number;
  context_heading?: string;
  is_staged?: boolean;
};

export type TextItem = {
  str: string;
  transform: number[];
  width: number;
  height: number;
};

export type PageTextContent = {
  pageIndex: number;
  items: TextItem[];
  viewport: { width: number; height: number; scale: number };
};

export type AnalysisResult = {
  result: ClauseResult[];
  doc_hash?: string;
  extracted_text?: string;
  extracted_text_preview?: string;
  page_count?: number;
  page_texts?: {
    page: number;
    text: string;
    start_char: number;
    end_char: number;
  }[];
  file_type?: string;
  status?: string;
  cached_at?: string;
  verification?: VerificationState;
  history?: VerificationAttempt[];
};

export type VerificationState = {
  doc_hash: string;
  unknown_count: number;
  show_verify_button: boolean;
  is_first_uploader: boolean;
  user_taught_count: number;
  can_verify: boolean;
  message: string;
};

export type VerificationAttempt = {
  attempt: number;
  verified_at: string;
  verified_by: string;
  clause_count: number;
  unknown_count: number;
  snapshot_hash: string;
  tx_hash: string;
  blockchain_link: string;
  geo_hash?: string | null;
  geo_summary?: string | null;
};

export type SearchMatch = {
  pageIndex: number;
  matchIndex: number; // index within the page (for multiple matches per page)
  charOffset: number; // character offset in normalized page text
  length: number; // normalized length of this match
};
