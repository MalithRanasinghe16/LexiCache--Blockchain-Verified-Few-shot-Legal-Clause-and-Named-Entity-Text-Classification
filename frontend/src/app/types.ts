// Shared TypeScript types for LexiCache frontend

export type ClauseResult = {
  clause_type: string;
  span: string;
  confidence: number;
  is_unknown?: boolean;
  needs_review?: boolean;
  source?: string;
  start_idx?: number;
  end_idx?: number;
  page_number?: number;
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
};
