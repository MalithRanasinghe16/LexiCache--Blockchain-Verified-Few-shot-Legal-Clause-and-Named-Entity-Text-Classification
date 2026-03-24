import { ClauseResult } from "../types";

const HEADING_PREFIX_RE =
  /^\s*(section|article|schedule|annex|appendix|exhibit|part|chapter)\b/i;
const NUMBERED_HEADING_RE =
  /^\s*((\d+(?:\.\d+){0,4})|([ivxlcdm]+))[\).:\-]?\s+[a-z]/i;
const BULLET_SUBTOPIC_RE =
  /^\s*(\([a-z0-9]{1,3}\)|[a-z0-9]{1,2}[\).])\s+[a-z]/i;

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function getUppercaseRatio(text: string): number {
  const lettersOnly = text.replace(/[^a-zA-Z]/g, "");
  if (!lettersOnly) return 0;

  let uppercaseCount = 0;
  for (const ch of lettersOnly) {
    if (ch >= "A" && ch <= "Z") {
      uppercaseCount += 1;
    }
  }
  return uppercaseCount / lettersOnly.length;
}

export function getClauseDisplayText(clause: ClauseResult): string {
  return normalizeWhitespace(
    clause.span_display || clause.span_exact || clause.span || "",
  );
}

export function isStructuralClause(clause: ClauseResult): boolean {
  const text = getClauseDisplayText(clause);
  if (!text) return true;

  const wordCount = text.split(" ").filter(Boolean).length;

  // Long content is very unlikely to be a heading/topic.
  if (wordCount >= 25 || text.length >= 180) {
    return false;
  }

  const contextHeading = normalizeWhitespace(clause.context_heading || "");
  if (contextHeading && text.toLowerCase() === contextHeading.toLowerCase()) {
    return true;
  }

  if (HEADING_PREFIX_RE.test(text)) return true;
  if (NUMBERED_HEADING_RE.test(text)) return true;
  if (BULLET_SUBTOPIC_RE.test(text)) return true;

  const uppercaseRatio = getUppercaseRatio(text);
  const hasSentencePunctuation = /[.!?;]$/.test(text) || text.includes(",");
  const isMostlyTitleChars = /^[a-zA-Z0-9\s\-:&/()'".]+$/.test(text);

  if (
    wordCount <= 14 &&
    uppercaseRatio >= 0.72 &&
    !hasSentencePunctuation &&
    isMostlyTitleChars
  ) {
    return true;
  }

  if (wordCount <= 8 && !hasSentencePunctuation) {
    return true;
  }

  return false;
}
