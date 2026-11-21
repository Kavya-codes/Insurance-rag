/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Fragment, useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  CornerDownRight,
  Download,
  FileText,
  History,
  Pause,
  Play,
  Search,
  Send,
  Share2,
  ShieldCheck,
  Square,
  Star,
  Trash2,
  Volume2,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface SourceCitation {
  index: number;
  title: string;
  section: string;
  source: string;
  score: number;
  text: string;
}

interface AskResponse {
  answer: string;
  sources: SourceCitation[];
  abstained: boolean;
  best_score: number;
  disclaimer: string;
}

interface EvalSummary {
  total_questions: number;
  answerable_questions: number;
  out_of_scope_questions: number;
  hit_at_1: number;
  hit_at_3: number;
  hit_at_5: number;
  mrr: number;
  keyword_coverage: number;
  abstention_accuracy: number;
  false_abstention_rate: number;
  faithfulness_score: number | null;
}

interface EvalRow {
  id: string;
  question: string;
  expected_doc: string | null;
  best_score: number;
  hit_at_1: boolean | null;
  hit_at_3: boolean | null;
  hit_at_5: boolean | null;
  keyword_coverage: number | null;
  abstained: boolean;
}

interface DocumentSummary {
  source: string;
  title: string;
  sections: string[];
  chunkCount: number;
}

interface ThreadTurn {
  id: string;
  question: string;
  response: AskResponse;
}

interface HistoryEntry {
  id: string;
  question: string;
  response: AskResponse;
  askedAt: string;
  priorTurns?: ThreadTurn[];
  rating?: number;
}

const RATING_LABELS: Record<number, string> = {
  1: "Not helpful",
  2: "Slightly helpful",
  3: "Somewhat helpful",
  4: "Helpful",
  5: "Very helpful",
};

const HISTORY_LIMIT = 5;
const SESSION_HISTORY_KEY = "insurance_rag_session_history_v1";

const HERO_ARTWORK = "/src/assets/images/hero_insurance_atelier_1795188554736.jpg";

interface PinPalette {
  name: string;
  surface: string;
  surfaceSoft: string;
  border: string;
  activeBorder: string;
  accentText: string;
  numeralText: string;
  citeIdle: string;
  citeActive: string;
  barFill: string;
  fallbackGradient: string;
}

const PIN_PALETTES: PinPalette[] = [
  {
    name: "Coral",
    surface: "bg-[#FFF1F2]",
    surfaceSoft: "bg-[#FFF1F2]/70",
    border: "border-rose-200",
    activeBorder: "border-rose-500 ring-2 ring-rose-500/20",
    accentText: "text-rose-800",
    numeralText: "text-rose-700",
    citeIdle: "bg-rose-50 text-rose-800 border-rose-200 hover:bg-rose-100",
    citeActive: "bg-rose-700 text-white border-rose-700",
    barFill: "#E11D48",
    fallbackGradient: "from-rose-100 via-amber-50 to-orange-100",
  },
  {
    name: "Teal",
    surface: "bg-[#F0FDFA]",
    surfaceSoft: "bg-[#F0FDFA]/75",
    border: "border-teal-200",
    activeBorder: "border-teal-600 ring-2 ring-teal-500/20",
    accentText: "text-teal-800",
    numeralText: "text-teal-700",
    citeIdle: "bg-teal-50 text-teal-800 border-teal-200 hover:bg-teal-100",
    citeActive: "bg-teal-700 text-white border-teal-700",
    barFill: "#0D9488",
    fallbackGradient: "from-teal-100 via-emerald-50 to-cyan-100",
  },
  {
    name: "Cobalt",
    surface: "bg-[#EFF6FF]",
    surfaceSoft: "bg-[#EFF6FF]/75",
    border: "border-blue-200",
    activeBorder: "border-blue-600 ring-2 ring-blue-500/20",
    accentText: "text-blue-800",
    numeralText: "text-blue-700",
    citeIdle: "bg-blue-50 text-blue-800 border-blue-200 hover:bg-blue-100",
    citeActive: "bg-blue-700 text-white border-blue-700",
    barFill: "#2563EB",
    fallbackGradient: "from-blue-100 via-sky-50 to-indigo-100",
  },
  {
    name: "Marigold",
    surface: "bg-[#FFFBEB]",
    surfaceSoft: "bg-[#FFFBEB]/80",
    border: "border-amber-200",
    activeBorder: "border-amber-500 ring-2 ring-amber-500/20",
    accentText: "text-amber-900",
    numeralText: "text-amber-700",
    citeIdle: "bg-amber-50 text-amber-900 border-amber-200 hover:bg-amber-100",
    citeActive: "bg-amber-700 text-white border-amber-700",
    barFill: "#D97706",
    fallbackGradient: "from-amber-100 via-yellow-50 to-orange-100",
  },
  {
    name: "Iris",
    surface: "bg-[#F5F3FF]",
    surfaceSoft: "bg-[#F5F3FF]/80",
    border: "border-violet-200",
    activeBorder: "border-violet-600 ring-2 ring-violet-500/20",
    accentText: "text-violet-800",
    numeralText: "text-violet-700",
    citeIdle: "bg-violet-50 text-violet-800 border-violet-200 hover:bg-violet-100",
    citeActive: "bg-violet-700 text-white border-violet-700",
    barFill: "#7C3AED",
    fallbackGradient: "from-violet-100 via-fuchsia-50 to-indigo-100",
  },
];

function getDocumentArtwork(doc: DocumentSummary, index: number): string {
  const key = `${doc.title} ${doc.source}`.toLowerCase();
  if (key.includes("pm-jay") || key.includes("ayushman") || key.includes("pmjay")) {
    return "/src/assets/images/pin_pmjay_shield_1795188554825.jpg";
  }
  if (key.includes("irdai") || key.includes("circular") || key.includes("protection") || key.includes("master")) {
    return "/src/assets/images/pin_irdai_regulations_1795188555438.jpg";
  }
  if (key.includes("arogya") || key.includes("health") || key.includes("sanjeevani") || key.includes("policy")) {
    return "/src/assets/images/pin_health_coverage_1795188554885.jpg";
  }
  const pool = [
    "/src/assets/images/pin_irdai_regulations_1795188555438.jpg",
    "/src/assets/images/pin_health_coverage_1795188554885.jpg",
    "/src/assets/images/pin_pmjay_shield_1795188554825.jpg",
    HERO_ARTWORK,
  ];
  return pool[index % pool.length];
}

function loadSessionHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(SESSION_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (item): item is HistoryEntry =>
          Boolean(item) &&
          typeof item.id === "string" &&
          typeof item.question === "string" &&
          typeof item.askedAt === "string" &&
          Boolean(item.response) &&
          typeof item.response.answer === "string" &&
          (item.rating === undefined ||
            (typeof item.rating === "number" &&
              Number.isInteger(item.rating) &&
              item.rating >= 1 &&
              item.rating <= 5))
      )
      .slice(0, HISTORY_LIMIT);
  } catch {
    return [];
  }
}

const SAMPLE_QUESTIONS = [
  {
    kicker: "Moratorium Period",
    question: "What is the moratorium period in health insurance?",
  },
  {
    kicker: "Insurer Switching",
    question: "How soon must I apply to switch my health insurer at renewal?",
  },
  {
    kicker: "AB PM-JAY Cover",
    question: "What is the annual coverage and co-payment under AB PM-JAY?",
  },
  {
    kicker: "Foreign Plans",
    question: "What is the monthly premium for US Medicare Part D?",
  },
];

interface FollowUpRule {
  keywords: string[];
  question: string;
}

const FOLLOW_UP_RULES: FollowUpRule[] = [
  {
    keywords: ["moratorium", "60 continuous months", "non-disclosure", "enhanced sum insured"],
    question: "Does the 60-month moratorium period apply immediately to an enhanced sum insured?",
  },
  {
    keywords: ["portability", "migration", "45 days", "60 days", "no claim bonus"],
    question: "Are waiting period credits and No Claim Bonus preserved when porting insurers?",
  },
  {
    keywords: ["pre-existing disease", "ped", "36 months", "slow-growing", "cataract", "hernia"],
    question: "What are the maximum waiting periods for Pre-Existing Diseases and listed ailments?",
  },
  {
    keywords: ["cashless", "pre-authorization", "discharge", "1 hour", "3 hours"],
    question: "What happens if an insurer delays final cashless discharge approval beyond 3 hours?",
  },
  {
    keywords: ["free look", "30 days", "proportionate risk premium", "medical examination"],
    question: "What deductions can an insurer make during a free look cancellation?",
  },
  {
    keywords: ["grace period", "monthly", "quarterly", "continuity"],
    question: "How long is the grace period for monthly versus annual health insurance renewals?",
  },
  {
    keywords: ["reimbursement", "15 days", "claims review committee", "repudiated"],
    question: "Can an insurer reject a health claim without a Claims Review Committee review?",
  },
  {
    keywords: ["lifelong renewability", "ayush", "ayurveda", "homeopathy", "unani"],
    question: "Are AYUSH treatments covered up to the full Sum Insured under health policies?",
  },
  {
    keywords: ["customer information sheet", "cis", "sub-limits", "exclusions"],
    question: "What details must an insurer disclose in the Customer Information Sheet (CIS)?",
  },
  {
    keywords: ["proposal", "underwriting", "15 days", "policyholder"],
    question: "How quickly must an insurer decide on an insurance proposal and share the proposal form?",
  },
  {
    keywords: ["cancellation", "7 days", "refund", "pro-rata"],
    question: "How much notice is required to cancel a retail policy and how is the refund calculated?",
  },
  {
    keywords: ["grievance", "bima bharosa", "igms", "ombudsman", "50 lakh"],
    question: "What is the compensation limit and filing deadline for the Insurance Ombudsman?",
  },
  {
    keywords: ["nomination", "beneficial nominee", "section 39", "spouse", "parents"],
    question: "Who qualifies as a beneficial nominee under Section 39 of the Insurance Act?",
  },
  {
    keywords: ["room rent", "icu", "iccu", "proportionate deduction", "24 consecutive hours"],
    question: "What happens when a patient chooses a room category above the eligible room rent limit?",
  },
  {
    keywords: ["cumulative bonus", "co-payment", "5%", "50%", "arogya sanjeevani"],
    question: "How does Cumulative Bonus accrue and what is the mandatory co-payment in Arogya Sanjeevani?",
  },
  {
    keywords: ["pre-hospitalization", "post-hospitalization", "30 days", "60 days", "15 days"],
    question: "How many days of pre-hospitalization and post-hospitalization expenses are covered?",
  },
  {
    keywords: ["ambulance", "cataract", "2,000", "40,000", "25%"],
    question: "What are the ambulance and cataract surgery sub-limits under standard policy wordings?",
  },
  {
    keywords: ["pm-jay", "ayushman bharat", "5 lakh", "senior citizen", "70 years", "hbp"],
    question: "How does AB PM-JAY cover senior citizens aged 70 and above across income groups?",
  },
];

function deriveSuggestedQuestions(
  sources: SourceCitation[],
  currentQuestion: string
): { question: string; matchedKeywords: string[] }[] {
  if (sources.length === 0) return [];

  const normalizedCurrent = currentQuestion.trim().toLowerCase();
  const currentTokens = new Set(
    normalizedCurrent.match(/[a-z0-9]+/g)?.filter((t) => t.length > 3) ?? []
  );

  const ranked = FOLLOW_UP_RULES.map((rule) => {
    const matchedKeywords: string[] = [];
    let score = 0;

    sources.forEach((src, srcIdx) => {
      const haystack = `${src.title}\n${src.section}\n${src.text}`.toLowerCase();
      const weight = srcIdx === 0 ? 2 : 1;
      for (const kw of rule.keywords) {
        if (haystack.includes(kw.toLowerCase())) {
          if (!matchedKeywords.includes(kw)) {
            matchedKeywords.push(kw);
          }
          score += weight;
        }
      }
    });

    const qLower = rule.question.toLowerCase();
    const candidateTokens = qLower.match(/[a-z0-9]+/g)?.filter((t) => t.length > 3) ?? [];
    const overlapCount = candidateTokens.filter((t) => currentTokens.has(t)).length;
    const similarity = candidateTokens.length > 0 ? overlapCount / candidateTokens.length : 0;

    return {
      question: rule.question,
      matchedKeywords,
      score,
      isDuplicate: qLower === normalizedCurrent || similarity >= 0.72,
    };
  })
    .filter((item) => item.matchedKeywords.length > 0 && !item.isDuplicate)
    .sort((a, b) => b.score - a.score || b.matchedKeywords.length - a.matchedKeywords.length);

  const selected: { question: string; matchedKeywords: string[] }[] = [];
  for (const item of ranked) {
    if (selected.length >= 4) break;
    selected.push({
      question: item.question,
      matchedKeywords: item.matchedKeywords.slice(0, 2),
    });
  }

  if (selected.length < 3) {
    const seenSections = new Set<string>();
    for (const src of sources) {
      if (selected.length >= 4) break;
      const cleanSection = src.section.replace(/^\d+\.\s*/, "").trim();
      if (!cleanSection || seenSections.has(cleanSection.toLowerCase())) continue;
      seenSections.add(cleanSection.toLowerCase());
      const fallbackQ = `What key rules or limits apply to ${cleanSection}?`;
      if (
        fallbackQ.toLowerCase() !== normalizedCurrent &&
        !selected.some((s) => s.question === fallbackQ)
      ) {
        selected.push({
          question: fallbackQ,
          matchedKeywords: [cleanSection],
        });
      }
    }
  }

  return selected;
}

function getPlainTextAnswer(answer: string): string {
  return answer
    .replace(/[ \t]*\[\d+\]/g, "")
    .replace(/[ \t]+([.,;:!?)])/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/^[ \t]+|[ \t]+$/gm, "")
    .trim();
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"ask" | "eval" | "docs">("ask");
  const [question, setQuestion] = useState("");
  const [askedQuestion, setAskedQuestion] = useState("");
  const [followUpQuestion, setFollowUpQuestion] = useState("");
  const [priorTurns, setPriorTurns] = useState<ThreadTurn[]>([]);
  const [expandedPriorTurns, setExpandedPriorTurns] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadSessionHistory());
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [openSources, setOpenSources] = useState<Record<number, boolean>>({});
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);
  const [hoveredSource, setHoveredSource] = useState<number | null>(null);
  const [exportStatus, setExportStatus] = useState<"exported" | "copied" | null>(null);
  const [plainCopied, setPlainCopied] = useState(false);
  const [hoveredStar, setHoveredStar] = useState<number | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareCopied, setShareCopied] = useState(false);
  const [speechState, setSpeechState] = useState<"idle" | "speaking" | "paused">("idle");
  const [brokenImages, setBrokenImages] = useState<Record<string, boolean>>({});

  const speechSupported =
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    typeof SpeechSynthesisUtterance !== "undefined";

  const [chunksCount, setChunksCount] = useState<number | null>(null);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [evalRows, setEvalRows] = useState<EvalRow[]>([]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [docFilter, setDocFilter] = useState("");

  const questionInputRef = useRef<HTMLTextAreaElement | null>(null);
  const followUpInputRef = useRef<HTMLInputElement | null>(null);
  const latestStateRef = useRef({
    question,
    followUpQuestion,
    loading,
    activeTab,
  });
  latestStateRef.current = { question, followUpQuestion, loading, activeTab };

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setChunksCount(typeof d.chunks_count === "number" ? d.chunks_count : 0))
      .catch(() => setChunksCount(null));

    fetch("/api/report")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.summary) setEvalSummary(d.summary);
        if (Array.isArray(d?.results)) setEvalRows(d.results);
      })
      .catch(() => {});

    fetch("/api/documents")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (Array.isArray(d?.documents)) setDocuments(d.documents);
      })
      .catch(() => {});

    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const sharedQuery = (params.get("q") ?? "").trim();
      if (sharedQuery) {
        setActiveTab("ask");
        setQuestion(sharedQuery);
        void submitQuestionRef.current(sharedQuery);
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(history));
    } catch {
      // ignore storage quota errors
    }
  }, [history]);

  useEffect(() => {
    return () => {
      if (speechSupported) {
        window.speechSynthesis.cancel();
      }
    };
  }, [speechSupported]);

  function stopSpeech() {
    if (speechSupported) {
      window.speechSynthesis.cancel();
    }
    setSpeechState("idle");
  }

  function handleClearResult() {
    stopSpeech();
    setAskedQuestion("");
    setFollowUpQuestion("");
    setPriorTurns([]);
    setExpandedPriorTurns({});
    setResult(null);
    setActiveHistoryId(null);
    setError(null);
    setOpenSources({});
    setHighlightedSource(null);
    setHoveredSource(null);
    setExportStatus(null);
    setPlainCopied(false);
    setHoveredStar(null);
    setShareUrl(null);
    setShareCopied(false);
  }

  const submitQuestionRef = useRef<
    (qText: string, options?: { isFollowUp?: boolean }) => Promise<void>
  >(async () => {});
  const clearResultRef = useRef<() => void>(() => {});
  clearResultRef.current = handleClearResult;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const hasModifier = event.metaKey || event.ctrlKey;

      if (hasModifier && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (latestStateRef.current.activeTab !== "ask") {
          setActiveTab("ask");
          window.requestAnimationFrame(() => {
            questionInputRef.current?.focus();
            questionInputRef.current?.select();
          });
        } else {
          questionInputRef.current?.focus();
          questionInputRef.current?.select();
        }
        return;
      }

      if (hasModifier && event.key === "Enter") {
        const isFollowUpFocused =
          typeof document !== "undefined" &&
          followUpInputRef.current !== null &&
          document.activeElement === followUpInputRef.current;

        if (isFollowUpFocused) {
          const currentFollowUp = latestStateRef.current.followUpQuestion.trim();
          if (currentFollowUp && !latestStateRef.current.loading) {
            event.preventDefault();
            void submitQuestionRef.current(latestStateRef.current.followUpQuestion, {
              isFollowUp: true,
            });
          }
          return;
        }

        const currentQ = latestStateRef.current.question.trim();
        if (currentQ && !latestStateRef.current.loading) {
          event.preventDefault();
          if (latestStateRef.current.activeTab !== "ask") {
            setActiveTab("ask");
          }
          void submitQuestionRef.current(latestStateRef.current.question);
        }
        return;
      }

      if (event.key === "Escape") {
        clearResultRef.current();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  async function submitQuestion(qText: string, options?: { isFollowUp?: boolean }) {
    const trimmed = qText.trim();
    if (!trimmed) return;
    const isFollowUp = Boolean(options?.isFollowUp && result);
    const nextPriorTurns: ThreadTurn[] =
      isFollowUp && result
        ? [
            ...priorTurns,
            {
              id: activeHistoryId ?? `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              question: (askedQuestion || question).trim(),
              response: result,
            },
          ]
        : [];

    stopSpeech();
    setLoading(true);
    setError(null);
    setHighlightedSource(null);
    setHoveredSource(null);
    setOpenSources({});
    setPlainCopied(false);
    setShareUrl(null);
    setShareCopied(false);

    try {
      const payload: {
        question: string;
        context?: { question: string; answer: string }[];
      } = { question: trimmed };
      if (isFollowUp && nextPriorTurns.length > 0) {
        payload.context = nextPriorTurns.slice(-4).map((turn) => ({
          question: turn.question,
          answer: getPlainTextAnswer(turn.response.answer),
        }));
      }

      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const rawText = await res.text();
      let body: Partial<AskResponse> & { detail?: string } = {};
      if (rawText) {
        try {
          body = JSON.parse(rawText) as Partial<AskResponse> & { detail?: string };
        } catch {
          if (!res.ok) {
            throw new Error(`Request failed (${res.status}). Please try again.`);
          }
          throw new Error("Received an unexpected response from the server.");
        }
      }
      if (!res.ok) {
        throw new Error(body.detail || `Unable to process request (${res.status}).`);
      }
      const answerData = body as AskResponse;
      const entryId = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const newEntry: HistoryEntry = {
        id: entryId,
        question: trimmed,
        response: answerData,
        askedAt: new Date().toISOString(),
        priorTurns: isFollowUp ? nextPriorTurns : undefined,
      };
      setPriorTurns(isFollowUp ? nextPriorTurns : []);
      setExpandedPriorTurns({});
      setFollowUpQuestion("");
      setAskedQuestion(trimmed);
      setExportStatus(null);
      setResult(answerData);
      setActiveHistoryId(entryId);
      setHistory((prev) => {
        const filtered = prev.filter(
          (item) => item.question.toLowerCase() !== trimmed.toLowerCase()
        );
        return [newEntry, ...filtered].slice(0, HISTORY_LIMIT);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }
  submitQuestionRef.current = submitQuestion;

  function handleSelectHistory(entry: HistoryEntry) {
    stopSpeech();
    setActiveTab("ask");
    setQuestion(entry.question);
    setAskedQuestion(entry.question);
    setFollowUpQuestion("");
    setPriorTurns(Array.isArray(entry.priorTurns) ? entry.priorTurns : []);
    setExpandedPriorTurns({});
    setResult(entry.response);
    setActiveHistoryId(entry.id);
    setError(null);
    setOpenSources({});
    setHighlightedSource(null);
    setHoveredSource(null);
    setExportStatus(null);
    setPlainCopied(false);
    setHoveredStar(null);
    setShareUrl(null);
    setShareCopied(false);
  }

  function handleRateAnswer(stars: number) {
    if (!result) return;
    if (stars < 1 || stars > 5) return;

    const currentEntry = activeHistoryId
      ? history.find((entry) => entry.id === activeHistoryId)
      : undefined;

    if (currentEntry) {
      const nextRating = currentEntry.rating === stars ? undefined : stars;
      setHistory((prev) =>
        prev.map((entry) =>
          entry.id === currentEntry.id ? { ...entry, rating: nextRating } : entry
        )
      );
      return;
    }

    const activeQ = (askedQuestion || question).trim();
    if (!activeQ) return;
    const entryId = activeHistoryId ?? `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const newEntry: HistoryEntry = {
      id: entryId,
      question: activeQ,
      response: result,
      askedAt: new Date().toISOString(),
      priorTurns: priorTurns.length > 0 ? priorTurns : undefined,
      rating: stars,
    };
    setActiveHistoryId(entryId);
    setHistory((prev) => {
      const filtered = prev.filter(
        (item) => item.question.toLowerCase() !== activeQ.toLowerCase()
      );
      return [newEntry, ...filtered].slice(0, HISTORY_LIMIT);
    });
  }

  function handleClearHistory() {
    setHistory([]);
    setActiveHistoryId(null);
    if (typeof window !== "undefined") {
      try {
        window.sessionStorage.removeItem(SESSION_HISTORY_KEY);
      } catch {
        // ignore storage errors
      }
    }
  }

  function handleClear() {
    stopSpeech();
    setQuestion("");
    setAskedQuestion("");
    setFollowUpQuestion("");
    setPriorTurns([]);
    setExpandedPriorTurns({});
    setResult(null);
    setActiveHistoryId(null);
    setError(null);
    setOpenSources({});
    setHighlightedSource(null);
    setHoveredSource(null);
    setExportStatus(null);
    setPlainCopied(false);
    setHoveredStar(null);
    setShareUrl(null);
    setShareCopied(false);
    if (typeof window !== "undefined" && window.location.search) {
      const cleanUrl = `${window.location.pathname}${window.location.hash}`;
      window.history.replaceState(null, "", cleanUrl);
    }
  }

  async function handleShare() {
    if (!result || typeof window === "undefined") return;
    const activeQ = (askedQuestion || question).trim();
    if (!activeQ) return;

    const url = new URL(window.location.href);
    url.search = "";
    url.searchParams.set("q", activeQ);
    const encodedUrl = url.toString();

    window.history.replaceState(null, "", `${url.pathname}${url.search}`);
    setShareUrl(encodedUrl);
    try {
      await navigator.clipboard.writeText(encodedUrl);
      setShareCopied(true);
      window.setTimeout(() => setShareCopied(false), 2500);
    } catch {
      setShareCopied(false);
    }
  }

  function handleReadAloud() {
    if (!speechSupported || !result) return;
    const synth = window.speechSynthesis;

    if (speechState === "speaking") {
      synth.pause();
      setSpeechState("paused");
      return;
    }

    if (speechState === "paused") {
      synth.resume();
      setSpeechState("speaking");
      return;
    }

    synth.cancel();
    const spokenText = getPlainTextAnswer(result.answer);
    if (!spokenText) return;

    const utterance = new SpeechSynthesisUtterance(spokenText);
    const voices = synth.getVoices();
    const preferredVoice =
      voices.find((v) => v.lang.toLowerCase() === "en-in") ??
      voices.find((v) => v.lang.toLowerCase().startsWith("en")) ??
      null;
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    utterance.rate = 1;
    utterance.onend = () => setSpeechState("idle");
    utterance.onerror = () => setSpeechState("idle");

    setSpeechState("speaking");
    synth.speak(utterance);
  }

  function buildAnswerMarkdown(qText: string, data: AskResponse): string {
    const timestamp = new Date().toISOString();
    const lines: string[] = [
      "# Indian Insurance RAG — Documentation Export",
      "",
      `- **Generated:** ${timestamp}`,
      `- **Status:** ${data.abstained ? "Abstained (insufficient evidence)" : "Grounded Answer"}`,
      `- **Best Retrieval Score:** ${data.best_score.toFixed(3)}`,
      "",
      "## Question",
      "",
      qText || "(Not recorded)",
      "",
      "## Answer",
      "",
      data.answer,
      "",
    ];

    if (data.sources.length > 0) {
      lines.push("## Cited Sources", "");
      for (const src of data.sources) {
        lines.push(`### [${src.index}] ${src.title}`, "");
        lines.push(`- **Section:** ${src.section}`);
        lines.push(`- **File:** \`${src.source}\``);
        lines.push(`- **Retrieval Score:** ${src.score.toFixed(3)}`, "");
        const quoted = src.text
          .split(/\r?\n/)
          .map((line) => (line.trim() === "" ? ">" : `> ${line}`))
          .join("\n");
        lines.push(quoted, "");
      }
    }

    lines.push("---", "", `*${data.disclaimer}*`, "");
    return lines.join("\n");
  }

  function buildMarkdownFilename(qText: string): string {
    const slug = qText
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48);
    const dateStamp = new Date().toISOString().slice(0, 10);
    return `insurance-rag-${slug || "answer"}-${dateStamp}.md`;
  }

  function handleExportMarkdown() {
    if (!result) return;
    const markdown = buildAnswerMarkdown(askedQuestion || question.trim(), result);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = buildMarkdownFilename(askedQuestion || question.trim());
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setExportStatus("exported");
    window.setTimeout(() => {
      setExportStatus((curr) => (curr === "exported" ? null : curr));
    }, 2500);
  }

  async function handleCopyMarkdown() {
    if (!result) return;
    const markdown = buildAnswerMarkdown(askedQuestion || question.trim(), result);
    try {
      await navigator.clipboard.writeText(markdown);
      setExportStatus("copied");
      window.setTimeout(() => {
        setExportStatus((curr) => (curr === "copied" ? null : curr));
      }, 2500);
    } catch {
      handleExportMarkdown();
    }
  }

  async function handleCopyPlainText() {
    if (!result) return;
    const plainText = getPlainTextAnswer(result.answer);
    if (!plainText) return;
    try {
      await navigator.clipboard.writeText(plainText);
      setPlainCopied(true);
      window.setTimeout(() => setPlainCopied(false), 2500);
    } catch {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = plainText;
        textarea.setAttribute("readonly", "");
        textarea.className = "fixed opacity-0 pointer-events-none";
        document.body.appendChild(textarea);
        textarea.select();
        const ok = document.execCommand("copy");
        textarea.remove();
        if (ok) {
          setPlainCopied(true);
          window.setTimeout(() => setPlainCopied(false), 2500);
        }
      } catch {
        setPlainCopied(false);
      }
    }
  }

  function handleCitationClick(indexNum: number) {
    setOpenSources((prev) => ({ ...prev, [indexNum]: true }));
    setHighlightedSource(indexNum);
    const el = document.getElementById(`source-card-${indexNum}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function renderAnswerText(answer: string) {
    const parts = answer.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = /^\[(\d+)\]$/.exec(part);
      if (!match) {
        return <Fragment key={idx}>{part}</Fragment>;
      }
      const num = Number(match[1]);
      const matchedSource = result?.sources.find((s) => s.index === num);
      const palette = PIN_PALETTES[(Math.max(1, num) - 1) % PIN_PALETTES.length];
      const isCitedActive = hoveredSource === num || highlightedSource === num;
      return (
        <button
          key={idx}
          type="button"
          title={
            matchedSource
              ? `[${num}] ${matchedSource.title} — ${matchedSource.section}`
              : `Jump to source [${num}]`
          }
          onMouseEnter={() => setHoveredSource(num)}
          onMouseLeave={() => setHoveredSource((curr) => (curr === num ? null : curr))}
          onFocus={() => setHoveredSource(num)}
          onBlur={() => setHoveredSource((curr) => (curr === num ? null : curr))}
          onClick={() => handleCitationClick(num)}
          className={`inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded-md border font-mono text-xs font-medium focus-visible:outline-2 focus-visible:outline-teal-700 cursor-pointer transition-colors duration-150 ${
            isCitedActive ? palette.citeActive : palette.citeIdle
          }`}
        >
          [{num}]
        </button>
      );
    });
  }

  const activeHistoryEntry = activeHistoryId
    ? history.find((entry) => entry.id === activeHistoryId) ?? null
    : null;
  const currentRating = activeHistoryEntry?.rating ?? null;
  const displayedRating = hoveredStar ?? currentRating ?? 0;
  const ratedHistoryEntries = history.filter(
    (entry): entry is HistoryEntry & { rating: number } =>
      typeof entry.rating === "number" && entry.rating >= 1 && entry.rating <= 5
  );
  const averageHistoryRating =
    ratedHistoryEntries.length > 0
      ? ratedHistoryEntries.reduce((sum, entry) => sum + entry.rating, 0) /
        ratedHistoryEntries.length
      : null;

  const suggestedQuestions =
    result && !result.abstained
      ? deriveSuggestedQuestions(result.sources, askedQuestion || question)
      : [];

  const normalizedDocFilter = docFilter.trim().toLowerCase();
  const filteredDocuments = normalizedDocFilter
    ? documents.filter(
        (doc) =>
          doc.title.toLowerCase().includes(normalizedDocFilter) ||
          doc.source.toLowerCase().includes(normalizedDocFilter)
      )
    : documents;

  return (
    <div className="min-h-screen bg-[#FBF9F5] text-stone-900">
      <header className="bg-white/90 backdrop-blur-sm border-b border-[#E5E0D8]">
        <div className="max-w-6xl mx-auto px-4 sm:px-8 h-16 flex items-center justify-between gap-8">
          <a
            href="/"
            className="font-display text-lg sm:text-xl font-bold tracking-tight text-stone-900 whitespace-nowrap shrink-0"
          >
            Indian Insurance RAG
          </a>

          <nav className="flex items-center gap-6 sm:gap-8 text-sm font-medium text-stone-600">
            <button
              type="button"
              onClick={() => setActiveTab("ask")}
              className={`min-h-[44px] px-1 whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer ${
                activeTab === "ask"
                  ? "border-rose-700 text-stone-900 font-semibold"
                  : "border-transparent hover:text-stone-900"
              }`}
            >
              Ask
            </button>
            <button
              type="button"
              onClick={() => {
                stopSpeech();
                setActiveTab("eval");
              }}
              className={`min-h-[44px] px-1 whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer ${
                activeTab === "eval"
                  ? "border-rose-700 text-stone-900 font-semibold"
                  : "border-transparent hover:text-stone-900"
              }`}
            >
              Evaluation
            </button>
            <button
              type="button"
              onClick={() => {
                stopSpeech();
                setActiveTab("docs");
              }}
              className={`min-h-[44px] px-1 whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer ${
                activeTab === "docs"
                  ? "border-rose-700 text-stone-900 font-semibold"
                  : "border-transparent hover:text-stone-900"
              }`}
            >
              Documents
            </button>
          </nav>

          <button
            type="button"
            onClick={handleClear}
            className="min-h-[40px] px-4 py-2 text-xs font-semibold text-stone-800 bg-[#F3F0EA] border border-[#E5E0D8] rounded-xl hover:bg-[#EAE4DA] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Reset
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
        {activeTab === "ask" && (
          <div className="space-y-10">
            <section className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.35fr)_320px] gap-8 items-stretch">
              <div className="flex flex-col justify-between gap-6 py-1">
                <div className="space-y-3">
                  <p className="text-xs font-semibold text-rose-700">
                    IRDAI Circulars · Policy Wordings · AB PM-JAY
                  </p>
                  <h1 className="font-display text-3xl sm:text-[2.65rem] font-bold tracking-tight text-stone-900 text-balance leading-[1.1]">
                    Clear answers from Indian health &amp; life policy rules.
                  </h1>
                  <p className="text-sm sm:text-base text-stone-600 leading-relaxed max-w-2xl">
                    Search across {documents.length || 4} official documents
                    {chunksCount !== null ? ` (${chunksCount} indexed passages)` : ""}. Every
                    answer cites its source passages directly, or declines if not covered.
                  </p>
                </div>

                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void submitQuestion(question);
                  }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <label
                      htmlFor="insurance-question"
                      className="block text-sm font-semibold text-stone-900"
                    >
                      Ask a question
                    </label>
                    <span className="hidden sm:inline text-xs text-stone-500 font-mono">
                      Cmd/Ctrl+K focus · Cmd/Ctrl+Enter ask · Esc clear
                    </span>
                  </div>
                  <div className="flex flex-col sm:flex-row gap-2.5 sm:items-end bg-white p-2 rounded-2xl border border-[#DFDAD0] focus-within:border-stone-500 transition-colors">
                    <textarea
                      ref={questionInputRef}
                      id="insurance-question"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="e.g., How fast must an insurer approve final cashless discharge?"
                      rows={2}
                      className="w-full rounded-xl bg-transparent px-3 py-2 text-base text-stone-900 placeholder:text-stone-400 focus:outline-none resize-none"
                    />
                    <button
                      type="submit"
                      disabled={loading}
                      className="min-h-[46px] px-5 rounded-xl bg-rose-700 text-white text-sm font-semibold hover:bg-rose-800 disabled:opacity-60 transition-colors inline-flex items-center justify-center gap-2 whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Search className="w-4 h-4" aria-hidden="true" />
                      <span>{loading ? "Searching..." : "Ask question"}</span>
                    </button>
                  </div>
                </form>
              </div>

              <div className="rounded-2xl border border-[#E5E0D8] bg-[#F3F0EA] overflow-hidden min-h-[210px]">
                {!brokenImages[HERO_ARTWORK] ? (
                  <img
                    src={HERO_ARTWORK}
                    alt="Color-tabbed Indian insurance policy folders and shield on a sunlit studio desk"
                    referrerPolicy="no-referrer"
                    onError={() => setBrokenImages((prev) => ({ ...prev, [HERO_ARTWORK]: true }))}
                    className="w-full h-full lg:h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center gap-2">
                    <ShieldCheck className="w-10 h-10 text-rose-700" aria-hidden="true" />
                    <span className="font-display text-sm font-bold text-stone-800">
                      Indian Insurance Policy Desk
                    </span>
                  </div>
                )}
              </div>
            </section>

            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_300px] gap-8 items-start">
              <div className="min-w-0 space-y-8">
                <section aria-label="Common questions">
                  <h2 className="font-display text-sm font-bold text-stone-700 mb-3">
                    Common Questions
                  </h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {SAMPLE_QUESTIONS.map((item, idx) => {
                      const palette = PIN_PALETTES[idx % PIN_PALETTES.length];
                      return (
                        <button
                          key={item.question}
                          type="button"
                          onClick={() => {
                            setQuestion(item.question);
                            void submitQuestion(item.question);
                          }}
                          className="group text-left rounded-2xl border border-[#E5E0D8] bg-white hover:border-stone-400 px-4 py-3.5 min-h-[44px] flex flex-col justify-between gap-1.5 transition-colors cursor-pointer"
                        >
                          <div className="flex items-center justify-between gap-2 w-full">
                            <span className={`text-xs font-semibold ${palette.accentText}`}>
                              {item.kicker}
                            </span>
                            <ArrowUpRight
                              className="w-4 h-4 text-stone-400 group-hover:text-stone-800 transition-colors shrink-0"
                              aria-hidden="true"
                            />
                          </div>
                          <span className="text-sm font-medium text-stone-900 leading-snug">
                            {item.question}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </section>

                {error && (
                  <section
                    role="alert"
                    className="bg-rose-50 border border-rose-200 text-rose-900 rounded-2xl p-4 sm:p-5 text-sm font-medium"
                  >
                    {error}
                  </section>
                )}

                {result && (
                  <div className="space-y-6">
                    {priorTurns.length > 0 && (
                      <section aria-label="Conversation context" className="space-y-2">
                        <div className="flex items-center justify-between gap-2 px-1">
                          <span className="text-xs font-semibold text-stone-600">
                            Earlier in this thread ({priorTurns.length})
                          </span>
                          <button
                            type="button"
                            onClick={() => {
                              setPriorTurns([]);
                              setExpandedPriorTurns({});
                            }}
                            className="text-xs font-medium text-stone-600 hover:text-stone-900 underline underline-offset-2 cursor-pointer"
                          >
                            Clear context
                          </button>
                        </div>
                        <div className="bg-white border border-[#E5E0D8] rounded-2xl divide-y divide-[#EDEAE3] overflow-hidden">
                          {priorTurns.map((turn, idx) => {
                            const isTurnExpanded = !!expandedPriorTurns[turn.id];
                            return (
                              <div key={turn.id}>
                                <button
                                  type="button"
                                  aria-expanded={isTurnExpanded}
                                  onClick={() =>
                                    setExpandedPriorTurns((prev) => ({
                                      ...prev,
                                      [turn.id]: !prev[turn.id],
                                    }))
                                  }
                                  className="w-full min-h-[44px] px-4 py-2.5 flex items-center justify-between gap-3 text-left hover:bg-[#FBF9F5] transition-colors cursor-pointer"
                                >
                                  <div className="min-w-0 text-xs sm:text-sm font-medium text-stone-800 truncate">
                                    <span className="font-mono text-xs text-stone-400 mr-2">
                                      Q{idx + 1}
                                    </span>
                                    {turn.question}
                                  </div>
                                  {isTurnExpanded ? (
                                    <ChevronUp
                                      className="w-4 h-4 text-stone-500 shrink-0"
                                      aria-hidden="true"
                                    />
                                  ) : (
                                    <ChevronDown
                                      className="w-4 h-4 text-stone-500 shrink-0"
                                      aria-hidden="true"
                                    />
                                  )}
                                </button>
                                {isTurnExpanded && (
                                  <div className="px-4 pb-3.5 pt-2 border-t border-[#EDEAE3] bg-[#FBF9F5] text-xs sm:text-sm text-stone-700 leading-relaxed whitespace-pre-wrap">
                                    {getPlainTextAnswer(turn.response.answer)}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </section>
                    )}

                    <section className="bg-white border border-[#E5E0D8] rounded-2xl p-5 sm:p-7 space-y-5">
                      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#EDEAE3]">
                        <div className="space-y-1 min-w-0">
                          <div className="flex items-center gap-2 text-xs font-semibold">
                            <ShieldCheck
                              className={`w-4 h-4 ${
                                result.abstained ? "text-amber-700" : "text-teal-700"
                              }`}
                              aria-hidden="true"
                            />
                            <span className={result.abstained ? "text-amber-900" : "text-teal-800"}>
                              {result.abstained
                                ? "Not found in indexed documents"
                                : "From indexed documents"}
                            </span>
                            <span className="text-stone-300" aria-hidden="true">
                              ·
                            </span>
                            <span className="font-mono tabular-nums font-normal text-stone-500">
                              Score {result.best_score.toFixed(3)}
                            </span>
                          </div>
                          {askedQuestion && (
                            <p className="text-xs sm:text-sm font-semibold text-stone-900 truncate">
                              {askedQuestion}
                            </p>
                          )}
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          {speechSupported && (
                            <>
                              <button
                                type="button"
                                onClick={handleReadAloud}
                                aria-pressed={speechState !== "idle"}
                                className={`min-h-[44px] px-3.5 py-2 rounded-xl border text-xs font-semibold inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer transition-colors ${
                                  speechState !== "idle"
                                    ? "border-teal-700 bg-teal-700 text-white hover:bg-teal-800"
                                    : "border-[#E5E0D8] bg-[#F4F1EA] hover:bg-[#EAE4DA] text-stone-800"
                                }`}
                              >
                                {speechState === "speaking" ? (
                                  <>
                                    <Pause className="w-3.5 h-3.5" aria-hidden="true" />
                                    <span>Pause</span>
                                  </>
                                ) : speechState === "paused" ? (
                                  <>
                                    <Play className="w-3.5 h-3.5" aria-hidden="true" />
                                    <span>Resume</span>
                                  </>
                                ) : (
                                  <>
                                    <Volume2 className="w-3.5 h-3.5 text-stone-600" aria-hidden="true" />
                                    <span>Read Aloud</span>
                                  </>
                                )}
                              </button>
                              {speechState !== "idle" && (
                                <button
                                  type="button"
                                  onClick={stopSpeech}
                                  aria-label="Stop reading aloud"
                                  className="min-h-[44px] px-3 py-2 rounded-xl border border-rose-200 bg-rose-50 hover:bg-rose-100 text-xs font-semibold text-rose-800 inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                                >
                                  <Square className="w-3.5 h-3.5" aria-hidden="true" />
                                  <span>Stop</span>
                                </button>
                              )}
                            </>
                          )}
                          <button
                            type="button"
                            onClick={handleCopyMarkdown}
                            className="min-h-[44px] px-3.5 py-2 rounded-xl border border-[#E5E0D8] bg-[#F4F1EA] hover:bg-[#EAE4DA] text-xs font-semibold text-stone-800 inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                          >
                            {exportStatus === "copied" ? (
                              <Check className="w-3.5 h-3.5 text-teal-700" aria-hidden="true" />
                            ) : (
                              <Copy className="w-3.5 h-3.5 text-stone-600" aria-hidden="true" />
                            )}
                            <span>{exportStatus === "copied" ? "Copied Markdown" : "Copy Markdown"}</span>
                          </button>
                          <button
                            type="button"
                            onClick={handleShare}
                            className="min-h-[44px] px-3.5 py-2 rounded-xl border border-[#E5E0D8] bg-[#F4F1EA] hover:bg-[#EAE4DA] text-xs font-semibold text-stone-800 inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                          >
                            {shareCopied ? (
                              <Check className="w-3.5 h-3.5 text-teal-700" aria-hidden="true" />
                            ) : (
                              <Share2 className="w-3.5 h-3.5 text-stone-600" aria-hidden="true" />
                            )}
                            <span>{shareCopied ? "Link Copied" : "Share"}</span>
                          </button>
                          <button
                            type="button"
                            onClick={handleExportMarkdown}
                            className="min-h-[44px] px-3.5 py-2 rounded-xl border border-stone-900 bg-stone-900 hover:bg-stone-800 text-xs font-semibold text-white inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer transition-colors"
                          >
                            {exportStatus === "exported" ? (
                              <Check className="w-3.5 h-3.5 text-emerald-300" aria-hidden="true" />
                            ) : (
                              <Download className="w-3.5 h-3.5" aria-hidden="true" />
                            )}
                            <span>{exportStatus === "exported" ? "Exported .md" : "Export Markdown"}</span>
                          </button>
                        </div>
                      </div>

                      {shareUrl && (
                        <div className="flex flex-col sm:flex-row sm:items-center gap-2 rounded-xl border border-[#E5E0D8] bg-[#FBF9F5] px-3.5 py-2.5 text-xs">
                          <span className="font-semibold text-stone-700 whitespace-nowrap">
                            Shareable URL:
                          </span>
                          <input
                            type="text"
                            readOnly
                            value={shareUrl}
                            onFocus={(e) => e.currentTarget.select()}
                            aria-label="Shareable question URL"
                            className="w-full min-w-0 rounded-lg border border-[#E5E0D8] bg-white px-2.5 py-1 font-mono text-xs text-stone-700 focus:outline-2 focus:outline-stone-700"
                          />
                        </div>
                      )}

                      {result.abstained && (
                        <div className="bg-amber-50 border border-amber-200 text-amber-950 rounded-xl px-4 py-3 text-sm font-medium">
                          Abstained — best retrieval score{" "}
                          <span className="font-mono tabular-nums">{result.best_score.toFixed(3)}</span>{" "}
                          did not meet confidence threshold.
                        </div>
                      )}

                      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                        <div className="min-w-0 flex-1 text-base text-stone-900 leading-[1.7] whitespace-pre-wrap">
                          {renderAnswerText(result.answer)}
                        </div>
                        <button
                          type="button"
                          onClick={handleCopyPlainText}
                          aria-label="Copy plain-text answer without citations to clipboard"
                          className="self-start shrink-0 min-h-[44px] px-3.5 py-2 rounded-xl border border-[#E5E0D8] bg-[#FBF9F5] hover:bg-[#F3F0EA] text-xs font-semibold text-stone-800 inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer transition-colors"
                        >
                          {plainCopied ? (
                            <Check className="w-3.5 h-3.5 text-teal-700" aria-hidden="true" />
                          ) : (
                            <Copy className="w-3.5 h-3.5 text-stone-500" aria-hidden="true" />
                          )}
                          <span>{plainCopied ? "Copied!" : "Copy to Clipboard"}</span>
                        </button>
                      </div>

                      <div
                        className="pt-4 border-t border-[#EDEAE3] flex flex-wrap items-center justify-between gap-3"
                        onMouseLeave={() => setHoveredStar(null)}
                      >
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="text-xs font-semibold text-stone-700">
                            Was this explanation helpful?
                          </span>
                          <div
                            className="inline-flex items-center gap-0.5"
                            role="group"
                            aria-label="Rate this answer from 1 to 5 stars"
                          >
                            {[1, 2, 3, 4, 5].map((star) => {
                              const isFilled = star <= displayedRating;
                              const isSelected = currentRating === star;
                              return (
                                <button
                                  key={star}
                                  type="button"
                                  onClick={() => handleRateAnswer(star)}
                                  onMouseEnter={() => setHoveredStar(star)}
                                  onFocus={() => setHoveredStar(star)}
                                  onBlur={() => setHoveredStar(null)}
                                  aria-pressed={isSelected}
                                  aria-label={`Rate ${star} out of 5 stars (${RATING_LABELS[star]})`}
                                  title={`${star}/5 — ${RATING_LABELS[star]}`}
                                  className="min-h-[44px] min-w-[38px] inline-flex items-center justify-center rounded-xl hover:bg-amber-50 focus-visible:outline-2 focus-visible:outline-amber-600 transition-colors cursor-pointer"
                                >
                                  <Star
                                    className={`w-5 h-5 transition-colors ${
                                      isFilled
                                        ? "fill-amber-400 text-amber-500"
                                        : "text-stone-300 hover:text-amber-400"
                                    }`}
                                    aria-hidden="true"
                                  />
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <div className="text-xs font-mono tabular-nums text-stone-600">
                          {hoveredStar ? (
                            <span className="text-amber-800 font-semibold">
                              {hoveredStar}/5 · {RATING_LABELS[hoveredStar]}
                            </span>
                          ) : currentRating ? (
                            <span className="text-amber-800 font-semibold">
                              {currentRating}/5 · {RATING_LABELS[currentRating]}
                            </span>
                          ) : (
                            <span className="text-stone-400">Select 1–5 stars</span>
                          )}
                        </div>
                      </div>

                      <p className="pt-4 border-t border-[#EDEAE3] text-xs text-stone-500">
                        {result.disclaimer}
                      </p>
                    </section>

                    <section
                      aria-label="Ask a follow-up"
                      className="bg-white border border-[#E5E0D8] rounded-2xl p-5 space-y-3"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <label
                          htmlFor="follow-up-question"
                          className="font-display text-sm font-bold text-stone-900 inline-flex items-center gap-2"
                        >
                          <CornerDownRight className="w-4 h-4 text-teal-700" aria-hidden="true" />
                          <span>Ask a follow-up</span>
                        </label>
                        {askedQuestion && (
                          <span className="text-xs text-stone-500 truncate max-w-full sm:max-w-md">
                            In response to: &ldquo;{askedQuestion}&rdquo;
                          </span>
                        )}
                      </div>
                      <form
                        onSubmit={(e) => {
                          e.preventDefault();
                          void submitQuestion(followUpQuestion, { isFollowUp: true });
                        }}
                        className="flex flex-col sm:flex-row gap-2.5 sm:items-center"
                      >
                        <input
                          ref={followUpInputRef}
                          id="follow-up-question"
                          type="text"
                          value={followUpQuestion}
                          onChange={(e) => setFollowUpQuestion(e.target.value)}
                          placeholder="Ask a follow-up about this answer (e.g., What are the exceptions?)..."
                          className="w-full min-h-[46px] rounded-xl border border-[#DFDAD0] bg-[#FBF9F5] px-4 py-2.5 text-sm sm:text-base text-stone-900 placeholder:text-stone-400 focus:bg-white focus:outline-2 focus:outline-teal-700"
                        />
                        <button
                          type="submit"
                          disabled={loading || !followUpQuestion.trim()}
                          className="min-h-[46px] px-5 rounded-xl bg-teal-700 text-white text-sm font-semibold hover:bg-teal-800 disabled:opacity-60 transition-colors inline-flex items-center justify-center gap-2 whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          <Send className="w-4 h-4" aria-hidden="true" />
                          <span>{loading ? "Searching..." : "Ask follow-up"}</span>
                        </button>
                      </form>
                    </section>

                    {result.sources.length > 0 && (
                      <section className="space-y-3">
                        <h2 className="font-display text-sm font-bold text-stone-700">
                          Sources ({result.sources.length})
                        </h2>
                        <div className="bg-white border border-[#E5E0D8] rounded-2xl divide-y divide-[#EDEAE3] overflow-hidden">
                          {result.sources.map((src) => {
                            const palette =
                              PIN_PALETTES[(Math.max(1, src.index) - 1) % PIN_PALETTES.length];
                            const isOpen = !!openSources[src.index];
                            const isHovered = hoveredSource === src.index;
                            const isHighlighted = highlightedSource === src.index || isHovered;
                            return (
                              <div
                                key={src.index}
                                id={`source-card-${src.index}`}
                                onMouseEnter={() => setHoveredSource(src.index)}
                                onMouseLeave={() =>
                                  setHoveredSource((curr) => (curr === src.index ? null : curr))
                                }
                                className={`transition-colors duration-150 ${
                                  isHighlighted ? palette.surface : "bg-white"
                                }`}
                              >
                                <button
                                  type="button"
                                  aria-expanded={isOpen}
                                  onClick={() =>
                                    setOpenSources((prev) => ({
                                      ...prev,
                                      [src.index]: !prev[src.index],
                                    }))
                                  }
                                  className="w-full min-h-[52px] px-4 sm:px-5 py-3.5 flex items-center justify-between gap-3 text-left cursor-pointer"
                                >
                                  <div className="min-w-0">
                                    <div className="text-sm font-semibold text-stone-900">
                                      <span className={`font-mono mr-1.5 ${palette.numeralText}`}>
                                        [{src.index}]
                                      </span>
                                      {src.title}
                                    </div>
                                    <div className="text-xs text-stone-600 mt-0.5 break-words">
                                      {src.section} · <span className="font-mono">{src.source}</span>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3 shrink-0">
                                    <span className="font-mono tabular-nums text-xs text-stone-500">
                                      {src.score.toFixed(3)}
                                    </span>
                                    {isOpen ? (
                                      <ChevronUp
                                        className="w-4 h-4 text-stone-500"
                                        aria-hidden="true"
                                      />
                                    ) : (
                                      <ChevronDown
                                        className="w-4 h-4 text-stone-500"
                                        aria-hidden="true"
                                      />
                                    )}
                                  </div>
                                </button>
                                {isOpen && (
                                  <div className="px-4 sm:px-5 pb-4 pt-3 border-t border-[#EDEAE3] bg-[#FBF9F5] text-sm text-stone-700 leading-relaxed whitespace-pre-wrap">
                                    {src.text}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </section>
                    )}

                    {suggestedQuestions.length > 0 && (
                      <section className="space-y-3">
                        <h2 className="font-display text-sm font-bold text-stone-700">
                          Suggested Follow-ups
                        </h2>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {suggestedQuestions.map((item) => (
                            <button
                              key={item.question}
                              type="button"
                              title={`Based on keywords: ${item.matchedKeywords.join(", ")}`}
                              onClick={() => {
                                void submitQuestion(item.question, { isFollowUp: true });
                              }}
                              className="min-h-[44px] px-4 py-3 rounded-2xl border border-[#E5E0D8] bg-white hover:border-stone-400 text-left text-xs sm:text-sm font-medium text-stone-800 flex items-center justify-between gap-2.5 transition-colors cursor-pointer"
                            >
                              <span>{item.question}</span>
                              <ArrowUpRight
                                className="w-4 h-4 text-teal-700 shrink-0"
                                aria-hidden="true"
                              />
                            </button>
                          ))}
                        </div>
                      </section>
                    )}
                  </div>
                )}
              </div>

              <aside
                aria-label="Recent session history"
                className="bg-white border border-[#E5E0D8] rounded-2xl overflow-hidden lg:sticky lg:top-6"
              >
                <div className="px-4 py-3.5 border-b border-[#EDEAE3] flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <History className="w-4 h-4 text-rose-700 shrink-0" aria-hidden="true" />
                    <h2 className="font-display text-sm font-bold text-stone-900">
                      Session History
                    </h2>
                    <span className="font-mono tabular-nums text-xs text-stone-500">
                      {history.length}/{HISTORY_LIMIT}
                    </span>
                  </div>
                  {history.length > 0 && (
                    <button
                      type="button"
                      onClick={handleClearHistory}
                      className="min-h-[36px] px-2.5 py-1 rounded-lg text-xs font-medium text-stone-600 hover:text-rose-700 hover:bg-rose-50 inline-flex items-center gap-1 cursor-pointer transition-colors whitespace-nowrap"
                    >
                      <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                      <span>Clear</span>
                    </button>
                  )}
                </div>

                {history.length === 0 ? (
                  <p className="p-4 text-xs text-stone-500 leading-relaxed">
                    Your last {HISTORY_LIMIT} questions and answers from this session will appear
                    here for quick switching.
                  </p>
                ) : (
                  <div>
                    {averageHistoryRating !== null && (
                      <div className="px-4 py-2.5 border-b border-[#EDEAE3] bg-[#FBF9F5] flex items-center justify-between text-xs text-stone-600 font-mono tabular-nums">
                        <span>{ratedHistoryEntries.length} rated</span>
                        <span className="inline-flex items-center gap-1 text-amber-800 font-semibold">
                          <Star
                            className="w-3.5 h-3.5 fill-amber-400 text-amber-500"
                            aria-hidden="true"
                          />
                          <span>Avg {averageHistoryRating.toFixed(1)}/5</span>
                        </span>
                      </div>
                    )}
                    <div className="divide-y divide-[#EDEAE3]">
                      {history.map((entry) => {
                        const isActive = activeHistoryId === entry.id;
                        const formattedTime = new Date(entry.askedAt).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        });
                        return (
                          <button
                            key={entry.id}
                            type="button"
                            onClick={() => handleSelectHistory(entry)}
                            aria-current={isActive ? "true" : undefined}
                            className={`w-full min-h-[48px] text-left px-4 py-3 transition-colors cursor-pointer ${
                              isActive ? "bg-[#F4F1EA]" : "bg-white hover:bg-[#FBF9F5]"
                            }`}
                          >
                            <div className="text-xs sm:text-sm font-semibold text-stone-900 line-clamp-2">
                              {entry.question}
                            </div>
                            <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px] text-stone-500 font-mono tabular-nums">
                              <span>
                                {entry.response.abstained
                                  ? "Abstained"
                                  : `${entry.response.sources.length} sources`}
                              </span>
                              <span className="inline-flex items-center gap-2">
                                {typeof entry.rating === "number" && (
                                  <span
                                    className="inline-flex items-center gap-0.5 text-amber-800 font-semibold"
                                    aria-label={`Rated ${entry.rating} out of 5 stars`}
                                  >
                                    <Star
                                      className="w-3 h-3 fill-amber-400 text-amber-500"
                                      aria-hidden="true"
                                    />
                                    <span>{entry.rating}/5</span>
                                  </span>
                                )}
                                <span>
                                  {entry.response.best_score.toFixed(2)} · {formattedTime}
                                </span>
                              </span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </aside>
            </div>
          </div>
        )}

        {activeTab === "eval" && (
          <div className="space-y-6">
            <section>
              <h1 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-stone-900">
                Golden Set Evaluation
              </h1>
              <p className="mt-1 text-sm text-stone-600">
                Measured on {evalSummary?.total_questions ?? 40} queries (
                {evalSummary?.answerable_questions ?? 34} answerable ·{" "}
                {evalSummary?.out_of_scope_questions ?? 6} out-of-scope) from{" "}
                <span className="font-mono">reports/latest.json</span>.
              </p>
            </section>

            {evalSummary ? (
              <section className="bg-white border border-[#E5E0D8] rounded-2xl grid grid-cols-2 sm:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[#EDEAE3]">
                <div className="p-4 sm:p-5">
                  <div className="text-xs font-semibold text-teal-800">Hit@1 / Hit@3 / Hit@5</div>
                  <div className="mt-1.5 text-lg sm:text-xl font-bold font-mono tabular-nums text-stone-900">
                    {(evalSummary.hit_at_1 * 100).toFixed(1)}% ·{" "}
                    {(evalSummary.hit_at_3 * 100).toFixed(0)}% ·{" "}
                    {(evalSummary.hit_at_5 * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-4 sm:p-5">
                  <div className="text-xs font-semibold text-blue-800">Mean Reciprocal Rank</div>
                  <div className="mt-1.5 text-lg sm:text-xl font-bold font-mono tabular-nums text-stone-900">
                    {evalSummary.mrr.toFixed(4)}
                  </div>
                </div>
                <div className="p-4 sm:p-5">
                  <div className="text-xs font-semibold text-violet-800">Keyword Coverage</div>
                  <div className="mt-1.5 text-lg sm:text-xl font-bold font-mono tabular-nums text-stone-900">
                    {(evalSummary.keyword_coverage * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 sm:p-5">
                  <div className="text-xs font-semibold text-amber-900">False Abstention Rate</div>
                  <div className="mt-1.5 text-lg sm:text-xl font-bold font-mono tabular-nums text-stone-900">
                    {(evalSummary.false_abstention_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </section>
            ) : null}

            <section className="bg-white border border-[#E5E0D8] rounded-2xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-[#E5E0D8] bg-[#FBF9F5] text-xs font-semibold text-stone-600">
                      <th className="py-3.5 px-4">ID</th>
                      <th className="py-3.5 px-4">Question</th>
                      <th className="py-3.5 px-4">Expected Doc</th>
                      <th className="py-3.5 px-4 text-right">Score</th>
                      <th className="py-3.5 px-4 text-center">Hit@3</th>
                      <th className="py-3.5 px-4 text-center">Abstained</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#EDEAE3]">
                    {evalRows.map((row) => (
                      <tr key={row.id} className="hover:bg-[#FBF9F5]">
                        <td className="py-3 px-4 font-mono text-xs text-stone-500">{row.id}</td>
                        <td className="py-3 px-4 text-stone-900 max-w-md">{row.question}</td>
                        <td className="py-3 px-4 font-mono text-xs text-stone-600">
                          {row.expected_doc ?? "OUT_OF_SCOPE"}
                        </td>
                        <td className="py-3 px-4 font-mono tabular-nums text-right text-stone-800">
                          {row.best_score.toFixed(3)}
                        </td>
                        <td className="py-3 px-4 text-center text-xs font-semibold">
                          <span
                            className={
                              row.hit_at_3 === null
                                ? "text-stone-400"
                                : row.hit_at_3
                                  ? "text-teal-700"
                                  : "text-rose-700"
                            }
                          >
                            {row.hit_at_3 === null ? "—" : row.hit_at_3 ? "Yes" : "No"}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center text-xs font-medium text-stone-700">
                          {row.abstained ? "Yes" : "No"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {activeTab === "docs" && (
          <div className="space-y-6">
            <section className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
              <div>
                <h1 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-stone-900">
                  Policy Library
                </h1>
                <p className="mt-1 text-sm text-stone-600">
                  Source files from <span className="font-mono">insurance-rag/data/raw/</span>,
                  split into ~900-character passages.
                </p>
              </div>
              <div className="w-full sm:w-80 shrink-0">
                <label htmlFor="doc-search-filter" className="sr-only">
                  Filter documents by title or filename
                </label>
                <div className="relative">
                  <Search
                    className="w-4 h-4 text-stone-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none"
                    aria-hidden="true"
                  />
                  <input
                    id="doc-search-filter"
                    type="text"
                    value={docFilter}
                    onChange={(e) => setDocFilter(e.target.value)}
                    placeholder="Filter by title or filename..."
                    className="w-full min-h-[44px] rounded-xl border border-[#DFDAD0] bg-white pl-10 pr-10 py-2 text-sm text-stone-900 placeholder:text-stone-400 focus:outline-2 focus:outline-rose-700"
                  />
                  {docFilter && (
                    <button
                      type="button"
                      onClick={() => setDocFilter("")}
                      aria-label="Clear document filter"
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 min-h-[36px] min-w-[36px] inline-flex items-center justify-center rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-100 cursor-pointer"
                    >
                      <X className="w-4 h-4" aria-hidden="true" />
                    </button>
                  )}
                </div>
              </div>
            </section>

            {filteredDocuments.length > 0 && (
              <section className="bg-white border border-[#E5E0D8] rounded-2xl p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div>
                    <h2 className="font-display text-base font-bold text-stone-900">
                      Passages per Document
                    </h2>
                    <p className="text-xs text-stone-500 mt-0.5">
                      Heading-aware chunks indexed across each source file
                    </p>
                  </div>
                  <div className="text-xs font-mono tabular-nums text-stone-600">
                    {filteredDocuments.length}
                    {normalizedDocFilter ? ` of ${documents.length}` : ""} sources ·{" "}
                    {filteredDocuments.reduce((sum, doc) => sum + doc.chunkCount, 0)} chunks
                  </div>
                </div>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={filteredDocuments.map((doc, idx) => ({
                        name:
                          doc.title.length > 26 ? `${doc.title.slice(0, 24).trim()}…` : doc.title,
                        fullTitle: doc.title,
                        source: doc.source,
                        chunks: doc.chunkCount,
                        sections: doc.sections.length,
                        fill: PIN_PALETTES[idx % PIN_PALETTES.length].barFill,
                      }))}
                      margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edeae3" />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: 11, fill: "#57534e" }}
                        tickLine={false}
                        axisLine={{ stroke: "#d6d3d1" }}
                        interval={0}
                      />
                      <YAxis
                        allowDecimals={false}
                        tick={{ fontSize: 11, fill: "#57534e" }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: "#fbf9f5" }}
                        content={({ active, payload }) => {
                          if (!active || !payload || payload.length === 0) return null;
                          const item = payload[0].payload as {
                            fullTitle: string;
                            source: string;
                            chunks: number;
                            sections: number;
                            fill: string;
                          };
                          return (
                            <div className="rounded-xl border border-[#E5E0D8] bg-white px-3.5 py-2.5 shadow-sm text-xs space-y-1 max-w-xs">
                              <div className="font-semibold text-stone-900">{item.fullTitle}</div>
                              <div className="font-mono text-stone-500 break-all">{item.source}</div>
                              <div className="font-mono tabular-nums font-semibold pt-0.5 text-stone-800">
                                {item.chunks} chunks · {item.sections} sections
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Bar dataKey="chunks" name="Chunks" radius={[8, 8, 0, 0]} maxBarSize={64}>
                        {filteredDocuments.map((doc, idx) => (
                          <Cell
                            key={doc.source}
                            fill={PIN_PALETTES[idx % PIN_PALETTES.length].barFill}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>
            )}

            {documents.length > 0 && filteredDocuments.length === 0 ? (
              <section className="bg-white border border-[#E5E0D8] rounded-2xl p-8 text-center space-y-3">
                <p className="text-sm text-stone-600">
                  No indexed documents match{" "}
                  <span className="font-semibold text-stone-900">"{docFilter.trim()}"</span>.
                </p>
                <button
                  type="button"
                  onClick={() => setDocFilter("")}
                  className="min-h-[44px] px-4 py-2 rounded-xl border border-[#E5E0D8] bg-[#FBF9F5] hover:bg-[#F3F0EA] text-xs font-semibold text-stone-800 cursor-pointer transition-colors"
                >
                  Clear filter
                </button>
              </section>
            ) : (
              <section className="columns-1 md:columns-2 gap-6 space-y-6">
                {filteredDocuments.map((doc, idx) => {
                  const palette = PIN_PALETTES[idx % PIN_PALETTES.length];
                  const artworkSrc = getDocumentArtwork(doc, idx);
                  const isImgBroken = !!brokenImages[artworkSrc];
                  return (
                    <article
                      key={doc.source}
                      className="break-inside-avoid bg-white border border-[#E5E0D8] rounded-2xl overflow-hidden transition-colors hover:border-stone-400"
                    >
                      <div
                        className={`aspect-[16/9] w-full overflow-hidden bg-gradient-to-br ${palette.fallbackGradient} border-b border-[#EDEAE3]`}
                      >
                        {!isImgBroken ? (
                          <img
                            src={artworkSrc}
                            alt={`Editorial cover illustration for ${doc.title}`}
                            referrerPolicy="no-referrer"
                            onError={() =>
                              setBrokenImages((prev) => ({ ...prev, [artworkSrc]: true }))
                            }
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center gap-2 p-4 text-center">
                            <BookOpen
                              className={`w-8 h-8 ${palette.accentText}`}
                              aria-hidden="true"
                            />
                            <span className="font-display text-sm font-bold text-stone-800">
                              {doc.title}
                            </span>
                          </div>
                        )}
                      </div>

                      <div className="p-5 sm:p-6 space-y-4">
                        <div className="flex items-start gap-3">
                          <FileText
                            className={`w-5 h-5 ${palette.accentText} shrink-0 mt-0.5`}
                            aria-hidden="true"
                          />
                          <div className="min-w-0">
                            <h2 className="font-display text-lg font-bold text-stone-900 leading-snug">
                              {doc.title}
                            </h2>
                            <p className="text-xs text-stone-500 font-mono tabular-nums mt-1 break-all">
                              {doc.source} · {doc.chunkCount} chunks · {doc.sections.length} sections
                            </p>
                          </div>
                        </div>

                        <ul className="divide-y divide-[#EDEAE3] border-t border-[#EDEAE3] text-xs text-stone-700">
                          {doc.sections.map((sec, secIdx) => (
                            <li key={sec} className="py-2 flex items-baseline gap-2.5">
                              <span
                                className={`font-mono tabular-nums text-[11px] font-semibold ${palette.numeralText} shrink-0`}
                              >
                                {String(secIdx + 1).padStart(2, "0")}
                              </span>
                              <span>{sec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </article>
                  );
                })}
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
