import "dotenv/config";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RAG_DIR = path.resolve(__dirname, "insurance-rag");
const CHUNKS_PATH = path.join(RAG_DIR, "data", "index", "chunks.json");
const RAW_DIR = path.join(RAG_DIR, "data", "raw");
const REPORT_PATH = path.join(RAG_DIR, "reports", "latest.json");

const ABSTENTION_TEXT = "I could not find this in the indexed documents.";
const DISCLAIMER_TEXT =
  "Disclaimer: This response provides general information based only on the indexed insurance documents and is not legal or financial advice.";
const ABSTENTION_THRESHOLD = Number(process.env.ABSTENTION_THRESHOLD ?? "0.25");
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-4-hydrogen";

interface IndexedChunk {
  chunk_id: string;
  text: string;
  title: string;
  section: string;
  source: string;
  chunk_index: number;
  char_count: number;
}

const INSURANCE_SYNONYMS: Record<string, string[]> = {
  reject: ["repudiation", "repudiated", "claim rejection"],
  rejected: ["repudiation", "repudiated", "claim rejection"],
  rejection: ["repudiation", "repudiated"],
  deny: ["repudiation", "rejection", "denial"],
  denied: ["repudiation", "rejection", "denial"],
  "switch insurer": ["portability", "migration"],
  switch: ["portability", "migration"],
  "change insurer": ["portability", "migration"],
  port: ["portability", "migration"],
  waiting: ["waiting period", "pre-existing disease", "PED"],
  ped: ["pre-existing disease", "36 months"],
  cancel: ["free look period", "policy cancellation", "refund"],
  cancellation: ["free look period", "policy cancellation", "refund"],
  grace: ["grace period", "continuity"],
  ncb: ["no claim bonus", "cumulative bonus"],
  bonus: ["cumulative bonus", "no claim bonus"],
  pmjay: ["PM-JAY", "Ayushman Bharat", "HBP"],
  "pm-jay": ["PM-JAY", "Ayushman Bharat", "HBP"],
  ayushman: ["PM-JAY", "Ayushman Bharat"],
  ombudsman: ["Insurance Ombudsman", "grievance redressal"],
  complaint: ["grievance redressal", "Bima Bharosa", "IGMS", "Insurance Ombudsman"],
  grievance: ["grievance redressal", "Bima Bharosa", "IGMS"],
  moratorium: ["moratorium period", "60 continuous months", "5 years"],
  cashless: ["cashless", "pre-authorization", "discharge authorization"],
  discharge: ["discharge authorization", "cashless", "3 hours"],
  ayush: ["AYUSH", "Ayurveda", "Homeopathy", "Unani", "Siddha"],
  cis: ["Customer Information Sheet"],
  nominee: ["nomination", "beneficial nominee", "Section 39"],
  "room rent": ["room rent", "proportionate deduction", "ICU"],
  copay: ["co-payment"],
};

const STOPWORDS = new Set([
  "the", "is", "are", "a", "an", "in", "on", "of", "for", "to", "and", "or",
  "what", "when", "how", "many", "do", "does", "my", "i", "it", "can", "must",
  "under", "with", "from", "by", "this", "that", "there", "who", "if", "any",
]);

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+(?:[-'][a-z0-9]+)*/g) ?? [];
}

function expandQuery(query: string): string {
  const lowered = query.toLowerCase();
  const tokenSet = new Set(tokenize(lowered));
  const additions: string[] = [];
  const triggers = Object.keys(INSURANCE_SYNONYMS).sort((a, b) => b.length - a.length);

  for (const trigger of triggers) {
    const matched =
      trigger.includes(" ") || trigger.includes("-")
        ? lowered.includes(trigger)
        : tokenSet.has(trigger);
    if (matched) {
      for (const exp of INSURANCE_SYNONYMS[trigger]) {
        if (!lowered.includes(exp.toLowerCase()) && !additions.includes(exp)) {
          additions.push(exp);
        }
      }
    }
  }
  return additions.length > 0 ? `${query.trim()} ${additions.join(" ")}` : query.trim();
}

function loadChunks(): IndexedChunk[] {
  if (!fs.existsSync(CHUNKS_PATH)) {
    return [];
  }
  const raw = fs.readFileSync(CHUNKS_PATH, "utf-8");
  return JSON.parse(raw) as IndexedChunk[];
}

function scorePassages(chunks: IndexedChunk[], query: string, topK: number) {
  if (chunks.length === 0) return [];
  const expanded = expandQuery(query);
  const qTokensExpanded = tokenize(expanded);
  const qContentTokens = Array.from(
    new Set(qTokensExpanded.filter((t) => !STOPWORDS.has(t)))
  );

  // Build BM25 document frequencies
  const docTokens = chunks.map((c) => tokenize(`${c.title}\n${c.section}\n${c.text}`));
  const avgdl =
    docTokens.reduce((acc, toks) => acc + toks.length, 0) / Math.max(docTokens.length, 1);
  const df = new Map<string, number>();
  for (const toks of docTokens) {
    for (const unique of new Set(toks)) {
      df.set(unique, (df.get(unique) ?? 0) + 1);
    }
  }

  const k1 = 1.5;
  const b = 0.75;
  const nDocs = chunks.length;

  const bm25Scores = docTokens.map((toks) => {
    const tfMap = new Map<string, number>();
    for (const t of toks) tfMap.set(t, (tfMap.get(t) ?? 0) + 1);
    let score = 0;
    for (const term of qTokensExpanded) {
      const freq = df.get(term) ?? 0;
      if (freq === 0) continue;
      const tf = tfMap.get(term) ?? 0;
      if (tf === 0) continue;
      const idf = Math.log((nDocs - freq + 0.5) / (freq + 0.5) + 1);
      const denom = tf + k1 * (1 - b + b * (toks.length / (avgdl || 1)));
      score += idf * ((tf * (k1 + 1)) / denom);
    }
    return score;
  });

  const bm25Ranks = bm25Scores
    .map((s, idx) => ({ idx, s }))
    .sort((a, b2) => b2.s - a.s)
    .slice(0, 20);

  const reranked = bm25Ranks.map(({ idx }) => {
    const chunk = chunks[idx];
    const pTokens = new Set(docTokens[idx]);
    const overlap = qContentTokens.filter((t) => pTokens.has(t)).length;
    let ratio = qContentTokens.length > 0 ? overlap / qContentTokens.length : 0;
    if (overlap <= 1 && qContentTokens.length >= 4) {
      ratio *= 0.4;
    }
    const score = Math.min(1, Number((ratio * 1.15).toFixed(4)));
    return { chunk, score };
  });

  reranked.sort((a, b2) => b2.score - a.score);
  return reranked.slice(0, topK);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractErrorDetails(err: unknown): { code?: number; status?: string; message: string } {
  const raw = err instanceof Error ? err.message : String(err ?? "Unknown Gemini API error.");
  try {
    const parsed = JSON.parse(raw) as {
      error?: { code?: number; status?: string; message?: string };
    };
    if (parsed?.error) {
      return {
        code: parsed.error.code,
        status: parsed.error.status,
        message: parsed.error.message || raw,
      };
    }
  } catch {
    // raw is plain text
  }
  const statusCode =
    typeof (err as { status?: unknown })?.status === "number"
      ? ((err as { status: number }).status)
      : undefined;
  return { code: statusCode, message: raw };
}

function isTransientGeminiError(code?: number, status?: string, message?: string): boolean {
  if (code === 503 || code === 429 || code === 500 || code === 502 || code === 504) {
    return true;
  }
  const upper = `${status ?? ""} ${message ?? ""}`.toUpperCase();
  return (
    upper.includes("UNAVAILABLE") ||
    upper.includes("HIGH DEMAND") ||
    upper.includes("RESOURCE_EXHAUSTED") ||
    upper.includes("ETIMEDOUT") ||
    upper.includes("ECONNRESET") ||
    upper.includes("FETCH FAILED")
  );
}

function formatFriendlyGeminiError(code?: number, status?: string, message?: string): { httpStatus: number; detail: string } {
  const upper = `${status ?? ""} ${message ?? ""}`.toUpperCase();
  if (code === 400 || code === 403 || upper.includes("API_KEY_INVALID") || upper.includes("PERMISSION_DENIED")) {
    return {
      httpStatus: 403,
      detail:
        "Gemini API authentication failed. Please check or update GEMINI_API_KEY in Settings > Secrets.",
    };
  }
  if (code === 429 || upper.includes("RESOURCE_EXHAUSTED")) {
    return {
      httpStatus: 429,
      detail:
        "Gemini API rate limit reached. Please wait a moment and try again, or check your quota in Settings > Secrets.",
    };
  }
  if (code === 503 || upper.includes("UNAVAILABLE") || upper.includes("HIGH DEMAND")) {
    return {
      httpStatus: 503,
      detail:
        "The Gemini model is temporarily experiencing high demand. Please try asking again in a few seconds.",
    };
  }
  return {
    httpStatus: 502,
    detail: message || "Unable to generate an answer from the Gemini API right now.",
  };
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

interface ConversationTurn {
  question: string;
  answer: string;
}

function isReferentialFollowUp(question: string): boolean {
  const tokens = tokenize(question).filter((t) => !STOPWORDS.has(t));
  if (tokens.length <= 3) {
    return true;
  }
  return /\b(it|its|this|that|these|those|they|them|their|above|same|former|latter|exception|exceptions|more|detail|details|explain|clarify|summarize|else|also|apply|applies)\b/i.test(
    question
  );
}

async function generateGroundedAnswer(
  ai: GoogleGenAI,
  question: string,
  contextBlock: string,
  priorTurns: ConversationTurn[] = []
): Promise<string> {
  const candidateModels =
    GEMINI_MODEL === "gemini-4-hydrogen"
      ? ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
      : [GEMINI_MODEL];

  const historyBlock =
    priorTurns.length > 0
      ? `Prior conversation context (use only to understand what follow-up references like "it" or "this" mean; still verify every fact strictly against the numbered Context passages below):\n${priorTurns
          .map((turn, i) => `Turn ${i + 1} Q: ${turn.question}\nTurn ${i + 1} A: ${turn.answer}`)
          .join("\n\n")}\n\n`
      : "";

  let lastErr: unknown;
  for (const model of candidateModels) {
    // On high-demand 503 for primary model, fail over quickly to gemini-3.1-flash-lite;
    // allow one short retry on the fallback model.
    const delays = model === "gemini-4-hydrogen" ? [0] : [0, 500];
    for (let attempt = 0; attempt < delays.length; attempt += 1) {
      if (delays[attempt] > 0) {
        await sleep(delays[attempt]);
      }
      try {
        const config: {
          systemInstruction: string;
          thinkingConfig?: { thinkingLevel: ThinkingLevel };
        } = {
          systemInstruction:
            'You are an Indian insurance information assistant. Answer STRICTLY and ONLY from the numbered context passages. Cite supporting passages inline as [1], [2]. Never invent numbers, dates, or clauses. If the passages do not contain the answer, respond with exact text: "I could not find this in the indexed documents."',
        };
        if (model.startsWith("gemini-4")) {
          config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
        }
        const response = await withTimeout(
          ai.models.generateContent({
            model,
            contents: `${historyBlock}Context passages:\n${contextBlock}\n\nQuestion: ${question}\n\nAnswer concisely with inline citations [1], [2] using ONLY the context passages above.`,
            config,
          }),
          12000,
          `Gemini (${model})`
        );
        return (response.text ?? "").trim() || ABSTENTION_TEXT;
      } catch (err) {
        lastErr = err;
        const details = extractErrorDetails(err);
        const msgUpper = details.message.toUpperCase();
        const isTimedOut = msgUpper.includes("TIMED OUT");
        if (!isTimedOut && !isTransientGeminiError(details.code, details.status, details.message)) {
          throw err;
        }
      }
    }
  }
  throw lastErr;
}

async function createServer() {
  const app = express();
  app.use(express.json());

  const chunks = loadChunks();

  const handleHealth = (_req: express.Request, res: express.Response) => {
    res.json({
      status: "ok",
      index_loaded: chunks.length > 0,
      chunks_count: chunks.length,
    });
  };

  const handleAsk = async (req: express.Request, res: express.Response) => {
    const question = typeof req.body?.question === "string" ? req.body.question.trim() : "";
    const topK =
      typeof req.body?.top_k === "number" && req.body.top_k >= 1 && req.body.top_k <= 20
        ? req.body.top_k
        : 5;

    const rawContext: unknown[] = Array.isArray(req.body?.context) ? req.body.context : [];
    const priorTurns: ConversationTurn[] = rawContext
      .filter(
        (item: unknown): item is ConversationTurn =>
          Boolean(item) &&
          typeof (item as { question?: unknown }).question === "string" &&
          typeof (item as { answer?: unknown }).answer === "string"
      )
      .slice(-4)
      .map((item: ConversationTurn): ConversationTurn => ({
        question: item.question.trim().slice(0, 500),
        answer: item.answer.trim().slice(0, 1200),
      }))
      .filter((item: ConversationTurn) => item.question.length > 0);

    if (!question) {
      res.status(422).json({ detail: "Question must not be blank." });
      return;
    }
    if (chunks.length === 0) {
      res.status(503).json({ detail: "Index not loaded. Run `insurance-rag ingest` first." });
      return;
    }

    let retrieved = scorePassages(chunks, question, topK);
    let bestScore = retrieved.length > 0 ? retrieved[0].score : 0;

    if (
      priorTurns.length > 0 &&
      bestScore < ABSTENTION_THRESHOLD &&
      isReferentialFollowUp(question)
    ) {
      const lastTurn = priorTurns[priorTurns.length - 1];
      const contextualRetrieved = scorePassages(
        chunks,
        `${lastTurn.question} ${question}`,
        topK
      );
      const contextualBest = contextualRetrieved.length > 0 ? contextualRetrieved[0].score : 0;
      if (contextualBest > bestScore) {
        retrieved = contextualRetrieved;
        bestScore = contextualBest;
      }
    }

    if (retrieved.length === 0 || bestScore < ABSTENTION_THRESHOLD) {
      res.json({
        answer: ABSTENTION_TEXT,
        sources: [],
        abstained: true,
        best_score: bestScore,
        disclaimer: DISCLAIMER_TEXT,
      });
      return;
    }

    const apiKey = (process.env.GEMINI_API_KEY ?? "").trim();
    if (!apiKey) {
      res.status(500).json({
        detail:
          "GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in Settings > Secrets to generate grounded answers.",
      });
      return;
    }

    const contextBlock = retrieved
      .map(
        (item, idx) =>
          `[${idx + 1}] Document: ${item.chunk.title} | Section: ${item.chunk.section} | File: ${item.chunk.source}\n${item.chunk.text}`
      )
      .join("\n\n");

    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });
      const rawText = await generateGroundedAnswer(ai, question, contextBlock, priorTurns);
      const abstained = rawText.toLowerCase().includes("could not find this in the indexed documents");

      res.json({
        answer: abstained ? ABSTENTION_TEXT : rawText,
        sources: abstained
          ? []
          : retrieved.map((item, idx) => ({
              index: idx + 1,
              title: item.chunk.title,
              section: item.chunk.section,
              source: item.chunk.source,
              score: item.score,
              text: item.chunk.text,
            })),
        abstained,
        best_score: bestScore,
        disclaimer: DISCLAIMER_TEXT,
      });
    } catch (err) {
      const details = extractErrorDetails(err);
      const friendly = formatFriendlyGeminiError(details.code, details.status, details.message);
      res.status(friendly.httpStatus).json({ detail: friendly.detail });
    }
  };

  app.get("/api/health", handleHealth);
  app.get("/health", handleHealth);
  app.post("/api/ask", handleAsk);
  app.post("/ask", handleAsk);

  app.get("/api/report", (_req, res) => {
    if (!fs.existsSync(REPORT_PATH)) {
      res.status(404).json({ detail: "Evaluation report not found." });
      return;
    }
    res.json(JSON.parse(fs.readFileSync(REPORT_PATH, "utf-8")));
  });

  app.get("/api/documents", (_req, res) => {
    const files = fs.existsSync(RAW_DIR)
      ? fs.readdirSync(RAW_DIR).filter((f) => !f.startsWith(".")).sort()
      : [];

    const bySource = new Map<
      string,
      { source: string; title: string; sections: string[]; chunkCount: number }
    >();
    for (const c of chunks) {
      const existing = bySource.get(c.source);
      if (!existing) {
        bySource.set(c.source, {
          source: c.source,
          title: c.title,
          sections: [c.section],
          chunkCount: 1,
        });
      } else {
        existing.chunkCount += 1;
        if (!existing.sections.includes(c.section)) {
          existing.sections.push(c.section);
        }
      }
    }

    res.json({
      files,
      documents: Array.from(bySource.values()),
      totalChunks: chunks.length,
    });
  });

  if (process.env.NODE_ENV === "production") {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  app.listen(3000, "0.0.0.0", () => {
    console.log("Indian Insurance RAG server listening on http://0.0.0.0:3000");
  });
}

createServer();
