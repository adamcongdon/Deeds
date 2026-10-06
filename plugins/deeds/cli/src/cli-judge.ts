/**
 * A local CLI judge: the same typed questions, policy and per-commit cache as Jev, with the judgment
 * request answered by a subscription-authenticated CLI instead of an API key. The primary CLI is
 * xAI's Grok Build (`grok -p` / `--prompt-file` with `--output-format json`). Cursor's CLI
 * (`cursor-agent -p --output-format json --mode ask`) is the other built-in preset. `DEEDS_CLI_CMD`
 * and `cli` in the deeds config drop another headless CLI in.
 *
 * The subprocess runs in a temporary directory, with the preset flags that keep it from editing the
 * repo (no `--always-approve` / `--yolo` / `--force`). Diffs reach it only as the already-redacted
 * state the Jev request carries. A failure throws, so the commit is left uncached and a rerun resumes.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { checkReply, type JevReply, type JevRequest, type JevTransport, type WireQuestion } from "./jev/judge.ts";
import type { Answer } from "./jev/policy.ts";

/** Default vendor for `--mode cli`. `--vendor grok` is an alias of this. */
export const GROK_CLI_VENDOR = "grok-cli";
/** `--vendor cursor` and `--vendor cursor-agent`. */
export const CURSOR_CLI_VENDOR = "cursor";

export const CLI_DEFAULT_CONCURRENCY = 2;
export const CLI_DEFAULT_TIMEOUT_MS = 180_000;
export const CLI_MAX_ATTEMPTS = 3;
/** A bare choice or score (no probability attached) is treated as a confident answer, above the policy thresholds. */
const BARE_CONFIDENCE = 0.9;
/** "no" / "false" for a noul, as a probability. A subtraction from 0.9 is not exactly 0.1. */
const BARE_NO = 0.1;

const CREDENTIAL_KEYS = new Set(["key", "access_token", "refresh_token", "api_key", "token", "id_token"]);

export type CliPrompt = "arg" | "stdin" | "file";

/** Everything analyze needs to spawn one judge CLI. Built by resolve.ts; the transport only runs it. */
export interface ResolvedCli {
  mode: "cli";
  /** grok-cli, cursor, or a custom id from `--vendor` / config. */
  vendor: string;
  /** What the report names. The cache id is `cacheModelId`, which also names the command. */
  modelId: string;
  /** Cache identity: vendor, command basename and the static argv, so two judges never share entries. */
  cacheModelId: string;
  cmd: string;
  args: string[];
  prompt: CliPrompt;
  promptFileFlag: string;
  /** When set, the temp directory is also passed as this flag (grok's `--cwd`). */
  cwdFlag?: string;
  timeoutMs: number;
  concurrency: number;
  maxAttempts: number;
}

export interface CliPreset {
  cmd: string;
  args: string[];
  prompt: CliPrompt;
  promptFileFlag: string;
  cwdFlag?: string;
}

/**
 * Grok does not read a piped prompt (headless docs: use `-p` or `--prompt-file`). The prompt file sits
 * in the temp cwd. Flags below are the documented read-only set: one turn, deny-by-default permissions,
 * no edit/shell/web/subagent tools, no always-approve. `--sandbox strict` is left out: a sandbox that
 * hides the temp prompt file would fail every commit, and this process never points the CLI at the repo.
 */
const GROK_PRESET: CliPreset = {
  cmd: "grok",
  prompt: "file",
  promptFileFlag: "--prompt-file",
  cwdFlag: "--cwd",
  args: [
    "--no-auto-update",
    "--output-format", "json",
    "--max-turns", "1",
    "--permission-mode", "dontAsk",
    "--disallowed-tools", "run_terminal_cmd,search_replace,web_search,web_fetch,Agent",
    "--disable-web-search",
    "--no-subagents",
    "--no-memory",
    "--no-plan",
    "--verbatim",
    "--no-alt-screen",
    "--deny", "Edit",
    "--deny", "Write",
    "--deny", "Bash",
  ],
};

/** Cursor print mode stays read-only in ask mode. The prompt is a positional argument after `-p`. */
const CURSOR_PRESET: CliPreset = {
  cmd: "cursor-agent",
  prompt: "arg",
  promptFileFlag: "--prompt-file",
  args: ["-p", "--output-format", "json", "--mode", "ask"],
};

/** `grok` / `grok-cli` and `cursor` / `cursor-agent` collapse to the preset ids. Anything else is custom. */
export function normalizeCliVendor(vendor: string): string {
  if (vendor === "grok" || vendor === "grok-cli") return GROK_CLI_VENDOR;
  if (vendor === "cursor" || vendor === "cursor-agent" || vendor === "agent") return CURSOR_CLI_VENDOR;
  return vendor;
}

export function cliPreset(vendor: string): CliPreset {
  if (vendor === GROK_CLI_VENDOR) return { ...GROK_PRESET, args: [...GROK_PRESET.args] };
  if (vendor === CURSOR_CLI_VENDOR) return { ...CURSOR_PRESET, args: [...CURSOR_PRESET.args] };
  return { cmd: vendor, args: [], prompt: "stdin", promptFileFlag: "--prompt-file" };
}

/** Vendor, binary basename and static args. A different command or flag set gets its own cache directory. */
export function cliCacheModelId(vendor: string, cmd: string, args: readonly string[]): string {
  const bin = basename(cmd) || cmd;
  const argsHash = createHash("sha256").update(args.join("\0")).digest("hex").slice(0, 8);
  return `${vendor}:${bin}:${argsHash}`;
}

/** Model flag for a preset, appended only when the user passed `--model`. */
export function cliModelArgs(vendor: string, model: string | undefined): string[] {
  if (!model) return [];
  if (vendor === GROK_CLI_VENDOR) return ["-m", model];
  return ["--model", model];
}

/**
 * The same questions Jev is asked, as one prompt. The model is told to answer and not to act.
 * `BEGIN_DEEDS_QUESTIONS` / `BEGIN_DEEDS_STATE` markers keep the two documents unambiguous. The end
 * markers are not suffixes of the begin markers, so a split on the begin line does not cut the JSON.
 */
export function buildJudgePrompt(req: JevRequest): string {
  const questions = Object.entries(req.questions).map(([id, q]) => ({
    id,
    type: q.type,
    instructions: q.instructions,
    ...(q.criteria !== undefined ? { criteria: q.criteria } : {}),
  }));
  return [
    "You are the deeds judge. You answer questions about one already-computed commit state.",
    "Do not use tools. Do not read, write or edit files. Do not run commands. Do not search the web. Only answer.",
    "Reply with a single JSON object and no other text. Each key is a question id.",
    "A noul value is a probability from 0 to 1 that the statement is true.",
    "A choice value is exactly one of that question's option keys.",
    "A score value is a number on that question's scale.",
    "Example: {\"capability_changed\":0.9,\"capability_change_kind\":\"no_capability_change\"}",
    "",
    "BEGIN_DEEDS_QUESTIONS",
    JSON.stringify(questions),
    "END_DEEDS_QUESTIONS",
    "",
    "BEGIN_DEEDS_STATE",
    JSON.stringify(req.state),
    "END_DEEDS_STATE",
  ].join("\n");
}

export interface CliInvocation {
  argv: string[];
  cwd: string;
  stdin?: string;
  cleanup: () => void;
}

/** Temp cwd plus the argv the CLI documents. Placeholders `{prompt}`, `{prompt-file}` and `{cwd}` are filled in. */
export function buildCliInvocation(cfg: Pick<ResolvedCli, "cmd" | "args" | "prompt" | "promptFileFlag" | "cwdFlag">, prompt: string): CliInvocation {
  const cwd = mkdtempSync(join(tmpdir(), "deeds-cli-"));
  const promptFile = join(cwd, "prompt.txt");
  let usedPrompt = false;
  let usedFile = false;
  let usedCwd = false;
  const args = cfg.args.map((a) => {
    if (a === "{prompt}") { usedPrompt = true; return prompt; }
    if (a === "{prompt-file}") { usedFile = true; return promptFile; }
    if (a === "{cwd}") { usedCwd = true; return cwd; }
    return a;
  });
  if (usedFile || (!usedPrompt && cfg.prompt === "file")) writeFileSync(promptFile, prompt);
  if (!usedFile && !usedPrompt && cfg.prompt === "file") args.push(cfg.promptFileFlag, promptFile);
  if (!usedPrompt && !usedFile && cfg.prompt === "arg") args.push(prompt);
  if (cfg.cwdFlag && !usedCwd && !args.includes(cfg.cwdFlag)) args.push(cfg.cwdFlag, cwd);
  const stdin = !usedPrompt && !usedFile && cfg.prompt === "stdin" ? prompt : undefined;
  return {
    argv: [cfg.cmd, ...args],
    cwd,
    ...(stdin !== undefined ? { stdin } : {}),
    cleanup: () => rmSync(cwd, { recursive: true, force: true }),
  };
}

/** One CLI execution. Tests replace this; the live one is `spawnCli`. */
export type CliRunner = (spec: { argv: string[]; cwd: string; stdin?: string; timeoutMs: number }) => Promise<{ stdout: string; stderr: string; exitCode: number; timedOut?: boolean }>;

class CliFailure extends Error {
  constructor(message: string, readonly retryable: boolean) {
    super(message);
    this.name = "CliFailure";
  }
}

/** Spawn `argv` in `cwd`. A timeout kills the process and is retryable. ENOENT is not. */
export const spawnCli: CliRunner = async (spec) => {
  let proc: ReturnType<typeof Bun.spawn>;
  try {
    proc = Bun.spawn(spec.argv, {
      cwd: spec.cwd,
      stdin: spec.stdin === undefined ? "ignore" : "pipe",
      stdout: "pipe",
      stderr: "pipe",
      env: { ...process.env, GROK_DISABLE_AUTOUPDATER: "1" },
    });
  } catch (err) {
    const code = (err as { code?: string }).code;
    const message = err instanceof Error ? err.message : String(err);
    throw new CliFailure(code === "ENOENT" ? `${spec.argv[0]} is not on PATH` : message, code !== "ENOENT");
  }
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill();
  }, spec.timeoutMs);
  const stdin = proc.stdin;
  if (spec.stdin !== undefined && stdin && typeof stdin !== "number") {
    stdin.write(spec.stdin);
    stdin.end();
  }
  if (typeof proc.stdout === "number" || typeof proc.stderr === "number" || proc.stdout === undefined || proc.stderr === undefined) {
    throw new CliFailure("CLI spawn returned no pipes", true);
  }
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  clearTimeout(timer);
  if (timedOut) throw new CliFailure(`timed out after ${spec.timeoutMs}ms`, true);
  return { stdout, stderr, exitCode, timedOut: false };
};

function tryJson(text: string): unknown | undefined {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

/** The JSON object that starts at `start` (`s[start] === "{"`), respecting strings, or undefined. */
function balancedObject(s: string, start: number): string | undefined {
  if (s[start] !== "{") return undefined;
  let depth = 0;
  let quote: '"' | "'" | null = null;
  let escape = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i]!;
    if (quote) {
      if (escape) escape = false;
      else if (c === "\\") escape = true;
      else if (c === quote) quote = null;
      continue;
    }
    if (c === '"' || c === "'") { quote = c; continue; }
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return s.slice(start, i + 1);
    }
  }
  return undefined;
}

function lastJsonObject(s: string): unknown | undefined {
  let end = s.length;
  while (end > 0) {
    const start = s.lastIndexOf("{", end - 1);
    if (start < 0) return undefined;
    const slice = balancedObject(s, start);
    if (slice) {
      const parsed = tryJson(slice);
      if (parsed !== undefined) return parsed;
    }
    end = start;
  }
  return undefined;
}

/** One or more JSON documents from a CLI's stdout. Pretty-printed JSON wins; otherwise NDJSON, otherwise the last object. */
export function cliJsonDocuments(stdout: string): unknown[] {
  const s = stdout.replace(/^\uFEFF/, "").trim();
  if (!s) throw new CliFailure("CLI produced no output", true);
  const whole = tryJson(s);
  if (whole !== undefined) return [whole];
  const lines: unknown[] = [];
  for (const line of s.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const parsed = tryJson(t);
    if (parsed !== undefined) lines.push(parsed);
  }
  if (lines.length) return lines;
  const last = lastJsonObject(s);
  if (last !== undefined) return [last];
  throw new CliFailure("CLI output is not JSON", true);
}

function usageOf(doc: Record<string, unknown>): { input: number; output: number } {
  const u = doc.usage;
  if (!u || typeof u !== "object" || Array.isArray(u)) return { input: 0, output: 0 };
  const o = u as Record<string, unknown>;
  const n = (v: unknown) => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0);
  // grok's json object documents usage.input_tokens and usage.output_tokens. Absent fields stay 0.
  return { input: n(o.input_tokens), output: n(o.output_tokens) };
}

function assertCliOk(doc: Record<string, unknown>): void {
  // Cursor's json result documents is_error and subtype. Grok's object has neither; absence is success.
  if (doc.is_error === true) {
    const detail = typeof doc.result === "string" ? doc.result.slice(0, 300) : "is_error";
    throw new CliFailure(`CLI reported an error: ${detail}`, true);
  }
  if (doc.type === "result" && typeof doc.subtype === "string" && doc.subtype !== "success") {
    throw new CliFailure(`CLI result subtype ${doc.subtype}`, true);
  }
}

/**
 * The model's text, from fields the CLIs actually document:
 * grok `--output-format json` uses `text` (and streaming-json uses `{type:"text",data}`);
 * Cursor's json result uses `result`. Question-id keys on the object itself are the answer,
 * for a CLI that printed the model's JSON with no envelope. Any other shape names its keys and fails.
 */
export function extractAnswerSource(docs: unknown[], questionIds: readonly string[]): { text?: string; direct?: unknown; usage: { input: number; output: number } } {
  const texts: string[] = [];
  const usage = { input: 0, output: 0 };
  for (const doc of docs) {
    if (!doc || typeof doc !== "object" || Array.isArray(doc)) continue;
    const o = doc as Record<string, unknown>;
    assertCliOk(o);
    const u = usageOf(o);
    if (u.input || u.output) { usage.input = u.input; usage.output = u.output; }
    if (typeof o.text === "string" && o.text.trim()) texts.push(o.text);
    else if (typeof o.result === "string" && o.result.trim()) texts.push(o.result);
    else if (o.type === "text" && typeof o.data === "string" && o.data.trim()) texts.push(o.data);
    else if (o.text && typeof o.text === "object") return { direct: o.text, usage };
    else if (o.result && typeof o.result === "object") return { direct: o.result, usage };
  }
  if (texts.length) return { text: texts.join(""), usage };
  for (const doc of docs) {
    if (doc && typeof doc === "object" && !Array.isArray(doc) && questionIds.some((id) => Object.hasOwn(doc as object, id))) {
      return { direct: doc, usage };
    }
  }
  const keys = [...new Set(docs.flatMap((d) => (d && typeof d === "object" && !Array.isArray(d) ? Object.keys(d as object) : [])))].slice(0, 12);
  throw new CliFailure(`CLI JSON has no text answer (top-level keys: ${keys.join(", ") || "none"})`, true);
}

/** JSON out of model text. A fenced block wins when it parses; otherwise the whole text, otherwise the last object. */
export function parseAnswerDocument(text: string): unknown {
  for (const m of text.matchAll(/```(?:json)?[^\n]*\n([\s\S]*?)```/gi)) {
    const parsed = tryJson(m[1]!.trim());
    if (parsed !== undefined) return parsed;
  }
  const stripped = text.trim().replace(/^\uFEFF/, "").replace(/^\s*```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();
  const direct = tryJson(stripped);
  if (direct !== undefined) return direct;
  const embedded = lastJsonObject(stripped);
  if (embedded !== undefined) return embedded;
  throw new CliFailure("model text has no JSON answers", true);
}

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : undefined;
}

function noulOf(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "yes" || s === "true") return BARE_CONFIDENCE;
    if (s === "no" || s === "false") return BARE_NO;
    const n = Number(s);
    if (Number.isFinite(n) && s !== "") return n;
  }
  const o = asRecord(v);
  if (!o) return undefined;
  if (typeof o.noul === "number") return o.noul;
  return undefined;
}

function choiceKey(q: WireQuestion, raw: string): string {
  const options = asRecord(q.criteria);
  if (!options) return raw;
  if (Object.hasOwn(options, raw)) return raw;
  const hit = Object.keys(options).find((k) => k.toLowerCase() === raw.toLowerCase());
  return hit ?? raw;
}

function answerFor(q: WireQuestion, v: unknown): Answer {
  if (q.type === "noul") {
    const noul = noulOf(v);
    if (noul === undefined) throw new CliFailure(`no probability for "${q.type}" answer`, true);
    return { type: "noul", noul };
  }
  if (q.type === "choice") {
    const o = asRecord(v);
    const raw = typeof v === "string" ? v : typeof o?.choice === "string" ? o.choice : undefined;
    if (!raw) throw new CliFailure("choice answer has no choice", true);
    const choice = choiceKey(q, raw);
    const probabilities = asRecord(o?.probabilities) as Record<string, number> | undefined;
    const confidence = typeof o?.confidence === "number" ? o.confidence : (typeof probabilities?.[choice] === "number" ? probabilities[choice] : BARE_CONFIDENCE);
    const probs: Record<string, number> = {};
    if (probabilities) for (const [k, p] of Object.entries(probabilities)) if (typeof p === "number") probs[choiceKey(q, k)] = p;
    if (probs[choice] === undefined) probs[choice] = confidence;
    return { type: "choice", choice, confidence, probabilities: probs };
  }
  const o = asRecord(v);
  const score = typeof v === "number" ? v : typeof o?.score === "number" ? o.score : undefined;
  if (score === undefined || !Number.isFinite(score)) throw new CliFailure("score answer has no score", true);
  const confidence = typeof o?.confidence === "number" ? o.confidence : BARE_CONFIDENCE;
  return { type: "score", score, confidence, probabilities: {} };
}

/** Typed answers for exactly the questions in the request, from model text or from a direct JSON object. */
export function answersFromModel(source: { text?: string; direct?: unknown }, questions: Record<string, WireQuestion>): Record<string, Answer> {
  const parsed = source.direct !== undefined ? source.direct : parseAnswerDocument(source.text ?? "");
  let doc = asRecord(parsed);
  if (!doc) throw new CliFailure("model answers are not a JSON object", true);
  const ids = Object.keys(questions);
  if (!ids.some((id) => Object.hasOwn(doc!, id)) && asRecord(doc.answers)) doc = asRecord(doc.answers)!;
  const answers: Record<string, Answer> = {};
  for (const [id, q] of Object.entries(questions)) {
    if (!Object.hasOwn(doc, id)) throw new CliFailure(`model answers have no "${id}"`, true);
    answers[id] = answerFor(q, doc[id]);
  }
  return answers;
}

/** Stdout of one CLI call into the reply `checkReply` accepts. */
export function replyFromCliStdout(stdout: string, req: JevRequest, model: string): JevReply {
  const docs = cliJsonDocuments(stdout);
  const source = extractAnswerSource(docs, Object.keys(req.questions));
  const answers = answersFromModel(source, req.questions);
  return checkReply({ model, answers, usage: { input_tokens: source.usage.input, output_tokens: source.usage.output } }, req);
}

function snippet(s: string): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > 200 ? `${t.slice(0, 200)}…` : t;
}

/** The judge transport: one CLI call per request, retried on timeout, non-zero exit and unparseable output. */
export function cliTransport(cfg: ResolvedCli, hooks: { run?: CliRunner; sleep?: (ms: number) => Promise<void> } = {}): JevTransport {
  const run = hooks.run ?? spawnCli;
  const sleep = hooks.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  return async (req) => {
    const prompt = buildJudgePrompt(req);
    let last = "CLI judge failed";
    for (let attempt = 1; attempt <= cfg.maxAttempts; attempt++) {
      const inv = buildCliInvocation(cfg, prompt);
      try {
        const result = await run({ argv: inv.argv, cwd: inv.cwd, ...(inv.stdin !== undefined ? { stdin: inv.stdin } : {}), timeoutMs: cfg.timeoutMs });
        if (result.timedOut) throw new CliFailure(`timed out after ${cfg.timeoutMs}ms`, true);
        if (result.exitCode !== 0) {
          const detail = snippet(result.stderr) || snippet(result.stdout) || `exit ${result.exitCode}`;
          throw new CliFailure(`${cfg.cmd} exit ${result.exitCode}: ${detail}`, true);
        }
        return replyFromCliStdout(result.stdout, req, cfg.modelId);
      } catch (err) {
        const retryable = err instanceof CliFailure ? err.retryable : true;
        last = err instanceof Error ? err.message : String(err);
        if (!retryable || attempt >= cfg.maxAttempts) throw new Error(`${cfg.vendor}: ${last}`);
        await sleep(500 * 2 ** (attempt - 1));
      } finally {
        inv.cleanup();
      }
    }
    throw new Error(`${cfg.vendor}: ${last}`);
  };
}

export interface CliProbe {
  vendor: string;
  cmd: string;
  onPath: boolean;
  /** True or false when a local check can tell; null when the only check would need a network the sandbox denies. */
  authenticated: boolean | null;
  detail: string;
}

function credentialIn(value: unknown, depth = 0): boolean {
  if (depth > 6 || !value || typeof value !== "object") return false;
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (CREDENTIAL_KEYS.has(k) && typeof v === "string" && v.trim().length >= 8) return true;
    if (credentialIn(v, depth + 1)) return true;
  }
  return false;
}

/** `~/.grok/auth.json` (or `$GROK_HOME/auth.json`): true when a credential field is present, false when the file is missing or empty of credentials. */
export function grokAuthFileHasCredential(home: string, env: Record<string, string | undefined>): { state: "yes" | "no" | "invalid"; path: string } {
  const root = env.GROK_HOME && env.GROK_HOME !== "" ? env.GROK_HOME.replace(/^~(?=\/|$)/, home) : join(home, ".grok");
  const path = join(root, "auth.json");
  if (!existsSync(path)) return { state: "no", path };
  let parsed: unknown;
  try {
    const raw = readFileSync(path, "utf8");
    parsed = JSON.parse(raw.length > 2_000_000 ? raw.slice(0, 2_000_000) : raw);
  } catch {
    return { state: "invalid", path };
  }
  return { state: credentialIn(parsed) ? "yes" : "no", path };
}

function commandOnPath(cmd: string, which: (cmd: string) => string | null): boolean {
  if (cmd.includes("/")) return existsSync(cmd);
  return which(cmd) !== null;
}

/**
 * PATH plus a local auth check. Grok: `XAI_API_KEY` or `auth.json`, no subprocess, so the offline
 * sandbox cannot make a logged-in CLI look logged out. Cursor: `CURSOR_API_KEY`, or `status` when
 * this process is not inside the network sandbox. A sandboxed status probe is skipped rather than run.
 */
export function probeCli(opts: {
  vendor: string;
  cmd: string;
  env: Record<string, string | undefined>;
  home: string;
  insideSandbox?: boolean;
  which?: (cmd: string) => string | null;
  runStatus?: (argv: string[]) => { exitCode: number; stdout: string; stderr: string };
}): CliProbe {
  const which = opts.which ?? ((cmd: string) => Bun.which(cmd));
  const onPath = commandOnPath(opts.cmd, which);
  const base: CliProbe = { vendor: opts.vendor, cmd: opts.cmd, onPath, authenticated: null, detail: "" };
  if (!onPath) {
    base.authenticated = false;
    base.detail = `${opts.cmd} is not on PATH`;
    return base;
  }
  const bin = basename(opts.cmd);
  const cursorBin = bin === "cursor-agent" || bin === "agent";
  if (bin === "grok" || (opts.vendor === GROK_CLI_VENDOR && !cursorBin)) {
    const file = grokAuthFileHasCredential(opts.home, opts.env);
    const key = (opts.env.XAI_API_KEY ?? "").trim() !== "";
    if (file.state === "yes") {
      base.authenticated = true;
      base.detail = `cached login at ${file.path}`;
    } else if (key) {
      base.authenticated = true;
      base.detail = "XAI_API_KEY is set";
    } else if (file.state === "invalid") {
      base.authenticated = false;
      base.detail = `${file.path} is not valid JSON; run grok login`;
    } else {
      base.authenticated = false;
      base.detail = `no XAI_API_KEY and no credential in ${file.path}; run grok login`;
    }
    return base;
  }
  if (opts.vendor === CURSOR_CLI_VENDOR || cursorBin) {
    if ((opts.env.CURSOR_API_KEY ?? "").trim() !== "") {
      base.authenticated = true;
      base.detail = "CURSOR_API_KEY is set";
      return base;
    }
    if (opts.insideSandbox) {
      base.authenticated = null;
      base.detail = "status probe skipped under the offline sandbox; set CURSOR_API_KEY or run cursor-agent status outside it";
      return base;
    }
    const runStatus = opts.runStatus ?? defaultStatus;
    try {
      const r = runStatus([opts.cmd, "status"]);
      const text = `${r.stdout}\n${r.stderr}`;
      const loggedOut = /not authenticated|not logged in/i.test(text);
      base.authenticated = r.exitCode === 0 && !loggedOut;
      base.detail = base.authenticated ? "cursor-agent status reports logged in" : "cursor-agent status reports logged out; run cursor-agent login";
    } catch (err) {
      base.authenticated = null;
      base.detail = `status probe failed: ${err instanceof Error ? err.message : String(err)}`;
    }
    return base;
  }
  base.authenticated = null;
  base.detail = onPath ? "custom CLI; auth is not probed" : base.detail;
  return base;
}

function defaultStatus(argv: string[]): { exitCode: number; stdout: string; stderr: string } {
  const r = Bun.spawnSync({ cmd: argv, stdout: "pipe", stderr: "pipe", stdin: "ignore", timeout: 5_000 });
  return { exitCode: r.exitCode ?? 1, stdout: r.stdout.toString(), stderr: r.stderr.toString() };
}

/** Split a command string on whitespace, keeping single- and double-quoted pieces. */
export function splitCommand(input: string): string[] {
  const out: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  for (const m of input.matchAll(re)) out.push(m[1] ?? m[2] ?? m[3] ?? "");
  return out.filter((s) => s !== "");
}
