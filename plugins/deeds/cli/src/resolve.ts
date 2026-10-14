/**
 * Input resolution for analyze: target, mode, vendor, key, model and window, each decided
 * here and nowhere else, so every command that analyzes resolves the same way.
 * The default judge is Jev on the user's own Jev key (TYPESAFE_API_KEY); `--mode fast` is an alias of it.
 * `--mode full` is the full-model reference read on Anthropic or OpenAI. `--mode cli` (or `--vendor grok-cli`
 * / `--vendor cursor`) judges with a local CLI and no API key. See cli-judge.ts.
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";
import {
  CLI_DEFAULT_CONCURRENCY,
  CLI_DEFAULT_TIMEOUT_MS,
  CLI_MAX_ATTEMPTS,
  GROK_CLI_VENDOR,
  cliCacheModelId,
  cliModelArgs,
  cliPreset,
  normalizeCliVendor,
  splitCommand,
  type CliPreset,
  type CliPrompt,
  type ResolvedCli,
} from "./cli-judge.ts";
export type { ResolvedCli };
import { DeedsError, EXIT } from "./contract.ts";
import { parseGithub } from "./git.ts";
import type { Vendor } from "./model.ts";
import { REPO_KINDS, type FactsOptions, type RepoKind } from "./jev/facts.ts";

/** jev: the judgment API (the default); full: one full-model read per commit; cli: a local headless CLI. */
export type Mode = "jev" | "full" | "cli";
export const MODES: readonly Mode[] = ["jev", "full", "cli"];
export const DEFAULT_MODE: Mode = "jev";
/** The Jev key variable, in the environment or behind the config pointer `keys.typesafe`. */
export const JEV_ENV_VAR = "TYPESAFE_API_KEY";

export const KEY_ENV: Record<Vendor, string> = { anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY" };
export const DEFAULT_MODEL: Record<Vendor, string> = { anthropic: "claude-sonnet-5-5", openai: "gpt-5.6-terra" };

export interface AnalyzeFlags {
  target: string;
  since: string;
  until?: string;
  mode: Mode;
  /** True when argv contained `--mode`. The default jev is not explicit, so `--vendor grok-cli` alone selects the CLI judge. */
  modeExplicit: boolean;
  vendor?: string;
  model?: string;
  /** Also write a self-contained HTML report to this path. */
  html?: string;
  /** Force colour on or off; unset means colour only on a terminal without NO_COLOR. */
  color?: boolean;
}

/** Parse analyze's argv. Pure. */
const KNOWN_CLI_VENDORS = new Set(["grok", "grok-cli", "cursor", "cursor-agent", "agent"]);

export function parseAnalyzeArgs(args: string[]): AnalyzeFlags {
  const flags: AnalyzeFlags = { target: ".", since: "90d", mode: DEFAULT_MODE, modeExplicit: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    const val = () => {
      const v = args[++i];
      if (!v) throw new DeedsError("usage", `${a} needs a value`, EXIT.usage);
      return v;
    };
    if (a === "--since") flags.since = val();
    else if (a === "--until") flags.until = val();
    else if (a === "--mode") {
      const v = val();
      if (v !== "jev" && v !== "fast" && v !== "full" && v !== "cli") throw new DeedsError("usage", "--mode is jev, full or cli (fast is an alias of jev)", EXIT.usage);
      flags.mode = v === "fast" || v === "jev" ? "jev" : v;
      flags.modeExplicit = true;
    } else if (a === "--vendor") {
      const v = val();
      if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(v)) throw new DeedsError("usage", "--vendor is anthropic, openai, grok-cli or cursor", EXIT.usage);
      flags.vendor = v;
    } else if (a === "--model") flags.model = val();
    else if (a === "--html") flags.html = val();
    else if (a === "--color") flags.color = true;
    else if (a === "--no-color") flags.color = false;
    else if (a.startsWith("-")) throw new DeedsError("usage", `unknown option ${a}`, EXIT.usage);
    else flags.target = a;
  }
  return flags;
}

/** "90d", "2w", "6m", "1y", "all", or anything git's --since understands. */
export function toSince(window: string): string {
  if (window === "all") return "1970-01-01";
  const m = /^(\d+)([dwmy])$/.exec(window);
  if (!m) return window;
  const unit = { d: "days", w: "weeks", m: "months", y: "years" }[m[2] as "d" | "w" | "m" | "y"];
  return `${m[1]} ${unit} ago`;
}

export type Target = { kind: "github"; owner: string; repo: string; url: string; label: string } | { kind: "local"; path: string; label: string };

/** A GitHub URL or a local folder. Existence and repo checks for a folder happen here. */
export function resolveTarget(target: string, cwd: string, isRepo: (p: string) => boolean): Target {
  const gh = parseGithub(target);
  if (gh) return { kind: "github", ...gh, label: `github.com/${gh.owner}/${gh.repo}` };
  const path = resolvePath(cwd, target);
  if (!existsSync(path) || !statSync(path).isDirectory()) throw new DeedsError("not_found", `no such folder: ${target}`, EXIT.error);
  if (!isRepo(path)) throw new DeedsError("not_a_repo", `${target} is not a git repository`, EXIT.error);
  return { kind: "local", path, label: path };
}

/** Read one variable out of a dotenv-style file. Returns undefined when the file or variable is absent. */
export function readEnvFile(path: string, name: string): string | undefined {
  if (!existsSync(path)) return undefined;
  for (const raw of readFileSync(path, "utf8").split("\n")) {
    const m = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/.exec(raw);
    if (!m || m[1] !== name) continue;
    const v = m[2]!.replace(/^(['"])(.*)\1$/, "$2");
    return v === "" ? undefined : v;
  }
  return undefined;
}

/**
 * `~/.config/deeds/config.json` may point a vendor's key at a variable inside an env file:
 *   { "keys": { "openai": { "envFile": "~/.secrets/.env", "var": "OPENAI_API_KEY" } } }
 * The process environment wins over the config file.
 */
function readConfig(home: string): { path: string; doc: Record<string, unknown> } | undefined {
  const path = join(home, ".config", "deeds", "config.json");
  if (!existsSync(path)) return undefined;
  let doc: unknown;
  try {
    doc = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw new DeedsError("bad_config", `${path} is not valid JSON`, EXIT.usage);
  }
  if (doc === null || typeof doc !== "object" || Array.isArray(doc)) throw new DeedsError("bad_config", `${path} must hold a JSON object`, EXIT.usage);
  return { path, doc: doc as Record<string, unknown> };
}

export type KeyOwner = Vendor | "typesafe";
const ENV_OF: Record<KeyOwner, string> = { ...KEY_ENV, typesafe: JEV_ENV_VAR };

export function configKey(vendor: KeyOwner, home: string): string | undefined {
  const cfg = readConfig(home);
  if (!cfg) return undefined;
  const { path, doc } = cfg;
  const entry = (doc as { keys?: Record<string, unknown> }).keys?.[vendor] as { envFile?: unknown; var?: unknown } | undefined;
  if (entry === undefined) return undefined;
  if (typeof entry.envFile !== "string" || entry.envFile === "") throw new DeedsError("bad_config", `${path}: keys.${vendor}.envFile must be a path`, EXIT.usage);
  const file = entry.envFile.replace(/^~(?=\/|$)/, home);
  const name = typeof entry.var === "string" && entry.var !== "" ? entry.var : ENV_OF[vendor];
  return readEnvFile(file, name);
}

export interface ResolvedModel {
  mode: "full";
  vendor: Vendor;
  apiKey: string;
  model: string;
  /** `<vendor>:<model>`, what the report names. */
  modelId: string;
}

export interface ResolvedJev {
  mode: "jev";
  apiKey: string;
}

export type ResolvedJudge = ResolvedJev | ResolvedModel | ResolvedCli;

function isKnownCliVendor(vendor: string): boolean {
  return KNOWN_CLI_VENDORS.has(vendor);
}

/** Decide the judge for a run: Jev on TYPESAFE_API_KEY by default, a full-model read under --mode full, a local CLI under --mode cli or --vendor grok-cli. */
export function resolveJudge(flags: Pick<AnalyzeFlags, "mode" | "modeExplicit" | "vendor" | "model">, env: Record<string, string | undefined>, home: string = env.HOME ?? homedir()): ResolvedJudge {
  const cliNamed = flags.vendor !== undefined && isKnownCliVendor(flags.vendor);
  if (cliNamed && flags.modeExplicit && flags.mode !== "cli") {
    throw new DeedsError("usage", `--vendor ${flags.vendor} is a local CLI judge; it does not apply to --mode ${flags.mode}`, EXIT.usage);
  }
  if (flags.mode === "cli" || cliNamed) {
    if (flags.vendor === "anthropic" || flags.vendor === "openai") {
      throw new DeedsError("usage", `--vendor ${flags.vendor} applies to --mode full, not --mode cli`, EXIT.usage);
    }
    return resolveCliJudge(flags, env, home);
  }
  if (flags.mode === "full") return resolveModel(flags, env, home);
  if (flags.vendor) throw new DeedsError("usage", "--vendor is anthropic or openai with --mode full, or grok-cli or cursor for the local CLI judge", EXIT.usage);
  if (flags.model) throw new DeedsError("usage", "--model applies to --mode full and --mode cli; the default judge is Jev", EXIT.usage);
  const apiKey = env[JEV_ENV_VAR] || configKey("typesafe", home);
  if (!apiKey) {
    throw new DeedsError(
      "missing_key",
      `set ${JEV_ENV_VAR} (your Jev key from typesafe.ai), or point keys.typesafe in ~/.config/deeds/config.json at it; deeds sends each commit's judged state to Jev with your own key (or run --mode full with ANTHROPIC_API_KEY or OPENAI_API_KEY, or --vendor grok-cli to judge with a local CLI and no key)`,
      EXIT.usage,
    );
  }
  return { mode: "jev", apiKey };
}

/** Decide vendor, key and model for the full-model reference read (--mode full). */
export function resolveModel(flags: Pick<AnalyzeFlags, "vendor" | "model">, env: Record<string, string | undefined>, home: string = env.HOME ?? homedir()): ResolvedModel {
  const keyFor = (v: Vendor) => env[KEY_ENV[v]] || configKey(v, home);
  if (flags.vendor !== undefined && flags.vendor !== "anthropic" && flags.vendor !== "openai") {
    throw new DeedsError("usage", "--mode full --vendor is anthropic or openai", EXIT.usage);
  }
  const vendor = (flags.vendor as Vendor | undefined) ?? (keyFor("anthropic") ? "anthropic" : keyFor("openai") ? "openai" : undefined);
  if (!vendor) throw new DeedsError("missing_key", "--mode full needs ANTHROPIC_API_KEY or OPENAI_API_KEY; deeds sends each commit's diff to that model with your own key", EXIT.usage);
  const apiKey = keyFor(vendor);
  if (!apiKey) throw new DeedsError("missing_key", `--vendor ${vendor} needs ${KEY_ENV[vendor]}; deeds sends each commit's diff to that model with your own key`, EXIT.usage);
  const model = flags.model ?? env.DEEDS_MODEL ?? DEFAULT_MODEL[vendor];
  return { mode: "full", vendor, apiKey, model, modelId: `${vendor}:${model}` };
}

interface CliOverlay {
  cmd?: string;
  args?: string[];
  prompt?: CliPrompt;
  promptFileFlag?: string;
  /** null clears the preset's cwd flag. */
  cwdFlag?: string | null;
  timeoutMs?: number;
  concurrency?: number;
}

function positiveInt(label: string, value: unknown): number | undefined {
  if (value === undefined) return undefined;
  const n = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  if (!Number.isInteger(n) || n < 1) throw new DeedsError("bad_config", `${label} must be a positive integer`, EXIT.usage);
  return n;
}

function cliOverlay(value: unknown, path: string, label: string): CliOverlay {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new DeedsError("bad_config", `${path}: ${label} must be an object`, EXIT.usage);
  const o = value as Record<string, unknown>;
  const out: CliOverlay = {};
  if (o.cmd !== undefined) {
    if (typeof o.cmd !== "string" || o.cmd === "") throw new DeedsError("bad_config", `${path}: ${label}.cmd must be a command`, EXIT.usage);
    out.cmd = o.cmd;
  }
  if (o.args !== undefined) {
    if (!Array.isArray(o.args) || !o.args.every((a) => typeof a === "string")) throw new DeedsError("bad_config", `${path}: ${label}.args must be a list of strings`, EXIT.usage);
    out.args = o.args as string[];
  }
  if (o.prompt !== undefined) {
    if (o.prompt !== "arg" && o.prompt !== "stdin" && o.prompt !== "file") throw new DeedsError("bad_config", `${path}: ${label}.prompt must be arg, stdin or file`, EXIT.usage);
    out.prompt = o.prompt;
  }
  if (o.promptFileFlag !== undefined) {
    if (typeof o.promptFileFlag !== "string" || o.promptFileFlag === "") throw new DeedsError("bad_config", `${path}: ${label}.promptFileFlag must be a flag`, EXIT.usage);
    out.promptFileFlag = o.promptFileFlag;
  }
  if (o.cwdFlag === null) out.cwdFlag = null;
  else if (o.cwdFlag !== undefined) {
    if (typeof o.cwdFlag !== "string" || o.cwdFlag === "") throw new DeedsError("bad_config", `${path}: ${label}.cwdFlag must be a flag`, EXIT.usage);
    out.cwdFlag = o.cwdFlag;
  }
  if (o.timeoutMs !== undefined) out.timeoutMs = positiveInt(`${path}: ${label}.timeoutMs`, o.timeoutMs);
  if (o.concurrency !== undefined) out.concurrency = positiveInt(`${path}: ${label}.concurrency`, o.concurrency);
  return out;
}

function applyOverlay(preset: CliPreset, overlay: CliOverlay): CliPreset {
  return {
    cmd: overlay.cmd ?? preset.cmd,
    args: overlay.args ? [...overlay.args] : preset.args,
    prompt: overlay.prompt ?? preset.prompt,
    promptFileFlag: overlay.promptFileFlag ?? preset.promptFileFlag,
    cwdFlag: overlay.cwdFlag === null ? undefined : (overlay.cwdFlag ?? preset.cwdFlag),
  };
}

/** `DEEDS_CLI_CMD`: a binary, or a full argv (quotes kept) that replaces the preset args. `{prompt}` and `{prompt-file}` choose how the prompt is delivered. */
function applyCommandString(preset: CliPreset, input: string): CliPreset {
  const placeholder = input.includes("{prompt}") || input.includes("{prompt-file}");
  if (!placeholder && !/\s/.test(input)) return { ...preset, cmd: input };
  const parts = splitCommand(input);
  const cmd = parts[0];
  if (!cmd) throw new DeedsError("bad_config", "DEEDS_CLI_CMD is empty", EXIT.usage);
  const args = parts.slice(1);
  const prompt: CliPrompt = args.includes("{prompt}") ? "arg" : args.includes("{prompt-file}") ? "file" : preset.prompt;
  return { ...preset, cmd, args, prompt };
}

/**
 * The local CLI judge. No API key. `--vendor` defaults to grok-cli. Preset flags come from cli-judge.ts;
 * `cli` in ~/.config/deeds/config.json and DEEDS_CLI_CMD / DEEDS_CLI_PROMPT / DEEDS_CLI_TIMEOUT_MS /
 * DEEDS_CLI_CONCURRENCY override them. A per-vendor block wins over the shared `cli` fields, and the
 * environment wins over the file.
 */
export function resolveCliJudge(flags: Pick<AnalyzeFlags, "vendor" | "model">, env: Record<string, string | undefined>, home: string = env.HOME ?? homedir()): ResolvedCli {
  const requested = flags.vendor ?? GROK_CLI_VENDOR;
  const vendor = normalizeCliVendor(requested);
  let preset = cliPreset(vendor);
  let timeoutMs: number | undefined;
  let concurrency: number | undefined;
  const cfg = readConfig(home);
  if (cfg?.doc.cli !== undefined) {
    const { path, doc } = cfg;
    const block = cliOverlay(doc.cli, path, "cli");
    preset = applyOverlay(preset, block);
    timeoutMs = block.timeoutMs ?? timeoutMs;
    concurrency = block.concurrency ?? concurrency;
    const vendors = (doc.cli as { vendors?: unknown }).vendors;
    if (vendors !== undefined) {
      if (vendors === null || typeof vendors !== "object" || Array.isArray(vendors)) throw new DeedsError("bad_config", `${path}: cli.vendors must be an object`, EXIT.usage);
      const table = vendors as Record<string, unknown>;
      const key = Object.keys(table).find((k) => normalizeCliVendor(k) === vendor);
      if (key !== undefined) {
        const specific = cliOverlay(table[key], path, `cli.vendors.${key}`);
        preset = applyOverlay(preset, specific);
        timeoutMs = specific.timeoutMs ?? timeoutMs;
        concurrency = specific.concurrency ?? concurrency;
      }
    }
  }
  const fromEnv = env.DEEDS_CLI_CMD;
  if (fromEnv !== undefined && fromEnv.trim() !== "") preset = applyCommandString(preset, fromEnv.trim());
  const promptEnv = env.DEEDS_CLI_PROMPT;
  if (promptEnv !== undefined && promptEnv !== "") {
    if (promptEnv !== "arg" && promptEnv !== "stdin" && promptEnv !== "file") {
      throw new DeedsError("bad_config", "DEEDS_CLI_PROMPT must be arg, stdin or file", EXIT.usage);
    }
    preset = { ...preset, prompt: promptEnv };
  }
  const model = flags.model;
  const args = [...preset.args, ...cliModelArgs(vendor, model)];
  const envTimeout = env.DEEDS_CLI_TIMEOUT_MS !== undefined && env.DEEDS_CLI_TIMEOUT_MS !== "" ? positiveInt("DEEDS_CLI_TIMEOUT_MS", env.DEEDS_CLI_TIMEOUT_MS) : undefined;
  const envConc = env.DEEDS_CLI_CONCURRENCY !== undefined && env.DEEDS_CLI_CONCURRENCY !== "" ? positiveInt("DEEDS_CLI_CONCURRENCY", env.DEEDS_CLI_CONCURRENCY) : undefined;
  return {
    mode: "cli",
    vendor,
    modelId: model ? `${vendor}:${model}` : vendor,
    cacheModelId: cliCacheModelId(vendor, preset.cmd, args),
    cmd: preset.cmd,
    args,
    prompt: preset.prompt,
    promptFileFlag: preset.promptFileFlag,
    ...(preset.cwdFlag ? { cwdFlag: preset.cwdFlag } : {}),
    timeoutMs: envTimeout ?? timeoutMs ?? CLI_DEFAULT_TIMEOUT_MS,
    concurrency: envConc ?? concurrency ?? CLI_DEFAULT_CONCURRENCY,
    maxAttempts: CLI_MAX_ATTEMPTS,
  };
}

/**
 * Repository settings for the judgment facts, from the same config file:
 *   { "excludeRepos": ["owner/name", "prefix*"], "repoKinds": { "name": "code" }, "repoProducts": { "name": "..." } }
 */
export function repoSettings(env: Record<string, string | undefined>, home: string = env.HOME ?? homedir()): FactsOptions {
  const cfg = readConfig(home);
  if (!cfg) return {};
  const { path, doc } = cfg;
  const out: { excludeRepos?: string[]; repoKinds?: Record<string, RepoKind>; repoProducts?: Record<string, string> } = {};
  if (doc.excludeRepos !== undefined) {
    if (!Array.isArray(doc.excludeRepos) || !doc.excludeRepos.every((x) => typeof x === "string")) {
      throw new DeedsError("bad_config", `${path}: excludeRepos must be a list of repo names`, EXIT.usage);
    }
    out.excludeRepos = doc.excludeRepos as string[];
  }
  for (const key of ["repoKinds", "repoProducts"] as const) {
    const v = doc[key];
    if (v === undefined) continue;
    if (v === null || typeof v !== "object" || Array.isArray(v)) throw new DeedsError("bad_config", `${path}: ${key} must map repo names to values`, EXIT.usage);
    for (const [name, val] of Object.entries(v)) {
      if (typeof val !== "string") throw new DeedsError("bad_config", `${path}: ${key}.${name} must be a string`, EXIT.usage);
      if (key === "repoKinds" && !(REPO_KINDS as readonly string[]).includes(val)) {
        throw new DeedsError("bad_config", `${path}: repoKinds.${name} must be one of ${REPO_KINDS.join(", ")}`, EXIT.usage);
      }
    }
    if (key === "repoKinds") out.repoKinds = v as Record<string, RepoKind>;
    else out.repoProducts = v as Record<string, string>;
  }
  return out;
}
