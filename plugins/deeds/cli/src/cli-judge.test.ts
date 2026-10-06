import { expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cacheKey } from "./analyze.ts";
import { loadCanon } from "./canon.ts";
import {
  buildCliInvocation,
  buildJudgePrompt,
  cliCacheModelId,
  cliPreset,
  cliTransport,
  probeCli,
  replyFromCliStdout,
  splitCommand,
  type ResolvedCli,
} from "./cli-judge.ts";
import { runAnalyze } from "./commands/analyze.ts";
import doctor from "./commands/doctor.ts";
import { DeedsError } from "./contract.ts";
import { judgeRepo, wireQuestions, type JevRequest } from "./jev/judge.ts";
import type { PolicyQuestion } from "./jev/policy.ts";
import { loadSystem, systemHash } from "./jev/system.ts";
import { parseAnalyzeArgs, resolveCliJudge, resolveJudge } from "./resolve.ts";

const SECRET = "sk-live-abcdefghijklmnopqrstuvwxyz";

const sampleReq = (): JevRequest => ({
  state: { product_diff: "added a route", files: "A app.ts [source]" },
  model: "jev-latest",
  questions: {
    capability_changed: { type: "noul", instructions: { question: "Does this change what a user can do?" } },
    capability_change_kind: {
      type: "choice",
      instructions: "Which change?",
      criteria: { new_capability: "new", no_capability_change: "none" },
    },
  },
});

function fenced(answers: unknown): string {
  return "```json\n" + JSON.stringify(answers) + "\n```";
}

const bare = { capability_changed: 0.91, capability_change_kind: "new_capability" };

test("grok json envelope: text field, fences stripped, usage kept", () => {
  const reply = replyFromCliStdout(JSON.stringify({
    text: fenced(bare),
    stopReason: "end_turn",
    sessionId: "abc",
    usage: { input_tokens: 11, output_tokens: 4 },
  }), sampleReq(), "grok-cli");
  expect(reply.model).toBe("grok-cli");
  expect(reply.usage).toEqual({ input_tokens: 11, output_tokens: 4 });
  expect(reply.answers.capability_changed).toEqual({ type: "noul", noul: 0.91 });
  const choice = reply.answers.capability_change_kind;
  expect(choice?.type).toBe("choice");
  if (choice?.type === "choice") {
    expect(choice.choice).toBe("new_capability");
    expect(choice.confidence).toBeGreaterThanOrEqual(0.6);
  }
});

test("cursor json envelope uses result, and a warning line before JSON is ignored", () => {
  const stdout = `update check skipped\n${JSON.stringify({
    type: "result",
    subtype: "success",
    is_error: false,
    result: fenced({ capability_changed: "no", capability_change_kind: "No_Capability_Change" }),
  })}`;
  const reply = replyFromCliStdout(stdout, sampleReq(), "cursor");
  expect(reply.answers.capability_changed).toEqual({ type: "noul", noul: 0.1 });
  const choice = reply.answers.capability_change_kind;
  expect(choice?.type).toBe("choice");
  if (choice?.type === "choice") expect(choice.choice).toBe("no_capability_change");
});

test("streaming text events and a direct answer object both parse", () => {
  const events = [
    JSON.stringify({ type: "text", data: "{\"capability_changed\":0.4," }),
    JSON.stringify({ type: "text", data: "\"capability_change_kind\":\"no_capability_change\"}" }),
  ].join("\n");
  const fromEvents = replyFromCliStdout(events, sampleReq(), "grok-cli");
  expect(fromEvents.answers.capability_changed).toEqual({ type: "noul", noul: 0.4 });

  const direct = replyFromCliStdout(JSON.stringify(bare), sampleReq(), "grok-cli");
  expect(direct.answers.capability_changed).toEqual({ type: "noul", noul: 0.91 });
  const wrapped = replyFromCliStdout(JSON.stringify({ text: JSON.stringify({ answers: bare }) }), sampleReq(), "grok-cli");
  expect(wrapped.answers.capability_change_kind).toMatchObject({ type: "choice", choice: "new_capability" });
});

test("a JSON object with no documented text field names its keys and fails", () => {
  expect(() => replyFromCliStdout(JSON.stringify({ sessionId: "abc", stopReason: "end_turn" }), sampleReq(), "grok-cli")).toThrow(/top-level keys: sessionId, stopReason/);
});

test("cursor is_error is a failure", () => {
  expect(() => replyFromCliStdout(JSON.stringify({ type: "result", subtype: "success", is_error: true, result: "nope" }), sampleReq(), "cursor")).toThrow(/nope/);
});

test("grok invocation is read-only, in a temp cwd, prompt via --prompt-file", () => {
  const preset = cliPreset("grok-cli");
  const inv = buildCliInvocation({ ...preset, cmd: "grok" }, buildJudgePrompt(sampleReq()));
  try {
    expect(inv.cwd).toContain("deeds-cli-");
    expect(inv.argv[0]).toBe("grok");
    expect(inv.argv).toContain("--output-format");
    expect(inv.argv).toContain("json");
    expect(inv.argv).toContain("--permission-mode");
    expect(inv.argv).toContain("dontAsk");
    expect(inv.argv).toContain("--max-turns");
    expect(inv.argv).toContain("--prompt-file");
    expect(inv.argv).toContain("--cwd");
    expect(inv.argv).toContain(inv.cwd);
    expect(inv.argv.join(" ")).not.toMatch(/yolo|always-approve|--force/);
    const file = inv.argv[inv.argv.indexOf("--prompt-file") + 1]!;
    const prompt = readFileSync(file, "utf8");
    expect(prompt).toContain("added a route");
    expect(prompt).toContain("Do not use tools");
    expect(prompt).not.toContain(SECRET);
  } finally {
    inv.cleanup();
  }
});

test("cursor invocation passes the prompt as an argument and stays in ask mode", () => {
  const preset = cliPreset("cursor");
  const inv = buildCliInvocation({ ...preset, cmd: "cursor-agent" }, "answer this");
  try {
    expect(inv.argv.slice(0, 6)).toEqual(["cursor-agent", "-p", "--output-format", "json", "--mode", "ask"]);
    expect(inv.argv.at(-1)).toBe("answer this");
    expect(inv.stdin).toBeUndefined();
    expect(inv.argv).not.toContain("--force");
  } finally {
    inv.cleanup();
  }
});

test("a stubbed subprocess answers, and a retry then a timeout both surface", async () => {
  const cfg: ResolvedCli = {
    ...resolveCliJudge({ vendor: "grok-cli" }, {}, mkdtempSync(join(tmpdir(), "deeds-home-"))),
    maxAttempts: 3,
    timeoutMs: 50,
  };
  let n = 0;
  const transport = cliTransport(cfg, {
    sleep: async () => {},
    run: async (spec) => {
      n++;
      expect(spec.cwd).toContain("deeds-cli-");
      expect(spec.argv).toContain("--prompt-file");
      if (n === 1) return { stdout: "not json", stderr: "warm up", exitCode: 1 };
      const file = spec.argv[spec.argv.indexOf("--prompt-file") + 1]!;
      expect(readFileSync(file, "utf8")).toContain("BEGIN_DEEDS_STATE");
      return { stdout: JSON.stringify({ text: fenced(bare), usage: { input_tokens: 2, output_tokens: 3 } }), stderr: "", exitCode: 0 };
    },
  });
  const reply = await transport(sampleReq());
  expect(n).toBe(2);
  expect(reply.usage?.input_tokens).toBe(2);

  const timingOut = cliTransport({ ...cfg, maxAttempts: 2 }, {
    sleep: async () => {},
    run: async () => { throw new Error("timed out after 50ms"); },
  });
  await expect(timingOut(sampleReq())).rejects.toThrow(/timed out after 50ms/);
});

test("spawned stub receives the prompt file in a temp cwd and a killed sleep times out", async () => {
  const dir = mkdtempSync(join(tmpdir(), "deeds-stub-"));
  const capture = join(dir, "capture.json");
  const script = join(dir, "stub.ts");
  writeFileSync(script, [
    'import { readFileSync, writeFileSync } from "node:fs";',
    "const args = process.argv.slice(2);",
    'const at = args.indexOf("--prompt-file");',
    'const prompt = at >= 0 ? readFileSync(args[at + 1], "utf8") : (args.at(-1) ?? "");',
    "writeFileSync(process.env.CAPTURE, JSON.stringify({ cwd: process.cwd(), args, prompt }));",
    'const questions = JSON.parse(prompt.split("BEGIN_DEEDS_QUESTIONS\\n")[1].split("\\nEND_DEEDS_QUESTIONS")[0]);',
    "const answers = {};",
    'for (const q of questions) answers[q.id] = q.type === "noul" ? 0.2 : q.type === "choice" ? Object.keys(q.criteria ?? {})[0] : 0;',
    'console.log(JSON.stringify({ text: JSON.stringify(answers), stopReason: "end_turn", usage: { input_tokens: 8, output_tokens: 1 } }));',
  ].join("\n"));
  const preset = cliPreset("grok-cli");
  const cfg: ResolvedCli = {
    mode: "cli",
    vendor: "grok-cli",
    modelId: "grok-cli",
    cacheModelId: "test",
    cmd: process.execPath,
    args: [script, ...preset.args],
    prompt: "file",
    promptFileFlag: "--prompt-file",
    cwdFlag: "--cwd",
    timeoutMs: 10_000,
    concurrency: 1,
    maxAttempts: 1,
  };
  const prev = process.env.CAPTURE;
  process.env.CAPTURE = capture;
  try {
    const reply = await cliTransport(cfg)(sampleReq());
    expect(reply.answers.capability_changed).toEqual({ type: "noul", noul: 0.2 });
    const seen = JSON.parse(readFileSync(capture, "utf8")) as { cwd: string; args: string[]; prompt: string };
    expect(seen.cwd).toContain("deeds-cli-");
    expect(seen.cwd).not.toBe(process.cwd());
    expect(seen.args).toContain("--permission-mode");
    expect(seen.args.join(" ")).not.toMatch(/yolo|always-approve/);
    expect(seen.prompt).toContain("added a route");
    expect(seen.prompt).not.toContain(SECRET);
  } finally {
    if (prev === undefined) delete process.env.CAPTURE;
    else process.env.CAPTURE = prev;
    rmSync(dir, { recursive: true, force: true });
  }

  const slow: ResolvedCli = {
    ...cfg,
    cmd: "sleep",
    args: ["30"],
    prompt: "stdin",
    cwdFlag: undefined,
    timeoutMs: 300,
    maxAttempts: 1,
  };
  await expect(cliTransport(slow)(sampleReq())).rejects.toThrow(/timed out/);
}, 15_000);

test("cache keys keep jev, grok and cursor apart", () => {
  const hash = systemHash(loadSystem());
  const canon = "canon";
  const jev = cacheKey(canon, "typesafe:jev-latest", "jev", hash);
  const grok = cacheKey(canon, cliCacheModelId("grok-cli", "grok", cliPreset("grok-cli").args), "cli", hash);
  const cursor = cacheKey(canon, cliCacheModelId("cursor", "cursor-agent", cliPreset("cursor").args), "cli", hash);
  expect(new Set([jev, grok, cursor]).size).toBe(3);
  expect(() => cacheKey(canon, "grok-cli", "cli")).toThrow(/question-system/);
});

test("resolve: grok-cli needs no API key; jev still does; config and DEEDS_CLI_CMD override", () => {
  const home = mkdtempSync(join(tmpdir(), "deeds-cfg-"));
  expect(() => resolveJudge(parseAnalyzeArgs(["."]), {}, home)).toThrow(/TYPESAFE_API_KEY/);
  const cli = resolveJudge(parseAnalyzeArgs([".", "--vendor", "grok-cli"]), {}, home);
  expect(cli.mode).toBe("cli");
  if (cli.mode !== "cli") return;
  expect(cli.vendor).toBe("grok-cli");
  expect(cli.cmd).toBe("grok");
  expect(cli.concurrency).toBe(2);
  expect(cli.prompt).toBe("file");
  expect(cli.cacheModelId.startsWith("grok-cli:grok:")).toBe(true);

  const cursor = resolveJudge(parseAnalyzeArgs(["--mode", "cli", "--vendor", "cursor"]), {}, home);
  expect(cursor.mode).toBe("cli");
  if (cursor.mode === "cli") expect(cursor.cmd).toBe("cursor-agent");

  expect(() => resolveJudge(parseAnalyzeArgs(["--mode", "full", "--vendor", "grok-cli"]), {}, home)).toThrow(/does not apply to --mode full/);
  expect(() => resolveJudge(parseAnalyzeArgs(["--mode", "jev", "--vendor", "cursor"]), { TYPESAFE_API_KEY: "k" }, home)).toThrow(/does not apply to --mode jev/);
  expect(() => resolveJudge(parseAnalyzeArgs(["--vendor", "nope"]), {}, home)).toThrow(/--vendor is anthropic or openai/);

  const customHome = mkdtempSync(join(tmpdir(), "deeds-cfg-"));
  mkdirSync(join(customHome, ".config", "deeds"), { recursive: true });
  writeFileSync(join(customHome, ".config", "deeds", "config.json"), JSON.stringify({
    cli: { concurrency: 4, vendors: { "grok-cli": { cmd: "/opt/grok" } } },
  }));
  const fromFile = resolveCliJudge({ vendor: "grok" }, {}, customHome);
  expect(fromFile.cmd).toBe("/opt/grok");
  expect(fromFile.concurrency).toBe(4);
  expect(fromFile.args).toContain("--output-format");

  const fromEnv = resolveCliJudge({ vendor: "cursor" }, { DEEDS_CLI_CMD: "my-agent --output-format json {prompt}", DEEDS_CLI_CONCURRENCY: "1" }, customHome);
  expect(fromEnv.cmd).toBe("my-agent");
  expect(fromEnv.args).toEqual(["--output-format", "json", "{prompt}"]);
  expect(fromEnv.prompt).toBe("arg");
  expect(fromEnv.concurrency).toBe(1);
  expect(fromEnv.cacheModelId).not.toBe(fromFile.cacheModelId);

  expect(splitCommand(`"/opt/My Grok/grok" --prompt-file {prompt-file}`)).toEqual(["/opt/My Grok/grok", "--prompt-file", "{prompt-file}"]);
});

test("doctor reports the configured CLI without calling it", async () => {
  const home = mkdtempSync(join(tmpdir(), "deeds-doc-"));
  const prev = { HOME: process.env.HOME, XAI_API_KEY: process.env.XAI_API_KEY, DEEDS_CLI_CMD: process.env.DEEDS_CLI_CMD, DEEDS_CLI_PROMPT: process.env.DEEDS_CLI_PROMPT };
  process.env.HOME = home;
  delete process.env.XAI_API_KEY;
  delete process.env.DEEDS_CLI_CMD;
  delete process.env.DEEDS_CLI_PROMPT;
  try {
    const result = await doctor.run({ args: [], json: false, cwd: home, commands: [], offlineEnforcement: "in-process" });
    const data = result.data as { cli: { vendor: string; onPath: boolean; authenticated: boolean | null; detail: string } };
    expect(data.cli.vendor).toBe("grok-cli");
    expect(data.cli.onPath).toBe(false);
    expect(result.text).toContain("cli judge:");
    expect(result.text).not.toMatch(/sk-|xai-|Bearer /);
  } finally {
    for (const [k, v] of Object.entries(prev)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
});

test("probe: grok auth is a local file or XAI_API_KEY, cursor status is skipped in the sandbox", () => {
  const home = mkdtempSync(join(tmpdir(), "deeds-auth-"));
  const missing = probeCli({ vendor: "grok-cli", cmd: "grok", env: {}, home, which: () => null });
  expect(missing.onPath).toBe(false);
  expect(missing.authenticated).toBe(false);

  mkdirSync(join(home, ".grok"), { recursive: true });
  writeFileSync(join(home, ".grok", "auth.json"), JSON.stringify({ "https://auth.x.ai::scope": { key: "abcdefghijklmnop" } }));
  const loggedIn = probeCli({ vendor: "grok-cli", cmd: "grok", env: {}, home, which: () => "/usr/bin/grok" });
  expect(loggedIn.onPath).toBe(true);
  expect(loggedIn.authenticated).toBe(true);
  expect(loggedIn.detail).toContain("auth.json");
  expect(loggedIn.detail).not.toContain("abcdefghijklmnop");

  const byKey = probeCli({ vendor: "grok-cli", cmd: "grok", env: { XAI_API_KEY: "xai-test" }, home: mkdtempSync(join(tmpdir(), "deeds-auth-")), which: () => "/bin/grok" });
  expect(byKey.authenticated).toBe(true);

  const sandboxed = probeCli({ vendor: "cursor", cmd: "cursor-agent", env: {}, home, insideSandbox: true, which: () => "/bin/cursor-agent" });
  expect(sandboxed.authenticated).toBeNull();
  expect(sandboxed.detail).toContain("sandbox");

  const status = probeCli({
    vendor: "cursor",
    cmd: "cursor-agent",
    env: {},
    home,
    which: () => "/bin/cursor-agent",
    runStatus: () => ({ exitCode: 0, stdout: "Logged in as a@b.c", stderr: "" }),
  });
  expect(status.authenticated).toBe(true);
});

function gitRepo(): string {
  const dir = mkdtempSync(join(tmpdir(), "deeds-repo-"));
  const git = (...args: string[]) => {
    const r = Bun.spawnSync(["git", "-C", dir, ...args], { stdout: "pipe", stderr: "pipe" });
    if (r.exitCode !== 0) throw new Error(r.stderr.toString());
  };
  git("init", "-b", "main");
  git("config", "user.email", "t@example.com");
  git("config", "user.name", "Test");
  writeFileSync(join(dir, "README.md"), "fixture\n");
  git("add", "README.md");
  git("commit", "-m", "init");
  // A root commit is settled without a judge call. The change under test needs one parent.
  writeFileSync(join(dir, "greet.ts"), `export function greet(name: string): string {\n  const token = "${SECRET}";\n  return name + token;\n}\n`);
  git("add", "greet.ts");
  git("commit", "-m", "add greet");
  return dir;
}

function canned(): string {
  const system = loadSystem() as unknown as { stages: { id: string; questions: PolicyQuestion[] }[] };
  const stage = system.stages.find((s) => s.id === "judge");
  if (!stage) throw new Error("no judge stage");
  const questions = wireQuestions(stage.questions);
  const answers: Record<string, unknown> = {};
  for (const [id, q] of Object.entries(questions)) {
    if (q.type === "noul") answers[id] = 0.05;
    else if (q.type === "choice") answers[id] = Object.keys((q.criteria ?? {}) as Record<string, unknown>)[0] ?? "none";
    else answers[id] = 0;
  }
  return fenced(answers);
}

test("judgeRepo redacts before the CLI sees the diff, and a second vendor does not reuse the cache", async () => {
  const repo = gitRepo();
  const cacheRoot = mkdtempSync(join(tmpdir(), "deeds-cache-"));
  const canon = loadCanon();
  expect(canon).toBeTruthy();
  const text = canned();
  const prompts: string[] = [];
  const run = async (spec: { argv: string[] }) => {
    const file = spec.argv[spec.argv.indexOf("--prompt-file") + 1]!;
    prompts.push(readFileSync(file, "utf8"));
    return { stdout: JSON.stringify({ text, usage: { input_tokens: 1, output_tokens: 1 } }), stderr: "", exitCode: 0 };
  };
  const base = resolveCliJudge({ vendor: "grok-cli" }, {}, mkdtempSync(join(tmpdir(), "deeds-home-")));
  const opts = {
    repo,
    repoName: "fixture",
    label: repo,
    since: "1970-01-01",
    canon: canon!,
    settings: {},
    concurrency: 1,
    cacheRoot,
    reportMode: "cli" as const,
    reportModel: "grok-cli",
  };
  const first = await judgeRepo({ ...opts, transport: cliTransport(base, { run, sleep: async () => {} }), modelId: base.cacheModelId });
  expect(first.complete).toBe(true);
  expect(first.mode).toBe("cli");
  expect(first.model).toBe("grok-cli");
  expect(first.redactions).toBeGreaterThan(0);
  expect(prompts).toHaveLength(1);
  expect(prompts[0]).not.toContain(SECRET);
  expect(prompts[0]).toContain("[REDACTED]");

  const again = await judgeRepo({ ...opts, transport: cliTransport(base, { run, sleep: async () => {} }), modelId: base.cacheModelId });
  expect(again.cached).toBe(again.commits);
  expect(prompts).toHaveLength(1);

  const cursor = resolveCliJudge({ vendor: "cursor" }, {}, opts.repo);
  await judgeRepo({ ...opts, transport: cliTransport({ ...cursor, prompt: "file", promptFileFlag: "--prompt-file" }, { run, sleep: async () => {} }), modelId: cursor.cacheModelId, reportModel: "cursor" });
  expect(prompts).toHaveLength(2);
});

test("a CLI failure is the incomplete error and the commit is not cached", async () => {
  const repo = gitRepo();
  const home = mkdtempSync(join(tmpdir(), "deeds-home-"));
  let thrown: unknown;
  try {
    await runAnalyze({ args: [repo, "--since", "all", "--vendor", "grok-cli"], json: true, cwd: repo }, {
      env: { HOME: home, DEEDS_CLI_CMD: "deeds-cli-judge-missing" },
      makeModel: () => { throw new Error("full mode"); },
      makeJev: () => { throw new Error("jev"); },
      makeCli: (r) => cliTransport({ ...r, maxAttempts: 1, timeoutMs: 2_000 }),
    });
  } catch (err) {
    thrown = err;
  }
  expect(thrown).toBeInstanceOf(DeedsError);
  expect((thrown as DeedsError).code).toBe("incomplete");
  expect((thrown as Error).message).toContain("deeds-cli-judge-missing");
  expect((thrown as Error).message).not.toContain(SECRET);
  // The failed commit is not cached, so a rerun tries it again.
  let second: unknown;
  try {
    await runAnalyze({ args: [repo, "--since", "all", "--vendor", "grok-cli"], json: true, cwd: repo }, {
      env: { HOME: home, DEEDS_CLI_CMD: "deeds-cli-judge-missing" },
      makeModel: () => { throw new Error("full mode"); },
      makeJev: () => { throw new Error("jev"); },
      makeCli: (r) => cliTransport({ ...r, maxAttempts: 1, timeoutMs: 2_000 }),
    });
  } catch (err) {
    second = err;
  }
  expect((second as DeedsError).code).toBe("incomplete");
});
