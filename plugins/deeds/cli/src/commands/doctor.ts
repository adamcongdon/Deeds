import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { probeCli } from "../cli-judge.ts";
import type { Command } from "../contract.ts";
import { resolveCliJudge } from "../resolve.ts";

/**
 * Not pure: it looks at the filesystem. It therefore runs under the OS network denial like every other
 * network-none command, and on a host with no usable sandbox it refuses unless `--allow-unsandboxed` is passed.
 * The CLI probe is local (PATH, auth.json, an env key). Cursor's `status` subprocess is skipped inside the
 * sandbox so a network denial cannot be reported as logged out.
 */
const doctor: Command = {
  name: "doctor",
  summary: "Report the offline sandbox and whether the local CLI judge is on PATH and signed in.",
  run(ctx) {
    const sandboxExec = existsSync("/usr/bin/sandbox-exec");
    const home = process.env.HOME ?? homedir();
    const cli = resolveCliJudge({ vendor: undefined, model: undefined }, process.env, home);
    const probe = probeCli({
      vendor: cli.vendor,
      cmd: cli.cmd,
      env: process.env,
      home,
      insideSandbox: ctx.offlineEnforcement === "os-sandbox",
    });
    const auth = probe.authenticated === null ? "unknown" : probe.authenticated ? "yes" : "no";
    const data = {
      platform: process.platform,
      sandboxExecPresent: sandboxExec,
      offlineEnforcement: ctx.offlineEnforcement,
      // Every non-pure network-none command needs the OS sandbox or the explicit override.
      sandboxRequired: true,
      cli: {
        vendor: probe.vendor,
        cmd: probe.cmd,
        onPath: probe.onPath,
        authenticated: probe.authenticated,
        detail: probe.detail,
      },
    };
    return {
      data,
      text: [
        `platform:            ${data.platform}`,
        `sandbox-exec:        ${sandboxExec ? "present" : "absent"}`,
        `offline enforcement: ${data.offlineEnforcement}`,
        `cli judge:           ${probe.vendor}`,
        `cli command:         ${probe.cmd} (${probe.onPath ? "on PATH" : "not on PATH"})`,
        `cli auth:            ${auth} (${probe.detail})`,
      ].join("\n"),
    };
  },
};

export default doctor;
