import { describe, expect, it } from "vitest";
import { acpRunnerOptions } from "./acpMention";
import { buildMentionItems } from "../components/MentionPickerMenu";
import type { ACPConfig } from "../../../api/types/acp";

const config: ACPConfig = {
  tool_enabled: true,
  runners: {
    opencode: {
      enabled: true,
      command: "opencode",
      args: ["acp"],
      env: {},
      trusted: true,
      tool_parse_mode: "update_detail",
      stdio_buffer_limit_bytes: 52428800,
    },
    codex: {
      enabled: false,
      command: "npx",
      args: ["-y", "@zed-industries/codex-acp"],
      env: {},
      trusted: false,
      tool_parse_mode: "update_detail",
      stdio_buffer_limit_bytes: 52428800,
    },
  },
};

describe("acpRunnerOptions", () => {
  it("lists enabled runners only", () => {
    expect(acpRunnerOptions(config)).toEqual([{ name: "opencode", label: "opencode" }]);
  });

  it("returns empty when the agent tool toggle is off", () => {
    expect(acpRunnerOptions({ ...config, tool_enabled: false })).toEqual([]);
  });

  it("returns empty for missing config", () => {
    expect(acpRunnerOptions(null)).toEqual([]);
    expect(acpRunnerOptions(undefined)).toEqual([]);
  });
});

describe("buildMentionItems with ACP runners", () => {
  it("filters runners by query", () => {
    const items = buildMentionItems("open", [], [], [], [], {
      acpRunners: [
        { name: "opencode", label: "opencode" },
        { name: "codex", label: "codex" },
      ],
    });
    expect(items).toEqual([{ kind: "acp_runner", name: "opencode", label: "opencode" }]);
  });

  it("orders runners after subagents and before files", () => {
    const items = buildMentionItems("e", [], [], [{ slug: "researcher", name: "" }], [
      { path: "a.txt", label: "a.txt" },
    ], {
      acpRunners: [{ name: "opencode", label: "opencode" }],
    });
    expect(items.map((item) => item.kind)).toEqual([
      "subagent",
      "acp_runner",
      "file",
    ]);
  });
});
