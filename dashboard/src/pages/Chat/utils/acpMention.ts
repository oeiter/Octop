import type { ACPConfig } from "../../../api/types/acp";

export interface AcpRunnerOption {
  name: string;
  label: string;
}

/**
 * Enabled runner options for the @ mention picker.
 * Empty when the agent's ``acp_runner`` tool toggle is off — the backend
 * passes ``@<name>`` through as plain text in that case, so the picker
 * hides the section too.
 */
export function acpRunnerOptions(
  config: ACPConfig | null | undefined,
): AcpRunnerOption[] {
  if (!config?.tool_enabled) return [];
  return Object.entries(config.runners ?? {})
    .filter(([, runner]) => runner?.enabled)
    .map(([name]) => ({ name, label: name }));
}
