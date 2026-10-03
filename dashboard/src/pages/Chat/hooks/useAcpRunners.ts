import { useEffect, useState } from "react";
import { acpApi } from "../../../api/modules/acp";
import { acpRunnerOptions, type AcpRunnerOption } from "../utils/acpMention";

/** Enabled ACP runner options for the current agent's @ mention picker. */
export function useAcpRunners(
  agentId: string | null | undefined,
): AcpRunnerOption[] {
  const [runners, setRunners] = useState<AcpRunnerOption[]>([]);

  useEffect(() => {
    if (!agentId) {
      setRunners([]);
      return;
    }
    let cancelled = false;
    acpApi
      .getConfig(agentId)
      .then((config) => {
        if (!cancelled) setRunners(acpRunnerOptions(config));
      })
      .catch(() => {
        if (!cancelled) setRunners([]);
      });
    return () => {
      cancelled = true;
    };
  }, [agentId]);

  return runners;
}
