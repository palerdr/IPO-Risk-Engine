import evidenceJson from "../../data/anthropic.v1.json";
import {
  evaluateScenario,
  MODEL_VERSION,
  solveRequiredGrowth,
  validateInputs,
  type Outcome,
  type ValuationInputs,
} from "./engine";

export interface EvidenceRecord {
  id: string;
  metric: string;
  value: number | null;
  unit: string | null;
  relation: "equal" | "greater_than" | null;
  status: "company_disclosure" | "unavailable";
  publishedAt: string | null;
  period: string | null;
  sourceUrl: string | null;
  sourceTitle: string | null;
  excerpt: string | null;
  note: string;
}
export interface EvidenceDataset {
  version: string;
  asOf: string;
  records: EvidenceRecord[];
}
export const EVIDENCE = evidenceJson as EvidenceDataset;
export const STARTING_REVENUE_REFERENCE = "may-revenue-run-rate";

export function exportScenario(raw: unknown): Outcome<{
  schemaVersion: string;
  modelVersion: string;
  evidence: EvidenceDataset;
  inputs: ValuationInputs;
  inputBasis: string;
  startingRevenueReference: string;
  valuation: ReturnType<typeof evaluateScenario>;
  reverseValuation: ReturnType<typeof solveRequiredGrowth>;
}> {
  const checked = validateInputs(raw);
  if (!checked.ok) return checked;
  const valuation = evaluateScenario(checked.value);
  if (!valuation.ok) return valuation;
  return {
    ok: true,
    value: {
      schemaVersion: "1.0.0",
      modelVersion: MODEL_VERSION,
      evidence: EVIDENCE,
      inputs: checked.value,
      inputBasis: "Scenario assumptions; no model input represents a verified forecast.",
      startingRevenueReference: STARTING_REVENUE_REFERENCE,
      valuation,
      reverseValuation: solveRequiredGrowth(checked.value),
    },
  };
}
