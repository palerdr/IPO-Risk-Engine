export const MODEL_VERSION = "1.0.0";
export const INPUT_VERSION = "1.0.0";

export interface ValuationInputs {
  version: typeof INPUT_VERSION;
  revenueB: number;
  growth: number;
  terminalGrowth: number;
  grossMarginStart: number;
  grossMarginEnd: number;
  opexStart: number;
  opexEnd: number;
  salesToCapital: number;
  taxRate: number;
  discountRate: number;
  netCashB: number | null;
  targetEquityB: number | null;
}

export interface ForecastYear {
  year: number;
  growth: number;
  revenueB: number;
  grossMargin: number;
  opexRatio: number;
  operatingIncomeB: number;
  taxB: number;
  reinvestmentB: number;
  cashFlowB: number;
  presentValueB: number;
  cumulativeCashFlowB: number;
}

export interface ValuationResult {
  modelVersion: typeof MODEL_VERSION;
  years: ForecastYear[];
  terminalYear: ForecastYear;
  forecastValueB: number;
  terminalValueB: number;
  discountedTerminalValueB: number;
  terminalContribution: number | null;
  enterpriseValueB: number;
  equityValueB: number | null;
  scenarioUpside: number | null;
  minimumCumulativeCashFlowB: number;
}
export type Outcome<T> = { ok: true; value: T } | { ok: false; errors: string[] };

export const DEFAULT_INPUTS: ValuationInputs = {
  version: INPUT_VERSION,
  revenueB: 47,
  growth: 0.3,
  terminalGrowth: 0.03,
  grossMarginStart: 0.5,
  grossMarginEnd: 0.7,
  opexStart: 0.8,
  opexEnd: 0.4,
  salesToCapital: 2,
  taxRate: 0.25,
  discountRate: 0.12,
  netCashB: null,
  targetEquityB: null,
};

export const LIMITS = {
  revenueB: [0.001, 100000],
  growth: [0, 1.5],
  terminalGrowth: [0, 0.1],
  grossMarginStart: [0, 1],
  grossMarginEnd: [0, 1],
  opexStart: [0, 2],
  opexEnd: [0, 2],
  salesToCapital: [0.1, 100],
  taxRate: [0, 1],
  discountRate: [0.001, 1],
  netCashB: [-100000, 100000],
  targetEquityB: [0.001, 100000],
} as const;
export type NumericField = keyof typeof LIMITS;

const FIELD_NAMES: Record<NumericField, string> = {
  revenueB: "Starting revenue",
  growth: "Growth",
  terminalGrowth: "Terminal growth",
  grossMarginStart: "Starting gross margin",
  grossMarginEnd: "Terminal gross margin",
  opexStart: "Starting operating expenses",
  opexEnd: "Terminal operating expenses",
  salesToCapital: "Sales-to-capital ratio",
  taxRate: "Tax rate",
  discountRate: "Discount rate",
  netCashB: "Net cash",
  targetEquityB: "Proposed equity valuation",
};

export function validateInputs(input: unknown): Outcome<ValuationInputs> {
  if (!input || typeof input !== "object" || Array.isArray(input))
    return { ok: false, errors: ["Provide a scenario object."] };
  const data = input as Record<string, unknown>;
  const errors: string[] = [];
  if (data.version !== INPUT_VERSION) errors.push("Use input version 1.0.0.");
  for (const key of Object.keys(LIMITS) as NumericField[]) {
    const value = data[key];
    if ((key === "netCashB" || key === "targetEquityB") && value === null) continue;
    const [min, max] = LIMITS[key];
    if (typeof value !== "number" || !Number.isFinite(value))
      errors.push(`${FIELD_NAMES[key]} needs a finite number.`);
    else if (value < min || value > max)
      errors.push(`${FIELD_NAMES[key]} must be between ${min} and ${max} in model units.`);
  }
  if (
    typeof data.discountRate === "number" &&
    typeof data.terminalGrowth === "number" &&
    data.discountRate <= data.terminalGrowth
  ) {
    errors.push("Discount rate must exceed terminal growth.");
  }
  if (errors.length) return { ok: false, errors };
  const clean = { version: INPUT_VERSION } as ValuationInputs;
  for (const key of Object.keys(LIMITS) as NumericField[]) clean[key] = data[key] as never;
  return { ok: true, value: clean };
}

export function evaluateScenario(raw: unknown): Outcome<ValuationResult> {
  const checked = validateInputs(raw);
  if (!checked.ok) return checked;
  const x = checked.value;
  const years: ForecastYear[] = [];
  let revenue = x.revenueB;
  let cumulative = 0;
  for (let year = 1; year <= 11; year++) {
    const growth =
      year <= 5 ? x.growth : x.growth + (x.terminalGrowth - x.growth) * ((year - 5) / 6);
    const progress = Math.min(1, (year - 1) / 9);
    const grossMargin = x.grossMarginStart + (x.grossMarginEnd - x.grossMarginStart) * progress;
    const opexRatio = x.opexStart + (x.opexEnd - x.opexStart) * progress;
    const previousRevenue = revenue;
    revenue *= 1 + growth;
    const operatingIncomeB = revenue * (grossMargin - opexRatio);
    const taxB = Math.max(0, operatingIncomeB) * x.taxRate;
    const reinvestmentB = (revenue - previousRevenue) / x.salesToCapital;
    const cashFlowB = operatingIncomeB - taxB - reinvestmentB;
    cumulative += cashFlowB;
    years.push({
      year,
      growth,
      revenueB: revenue,
      grossMargin,
      opexRatio,
      operatingIncomeB,
      taxB,
      reinvestmentB,
      cashFlowB,
      presentValueB: cashFlowB / (1 + x.discountRate) ** year,
      cumulativeCashFlowB: cumulative,
    });
  }
  const terminalYear = years.pop()!;
  const forecastValueB = years.reduce((sum, row) => sum + row.presentValueB, 0);
  const terminalValueB = terminalYear.cashFlowB / (x.discountRate - x.terminalGrowth);
  const discountedTerminalValueB = terminalValueB / (1 + x.discountRate) ** 10;
  const enterpriseValueB = forecastValueB + discountedTerminalValueB;
  const equityValueB = x.netCashB === null ? null : enterpriseValueB + x.netCashB;
  if (
    ![
      forecastValueB,
      terminalValueB,
      discountedTerminalValueB,
      enterpriseValueB,
      equityValueB ?? 0,
    ].every(Number.isFinite)
  ) {
    return { ok: false, errors: ["The calculation exceeds the supported numeric range."] };
  }
  return {
    ok: true,
    value: {
      modelVersion: MODEL_VERSION,
      years,
      terminalYear,
      forecastValueB,
      terminalValueB,
      discountedTerminalValueB,
      enterpriseValueB,
      equityValueB,
      terminalContribution:
        enterpriseValueB > 0 ? discountedTerminalValueB / enterpriseValueB : null,
      scenarioUpside:
        equityValueB === null || x.targetEquityB === null
          ? null
          : equityValueB / x.targetEquityB - 1,
      minimumCumulativeCashFlowB: Math.min(0, ...years.map((row) => row.cumulativeCashFlowB)),
    },
  };
}

export interface GrowthSolution {
  growth: number;
  impliedEquityB: number;
  residualB: number;
}
export function solveRequiredGrowth(raw: unknown): Outcome<GrowthSolution> {
  const checked = validateInputs(raw);
  if (!checked.ok) return checked;
  const x = checked.value;
  if (x.netCashB === null || x.targetEquityB === null)
    return {
      ok: false,
      errors: ["Enter net cash and a proposed equity valuation to solve for growth."],
    };
  const target = x.targetEquityB;
  const residual = (growth: number) => {
    const result = evaluateScenario({ ...x, growth });
    if (!result.ok) return NaN;
    return result.value.equityValueB! - target;
  };
  let lo = 0,
    hi = 1.5,
    flo = residual(lo),
    fhi = residual(hi);
  const tolerance = Math.max(1e-8, target * 1e-10);
  const solved = (growth: number, error: number): Outcome<GrowthSolution> => ({
    ok: true,
    value: { growth, impliedEquityB: target + error, residualB: error },
  });
  if (Math.abs(flo) <= tolerance) return solved(lo, flo);
  if (Math.abs(fhi) <= tolerance) return solved(hi, fhi);
  if (!Number.isFinite(flo) || !Number.isFinite(fhi) || Math.sign(flo) === Math.sign(fhi))
    return { ok: false, errors: ["No solution within this range (0–150% growth)."] };
  for (let iteration = 0; iteration < 100; iteration++) {
    const middle = (lo + hi) / 2;
    const fm = residual(middle);
    if (!Number.isFinite(fm))
      return { ok: false, errors: ["The solver exceeds the supported numeric range."] };
    if (Math.abs(fm) <= tolerance) return solved(middle, fm);
    if (Math.sign(fm) === Math.sign(flo)) {
      lo = middle;
      flo = fm;
    } else {
      hi = middle;
      fhi = fm;
    }
  }
  return { ok: false, errors: ["The solver did not converge within 100 iterations."] };
}

export interface SensitivityRow {
  discountRate: number;
  cells: { operatingMargin: number; enterpriseValueB: number | null }[];
}
export function sensitivity(input: ValuationInputs): SensitivityRow[] {
  return [-0.02, -0.01, 0, 0.01, 0.02].map((delta) => ({
    discountRate: input.discountRate + delta,
    cells: [-0.1, -0.05, 0, 0.05, 0.1].map((marginDelta) => {
      const operatingMargin = input.grossMarginEnd - input.opexEnd + marginDelta;
      const r = evaluateScenario({
        ...input,
        discountRate: input.discountRate + delta,
        opexEnd: input.opexEnd - marginDelta,
      });
      return { operatingMargin, enterpriseValueB: r.ok ? r.value.enterpriseValueB : null };
    }),
  }));
}
