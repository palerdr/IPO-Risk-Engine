import {
  DEFAULT_INPUTS,
  INPUT_VERSION,
  LIMITS,
  validateInputs,
  type NumericField,
  type Outcome,
  type ValuationInputs,
} from "./engine";
export type FormValues = Record<NumericField, string>;
export const PERCENT_FIELDS = new Set<NumericField>([
  "growth",
  "terminalGrowth",
  "grossMarginStart",
  "grossMarginEnd",
  "opexStart",
  "opexEnd",
  "taxRate",
  "discountRate",
]);
export function toForm(input: ValuationInputs = DEFAULT_INPUTS): FormValues {
  return Object.fromEntries(
    (Object.keys(LIMITS) as NumericField[]).map((key) => {
      const value = input[key];
      if (value === null) return [key, ""];
      const scale = PERCENT_FIELDS.has(key) ? 100 : 1;
      return [key, String(Number((value * scale).toPrecision(12)))];
    }),
  ) as FormValues;
}
export function parseForm(form: FormValues): Outcome<ValuationInputs> {
  const data: Record<string, unknown> = { version: INPUT_VERSION };
  for (const key of Object.keys(LIMITS) as NumericField[]) {
    const raw = form[key].trim();
    if (raw === "") {
      const optional = key === "netCashB" || key === "targetEquityB";
      data[key] = optional ? null : Number.NaN;
    } else {
      data[key] = Number(raw) / (PERCENT_FIELDS.has(key) ? 100 : 1);
    }
  }
  return validateInputs(data);
}
