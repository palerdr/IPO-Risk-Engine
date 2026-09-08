import { describe, expect, it } from 'vitest';
import { DEFAULT_INPUTS, evaluateScenario, sensitivity, solveRequiredGrowth, validateInputs, type ValuationInputs } from '../lib/valuation/engine';
import { EVIDENCE, exportScenario } from '../lib/valuation/scenario';
import { parseForm, toForm } from '../lib/valuation/form';

const stable: ValuationInputs = { ...DEFAULT_INPUTS, revenueB: 100, growth: 0, terminalGrowth: 0, grossMarginStart: .5, grossMarginEnd: .5, opexStart: .2, opexEnd: .2, taxRate: .2, discountRate: .1 };
function value(input: ValuationInputs) { const result = evaluateScenario(input); if (!result.ok) throw new Error(result.errors.join(' ')); return result.value; }

describe('DCF financial fixtures', () => {
  it('values a flat $24B cash-flow perpetuity at $240B', () => {
    const r = value(stable);
    expect(r.enterpriseValueB).toBeCloseTo(240, 9);
    expect(r.years[0].operatingIncomeB).toBeCloseTo(30);
    expect(r.years[0].taxB).toBeCloseTo(6);
    expect(r.years[0].cashFlowB).toBeCloseTo(24);
    expect(r.years[0].presentValueB).toBeCloseTo(24 / 1.1);
    expect(r.forecastValueB).toBeCloseTo(24 * (1 - 1.1 ** -10) / .1, 9);
    expect(r.discountedTerminalValueB).toBeCloseTo(240 / 1.1 ** 10, 9);
  });
  it('values a growing perpetuity with reinvestment against a hand calculation', () => {
    const r = value({ ...stable, growth: .03, terminalGrowth: .03, salesToCapital: 2 });
    const firstCashFlow = 103 * .3 * .8 - 3 / 2;
    expect(r.enterpriseValueB).toBeCloseTo(firstCashFlow / (.1 - .03), 8);
    expect(r.terminalYear.reinvestmentB).toBeCloseTo(r.years[9].revenueB * .03 / 2, 8);
    expect(r.terminalYear.cashFlowB).toBeCloseTo(firstCashFlow * 1.03 ** 10, 8);
  });
  it('keeps net cash unknown until the caller supplies it, including explicit zero', () => {
    expect(value(stable).equityValueB).toBeNull();
    expect(value({ ...stable, netCashB: 0 }).equityValueB).toBeCloseTo(240);
    expect(value({ ...stable, netCashB: 20, targetEquityB: 200 }).scenarioUpside).toBeCloseTo(.3);
    expect(value({ ...stable, netCashB: -20 }).equityValueB).toBeCloseTo(220);
  });
  it('fades growth over six steps and margins over ten forecast years', () => {
    const r = value(DEFAULT_INPUTS);
    expect(r.years[4].growth).toBeCloseTo(.3);
    expect(r.years[5].growth).toBeCloseTo(.255);
    expect(r.years[9].growth).toBeCloseTo(.075);
    expect(r.terminalYear.growth).toBeCloseTo(.03);
    expect(r.years[0].grossMargin).toBeCloseTo(.5);
    expect(r.years[9].grossMargin).toBeCloseTo(.7);
    expect(r.terminalYear.opexRatio).toBeCloseTo(.4);
  });
  it('charges no tax on losses and measures the worst cumulative funding gap', () => {
    const r = value(DEFAULT_INPUTS);
    expect(r.years[0].cashFlowB).toBeCloseTo(-25.38);
    expect(r.years[0].taxB).toBe(0);
    let running = 0, minimum = 0;
    for (const row of r.years) { running += row.cashFlowB; minimum = Math.min(minimum, running); }
    expect(r.minimumCumulativeCashFlowB).toBeCloseTo(minimum);
    expect(r.terminalContribution).toBeGreaterThan(1);
  });
  it('retains negative enterprise values without presenting common share prices', () => {
    const r = value({ ...stable, opexStart: .8, opexEnd: .8 });
    expect(r.enterpriseValueB).toBeLessThan(0);
    expect(r.terminalContribution).toBeNull();
    expect(r).not.toHaveProperty('perShareValue');
  });
  it('reduces positive-cash-flow value as the discount rate or expenses rise', () => {
    expect(value({ ...stable, discountRate: .12 }).enterpriseValueB).toBeLessThan(value(stable).enterpriseValueB);
    expect(value({ ...stable, opexStart: .3, opexEnd: .3 }).enterpriseValueB).toBeLessThan(value(stable).enterpriseValueB);
  });
});

describe('input boundaries and form conversion', () => {
  it.each([NaN, Infinity, -Infinity, '47', undefined])('rejects non-numeric input %s', revenueB => {
    expect(evaluateScenario({ ...stable, revenueB }).ok).toBe(false);
  });
  it.each(['growth', 'grossMarginStart', 'grossMarginEnd', 'opexStart', 'opexEnd', 'salesToCapital', 'taxRate', 'discountRate', 'netCashB', 'targetEquityB'])('rejects infinity for %s', key => {
    expect(evaluateScenario({ ...stable, [key]: Infinity }).ok).toBe(false);
  });
  it('rejects invalid bounds and terminal discount relationships', () => {
    for (const patch of [{ growth: -1 }, { growth: 2 }, { salesToCapital: 0 }, { discountRate: .03, terminalGrowth: .03 }, { discountRate: .02, terminalGrowth: .03 }, { targetEquityB: 0 }, { grossMarginEnd: 1.01 }, { version: '2' }]) {
      expect(evaluateScenario({ ...stable, ...patch }).ok).toBe(false);
    }
    expect(validateInputs(null).ok).toBe(false);
    expect(validateInputs([]).ok).toBe(false);
  });
  it('preserves empty optional fields and rejects an empty required field', () => {
    const form = toForm();
    expect(form.growth).toBe('30');
    expect(parseForm(form)).toEqual({ ok: true, value: DEFAULT_INPUTS });
    expect(parseForm({ ...form, netCashB: '0' })).toEqual({ ok: true, value: { ...DEFAULT_INPUTS, netCashB: 0 } });
    expect(parseForm({ ...form, revenueB: '' }).ok).toBe(false);
  });
  it('marks invalid sensitivity cells and reproduces the central value', () => {
    const rows = sensitivity(stable);
    expect(rows[2].cells[2].enterpriseValueB).toBeCloseTo(value(stable).enterpriseValueB);
    const nearTerminal = sensitivity({ ...stable, terminalGrowth: .03, discountRate: .04 });
    expect(nearTerminal[0].cells[2].enterpriseValueB).toBeNull();
  });
});

describe('reverse valuation', () => {
  it('recovers a known growth rate and leaves inputs untouched', () => {
    const input = { ...DEFAULT_INPUTS, growth: .4, netCashB: 10 };
    const target = value(input).equityValueB!;
    const result = solveRequiredGrowth({ ...input, growth: .1, targetEquityB: target });
    expect(result.ok).toBe(true);
    if (result.ok) { expect(result.value.growth).toBeCloseTo(.4, 7); expect(Math.abs(result.value.residualB)).toBeLessThan(target * 1e-8); }
    expect(input.growth).toBe(.4);
  });
  it('handles a root at the bracket endpoint', () => {
    const result = solveRequiredGrowth({ ...stable, netCashB: 0, targetEquityB: 240 });
    expect(result.ok && result.value.growth).toBe(0);
  });
  it('reports an unreachable target without inventing a growth rate', () => {
    expect(solveRequiredGrowth({ ...stable, netCashB: 0, targetEquityB: 1 })).toEqual({ ok: false, errors: ['No solution within this range (0–150% growth).'] });
  });
  it('requires explicit net cash and a target', () => {
    expect(solveRequiredGrowth(DEFAULT_INPUTS).ok).toBe(false);
    expect(solveRequiredGrowth({ ...DEFAULT_INPUTS, netCashB: 0 }).ok).toBe(false);
  });
});

describe('evidence and reproduction', () => {
  it('exports enough context to reproduce the valuation after a JSON round trip', () => {
    const r = exportScenario({ ...DEFAULT_INPUTS, netCashB: 0, targetEquityB: 965 });
    expect(r.ok).toBe(true);
    if (r.ok) {
      const copy = JSON.parse(JSON.stringify(r.value));
      expect(evaluateScenario(copy.inputs)).toEqual(copy.valuation);
      expect(solveRequiredGrowth(copy.inputs)).toEqual(copy.reverseValuation);
      expect(copy.evidence.version).toBe('1.0.0');
      expect(copy.evidence.records.some((item: { id: string }) => item.id === copy.startingRevenueReference)).toBe(true);
    }
  });
  it('does not export invalid scenarios', () => expect(exportScenario({ ...DEFAULT_INPUTS, revenueB: NaN }).ok).toBe(false));
  it('preserves the reported run-rate lower bound and missing annual revenue', () => {
    expect(EVIDENCE.records.find(record => record.id === 'may-revenue-run-rate')).toMatchObject({ value: 47, relation: 'greater_than' });
    expect(EVIDENCE.records.find(record => record.id === 'annual-revenue')).toMatchObject({ value: null, status: 'unavailable' });
    expect(EVIDENCE.records.find(record => record.id === 'diluted-shares')?.value).toBeNull();
  });
  it('gives disclosed records dated HTTPS sources and unique stable identifiers', () => {
    expect(new Set(EVIDENCE.records.map(record => record.id)).size).toBe(EVIDENCE.records.length);
    for (const record of EVIDENCE.records.filter(record => record.status === 'company_disclosure')) {
      expect(record.sourceUrl).toMatch(/^https:\/\//);
      expect(record.publishedAt).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(record.publishedAt! <= EVIDENCE.asOf).toBe(true);
    }
  });
});
