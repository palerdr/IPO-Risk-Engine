// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import App from '../App';
import { evaluateScenario } from '../lib/valuation/engine';

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const change = (name: string, value: string) => fireEvent.change(screen.getByLabelText(name), { target: { value } });
const outputValue = () => document.querySelector('.valuation-hero .value')?.textContent;

describe('workbench interaction', () => {
  it('renders the full local demo with missing-data boundaries and dated sources', () => {
    render(<App />);
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Anthropic.');
    expect(screen.getByText('Needs net cash')).toBeTruthy();
    expect(screen.getAllByRole('table')).toHaveLength(2);
    const forecast = screen.getByRole('table', { name: /Forecast and terminal year/ });
    expect(within(forecast).getAllByRole('row')).toHaveLength(12);
    const sensitivity = screen.getByRole('table', { name: /Enterprise value by discount/ });
    expect(within(sensitivity).getAllByRole('cell')).toHaveLength(25);
    expect(screen.getAllByRole('link', { name: /Anthropic Series H announcement/ })[0].getAttribute('href')).toBe('https://www.anthropic.com/news/series-h');
    expect(screen.getByText(/Per-share results need/)).toBeTruthy();
  });

  it('recalculates on input changes, shows invalid inputs, and restores the default', () => {
    render(<App />);
    const initial = outputValue();
    change('Discount rate / WACC', '14');
    expect(outputValue()).not.toBe(initial);
    change('Normalized starting revenue', '');
    expect(screen.getByRole('alert').textContent).toContain('Starting revenue needs a finite number');
    expect((screen.getByRole('button', { name: /Export scenario/ }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Reset assumptions' }));
    expect(screen.queryByRole('alert')).toBeNull();
    expect(outputValue()).toBe(initial);
    expect((screen.getByLabelText('Net cash assumption') as HTMLInputElement).value).toBe('');
  });

  it('solves a target, applies the growth rate, and retains the equity bridge', () => {
    render(<App />);
    change('Net cash assumption', '0');
    change('Proposed IPO equity valuation', '965');
    fireEvent.click(screen.getByRole('radio', { name: 'Required growth' }));
    const implied = document.querySelector('.reverse-value');
    expect(implied?.textContent).toMatch(/\d+\.\d+%/);
    fireEvent.click(screen.getByRole('button', { name: /Apply growth to scenario/ }));
    expect((screen.getByRole('radio', { name: 'Forward valuation' }) as HTMLInputElement).checked).toBe(true);
    expect(outputValue()).toBe('$965B');
    expect(document.querySelector('.bridge')?.textContent).toContain('$965B');
  });

  it('downloads a reproducible scenario through the export button', async () => {
    render(<App />);
    change('Net cash assumption', '0');
    let blob: Blob | undefined;
    vi.stubGlobal('URL', class extends URL {
      static createObjectURL(value: Blob) { blob = value; return 'blob:scenario'; }
      static revokeObjectURL() {}
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByRole('button', { name: /Export scenario/ }));
    expect(click).toHaveBeenCalledOnce();
    const text = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsText(blob!);
    });
    const exported = JSON.parse(text);
    expect(evaluateScenario(exported.inputs)).toEqual(exported.valuation);
    expect(exported.evidence.version).toBe('1.0.0');
    expect(screen.getByRole('status').textContent).toContain('scenario export includes');
  });
});
