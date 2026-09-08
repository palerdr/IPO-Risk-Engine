import { useState } from "react";
import { downloadJson } from "./lib/download";
import {
  DEFAULT_INPUTS,
  evaluateScenario,
  solveRequiredGrowth,
  sensitivity,
  type NumericField,
} from "./lib/valuation/engine";
import { parseForm, toForm, PERCENT_FIELDS } from "./lib/valuation/form";
import { EVIDENCE, exportScenario } from "./lib/valuation/scenario";

const money = (value: number) =>
  `$${value.toLocaleString("en-US", { maximumFractionDigits: 1 })}B`;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const labels: Record<NumericField, string> = {
  targetEquityB: "Assumed entry equity value ($B)",
  netCashB: "Net cash assumption ($B)",
  growth: "Revenue growth in years 1–5 (%)",
  discountRate: "Discount rate (%)",
  revenueB: "Normalized starting revenue ($B)",
  grossMarginStart: "Year-1 gross margin (%)",
  grossMarginEnd: "Year-10 gross margin (%)",
  opexStart: "Year-1 operating expenses / revenue (%)",
  opexEnd: "Year-10 operating expenses / revenue (%)",
  salesToCapital: "Sales-to-capital ratio",
  terminalGrowth: "Terminal growth (%)",
  taxRate: "Tax rate (%)",
};
const primary: NumericField[] = [
  "targetEquityB",
  "netCashB",
  "growth",
  "discountRate",
];

export default function Valuation() {
  const [form, setForm] = useState(() => toForm());
  const [message, setMessage] = useState("");
  const parsed = parseForm(form);
  const calculated = parsed.ok ? evaluateScenario(parsed.value) : parsed;
  const result = calculated.ok ? calculated.value : null;
  const reverse = parsed.ok ? solveRequiredGrowth(parsed.value) : parsed;
  function field(key: NumericField) {
    return (
      <div className="field" key={key}>
        <label htmlFor={key}>{labels[key]}</label>
        <div className="input-wrap">
          <input
            id={key}
            type="number"
            step="any"
            value={form[key]}
            onChange={(event) =>
              setForm({ ...form, [key]: event.target.value })
            }
          />
        </div>
      </div>
    );
  }
  function download() {
    if (!parsed.ok) return;
    const output = exportScenario(parsed.value);
    if (!output.ok) return;
    downloadJson(output.value, "anthropic-valuation.json");
    setMessage("You exported the scenario with its source evidence.");
  }
  return (
    <section aria-label="Anthropic valuation">
      <div className="workspace-heading">
        <h1>Anthropic valuation</h1>
        <span className="badge">PRE-LISTING CASE STUDY</span>
      </div>
      <p className="page-description">
        You estimate long-term value from business assumptions. This calculator
        does not use the post-listing risk classifier.
      </p>
      <div className="boundary">
        You have no Anthropic post-listing prices in this evidence snapshot. A
        trading-risk score remains unavailable.
      </div>
      <div className="panel scenario-controls">
        <div className="section-title">
          <h2>Compare an assumed entry valuation</h2>
          <button
            className="text-button"
            onClick={() =>
              setForm(
                toForm({ ...DEFAULT_INPUTS, netCashB: 0, targetEquityB: 965 }),
              )
            }
          >
            Load example · $0 net cash
          </button>
        </div>
        <div className="primary-fields">{primary.map(field)}</div>
        <p className="note">
          You must enter net cash to estimate equity value. A blank value means
          unknown; 0 is an explicit assumption.
        </p>
        <details className="advanced-assumptions">
          <summary>Advanced assumptions</summary>
          <div className="advanced-fields">
            {(Object.keys(labels) as NumericField[])
              .filter((key) => !primary.includes(key))
              .map(field)}
          </div>
          <p className="note">
            You use $47B as an illustrative annual revenue assumption referenced
            to a disclosed run rate. The run rate does not establish recognized
            annual revenue.
          </p>
        </details>
      </div>
      {!calculated.ok && (
        <div className="error-panel" role="alert">
          {calculated.errors.join(" ")}
        </div>
      )}
      {result && (
        <>
          <div className="metric-grid">
            <div className="metric-card">
              <p>Modeled business value</p>
              <strong>{money(result.enterpriseValueB)}</strong>
              <span>Enterprise value · ten-year DCF</span>
            </div>
            <div className="metric-card">
              <p>Modeled equity value</p>
              <strong>
                {result.equityValueB === null
                  ? "Needs net cash"
                  : money(result.equityValueB)}
              </strong>
              <span>Business value + net cash</span>
            </div>
            <div className="metric-card">
              <p>Value versus entry</p>
              <strong>
                {result.scenarioUpside === null
                  ? "Not compared"
                  : percent(result.scenarioUpside)}
              </strong>
              <span>Scenario gap · no expected-return claim</span>
            </div>
            <div className="metric-card">
              <p>Required revenue growth</p>
              <strong>
                {reverse.ok
                  ? percent(reverse.value.growth)
                  : "Needs entry inputs"}
              </strong>
              <span>Annual growth in years 1–5</span>
            </div>
          </div>
          <div className="panel">
            <h2>Test the growth required by your entry assumption</h2>
            <p className="note">
              You hold the other assumptions fixed and solve for revenue growth.
              This calculation does not estimate the chance of reaching that
              growth.
            </p>
            {reverse.ok ? (
              <button
                className="outline"
                onClick={() =>
                  setForm({
                    ...form,
                    growth: String(
                      reverse.value.growth *
                        (PERCENT_FIELDS.has("growth") ? 100 : 1),
                    ),
                  })
                }
              >
                Apply required growth
              </button>
            ) : (
              <p>{reverse.errors.join(" ")}</p>
            )}
          </div>
          <details className="panel">
            <summary>Forecast and valuation sensitivity</summary>
            <div className="table-scroll">
              <table>
                <caption>Annual forecast · USD billions</caption>
                <thead>
                  <tr>
                    <th>Year</th>
                    <th>Revenue</th>
                    <th>Operating income</th>
                    <th>Reinvestment</th>
                    <th>Free cash flow</th>
                  </tr>
                </thead>
                <tbody>
                  {[...result.years, result.terminalYear].map((row) => (
                    <tr key={row.year}>
                      <th>{row.year === 11 ? "11 · terminal" : row.year}</th>
                      <td>{money(row.revenueB)}</td>
                      <td>{money(row.operatingIncomeB)}</td>
                      <td>{money(row.reinvestmentB)}</td>
                      <td>{money(row.cashFlowB)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {parsed.ok && (
              <div className="table-scroll">
                <table>
                  <caption>
                    Enterprise value by discount rate and terminal operating
                    margin
                  </caption>
                  <thead>
                    <tr>
                      <th>Discount rate / margin</th>
                      {sensitivity(parsed.value)[0].cells.map((cell) => (
                        <th key={cell.operatingMargin}>
                          {percent(cell.operatingMargin)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sensitivity(parsed.value).map((row) => (
                      <tr key={row.discountRate}>
                        <th>{percent(row.discountRate)}</th>
                        {row.cells.map((cell) => (
                          <td key={cell.operatingMargin}>
                            {cell.enterpriseValueB === null
                              ? "Unavailable"
                              : money(cell.enterpriseValueB)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="note">
              Terminal contribution:{" "}
              {result.terminalContribution === null
                ? "undefined"
                : percent(result.terminalContribution)}
              . Lowest cumulative cash flow:{" "}
              {money(result.minimumCumulativeCashFlowB)}. A contribution above
              100% means forecast-period cash flows have a negative present
              value.
            </p>
          </details>
        </>
      )}
      <details className="panel">
        <summary>Sources and model limits</summary>
        <p>
          You discount ten years of unlevered cash flow and a terminal value.
          You include terminal reinvestment. The model omits tax loss
          carryforwards and future financing dilution. Per-share values require
          a supported diluted share count.
        </p>
        <p className="note">
          Evidence snapshot: {EVIDENCE.asOf}. This dataset contains no public
          prospectus.
        </p>
        <div className="evidence-grid">
          {EVIDENCE.records.map((record) => (
            <article key={record.id} className="panel evidence-card">
              <h3>{record.metric}</h3>
              <p>{record.note}</p>
              {record.sourceUrl && (
                <a href={record.sourceUrl} target="_blank" rel="noreferrer">
                  {record.sourceTitle}
                </a>
              )}
            </article>
          ))}
        </div>
      </details>
      <div className="intro-actions">
        <button onClick={download} disabled={!result}>
          Export valuation
        </button>
        <button
          className="outline"
          onClick={() => {
            setForm(toForm());
            setMessage("");
          }}
        >
          Reset assumptions
        </button>
      </div>
      <p role="status">{message}</p>
    </section>
  );
}
