import { useState } from "react";
import riskDemo from "./data/risk-demo.json";
import {
  DEFAULT_INPUTS,
  evaluateScenario,
  LIMITS,
  MODEL_VERSION,
  sensitivity,
  solveRequiredGrowth,
  type NumericField,
  type ValuationInputs,
  type ValuationResult,
} from "./lib/valuation/engine";
import { EVIDENCE, exportScenario } from "./lib/valuation/scenario";
import {
  parseForm,
  PERCENT_FIELDS,
  toForm,
  type FormValues,
} from "./lib/valuation/form";

const money = (n: number) =>
  `${n < 0 ? "−" : ""}$${Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: 1 })}B`;
const number = (n: number) =>
  n.toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
const pct = (n: number) => `${number(n * 100)}%`;
const fields: {
  key: NumericField;
  label: string;
  unit: string;
  hint?: string;
}[] = [
  {
    key: "revenueB",
    label: "Normalized starting revenue",
    unit: "$B",
    hint: "Illustrative $47B starting point, referenced to the May run rate. This is an assumption about annual revenue.",
  },
  { key: "growth", label: "Revenue growth · years 1–5", unit: "%" },
  { key: "grossMarginStart", label: "Gross margin · year 1", unit: "%" },
  { key: "grossMarginEnd", label: "Gross margin · year 10", unit: "%" },
  {
    key: "opexStart",
    label: "Operating expenses · year 1",
    unit: "% of revenue",
  },
  {
    key: "opexEnd",
    label: "Operating expenses · year 10",
    unit: "% of revenue",
  },
  {
    key: "salesToCapital",
    label: "Sales-to-capital ratio",
    unit: "×",
    hint: "Revenue increase per dollar of net reinvestment.",
  },
  { key: "discountRate", label: "Discount rate / WACC", unit: "%" },
  { key: "terminalGrowth", label: "Terminal growth", unit: "%" },
  { key: "taxRate", label: "Tax on positive operating income", unit: "%" },
  {
    key: "netCashB",
    label: "Net cash assumption",
    unit: "$B",
    hint: "Cash less debt and other senior claims. Enter 0 to assume no net cash; a blank value means unknown.",
  },
  {
    key: "targetEquityB",
    label: "Proposed IPO equity valuation",
    unit: "$B",
    hint: "Your assumed entry valuation. The private funding reference is $965B as of May 28.",
  },
];

function Field({
  field,
  form,
  setField,
}: {
  field: (typeof fields)[number];
  form: FormValues;
  setField: (key: NumericField, value: string) => void;
}) {
  const factor = PERCENT_FIELDS.has(field.key) ? 100 : 1;
  return (
    <div className="field">
      <label htmlFor={field.key}>{field.label}</label>
      <div className="input-wrap">
        <input
          id={field.key}
          type="number"
          inputMode="decimal"
          value={form[field.key]}
          onChange={(e) => setField(field.key, e.target.value)}
          min={LIMITS[field.key][0] * factor}
          max={LIMITS[field.key][1] * factor}
          step="any"
          aria-describedby={field.hint ? `${field.key}-hint` : undefined}
        />
        <span>{field.unit}</span>
      </div>
      {field.hint ? (
        <p id={`${field.key}-hint`} className="field-hint">
          {field.hint}
        </p>
      ) : null}
    </div>
  );
}

function CashFlowChart({ result }: { result: ValuationResult }) {
  const values = result.years.map((row) => row.cashFlowB);
  const min = Math.min(0, ...values),
    max = Math.max(1, ...values);
  const y = (value: number) => 185 - ((value - min) / (max - min)) * 155;
  const zero = y(0);
  return (
    <figure className="cash-chart">
      <figcaption>
        <span>Path to free cash flow</span>
        <span className="muted">Annual, USD billions</span>
      </figcaption>
      <svg
        viewBox="0 0 720 228"
        role="img"
        aria-label="Projected annual free cash flow. The forecast table contains the numeric values."
      >
        {[min, (max + min) / 2, max].map((v, i) => (
          <g key={i}>
            <line x1="57" x2="702" y1={y(v)} y2={y(v)} stroke="var(--border)" />
            <text
              x="47"
              y={y(v) + 5}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="14"
            >
              {Math.round(v)}
            </text>
          </g>
        ))}
        <line x1="57" x2="702" y1={zero} y2={zero} stroke="var(--muted)" />
        {values.map((v, i) => (
          <g key={i}>
            <rect
              x={71 + i * 63}
              y={Math.min(zero, y(v))}
              width="29"
              height={Math.max(1, Math.abs(y(v) - zero))}
              rx="2"
              fill={v >= 0 ? "var(--accent)" : "var(--negative)"}
            />
            <text
              x={85.5 + i * 63}
              y="215"
              textAnchor="middle"
              fill="var(--muted)"
              fontSize="14"
            >
              Y{i + 1}
            </text>
            <title>{`Year ${i + 1}: ${money(v)}`}</title>
          </g>
        ))}
      </svg>
    </figure>
  );
}

function ValuationSummary({ result }: { result: ValuationResult }) {
  return (
    <div className="metric-grid">
      <div className="metric-card valuation-hero">
        <p>Modeled business value</p>
        <h2 className="value">{money(result.enterpriseValueB)}</h2>
        <span>Enterprise value · ten-year DCF</span>
      </div>
      <div className="metric-card">
        <p>Modeled equity value</p>
        <strong>
          {result.equityValueB === null
            ? "Needs net cash"
            : money(result.equityValueB)}
        </strong>
        <span>Business value + cash less debt</span>
      </div>
      <div className="metric-card">
        <p>Value versus entry</p>
        <strong
          className={
            result.scenarioUpside !== null && result.scenarioUpside < 0
              ? "negative"
              : ""
          }
        >
          {result.scenarioUpside === null
            ? "Not compared"
            : pct(result.scenarioUpside)}
        </strong>
        <span>Modeled equity / assumed entry − 1</span>
      </div>
      <div className="metric-card">
        <p>Anthropic drawdown risk</p>
        <strong className="pending-value">No score</strong>
        <span>No post-listing prices in this dataset</span>
      </div>
    </div>
  );
}

function ValueComparison({
  result,
  input,
}: {
  result: ValuationResult;
  input: ValuationInputs;
}) {
  const comparable =
    result.equityValueB !== null && input.targetEquityB !== null;
  const values = comparable
    ? [
        { label: "Modeled equity", value: result.equityValueB! },
        { label: "Assumed entry", value: input.targetEquityB! },
      ]
    : [{ label: "Modeled business value", value: result.enterpriseValueB }];
  const scale = Math.max(1, ...values.map((item) => Math.abs(item.value)));
  return (
    <section className="panel comparison-chart">
      <h2>Value versus entry</h2>
      <p className="note">
        {comparable
          ? "Equity values · USD billions"
          : "Set net cash and an entry value to compare equity values."}
      </p>
      <div className="comparison-bars">
        {values.map((item, index) => (
          <div key={item.label} className="comparison-row">
            <div>
              <span>{item.label}</span>
              <strong>{money(item.value)}</strong>
            </div>
            <div className="bar-track">
              <div
                className={index === 0 ? "bar modeled" : "bar entry"}
                style={{ width: `${(Math.abs(item.value) / scale) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="note">
        {comparable
          ? "You compare a scenario with your entry assumption. This gap is not an expected stock return."
          : "The private funding valuation measures equity. You need the cash-and-debt bridge before comparing it with business value."}
      </p>
    </section>
  );
}

function RiskEngine() {
  return (
    <section className="risk-view" aria-label="Risk engine walkthrough">
      <div className="panel risk-explanation">
        <span className="badge">POST-LISTING MODEL</span>
        <h2>Estimate drawdown risk after an IPO</h2>
        <p>
          You use the original Python engine to estimate the chance of a 20% or
          greater drawdown over the next 20 trading sessions. For this RIVER
          walkthrough, you first need 61 sessions of price and volume history.
        </p>
        <div className="boundary">
          <strong>Anthropic: no score available.</strong> This dataset contains
          no post-listing bars. The valuation calculator does not call this
          model, and a DCF value does not determine drawdown risk.
        </div>
      </div>
      <div className="section-title">
        <div>
          <h2>Original engine · saved walkthrough</h2>
          <p className="note">
            You are viewing output from a Python run on synthetic prices. No
            Anthropic prices or live predictions appear below.
          </p>
        </div>
        <span className="badge">SYNTHETIC DATA</span>
      </div>
      <div className="metric-grid">
        <div className="metric-card">
          <p>Synthetic IPOs</p>
          <strong>
            {riskDemo.rows.train +
              riskDemo.rows.validation +
              riskDemo.rows.test}
          </strong>
          <span>Seed {riskDemo.seed} · reproducible sample</span>
        </div>
        <div className="metric-card">
          <p>Model inputs</p>
          <strong>{riskDemo.features}</strong>
          <span>Features from three price windows</span>
        </div>
        <div className="metric-card">
          <p>Example event probability</p>
          <strong>{pct(riskDemo.example.adverse_probability)}</strong>
          <span>{riskDemo.example.symbol} · synthetic</span>
        </div>
        <div className="metric-card">
          <p>Example policy label</p>
          <strong>{riskDemo.example.action}</strong>
          <span>Research label · no order execution</span>
        </div>
      </div>
      <div className="chart-grid">
        <section className="panel">
          <h2>From price history to risk label</h2>
          <ol className="pipeline">
            <li>
              <strong>Observe prices</strong>
              <span>
                You supply daily price bars and volume for sessions 0–60.
              </span>
            </li>
            <li>
              <strong>Measure trading behavior</strong>
              <span>
                You calculate volatility and liquidity features across FLOP,
                TURN, and RIVER windows.
              </span>
            </li>
            <li>
              <strong>Estimate event probability</strong>
              <span>
                You combine Ridge and KNN estimates, then calibrate their
                scores. This run selected the {riskDemo.calibrator} calibrator.
              </span>
            </li>
            <li>
              <strong>Assign a research label</strong>
              <span>
                You apply fixed thresholds: below 30% → SIZE_UP; 30% to below
                60% → SMALL_BET; 60% or above → FOLD.
              </span>
            </li>
          </ol>
        </section>
        <section className="panel">
          <h2>Research labels across the test set</h2>
          <p className="note">
            {riskDemo.rows.test} synthetic IPOs · fixed default thresholds
          </p>
          <div className="comparison-bars">
            {["SIZE_UP", "SMALL_BET", "FOLD"].map((label) => {
              const count =
                (riskDemo.actions as Record<string, number>)[label] ?? 0;
              return (
                <div className="comparison-row" key={label}>
                  <div>
                    <span>{label}</span>
                    <strong>{count}</strong>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar modeled"
                      style={{
                        width: `${(count / riskDemo.rows.test) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="note">
            You train on {riskDemo.rows.train} examples and reserve{" "}
            {riskDemo.rows.validation} for validation. This walkthrough leaves
            validation unused and shows code execution without establishing
            predictive accuracy.
          </p>
        </section>
      </div>
      <details className="panel">
        <summary>Inspect the run and research limits</summary>
        <p>You can reproduce this saved output from the repository root:</p>
        <pre>uv run python -m scripts.demo_risk_engine</pre>
        <p>
          You need audited historical data and a validated fitted model before
          interpreting real-company scores. The existing threshold optimizer can
          violate its requested risk cap; this walkthrough uses fixed
          thresholds.
        </p>
        <pre>{JSON.stringify(riskDemo, null, 2)}</pre>
      </details>
    </section>
  );
}

function Sensitivity({ input }: { input: ValuationInputs }) {
  const rows = sensitivity(input);
  return (
    <section className="panel sensitivity">
      <div className="section-title">
        <h2>Valuation sensitivity</h2>
        <span className="muted">Enterprise value · $B</span>
      </div>
      <p className="note">
        Discount rate by terminal operating margin. Change terminal operating
        expenses while holding gross margin fixed.
      </p>
      <div className="table-scroll">
        <table>
          <caption className="sr-only">
            Enterprise value by discount rate and terminal operating margin
          </caption>
          <thead>
            <tr>
              <th scope="col">WACC / margin</th>
              {rows[0].cells.map((cell) => (
                <th scope="col" key={cell.operatingMargin}>
                  {pct(cell.operatingMargin)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={row.discountRate}>
                <th scope="row">{pct(row.discountRate)}</th>
                {row.cells.map((cell, j) => (
                  <td
                    key={j}
                    className={i === 2 && j === 2 ? "selected-cell" : ""}
                  >
                    {cell.enterpriseValueB === null
                      ? "Invalid"
                      : number(cell.enterpriseValueB)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="field-hint">
        The teal cell marks your current assumptions. Invalid cells fall outside
        model bounds.
      </p>
    </section>
  );
}

export default function App() {
  const [form, setForm] = useState<FormValues>(() => toForm());
  const [view, setView] = useState("valuation");
  const [mode, setMode] = useState("forward");
  const [exportMessage, setExportMessage] = useState("");
  const parsed = parseForm(form);
  const evaluation = parsed.ok ? evaluateScenario(parsed.value) : parsed;
  const result = evaluation.ok ? evaluation.value : null;
  const reverse = parsed.ok ? solveRequiredGrowth(parsed.value) : parsed;
  const setField = (key: NumericField, value: string) =>
    setForm((previous) => ({ ...previous, [key]: value }));

  function download() {
    if (!parsed.ok) return;
    const exported = exportScenario(parsed.value);
    if (!exported.ok) {
      setExportMessage(exported.errors.join(" "));
      return;
    }
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(exported.value, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `anthropic-scenario-v${MODEL_VERSION}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setExportMessage(
      "Your scenario export includes the inputs and source evidence.",
    );
  }

  return (
    <main className="workbench" id="top">
      <header className="masthead">
        <a className="brand" href="#top">
          <span className="brand-icon" aria-hidden="true">
            ↗
          </span>{" "}
          IPO / RESEARCH
        </a>
        <span className="snapshot-date">
          Evidence snapshot · {EVIDENCE.asOf}
        </span>
      </header>
      <section className="intro">
        <div>
          <h1>
            Anthropic<span className="title-dot">.</span>
          </h1>
          <p>
            Evaluate an entry valuation before listing. Assess trading risk
            after listing.
          </p>
        </div>
        <div className="intro-actions" hidden={view !== "valuation"}>
          <button
            className="outline"
            onClick={() => {
              setForm(toForm(DEFAULT_INPUTS));
              setExportMessage("");
            }}
          >
            Reset assumptions
          </button>
          <button onClick={download} disabled={!result}>
            Export scenario ↗
          </button>
        </div>
      </section>
      <nav className="view-controls" aria-label="Research views">
        <button
          aria-pressed={view === "valuation"}
          onClick={() => setView("valuation")}
        >
          Valuation
        </button>
        <button aria-pressed={view === "risk"} onClick={() => setView("risk")}>
          Risk engine <span>Post-IPO</span>
        </button>
      </nav>
      <p className="sr-only" role="status">
        {exportMessage}
      </p>
      <div hidden={view !== "valuation"}>
        <div className="workspace-heading">
          <h2>Valuation overview</h2>
          <span className="badge">ILLUSTRATIVE SCENARIO</span>
        </div>
        {result ? <ValuationSummary result={result} /> : null}
        <section
          className="panel scenario-controls"
          aria-label="Scenario assumptions"
        >
          <div className="section-title">
            <h2>Set your assumptions</h2>
            <button
              className="text-button"
              onClick={() => {
                setForm(
                  toForm({
                    ...DEFAULT_INPUTS,
                    targetEquityB: 965,
                    netCashB: 0,
                  }),
                );
                setMode("forward");
                setExportMessage("");
              }}
            >
              Load example · $0 net cash
            </button>
          </div>
          <div className="primary-fields">
            {[fields[11], fields[10], fields[1], fields[7]].map((field) => (
              <Field
                key={field.key}
                field={field}
                form={form}
                setField={setField}
              />
            ))}
          </div>
          <details className="advanced-assumptions">
            <summary>
              Advanced assumptions{" "}
              <span>Revenue, margins and reinvestment</span>
            </summary>
            <div className="advanced-fields">
              {fields
                .filter(
                  (field) =>
                    ![
                      "targetEquityB",
                      "netCashB",
                      "growth",
                      "discountRate",
                    ].includes(field.key),
                )
                .map((field) => (
                  <Field
                    key={field.key}
                    field={field}
                    form={form}
                    setField={setField}
                  />
                ))}
            </div>
            <p className="note">
              Per-share results need a supported diluted share count. This
              evidence version has none.
            </p>
          </details>
        </section>
        {!evaluation.ok ? (
          <div role="alert" className="error-panel">
            <h2>Review your inputs</h2>
            {evaluation.errors.map((error) => (
              <p key={error}>{error}</p>
            ))}
          </div>
        ) : null}
        <section className="decision-panel">
          <fieldset className="mode-controls">
            <legend className="sr-only">Valuation mode</legend>
            {[
              ["forward", "Forward valuation"],
              ["reverse", "Required growth"],
            ].map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="mode"
                  value={value}
                  checked={mode === value}
                  onChange={() => setMode(value)}
                />
                {label}
              </label>
            ))}
          </fieldset>
          <div hidden={mode !== "forward"}>
            {result ? (
              <div className="takeaway">
                <h2>
                  {result.scenarioUpside === null
                    ? "Complete the entry comparison"
                    : result.scenarioUpside < 0
                      ? "Your modeled value falls below the entry assumption"
                      : "Your modeled value meets or exceeds the entry assumption"}
                </h2>
                <p>
                  {result.scenarioUpside === null
                    ? "Enter net cash and an assumed equity valuation, or load the example. Then change growth to see the effect on value."
                    : `At ${pct(parsed.ok ? parsed.value.growth : 0)} growth in years 1–5, you estimate ${money(result.equityValueB!)} of equity value. Select Required growth to find the growth rate that would support your entry assumption.`}
                </p>
              </div>
            ) : null}
          </div>
          <div hidden={mode !== "reverse"} className="reverse-panel">
            <h2>Growth needed to support your entry valuation</h2>
            {reverse.ok ? (
              <>
                <p className="reverse-value">
                  {pct(reverse.value.growth)}
                  <span> per year in years 1–5</span>
                </p>
                <p>
                  You reach {money(reverse.value.impliedEquityB)} of equity
                  value with this growth rate and the other assumptions held
                  fixed.
                </p>
                <button
                  className="outline"
                  onClick={() => {
                    setField("growth", String(reverse.value.growth * 100));
                    setMode("forward");
                  }}
                >
                  Apply growth to scenario ↗
                </button>
              </>
            ) : (
              <p className="callout">{reverse.errors.join(" ")}</p>
            )}
            <p className="note">
              You solve for a growth rate within 0–150%. This calculation does
              not estimate the chance of reaching that growth.
            </p>
          </div>
        </section>
        {result && parsed.ok ? (
          <>
            <div className="chart-grid">
              <ValueComparison result={result} input={parsed.value} />
              <section className="panel">
                <CashFlowChart result={result} />
                <p className="note">
                  You forecast the cash left after operating costs and
                  reinvestment. Negative years consume cash before financing.
                </p>
              </section>
            </div>
            <details className="panel">
              <summary>Valuation sensitivity</summary>
              <Sensitivity input={parsed.value} />
            </details>
            <details className="panel bridge">
              <summary>Business value to equity · calculation detail</summary>
              <div>
                <span>Present value · years 1–10</span>
                <strong>{money(result.forecastValueB)}</strong>
              </div>
              <div>
                <span>Present value · terminal cash flows</span>
                <strong>{money(result.discountedTerminalValueB)}</strong>
              </div>
              <div className="total">
                <span>Enterprise value</span>
                <strong>{money(result.enterpriseValueB)}</strong>
              </div>
              <div>
                <span>Net cash assumption</span>
                <strong>
                  {parsed.value.netCashB === null
                    ? "Unknown"
                    : money(parsed.value.netCashB)}
                </strong>
              </div>
              <div className="total">
                <span>Equity value</span>
                <strong>
                  {result.equityValueB === null
                    ? "Unavailable"
                    : money(result.equityValueB)}
                </strong>
              </div>
              <div>
                <span>Terminal contribution to value</span>
                <strong>
                  {result.terminalContribution === null
                    ? "Undefined"
                    : pct(result.terminalContribution)}
                </strong>
              </div>
              <div>
                <span>Lowest cumulative cash flow</span>
                <strong>{money(result.minimumCumulativeCashFlowB)}</strong>
              </div>
              {result.terminalContribution !== null &&
              result.terminalContribution > 1 ? (
                <p className="note">
                  Terminal contribution exceeds 100% because the forecast-period
                  cash flows have a negative present value.
                </p>
              ) : null}
              {result.enterpriseValueB <= 0 ? (
                <p className="note">
                  You obtain a non-positive enterprise value in this scenario.
                  This does not imply a negative common-stock price.
                </p>
              ) : null}
              <p className="note">
                Future capital raises and dilution can change the value
                attributable to current shareholders.
              </p>
            </details>
          </>
        ) : null}
        {result ? (
          <details className="panel forecast">
            <summary>Annual forecast table</summary>
            <div className="section-title">
              <div>
                <p className="eyebrow">Inspect the forecast</p>
                <h2>Annual cash flows</h2>
              </div>
              <span className="muted">
                USD billions · relative forecast years
              </span>
            </div>
            <div className="table-scroll">
              <table>
                <caption className="sr-only">
                  Forecast and terminal year cash flows in billions of US
                  dollars
                </caption>
                <thead>
                  <tr>
                    {[
                      "Year",
                      "Revenue",
                      "Growth",
                      "Gross margin",
                      "Op. expenses / rev.",
                      "Op. income",
                      "Cash tax",
                      "Reinvestment",
                      "Free cash flow",
                      "PV of cash flow",
                    ].map((title) => (
                      <th scope="col" key={title}>
                        {title}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...result.years, result.terminalYear].map((row) => (
                    <tr key={row.year}>
                      <th scope="row">
                        {row.year === 11 ? "11 · terminal" : row.year}
                      </th>
                      <td>{number(row.revenueB)}</td>
                      <td>{pct(row.growth)}</td>
                      <td>{pct(row.grossMargin)}</td>
                      <td>{pct(row.opexRatio)}</td>
                      <td>{number(row.operatingIncomeB)}</td>
                      <td>{number(row.taxB)}</td>
                      <td>{number(row.reinvestmentB)}</td>
                      <td className={row.cashFlowB < 0 ? "negative" : ""}>
                        {number(row.cashFlowB)}
                      </td>
                      <td>
                        {row.year === 11
                          ? "In terminal value"
                          : number(row.presentValueB)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ) : null}
        <details className="panel methodology">
          <summary>Model conventions and equations</summary>
          <p>
            You project ten years of unlevered cash flow. Growth stays constant
            for five years, then moves in equal steps to terminal growth in year
            11. Margins move from year-1 to year-10 inputs.
          </p>
          <p>
            You expense inference compute through gross margin and training
            compute through operating expenses. Reinvestment covers net capital
            expenditure and working capital. Keep these costs separate.
          </p>
          <details>
            <summary>Inspect the calculation</summary>
            <p className="formula">
              FCFF = operating income − cash tax − reinvestment
            </p>
            <p>
              Reinvestment = revenue increase ÷ sales-to-capital ratio. Cash tax
              applies to positive operating income; the model omits loss
              carryforwards. Year-11 reinvestment continues at terminal growth.
            </p>
            <p className="formula">
              Terminal value = year-11 FCFF ÷ (WACC − terminal growth)
            </p>
            <p>
              You discount annual cash flows at year-end and terminal value from
              the end of year 10. Negative enterprise values remain visible.
              Common-equity limited liability and future dilution need separate
              analysis.
            </p>
            <a
              href="https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/growthvaluedrivers.htm"
              target="_blank"
              rel="noreferrer"
            >
              Read the reinvestment methodology ↗
            </a>
          </details>
        </details>
        <details className="panel evidence-section" id="evidence">
          <summary>Sources and missing disclosures</summary>
          <div className="section-title">
            <div>
              <p className="eyebrow">Trace the inputs</p>
              <h2>Source ledger</h2>
            </div>
            <span className="muted">Evidence v{EVIDENCE.version}</span>
          </div>
          <div className="evidence-grid">
            {EVIDENCE.records.map((record) => (
              <article key={record.id} className="panel evidence-card">
                <div className="section-title">
                  <span
                    className={
                      record.status === "company_disclosure"
                        ? "disclosure-label"
                        : "unknown-label"
                    }
                  >
                    {record.status === "company_disclosure"
                      ? "Company disclosure"
                      : "Unavailable"}
                  </span>
                  <span className="muted">
                    {record.publishedAt ?? "Awaiting evidence"}
                  </span>
                </div>
                <h3>{record.metric}</h3>
                {record.value !== null ? (
                  <p className="evidence-value">
                    {record.relation === "greater_than" ? "> " : ""}
                    {money(record.value)}
                  </p>
                ) : null}
                <p>{record.note}</p>
                {record.excerpt ? (
                  <blockquote>“{record.excerpt}”</blockquote>
                ) : null}
                {record.sourceUrl ? (
                  <a href={record.sourceUrl} target="_blank" rel="noreferrer">
                    {record.sourceTitle} ↗
                  </a>
                ) : null}
              </article>
            ))}
          </div>
        </details>
      </div>
      {view === "risk" ? <RiskEngine /> : null}
      <footer>
        <span>
          Evidence v{EVIDENCE.version} · Valuation model v{MODEL_VERSION}
        </span>
        <span>Fixed evidence snapshot · no live data feed</span>
      </footer>
    </main>
  );
}
