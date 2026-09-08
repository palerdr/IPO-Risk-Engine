import { useState } from "react";
import report from "./data/research.json";
import Valuation from "./Valuation";

type Prediction = (typeof report.predictions)[number];
type ModelId = keyof Prediction["probabilities"];
const pct = (value: number | null) =>
  value === null ? "Unavailable" : `${(value * 100).toFixed(1)}%`;
const fixed = (value: number | null) =>
  value === null ? "Unavailable" : value.toFixed(3);
const names: Record<string, string> = {
  return_19_sessions: "Return since first close",
  return_5_sessions: "Last five-session return",
  daily_volatility: "Daily return volatility",
  observed_drawdown: "Observed maximum drawdown",
  worst_daily_return: "Worst daily return",
  volume_change: "Volume change: last / first five sessions",
  market_return: "SPY return over the observation window",
  market_volatility: "SPY daily volatility",
};

function PriceChart({
  prediction,
  revealed,
}: {
  prediction: Prediction;
  revealed: boolean;
}) {
  const visible = prediction.path.slice(0, revealed ? 40 : 20);
  const min = Math.min(80, ...visible.map((point) => point.value)) * 0.97;
  const max = Math.max(110, ...visible.map((point) => point.value)) * 1.03;
  const x = (i: number) => 55 + (i / 39) * 625;
  const y = (value: number) => 228 - ((value - min) / (max - min)) * 185;
  const line = (from: number, to: number) =>
    prediction.path
      .slice(from, to)
      .map((point, index) => `${x(from + index)},${y(point.value)}`)
      .join(" ");
  return (
    <figure className="price-chart">
      <figcaption>
        <h2>Price path around the prediction</h2>
        <span className="note">Adjusted close · session 20 = 100</span>
      </figcaption>
      <svg
        viewBox="0 0 720 290"
        role="img"
        aria-label={
          revealed
            ? "Observed and subsequent prices over 40 trading sessions"
            : "First 20 sessions only; subsequent prices remain hidden"
        }
      >
        {[min, (min + max) / 2, max].map((value) => (
          <g key={value}>
            <line
              x1="55"
              x2="680"
              y1={y(value)}
              y2={y(value)}
              stroke="var(--border)"
            />
            <text x="43" y={y(value) + 5} textAnchor="end">
              {value.toFixed(0)}
            </text>
          </g>
        ))}
        <line
          x1={x(19)}
          x2={x(19)}
          y1="24"
          y2="240"
          stroke="var(--muted)"
          strokeDasharray="5 5"
        />
        <text x={x(19) - 8} y="18" textAnchor="end">
          Prediction date
        </text>
        <polyline
          points={line(0, 20)}
          stroke="var(--accent)"
          strokeWidth="3"
          fill="none"
        />
        {revealed ? (
          <polyline
            points={line(19, 40)}
            stroke="#eac077"
            strokeWidth="3"
            fill="none"
          />
        ) : (
          <text x="520" y="133" textAnchor="middle">
            Future prices hidden
          </text>
        )}
        {[0, 9, 19, 29, 39].map((index) => (
          <text key={index} x={x(index)} y="264" textAnchor="middle">
            {index + 1}
          </text>
        ))}
        <text x="368" y="286" textAnchor="middle">
          Trading session since listing
        </text>
      </svg>
    </figure>
  );
}

function Calibration({ model }: { model: (typeof report.models)[number] }) {
  const x = (value: number) => 52 + value * 285;
  const y = (value: number) => 242 - value * 190;
  return (
    <figure className="calibration-chart">
      <figcaption>
        <h2>Probability calibration</h2>
        <p className="note">
          Compare predicted probabilities with observed event rates.
        </p>
      </figcaption>
      <svg
        viewBox="0 0 410 300"
        role="img"
        aria-label={`Calibration plot for ${model.name}; the table below contains counts and values`}
      >
        {[0, 0.5, 1].map((value) => (
          <g key={value}>
            <line
              x1="52"
              x2="337"
              y1={y(value)}
              y2={y(value)}
              stroke="var(--border)"
            />
            <text x="40" y={y(value) + 4} textAnchor="end">
              {`${Math.round(value * 100)}%`}
            </text>
            <text x={x(value)} y="264" textAnchor="middle">
              {`${Math.round(value * 100)}%`}
            </text>
          </g>
        ))}
        <line
          x1={x(0)}
          x2={x(1)}
          y1={y(0)}
          y2={y(1)}
          stroke="var(--muted)"
          strokeDasharray="5 5"
        />
        {model.calibration.map((bin) => (
          <g key={bin.lower}>
            <circle
              cx={x(bin.mean_probability)}
              cy={y(bin.event_rate)}
              r="6"
              fill="var(--accent)"
            />
            <title>{`${bin.n} listings: predicted ${pct(bin.mean_probability)}, observed ${pct(bin.event_rate)}`}</title>
          </g>
        ))}
        <text x="195" y="290" textAnchor="middle">
          Predicted probability
        </text>
        <text x="52" y="25">
          Observed event rate
        </text>
      </svg>
      <details>
        <summary>Bin counts and values</summary>
        <div className="table-scroll">
          <table>
            <caption>Calibration bins</caption>
            <thead>
              <tr>
                <th>Listings</th>
                <th>Mean probability</th>
                <th>Observed rate</th>
              </tr>
            </thead>
            <tbody>
              {model.calibration.map((bin) => (
                <tr key={bin.lower}>
                  <th>{bin.n}</th>
                  <td>{pct(bin.mean_probability)}</td>
                  <td>{pct(bin.event_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

export default function App() {
  const [view, setView] = useState("research");
  const [selectedSymbol, setSelectedSymbol] = useState(
    report.predictions[0].symbol,
  );
  const [modelId, setModelId] = useState<ModelId>("logistic");
  const [revealed, setRevealed] = useState(false);
  const [threshold, setThreshold] = useState(0.3);
  const [downloadMessage, setDownloadMessage] = useState("");
  const prediction = report.predictions.find(
    (row) => row.symbol === selectedSymbol,
  )!;
  const model = report.models.find((row) => row.id === modelId)!;
  const fold = report.folds.find((row) => row.id === prediction.fold)!;
  const screen = model.screening.find((row) => row.threshold === threshold)!;
  const logistic = report.models.find((row) => row.id === "logistic")!;
  const baseline = report.models.find((row) => row.id === "base_rate")!;
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "ipo-risk-study-v2.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setDownloadMessage(
      "You exported the study with predictions and frozen logistic models.",
    );
  }
  return (
    <main className="workbench">
      <header className="masthead">
        <span className="brand">
          <span className="brand-icon" aria-hidden="true">
            ↗
          </span>{" "}
          IPO / RESEARCH
        </span>
        <span className="snapshot-date">
          Frozen price snapshot · {report.provenance.retrieved_at.slice(0, 10)}
        </span>
      </header>
      <nav className="view-controls app-nav" aria-label="Research views">
        <button
          aria-pressed={view === "research"}
          onClick={() => setView("research")}
        >
          Risk research
        </button>
        <button
          aria-pressed={view === "valuation"}
          onClick={() => setView("valuation")}
        >
          Anthropic valuation
        </button>
      </nav>
      <div hidden={view !== "valuation"}>
        <Valuation />
      </div>
      <div hidden={view !== "research"}>
        <section className="intro research-intro">
          <div>
            <h1>IPO drawdown risk</h1>
            <p>
              You observe 20 trading sessions, then estimate the chance of a 20%
              drawdown over the next 20.
            </p>
          </div>
          <button className="outline" onClick={download}>
            Export study
          </button>
        </section>
        <div className="study-notice">
          <span className="badge">HISTORICAL RESEARCH</span>
          <p>
            You are viewing a curated sample of real listings. These results do
            not establish performance across the IPO population.
          </p>
        </div>
        <div className="metric-grid" aria-label="Study summary">
          <div className="metric-card">
            <p>Held-out listings</p>
            <strong>{report.cohort.evaluated}</strong>
            <span>
              {report.cohort.eligible} eligible ·{" "}
              {report.cohort.exclusions.length} excluded
            </span>
          </div>
          <div className="metric-card">
            <p>Observed drawdown events</p>
            <strong>
              {logistic.events} / {logistic.n}
            </strong>
            <span>At least 20% within the target window</span>
          </div>
          <div className="metric-card">
            <p>Logistic Brier score</p>
            <strong>{fixed(logistic.brier)}</strong>
            <span>
              {fixed(baseline.brier)} for the training-rate baseline · lower is
              better
            </span>
          </div>
          <div className="metric-card">
            <p>Brier skill versus baseline</p>
            <strong>{pct(logistic.brier_skill)}</strong>
            <span>
              Exploratory interval:{" "}
              {pct(logistic.brier_skill_interval?.[0] ?? null)} to{" "}
              {pct(logistic.brier_skill_interval?.[1] ?? null)}
            </span>
          </div>
        </div>
        <section className="panel replay-panel" aria-label="Historical replay">
          <div className="section-title">
            <div>
              <h2>Replay a listing</h2>
              <p className="note">
                You see the prediction before you reveal the outcome. The
                default case is the first held-out listing by observation date.
              </p>
            </div>
            <span className="badge">OUT-OF-SAMPLE</span>
          </div>
          <div className="replay-layout">
            <div className="replay-controls">
              <div className="field">
                <label htmlFor="listing">Listing</label>
                <select
                  id="listing"
                  value={selectedSymbol}
                  onChange={(event) => {
                    setSelectedSymbol(event.target.value);
                    setRevealed(false);
                  }}
                >
                  {report.predictions.map((row) => (
                    <option key={row.symbol} value={row.symbol}>
                      {row.symbol} · {row.as_of}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="model">Prediction model</label>
                <select
                  id="model"
                  value={modelId}
                  onChange={(event) =>
                    setModelId(event.target.value as ModelId)
                  }
                >
                  {report.models.map((row) => (
                    <option key={row.id} value={row.id}>
                      {row.name}
                      {row.id === "logistic" ? " · primary" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <p className="note">
                Prediction as of {prediction.as_of}
                <br />
                Training outcomes end {fold.train_label_end}
              </p>
              <div className="replay-probability">
                <span>Estimated drawdown-event probability</span>
                <strong>{pct(prediction.probabilities[modelId])}</strong>
                <p className="note">
                  Training event rate: {pct(prediction.probabilities.base_rate)}
                  . You have {fold.train_count} training listings for this fold.
                </p>
              </div>
              <button onClick={() => setRevealed(!revealed)}>
                {revealed ? "Hide outcome" : "Reveal outcome"}
              </button>
              {revealed && (
                <div className="outcome" role="status">
                  <strong>
                    {prediction.event
                      ? "The drawdown event occurred."
                      : "The drawdown event did not occur."}
                  </strong>
                  <p>
                    Maximum drawdown: {pct(prediction.drawdown)}
                    <br />
                    Outcome window ends {prediction.label_end}.
                  </p>
                </div>
              )}
            </div>
            <PriceChart prediction={prediction} revealed={revealed} />
          </div>
          <details className="replay-audit">
            <summary>Inspect the observed inputs and source</summary>
            <p className="note">
              You calculate these inputs from sessions 1–20. Future prices do
              not enter the features. Sector metadata is display-only.
            </p>
            <div className="table-scroll">
              <table>
                <caption>Observed features for {prediction.symbol}</caption>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(prediction.features).map(([key, value]) => (
                    <tr key={key}>
                      <th>{names[key]}</th>
                      <td>{pct(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              <a href={prediction.source_url} target="_blank" rel="noreferrer">
                Yahoo Finance price history for {prediction.symbol}
              </a>
            </p>
            <p className="note">
              Listing date: {prediction.listing_date}. Sector:{" "}
              {prediction.sector}. Retrieval:{" "}
              {prediction.retrieved_at.slice(0, 10)}. You use the vendor's
              current adjusted-price history.
            </p>
          </details>
        </section>
        <section className="panel" aria-label="Model comparison">
          <h2>Compare the model families</h2>
          <p className="note">
            You evaluate the same held-out listings with fixed model settings.
            Logistic regression remains the primary model; you do not select a
            winner from these test results.
          </p>
          <div className="table-scroll">
            <table>
              <caption>Held-out probability forecast quality</caption>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Brier ↓</th>
                  <th>Log loss ↓</th>
                  <th>Average precision ↑</th>
                  <th>Brier skill ↑</th>
                </tr>
              </thead>
              <tbody>
                {report.models.map((row) => (
                  <tr
                    key={row.id}
                    className={row.id === "logistic" ? "primary-model" : ""}
                  >
                    <th>
                      {row.name}
                      {row.id === "logistic" ? " · primary" : ""}
                    </th>
                    <td>{fixed(row.brier)}</td>
                    <td>{fixed(row.log_loss)}</td>
                    <td>{fixed(row.average_precision)}</td>
                    <td>{pct(row.brier_skill)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <details className="panel">
          <summary>Calibration and screening tradeoffs</summary>
          <p className="note">
            Selected model: {model.name}. You can change the model in the replay
            controls.
          </p>
          <div className="chart-grid">
            <Calibration model={model} />
            <section className="screening">
              <h2>Screening at a fixed cutoff</h2>
              <p className="note">
                You flag a listing when its event probability meets the cutoff.
                You do not assign a position size.
              </p>
              <div className="field">
                <label htmlFor="threshold">Probability cutoff</label>
                <select
                  id="threshold"
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                >
                  {model.screening.map((row) => (
                    <option key={row.threshold} value={row.threshold}>
                      {pct(row.threshold)}
                    </option>
                  ))}
                </select>
              </div>
              <dl className="stat-list">
                <div>
                  <dt>Listings flagged</dt>
                  <dd>
                    {screen.flagged} / {model.n}
                  </dd>
                </div>
                <div>
                  <dt>Events caught</dt>
                  <dd>
                    {screen.caught} / {screen.events}
                  </dd>
                </div>
                <div>
                  <dt>Precision among flagged listings</dt>
                  <dd>{pct(screen.precision)}</dd>
                </div>
                <div>
                  <dt>Severe events missed (≥30% drawdown)</dt>
                  <dd>
                    {screen.missed_severe} / {screen.severe_events}
                  </dd>
                </div>
              </dl>
              <p className="note">
                You can inspect the tradeoff without claiming an enforced risk
                cap.
              </p>
            </section>
          </div>
        </details>
        <details className="panel">
          <summary>Chronological splits and model settings</summary>
          <p>{report.protocol.split}</p>
          <div className="table-scroll">
            <table>
              <caption>Temporal evaluation folds</caption>
              <thead>
                <tr>
                  <th>Fold</th>
                  <th>Training outcomes end</th>
                  <th>Test observations</th>
                  <th>Train / test</th>
                  <th>Unmatured labels excluded</th>
                </tr>
              </thead>
              <tbody>
                {report.folds.map((row) => (
                  <tr key={row.id}>
                    <th>{row.id}</th>
                    <td>{row.train_label_end}</td>
                    <td>
                      {row.test_start} to {row.test_end}
                    </td>
                    <td>
                      {row.train_count} / {row.test_count}
                    </td>
                    <td>{row.purged_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>{report.protocol.calibration}</p>
          <p className="note">{report.protocol.uncertainty}</p>
          <pre>{JSON.stringify(report.model_config, null, 2)}</pre>
        </details>
        <details className="panel">
          <summary>
            Data coverage and exclusions · {report.cohort.exclusions.length}{" "}
            listings
          </summary>
          <p>{report.cohort.selection}</p>
          <p className="note">
            You retain {report.cohort.eligible} of {report.cohort.requested}{" "}
            registry entries after checking dates and daily sessions. You
            evaluate the later date blocks.
          </p>
          <div className="table-scroll">
            <table>
              <caption>Excluded listings</caption>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Registry listing date</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {report.cohort.exclusions.map((row) => (
                  <tr key={row.symbol}>
                    <th>{row.symbol}</th>
                    <td>{row.listing_date}</td>
                    <td className="reason-cell">{row.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <details className="panel">
          <summary>Study limits and reproduction</summary>
          {report.limitations.map((text) => (
            <p key={text}>{text}</p>
          ))}
          <p className="note">
            You evaluate adjusted-close maximum drawdown, including the
            prediction-date close as the initial peak. Daily closing prices do
            not capture intraday extremes.
          </p>
          <p>You can reproduce this report from the frozen input file:</p>
          <pre>uv run ipo-research evaluate</pre>
          <p className="note">
            Input SHA-256:{" "}
            <span className="hash">{report.provenance.input_sha256}</span>
          </p>
        </details>
      </div>
      <p className="sr-only" role="status">
        {downloadMessage}
      </p>
      <footer>
        <span>IPO Risk Research · study v{report.schema_version}</span>
        <span>Frozen data · no live inference or order execution</span>
      </footer>
    </main>
  );
}
