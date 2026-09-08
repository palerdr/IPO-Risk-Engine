import { useState } from "react";
import { report, type ModelId } from "./lib/research";
import { pct, fixed } from "./lib/format";
import { downloadJson } from "./lib/download";
import { Calibration, PriceChart } from "./components/ResearchCharts";
import Valuation from "./Valuation";

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
    downloadJson(report, "ipo-risk-study-v2.json");
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
            not establish performance across the IPO population. You have no
            validated allocation policy; a peak-to-trough event can occur while
            your position remains above its entry price.
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
