import type { Prediction, ResearchModel } from "../lib/research";
import { pct } from "../lib/format";

export function PriceChart({
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

export function Calibration({ model }: { model: ResearchModel }) {
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
