export const pct = (value: number | null) =>
  value === null ? "Unavailable" : `${(value * 100).toFixed(1)}%`;

export const fixed = (value: number | null) =>
  value === null ? "Unavailable" : value.toFixed(3);
