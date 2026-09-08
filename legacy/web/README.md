# Anthropic valuation workbench

You can calculate a ten-year enterprise valuation and solve for the revenue growth required by a proposed IPO equity valuation. You can inspect the financial assumptions and export the result with its source evidence.

Use Node 22.13 or a later supported release.

```sh
npm ci
npm run dev
```

```sh
npm run build
npm run preview
```

The build runs TypeScript checks and 41 tests, then writes a static site to `dist/`. The application uses React and React DOM at runtime. It needs no backend or API credentials. Native HTML controls and plain CSS define the interface.

**Read the code in this order**

| File | Responsibility |
|---|---|
| `lib/valuation/engine.ts` | Input validation, cash-flow forecast, reverse solver, sensitivity |
| `lib/valuation/form.ts` | Convert form strings and percentages into model inputs |
| `lib/valuation/scenario.ts` | Attach versioned source evidence to JSON exports |
| `App.tsx` | Display inputs and results; respond to user actions |

`main.tsx` mounts the application. `data/anthropic.v1.json` contains reviewed evidence. The existing financial tests remain unchanged; interaction tests cover the workflow in a simulated DOM.

**Financial conventions**

You enter money in USD billions. The interface accepts percentages, while the engine accepts decimal rates. Year 0 contains normalized starting revenue. Growth stays constant through years 1–5, then moves in six equal steps to terminal growth in year 11. Margins move from year-1 inputs to year-10 targets over nine intervals.

```text
operating income = revenue × (gross margin − operating expense ratio)
cash tax = max(operating income, 0) × tax rate
reinvestment = revenue increase / sales-to-capital ratio
free cash flow = operating income − cash tax − reinvestment
terminal value = year-11 free cash flow / (WACC − terminal growth)
enterprise value = PV(years 1–10 cash flows) + PV(terminal value)
equity value = enterprise value + assumed net cash
```

You discount cash flows at year-end and terminal value from the end of year 10. Year-11 cash flow includes reinvestment. Net cash means cash less debt and other senior claims. A blank value blocks equity output; zero represents an explicit assumption. Per-share results need a supported diluted share count, which this evidence version lacks.

You expense inference compute through gross margin and training compute through operating expenses. Reinvestment covers net capital expenditure and working capital. The model omits tax loss carryforwards and future financing dilution.

The reverse solver uses bisection over 0–150% growth. Its tolerance is `max(1e-8, target × 1e-10)` USD billions, with a 100-iteration limit. An unbracketed target produces an error. A mathematical solution does not establish economic feasibility or a forecast probability.

The sensitivity table varies WACC by ±1 and ±2 percentage points, and terminal operating margin by ±5 and ±10 points. It changes terminal operating expenses with gross margin fixed. Invalid cells remain unavailable.

**Reproduce and update evidence**

Enter net cash `0` and proposed equity value `965`, then select Required growth. With the other default inputs, you obtain about 54.44% annual growth in years 1–5. Apply that result to reproduce the target in forward valuation.

Choose Export scenario to save inputs and outputs with the evidence dataset. Pass the exported `inputs` to `evaluateScenario` and `solveRequiredGrowth` at the recorded model version to reproduce the results. The tests check this round trip.

Anthropic's [May 28 funding announcement](https://www.anthropic.com/news/series-h) reports a $965B post-money valuation and revenue run rate above $47B. The $47B normalized starting revenue in the calculator remains an illustrative assumption. Its run-rate reference does not establish annual recognized revenue. The [June 1 announcement](https://www.anthropic.com/news/confidential-draft-s1-sec) describes a confidential draft filing. This evidence version contains no public prospectus.

Preserve version 1 when you add reviewed disclosures. Create a new evidence file and update the active import. Preserve publication dates and metric definitions. Consult [Damodaran's growth valuation guidance](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/growthvaluedrivers.htm) for the link between growth and reinvestment.

Read the [project guide](../docs/PROJECT_GUIDE.md) for demonstration steps and resume claims. The app does not save edits after a reload and has no live data refresh. The Sites project has no published version; the prior source upload failed on a connection timeout.

**Original risk engine in the dashboard**

Open Risk engine to inspect a saved run from the Python engine on synthetic prices. You can see the feature-to-policy flow and the required post-listing inputs. The Anthropic valuation view does not call the risk engine; the evidence snapshot contains no Anthropic post-listing bars.

You can refresh the saved walkthrough from the repository root:

```sh
uv run python -m scripts.demo_risk_engine > web/data/risk-demo.json
```

The Python test suite checks this snapshot against a fresh run. The browser reads the saved JSON and makes no live inference call.
