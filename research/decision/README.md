# Allocator decision study

You have no demonstrated allocation-policy benefit from the current classifiers. You can reproduce the critique with `uv run --extra challengers ipo-challengers audit` and inspect the [executed companion notebook](audit.ipynb). The audit preserves the frozen study and uses its held-out issuer IDs.

You compare single-feature rankings on matching cases. For pre-IPO offer-price comparisons, you exclude 27 cases without an offer price and rescore the models on the remaining 953 cases. You selected the features and directions after viewing the study, so these comparisons remain retrospective diagnostics.

You can reproduce 342 of 496 pre-IPO drawdown events with a nonnegative close-20/offer-price proxy and 87 of 446 day-20 events with a nonnegative close-40/close-20 proxy. The pre-IPO calculation divides vendor-adjusted closes by nominal offer prices. You need a common share basis before calling those ratios allocator returns. You should also distinguish a nonnegative terminal return from a path that stayed above entry.

You will study an allocator who can accept or decline a fixed allocation at the final offer price and exit at close 20. You will evaluate continuous cash-relative returns and downside-weighted policy loss. The draft [protocol](protocol.json) defines this actor and the required data checks. You have no fitted allocator model or execution backtest. The protocol remains a design draft until you freeze implementation details and pass its activation checks.

You need document-level availability cutoffs for EDGAR features. An acceptance timestamp establishes when the SEC accepted a document; a later final prospectus cannot establish what an allocator knew before the acceptance deadline. Rule 424(b)(4) permits a filing after pricing. You can use a later document for reconciliation if you keep the earlier source for each prediction-time input. [Rule 424](https://www.govinfo.gov/content/pkg/CFR-2025-title17-vol3/pdf/CFR-2025-title17-vol3-sec230-424.pdf)

You must derive event dates from the applicable terms. FINRA specifies a ten-day research quiet-period minimum with exemptions, including emerging growth companies. Bowhead's prospectus specifies a 30-day overallotment option and permits underwriters to stop stabilization at any time. Neither supports a universal session-25 or session-30 feature. [FINRA Rule 2241](https://www.finra.org/rules-guidance/rulebooks/finra-rules/2241), [Bowhead prospectus](https://ir.bowheadspecialty.com/sec-filings/all-sec-filings/content/0001628280-24-025249/bowheadspecialtyholdingsin.htm)

You cannot pre-register earlier 2026 outcomes on September 7, 2026. You must start the prospective cohort after the final protocol and implementation commit, then preserve dated forecasts before their outcome windows. You can use prior outcomes for development with that status stated.

You should build the Anthropic presentation around comparable deal terms and offer-based price paths. The current registry does not establish the claimed count of thirty comparable large-cap technology IPOs. You need explicit sector and size rules before you count that cohort.

You can rerun the notebook from the repository root with the optional notebook tools:

```sh
uv run --extra challengers --with nbconvert --with ipykernel python -m jupyter nbconvert --execute --to notebook --inplace research/decision/audit.ipynb
```

You retain `audit.py` as the notebook's entry point. The CLI and notebook call the same audit implementation.
