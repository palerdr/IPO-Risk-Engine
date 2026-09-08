// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import App from "../App";
import riskDemo from "../data/risk-demo.json";

afterEach(cleanup);

describe("focused dashboard presentation", () => {
  it("keeps four decision inputs in view and preserves the detailed model behind disclosures", () => {
    render(<App />);
    const assumptions = screen.getByRole("region", {
      name: "Scenario assumptions",
    });
    expect(assumptions.querySelectorAll(".primary-fields input")).toHaveLength(
      4,
    );
    const advanced = assumptions.querySelector("details")!;
    expect(advanced.open).toBe(false);
    expect(advanced.querySelectorAll("input")).toHaveLength(8);
    for (const title of [
      "Annual forecast table",
      "Sources and missing disclosures",
    ]) {
      expect(screen.getByText(title).closest("details")!.open).toBe(false);
    }
    expect(screen.getByText("No score")).toBeTruthy();
  });

  it("loads a disclosed cash assumption and compares equity values on the same basis", () => {
    render(<App />);
    fireEvent.click(
      screen.getByRole("button", { name: "Load example · $0 net cash" }),
    );
    expect(
      (screen.getByLabelText("Net cash assumption") as HTMLInputElement).value,
    ).toBe("0");
    expect(
      (
        screen.getByLabelText(
          "Proposed IPO equity valuation",
        ) as HTMLInputElement
      ).value,
    ).toBe("965");
    expect(
      screen.getByText("Your modeled value falls below the entry assumption"),
    ).toBeTruthy();
    expect(screen.getByText("Equity values · USD billions")).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "Required growth" }));
    expect(document.querySelector(".reverse-value")!.textContent).toContain(
      "54.4%",
    );
  });

  it("explains the original engine without assigning its synthetic score to Anthropic", () => {
    render(<App />);
    fireEvent.click(
      screen.getByRole("button", { name: "Risk engine Post-IPO" }),
    );
    const walkthrough = screen.getByRole("region", {
      name: "Risk engine walkthrough",
    });
    expect(
      within(walkthrough).getByText("Anthropic: no score available."),
    ).toBeTruthy();
    expect(walkthrough.textContent).toContain("61 sessions");
    expect(walkthrough.textContent).toContain("synthetic prices");
    expect(
      within(walkthrough).getByText(`${riskDemo.example.symbol} · synthetic`),
    ).toBeTruthy();
    expect(within(walkthrough).getByText("18.2%")).toBeTruthy();
    expect(
      screen.queryByRole("region", { name: "Scenario assumptions" }),
    ).toBeNull();
    expect(
      within(walkthrough).queryByRole("button", { name: /Run|Predict/ }),
    ).toBeNull();
  });

  it("retains valuation edits after visiting the risk-engine walkthrough", () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Revenue growth · years 1–5"), {
      target: { value: "45" },
    });
    const value = document.querySelector(".valuation-hero .value")!.textContent;
    fireEvent.click(
      screen.getByRole("button", { name: "Risk engine Post-IPO" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Valuation" }));
    expect(
      (screen.getByLabelText("Revenue growth · years 1–5") as HTMLInputElement)
        .value,
    ).toBe("45");
    expect(document.querySelector(".valuation-hero .value")!.textContent).toBe(
      value,
    );
  });
});
