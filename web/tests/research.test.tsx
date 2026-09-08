// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import App from "../App";
import study from "../data/research.json";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("risk research workflow", () => {
  it("leads with the real held-out study and hides the replay outcome", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "IPO drawdown risk",
    );
    expect(
      screen.getByRole("img", { name: /First 20 sessions only/ }),
    ).toBeTruthy();
    expect(screen.queryByText("The drawdown event occurred.")).toBeNull();
    expect(screen.queryByText("The drawdown event did not occur.")).toBeNull();
    expect(screen.getByLabelText("Listing").getAttribute("id")).toBe("listing");
    expect(document.body.textContent).not.toMatch(
      /FOLD|SIZE_UP|SMALL_BET|RIVER|SYNTHETIC/,
    );
  });

  it("reveals the true result and hides it again after changing listings", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Reveal outcome" }));
    expect(
      screen.getByRole("img", { name: /Observed and subsequent prices/ }),
    ).toBeTruthy();
    const text = study.predictions[0].event
      ? "The drawdown event occurred."
      : "The drawdown event did not occur.";
    expect(screen.getByText(text)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Listing"), {
      target: { value: study.predictions[1].symbol },
    });
    expect(screen.getByRole("button", { name: "Reveal outcome" })).toBeTruthy();
    expect(screen.queryByText(text)).toBeNull();
  });

  it("uses the selected model and cutoff for the screening counts", () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Prediction model"), {
      target: { value: "boosted_trees" },
    });
    fireEvent.change(screen.getByLabelText("Probability cutoff"), {
      target: { value: "0.5" },
    });
    const selected = study.models.find(
      (model) => model.id === "boosted_trees",
    )!;
    const counts = selected.screening.find((row) => row.threshold === 0.5)!;
    const panel = document.querySelector(".screening")! as HTMLElement;
    expect(
      within(panel).getByText(`${counts.flagged} / ${selected.n}`),
    ).toBeTruthy();
    expect(
      document.querySelector(".replay-probability strong")!.textContent,
    ).toBe(
      `${(study.predictions[0].probabilities.boosted_trees * 100).toFixed(1)}%`,
    );
  });

  it("keeps Anthropic valuation separate and runs the reverse solver", () => {
    render(<App />);
    fireEvent.click(
      screen.getByRole("button", { name: "Anthropic valuation" }),
    );
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "Anthropic valuation",
    );
    expect(
      screen.getByText(/trading-risk score remains unavailable/),
    ).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: "Load example · $0 net cash" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Apply required growth" }),
    );
    expect(screen.getAllByText("$965B").length).toBeGreaterThanOrEqual(1);
    fireEvent.click(screen.getByRole("button", { name: "Risk research" }));
    expect(
      screen.getByRole("region", { name: "Historical replay" }),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Anthropic valuation" }));
    expect((screen.getByLabelText("Assumed entry equity value ($B)") as HTMLInputElement).value).toBe("965");
  });

  it("exports the exact study with provenance and frozen model parameters", async () => {
    render(<App />);
    let blob: Blob | undefined;
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL(value: Blob) {
          blob = value;
          return "blob:study";
        }
        static revokeObjectURL() {}
      },
    );
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    fireEvent.click(screen.getByRole("button", { name: "Export study" }));
    expect(click).toHaveBeenCalledOnce();
    const contents = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(blob!);
    });
    expect(JSON.parse(contents)).toEqual(study);
  });
});
