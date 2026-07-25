import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { NoticeText } from "@/components/NoticeText";

/**
 * The shared notice renderer (Rider A item 1) that replaces the four
 * near-identical copies previously living in LineDetailPanel (x2),
 * PersonPicker, and WorklistRows. Pins the exact markup contract those four
 * copies shared: role="alert" for conflict/network, role="status" for
 * timeout (non-error tone), and the kind→color mapping.
 */
describe("NoticeText", () => {
  it("renders nothing when notice is null", () => {
    const { container } = render(<NoticeText notice={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a timeout notice with role=status and the neutral tone", () => {
    render(<NoticeText notice={{ kind: "timeout", message: "Still processing — check back later." }} />);
    const el = screen.getByRole("status");
    expect(el).toHaveTextContent("Still processing — check back later.");
    expect(el.className).toContain("text-neutral-500");
    expect(el.className).not.toContain("text-red-600");
  });

  it("renders a conflict notice with role=alert and the red tone", () => {
    render(<NoticeText notice={{ kind: "conflict", message: "Speaker was already renamed." }} />);
    const el = screen.getByRole("alert");
    expect(el).toHaveTextContent("Speaker was already renamed.");
    expect(el.className).toContain("text-red-600");
  });

  it("renders a network notice with role=alert and the red tone", () => {
    render(<NoticeText notice={{ kind: "network", message: "Could not reach the server." }} />);
    const el = screen.getByRole("alert");
    expect(el.className).toContain("text-red-600");
  });

  it("defaults to mt-1 and accepts a className override for the mt-2 call sites", () => {
    const { rerender } = render(
      <NoticeText notice={{ kind: "conflict", message: "x" }} />,
    );
    expect(screen.getByRole("alert").className).toContain("mt-1");

    rerender(<NoticeText notice={{ kind: "conflict", message: "x" }} className="mt-2" />);
    expect(screen.getByRole("alert").className).toContain("mt-2");
  });
});
