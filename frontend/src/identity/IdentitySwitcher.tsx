"use client";

import { useState } from "react";
import { PRESET_USER_IDS, useIdentity } from "./IdentityProvider";

const CUSTOM_OPTION = "__custom__";

/** Header dev-identity switcher: preset users + free-text entry. */
export function IdentitySwitcher() {
  const { userId, setUserId } = useIdentity();
  const isPreset = (PRESET_USER_IDS as readonly string[]).includes(userId);
  const [customValue, setCustomValue] = useState(isPreset ? "" : userId);
  const [showCustomInput, setShowCustomInput] = useState(!isPreset);

  function handleSelectChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const next = event.target.value;
    if (next === CUSTOM_OPTION) {
      setShowCustomInput(true);
      return;
    }
    setShowCustomInput(false);
    setUserId(next);
  }

  function handleCustomSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (customValue.trim()) {
      setUserId(customValue);
    }
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <label htmlFor="identity-select" className="text-neutral-500">
        User
      </label>
      <select
        id="identity-select"
        value={showCustomInput ? CUSTOM_OPTION : userId}
        onChange={handleSelectChange}
        className="rounded border border-neutral-300 bg-white px-2 py-1"
      >
        {PRESET_USER_IDS.map((preset) => (
          <option key={preset} value={preset}>
            {preset}
          </option>
        ))}
        <option value={CUSTOM_OPTION}>Custom…</option>
      </select>
      {showCustomInput && (
        <form onSubmit={handleCustomSubmit} className="flex items-center gap-1">
          <input
            aria-label="Custom user id"
            value={customValue}
            onChange={(event) => setCustomValue(event.target.value)}
            placeholder="user id"
            className="w-28 rounded border border-neutral-300 px-2 py-1"
          />
          <button
            type="submit"
            className="rounded border border-neutral-300 px-2 py-1 hover:bg-neutral-100"
          >
            Set
          </button>
        </form>
      )}
    </div>
  );
}
