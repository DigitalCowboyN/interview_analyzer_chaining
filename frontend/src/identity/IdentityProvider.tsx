"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

const STORAGE_KEY = "interview-analyzer:user-id";
const DEFAULT_USER_ID = "dev";

/** A small set of preset dev identities offered in the switcher, alongside free-text entry. */
export const PRESET_USER_IDS = ["dev", "reviewer", "admin"] as const;

interface IdentityContextValue {
  userId: string;
  setUserId: (userId: string) => void;
}

const IdentityContext = createContext<IdentityContextValue | null>(null);

function readStoredUserId(): string {
  if (typeof window === "undefined") return DEFAULT_USER_ID;
  try {
    return window.localStorage.getItem(STORAGE_KEY) || DEFAULT_USER_ID;
  } catch {
    // localStorage can throw in private-browsing / disabled-storage contexts.
    return DEFAULT_USER_ID;
  }
}

/**
 * Dev identity provider: no auth, just a user id persisted in localStorage
 * and sent as `X-User-ID` on every API request. Default "dev".
 */
export function IdentityProvider({ children }: { children: ReactNode }) {
  // Start with the default on both server and first client render so hydration matches;
  // sync from localStorage on mount.
  const [userId, setUserIdState] = useState<string>(DEFAULT_USER_ID);

  useEffect(() => {
    setUserIdState(readStoredUserId());
  }, []);

  const setUserId = useCallback((next: string) => {
    const trimmed = next.trim();
    if (!trimmed) return;
    setUserIdState(trimmed);
    try {
      window.localStorage.setItem(STORAGE_KEY, trimmed);
    } catch {
      // Ignore persistence failures — in-memory state still updates.
    }
  }, []);

  const value = useMemo(() => ({ userId, setUserId }), [userId, setUserId]);

  return (
    <IdentityContext.Provider value={value}>
      {children}
    </IdentityContext.Provider>
  );
}

/** Read/set the current dev user id. Must be used within an IdentityProvider. */
export function useIdentity(): IdentityContextValue {
  const ctx = useContext(IdentityContext);
  if (!ctx) {
    throw new Error("useIdentity must be used within an IdentityProvider");
  }
  return ctx;
}

/** Non-hook accessor for the current user id — used by apiFetch outside of React. */
export function getCurrentUserId(): string {
  return readStoredUserId();
}
