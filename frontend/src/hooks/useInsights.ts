import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";
import type { Insight } from "@/lib/insights";

const LENSES = ["meeting_minutes", "persona"] as const;

interface LensItemRow {
  item_id: string;
  node_type: string;
  confidence: number;
  locked: boolean;
  supporting_fragment_ids: string[];
  fields: Record<string, unknown>;
}

/** Every lens item for an interview across both lenses (existing
 * `GET /interviews/{id}/lenses/{lens}/items`; src/api/routers/queries.py).
 * Uses `Promise.allSettled` rather than `Promise.all` so a single lens
 * being down (e.g. that lens hasn't run yet) doesn't blank out the other
 * lens's insights — only throws (isError) when BOTH lenses reject. */
export function useInsights(interviewId: string) {
  return useQuery({
    queryKey: queryKeys.insights(interviewId),
    queryFn: async (): Promise<Insight[]> => {
      const settled = await Promise.allSettled(
        LENSES.map(async (lens): Promise<Insight[]> => {
          const data = (await apiGet("/interviews/{interview_id}/lenses/{lens}/items", {
            params: { interview_id: interviewId, lens },
            query: { limit: 500 },
          })) as { items: LensItemRow[] };
          return data.items.map((row) => ({
            item_id: row.item_id,
            node_type: row.node_type,
            lens,
            text: String(row.fields?.text ?? ""),
            confidence: row.confidence,
            locked: row.locked,
            supporting_fragment_ids: row.supporting_fragment_ids ?? [],
          }));
        }),
      );

      const fulfilled = settled.filter(
        (result): result is PromiseFulfilledResult<Insight[]> => result.status === "fulfilled",
      );
      if (fulfilled.length === 0) {
        const firstRejected = settled.find(
          (result): result is PromiseRejectedResult => result.status === "rejected",
        );
        throw firstRejected?.reason ?? new Error("Both lenses failed to load.");
      }
      return fulfilled.flatMap((result) => result.value);
    },
    enabled: Boolean(interviewId),
  });
}
