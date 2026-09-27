"use client";

import { useParams } from "next/navigation";
import type { ReactNode } from "react";
import { ProjectTabs } from "@/components/ProjectTabs";

export default function ProjectLayout({ children }: { children: ReactNode }) {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <>
      <ProjectTabs projectId={projectId} />
      {children}
    </>
  );
}
