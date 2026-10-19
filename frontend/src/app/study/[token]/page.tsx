import { StudyClient } from "./study-client";

export default async function StudyPage({
  params,
  searchParams,
}: {
  params: Promise<{ token: string }>;
  searchParams: Promise<{ mode?: string }>;
}) {
  const { token } = await params;
  const query = await searchParams;
  const mode = query.mode === "pilot" || query.mode === "demo" ? query.mode : "main";
  return <StudyClient token={token} mode={mode} />;
}
