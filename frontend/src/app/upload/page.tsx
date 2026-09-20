import type { UploadMeta } from "@/lib/types";
import { UploadForm } from "@/components/UploadForm";
import { fetchUploads } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function UploadPage() {
  let uploads: UploadMeta[] = [];
  let error: string | null = null;
  try {
    uploads = await fetchUploads();
  } catch (err) {
    error = err instanceof Error ? err.message : "API unavailable";
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="font-display text-4xl tracking-tight text-ink">
          Test uploads
        </h1>
        <p className="max-w-2xl text-ink-mute">
          Add new CV and job-description pairs for manual testing. Files are
          stored locally for later baseline / InterviewIQ runs.
        </p>
      </section>

      {error ? (
        <div className="rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          API unavailable: {error}
        </div>
      ) : null}

      <UploadForm initialUploads={uploads} />
    </div>
  );
}
