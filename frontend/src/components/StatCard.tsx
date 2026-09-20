type StatCardProps = {
  label: string;
  value: string;
  hint?: string;
};

export function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="rounded-xl border border-paper-line bg-paper-card p-4 shadow-panel">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-mute">
        {label}
      </p>
      <p className="mt-2 font-display text-2xl text-ink">{value}</p>
      {hint ? <p className="mt-1 text-xs text-ink-mute">{hint}</p> : null}
    </div>
  );
}
