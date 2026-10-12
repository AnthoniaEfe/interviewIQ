type Props = {
  status: string | null | undefined;
};

export function StatusPill({ status }: Props) {
  const tone =
    status === "completed" ||
    status === "SCALE" ||
    status === "win" ||
    status === "HOLD"
      ? "bg-accent-soft text-accent-deep"
      : status === "failed" ||
          status === "DIAGNOSE" ||
          status === "lose" ||
          status === "FAIL"
        ? "bg-warn-soft text-warn"
        : status === "tie" || status === "SCALE_BLOCKED_BY_BUDGET"
          ? "bg-paper text-ink-mute"
          : "bg-paper text-ink-mute";
  return (
    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${tone}`}>
      {status || "unknown"}
    </span>
  );
}
