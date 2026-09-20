import Link from "next/link";

const links = [
  { href: "/", label: "Experiments" },
  { href: "/upload", label: "Upload pairs" },
];

export function SiteHeader() {
  return (
    <header className="border-b border-paper-line/80 bg-paper-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-end justify-between gap-6 px-4 py-5 sm:px-6">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent-deep">
            Research console
          </p>
          <Link href="/" className="font-display text-3xl tracking-tight text-ink">
            InterviewIQ
          </Link>
          <p className="mt-1 max-w-xl text-sm text-ink-mute">
            Profile-gap interview generation — experiment metrics, cost, latency,
            and ad-hoc CV/JD testing.
          </p>
        </div>
        <nav className="flex gap-1">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-md px-3 py-2 text-sm font-medium text-ink-soft transition hover:bg-accent-soft hover:text-accent-deep"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
