const FORBIDDEN_KEYS = new Set([
  "condition",
  "condition_id",
  "arm",
  "treatment",
  "baseline",
  "pair_id",
  "profile_id",
  "jd_id",
]);

const FORBIDDEN_VALUES = new Set(["T", "B1", "treatment", "profile-aware"]);

export function payloadIsBlind(value) {
  const parsed = typeof value === "string" ? JSON.parse(value) : value;
  return !hasLeak(parsed);
}

function hasLeak(node) {
  if (Array.isArray(node)) return node.some(hasLeak);
  if (node && typeof node === "object") {
    for (const [key, child] of Object.entries(node)) {
      if (FORBIDDEN_KEYS.has(key) || hasLeak(child)) return true;
    }
    return false;
  }
  return typeof node === "string" && FORBIDDEN_VALUES.has(node);
}
