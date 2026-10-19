import assert from "node:assert/strict";
import test from "node:test";

import { payloadIsBlind } from "./guard.mjs";

test("blind payload passes", () => {
  const payload = {
    items: [{ item_id: "item-001", set_x: ["How did you close the books?"], set_y: ["Tell me about SAP."] }],
  };
  assert.equal(payloadIsBlind(JSON.stringify(payload)), true);
});

test("condition label fails", () => {
  assert.equal(payloadIsBlind(JSON.stringify({ condition: "T" })), false);
  assert.equal(payloadIsBlind(JSON.stringify({ set_x: ["profile-aware"] })), false);
});
