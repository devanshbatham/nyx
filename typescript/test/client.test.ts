import assert from "node:assert/strict";
import test from "node:test";

import { NyxApiError, NyxClient } from "../src/index.js";

test("sends authenticated systemone requests", async () => {
  const client = new NyxClient({
    apiKey: "secret",
    baseUrl: "https://nyx.invalid/",
    fetch: async (input, init) => {
      assert.equal(input, "https://nyx.invalid/v1/systemone");
      assert.equal(new Headers(init?.headers).get("Authorization"), "Bearer secret");
      assert.deepEqual(JSON.parse(String(init?.body)), {
        state: "hello",
        questions: { sentiment: { type: "noul", instructions: "Positive?" } },
        model: "nyx",
      });
      return Response.json({ model: "nyx", answers: {}, usage: { input_tokens: 1, output_tokens: 1 } });
    },
  });
  const response = await client.systemone({
    state: "hello",
    questions: { sentiment: { type: "noul", instructions: "Positive?" } },
  });
  assert.equal(response.model, "nyx");
});

test("raises typed HTTP errors", async () => {
  const client = new NyxClient({
    apiKey: "secret",
    fetch: async () => new Response("denied", { status: 401 }),
  });
  await assert.rejects(client.models(), (error: unknown) => {
    assert.ok(error instanceof NyxApiError);
    assert.equal(error.status, 401);
    assert.equal(error.body, "denied");
    return true;
  });
});
