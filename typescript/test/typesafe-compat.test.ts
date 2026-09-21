import assert from "node:assert/strict";
import test from "node:test";

import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

test("official TypeSafe JavaScript SDK is drop-in", async () => {
  const seen: Array<{ url: string; body?: Record<string, unknown> }> = [];
  const fetchMock = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const authorization = new Headers(init?.headers).get("Authorization");
    assert.equal(authorization, "Bearer secret");
    if (url.endsWith("/v1/systemone")) {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      seen.push({ url, body });
      return Response.json(
        {
          model: "nyx",
          answers: {
            route: { type: "choice", choice: "billing", confidence: 0.9, probabilities: { billing: 0.9, other: 0.1 } },
            urgency: { type: "score", score: 1.8, confidence: 0.8, legend: { "0": "low", "1": "medium", "2": "high" }, probabilities: { "0": 0.1, "1": 0.0, "2": 0.9 } },
            accepted: { type: "noul", noul: 0.75 },
          },
          usage: { input_tokens: 42, output_tokens: 3 },
        },
        { headers: { "x-typesafe-request-id": "req-systemone" } },
      );
    }
    assert.ok(url.endsWith("/v1/models"));
    seen.push({ url });
    return Response.json(
      { models: [{ name: "nyx", description: "nyx decision model", release_date: "2026-09-21" }] },
      { headers: { "x-typesafe-request-id": "req-models" } },
    );
  };

  const client = new TypeSafeClient({
    apiKey: "secret",
    baseURL: "https://nyx.invalid/",
    defaultModel: "jev-latest",
    fetch: fetchMock,
  });
  const request = client.systemOne({
    state: "I was charged twice.",
    questions: {
      route: choice("Route it", { billing: null, other: null }),
      urgency: score("How urgent?", ["low", "medium", "high"]),
      accepted: noul("Is accepted?"),
    },
  });
  const { data, requestId } = await request.withResponse();
  const models = await client.models.list();

  assert.equal(seen.length, 2);
  assert.equal(seen[0]!.url, "https://nyx.invalid/v1/systemone");
  assert.equal(seen[0]!.body?.model, "jev-latest");
  assert.equal(data.model, "nyx");
  assert.equal(data.answers.route.choice, "billing");
  assert.equal(data.answers.urgency.score, 1.8);
  assert.equal(data.answers.accepted.noul, 0.75);
  assert.equal(requestId, "req-systemone");
  assert.equal(models[0]!.name, "nyx");
  assert.equal(seen[1]!.url, "https://nyx.invalid/v1/models");
});
