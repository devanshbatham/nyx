export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export type ChoiceQuestion = {
  type: "choice";
  instructions?: JsonValue;
  criteria: Record<string, JsonValue>;
};

export type ScoreQuestion = {
  type: "score";
  instructions?: JsonValue;
  criteria: JsonValue[];
};

export type NoulQuestion = {
  type: "noul";
  instructions?: JsonValue;
  criteria?: { true?: JsonValue; false?: JsonValue } | null;
};

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export type SystemOneRequest = {
  state: JsonValue;
  questions: Record<string, Question>;
  model?: string;
};

export type Answer =
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; probabilities: Record<string, number>; legend: Record<string, JsonValue> }
  | { type: "noul"; noul: number };

export type SystemOneResponse = {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
};

export class APIError extends Error {
  constructor(public readonly status: number, public readonly body: string) {
    super(`nyx API returned HTTP ${status}`);
    this.name = "APIError";
  }
}

export type ClientOptions = {
  apiKey: string;
  baseUrl?: string;
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
};

export class Client {
  readonly #apiKey: string;
  readonly #baseUrl: string;
  readonly #timeoutMs: number;
  readonly #fetch: typeof globalThis.fetch;

  constructor(options: ClientOptions) {
    if (!options.apiKey) throw new TypeError("apiKey is required");
    this.#apiKey = options.apiKey;
    this.#baseUrl = (options.baseUrl ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    this.#timeoutMs = options.timeoutMs ?? 120_000;
    this.#fetch = options.fetch ?? globalThis.fetch;
    if (!this.#fetch) throw new TypeError("A Fetch API implementation is required");
  }

  async systemone(request: SystemOneRequest): Promise<SystemOneResponse> {
    return this.#request<SystemOneResponse>("/v1/systemone", {
      method: "POST",
      body: JSON.stringify({ ...request, model: request.model ?? "nyx" }),
    });
  }

  async models(): Promise<{ models: Array<{ name: string; description: string; release_date: string }> }> {
    return this.#request("/v1/models", { method: "GET" });
  }

  async #request<T>(path: string, init: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.#timeoutMs);
    try {
      const response = await this.#fetch(`${this.#baseUrl}${path}`, {
        ...init,
        headers: {
          Authorization: `Bearer ${this.#apiKey}`,
          "Content-Type": "application/json",
          ...init.headers,
        },
        redirect: "error",
        signal: controller.signal,
      });
      if (!response.ok) throw new APIError(response.status, await response.text());
      return (await response.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }
}
