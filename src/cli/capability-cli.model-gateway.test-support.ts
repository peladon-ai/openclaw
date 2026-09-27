import { expect, it, vi, type Mock } from "vitest";

type GatewayCall = {
  method?: unknown;
  timeoutMs?: number;
  params?: Record<string, unknown>;
};

type ModelGatewayFixture = {
  runCapability: (domain: string, action: string, ...argv: string[]) => Promise<void>;
  mocks: {
    callGateway: Mock;
    acquireSimpleCompletionModelForAgent: Mock;
    runtime: { error: Mock; writeJson: Mock };
  };
  firstGatewayCall: () => GatewayCall | undefined;
  firstJsonOutput: () => { outputs?: unknown } | undefined;
  expectRuntimeErrorContains: (expected: string) => void;
};

// Register against the capability suite's existing mocks and per-test reset lifecycle.
export function registerModelGatewayTests({
  runCapability,
  mocks,
  firstGatewayCall,
  firstJsonOutput,
  expectRuntimeErrorContains,
}: ModelGatewayFixture): void {
  it("propagates gateway deadline and stable identity before dispatch without logging the prompt", async () => {
    const requestId = "12345678-1234-4234-8234-123456789abc";
    await runCapability(
      "model",
      "run",
      "--gateway",
      "--prompt",
      "private prompt",
      "--timeout-ms",
      "900000",
      "--request-id",
      requestId,
      "--json",
    );
    expect(mocks.callGateway).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        timeoutMs: 900_000,
        expectFinal: true,
        params: expect.objectContaining({
          idempotencyKey: requestId,
          sessionId: `model-run-${requestId}`,
          sessionKey: `agent:main:explicit:model-run-${requestId}`,
        }),
      }),
    );
    expect(mocks.runtime.error).toHaveBeenCalledWith(
      JSON.stringify({
        event: "model.run.request",
        requestId,
        sessionId: `model-run-${requestId}`,
        sessionKey: `agent:main:explicit:model-run-${requestId}`,
        timeoutMs: 900_000,
      }),
    );
    expect(mocks.runtime.error).toHaveBeenCalledBefore(mocks.callGateway);
    expect(JSON.stringify(mocks.runtime.error.mock.calls)).not.toContain("private prompt");
  });

  it.each(["0", "-1", "1.5", "NaN", "Infinity", "3600001", "120000ms", ""])(
    "rejects invalid gateway deadline %j before dispatch",
    async (timeout) => {
      await expect(
        runCapability("model", "run", "--gateway", "--prompt", "hello", "--timeout-ms", timeout),
      ).rejects.toThrow("exit 1");
      expect(mocks.callGateway).not.toHaveBeenCalled();
      expectRuntimeErrorContains("--timeout-ms must be an integer from 1 to 3600000.");
    },
  );

  it.each(["1", "3600000"])("accepts gateway deadline boundary %s", async (timeout) => {
    await runCapability("model", "run", "--gateway", "--prompt", "hello", "--timeout-ms", timeout);
    expect(mocks.callGateway).toHaveBeenCalledWith(
      expect.objectContaining({ timeoutMs: Number(timeout) }),
    );
  });

  it("rejects malformed request identities before dispatch", async () => {
    await expect(
      runCapability(
        "model",
        "run",
        "--gateway",
        "--prompt",
        "hello",
        "--request-id",
        "bad\nidentity",
      ),
    ).rejects.toThrow("exit 1");
    expect(mocks.callGateway).not.toHaveBeenCalled();
    expectRuntimeErrorContains("--request-id must be a UUID.");
  });

  it.each([
    ["--timeout-ms", "900000"],
    ["--request-id", "12345678-1234-4234-8234-123456789abc"],
  ])("rejects gateway-only option %s for local runs", async (option, value) => {
    await expect(runCapability("model", "run", "--prompt", "hello", option, value)).rejects.toThrow(
      "exit 1",
    );
    expect(mocks.callGateway).not.toHaveBeenCalled();
    expect(mocks.acquireSimpleCompletionModelForAgent).not.toHaveBeenCalled();
  });

  it.each([120_000, 900_000])(
    "honors the %i ms gateway deadline across the old 120 second boundary",
    async (deadline) => {
      vi.useFakeTimers();
      try {
        mocks.callGateway.mockImplementationOnce(
          async (options: { method: string; timeoutMs?: number }) => {
            await new Promise<void>((resolve, reject) => {
              const response = setTimeout(() => {
                clearTimeout(timeout);
                resolve();
              }, 120_001);
              const timeout = setTimeout(() => {
                clearTimeout(response);
                reject(new Error("gateway timeout: outcome unknown"));
              }, options.timeoutMs);
            });
            return {
              result: {
                payloads: [{ text: "delayed reply" }],
                meta: { agentMeta: { provider: "fixture", model: "fixture" } },
              },
            };
          },
        );
        const run = runCapability(
          "model",
          "run",
          "--gateway",
          "--prompt",
          "hello",
          ...(deadline === 120_000 ? [] : ["--timeout-ms", String(deadline)]),
          "--json",
        );
        const settled =
          deadline === 120_000
            ? expect(run).rejects.toThrow("exit 1")
            : expect(run).resolves.toBeUndefined();
        await vi.runAllTimersAsync();
        await settled;
        expect(mocks.callGateway).toHaveBeenCalledTimes(1);
        if (deadline === 900_000) {
          expect(firstJsonOutput()?.outputs).toEqual([{ text: "delayed reply" }]);
        } else {
          expect(mocks.runtime.writeJson).not.toHaveBeenCalled();
        }
      } finally {
        vi.useRealTimers();
      }
    },
  );

  it("does not resubmit after a gateway disconnect with unknown outcome", async () => {
    mocks.callGateway.mockRejectedValueOnce(new Error("connection closed"));
    await expect(runCapability("model", "run", "--gateway", "--prompt", "hello")).rejects.toThrow(
      "exit 1",
    );
    expect(mocks.callGateway).toHaveBeenCalledTimes(1);
    expect(mocks.runtime.writeJson).not.toHaveBeenCalled();
    expect(mocks.runtime.error.mock.calls[0]?.[0]).toContain('"event":"model.run.request"');
  });

  it("runs gateway model probes in fresh raw sessions without chat-agent prompt policy or tools", async () => {
    await runCapability("model", "run", "--prompt", "hello", "--gateway", "--json");

    const gatewayCall = firstGatewayCall();
    const sessionId = gatewayCall?.params?.sessionId;
    expect(gatewayCall?.method).toBe("agent");
    expect(typeof sessionId).toBe("string");
    if (typeof sessionId !== "string") {
      throw new Error("expected gateway model run session id");
    }
    expect(gatewayCall?.timeoutMs).toBe(120_000);
    expect(gatewayCall?.params?.idempotencyKey).toBe(sessionId.slice("model-run-".length));
    expect(sessionId).toEqual(expect.stringMatching(/^model-run-[0-9a-f-]{36}$/));
    expect(gatewayCall?.params?.sessionKey).toBe(`agent:main:explicit:${sessionId}`);
    expect(gatewayCall?.params?.cleanupBundleMcpOnRunEnd).toBe(true);
    expect(gatewayCall?.params?.modelRun).toBe(true);
    expect(gatewayCall?.params?.promptMode).toBe("none");

    await runCapability("model", "run", "--prompt", "again", "--gateway", "--json");

    const gatewayCalls = mocks.callGateway.mock.calls as unknown as Array<[GatewayCall]>;
    const nextGatewayCall = gatewayCalls[1]?.[0];
    const nextSessionId = nextGatewayCall?.params?.sessionId;
    expect(nextGatewayCall?.method).toBe("agent");
    expect(typeof nextSessionId).toBe("string");
    if (typeof nextSessionId !== "string") {
      throw new Error("expected second gateway model run session id");
    }
    expect(nextSessionId).toEqual(expect.stringMatching(/^model-run-[0-9a-f-]{36}$/));
    expect(nextGatewayCall?.params?.sessionKey).toBe(`agent:main:explicit:${nextSessionId}`);
    expect(nextSessionId).not.toBe(sessionId);
  });
}
