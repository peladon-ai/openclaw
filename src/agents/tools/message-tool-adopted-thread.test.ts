import { afterEach, describe, expect, it, vi } from "vitest";
import { createReplyOperation } from "../../auto-reply/reply/reply-run-registry.js";
import type { ChannelPlugin } from "../../channels/plugins/types.public.js";
import { clearRuntimeConfigSnapshot, setRuntimeConfigSnapshot } from "../../config/config.js";
import * as transcriptRuntime from "../../config/sessions/transcript.runtime.js";
import {
  mintMessageActionTurnCapability,
  revokeMessageActionTurnCapability,
} from "../../gateway/message-action-turn-capability.js";
import { runMessageAction } from "../../infra/outbound/message-action-runner.js";
import { workspaceConfig } from "../../infra/outbound/message-action-runner.test-support.js";
import { resetPluginRuntimeStateForTest, setActivePluginRegistry } from "../../plugins/runtime.js";
import { createOutboundTestPlugin, createTestRegistry } from "../../test-utils/channel-plugins.js";
import { readEmbeddedMessageDeliveryFact } from "../embedded-agent-message-delivery.js";
import { resolveEmbeddedRunAttemptTerminalState } from "../embedded-agent-runner/run/terminal-outcome.js";
import { resolveSettledTurnFinalizationRequest } from "../embedded-agent-runner/run/terminal-resolution.js";
import {
  buildEmbeddedRunnerAssistant,
  makeEmbeddedRunnerAttempt,
} from "../test-helpers/embedded-agent-runner-e2e-fixtures.js";
import { readToolResultDetails } from "../tool-result-error.js";
import { createMessageTool } from "./message-tool-execution.js";

// The transport and its owner proof are fixtures; core send routing, source
// classification, reply-operation bookkeeping, and finalizer eligibility are real.
describe("core message sends into an adopted source thread", () => {
  afterEach(() => {
    clearRuntimeConfigSnapshot();
    resetPluginRuntimeStateForTest();
  });

  it.each([
    { mode: "delivered", completes: true },
    { mode: "mirror failed", completes: true },
    { mode: "unrelated target", completes: false },
    { mode: "contradictory recipient", completes: false },
    { mode: "mixed physical threads", completes: false },
    { mode: "failed send", completes: false },
    { mode: "progress", completes: false },
    { mode: "dry run", completes: false },
    { mode: "all chunks identified", completes: true },
    { mode: "first chunk missing identity", completes: false },
    { mode: "last chunk missing identity", completes: false },
    { mode: "first chunk not sent", completes: false },
    { mode: "first chunk wrong recipient", completes: false },
    { mode: "first chunk wrong thread", completes: false },
  ])(
    "keeps one-send finalization and missing-reply recovery for $mode",
    async ({ mode, completes }) => {
      const chunked = mode.includes("chunk");
      let physicalSendCount = 0;
      const sendText = vi.fn(async ({ to }: { to: string }) => {
        physicalSendCount += 1;
        if (mode === "failed send") {
          throw new Error("synthetic provider failure");
        }
        const recipient =
          mode === "contradictory recipient" ||
          (mode === "first chunk wrong recipient" && physicalSendCount === 1)
            ? "other-room"
            : to;
        const notSent = mode === "first chunk not sent" && physicalSendCount === 1;
        const messageId =
          notSent ||
          (mode === "first chunk missing identity" && physicalSendCount === 1) ||
          (mode === "last chunk missing identity" && physicalSendCount === 2)
            ? ""
            : `outbound-${physicalSendCount}`;
        return {
          channel: "workspace",
          ...(notSent ? { outcome: "not_sent" as const } : {}),
          messageId,
          target: { kind: "channel" as const, id: recipient },
          receipt: {
            platformMessageIds: messageId ? [messageId] : [],
            parts: messageId
              ? [
                  {
                    platformMessageId: messageId,
                    kind: "text" as const,
                    index: 0,
                    raw: { channelId: recipient },
                    ...(mode === "mixed physical threads" ||
                    (mode === "first chunk wrong thread" && physicalSendCount === 1)
                      ? { threadId: "other-room" }
                      : {}),
                  },
                ]
              : [],
            sentAt: 1,
          },
        };
      });
      const handleAction = vi.fn(async () => {
        throw new Error("generic sends must use the core outbound path");
      });
      const plugin: ChannelPlugin = {
        ...createOutboundTestPlugin({
          id: "workspace",
          capabilities: { chatTypes: ["channel"] },
          messaging: {
            normalizeTarget: (target) => target.replace(/^channel:/u, ""),
            targetResolver: { looksLikeId: () => true },
          },
          outbound: {
            deliveryMode: "direct",
            sendText,
            ...(chunked
              ? {
                  textChunkLimit: 10,
                  chunker: (text) => [text.slice(0, 10), text.slice(10)],
                }
              : {}),
          },
        }),
        config: {
          listAccountIds: () => ["default"],
          resolveAccount: () => ({ enabled: true }),
          isConfigured: () => true,
        },
        actions: {
          describeMessageTool: () => ({ actions: ["send"] }),
          prepareSendPayload: ({ payload }) => payload,
          handleAction,
          messageActionTargetAliases: {
            send: {
              aliases: [],
              matchesCurrentConversation: ({ args, accountId, toolContext }) =>
                accountId === "default" &&
                toolContext.currentChannelId === "parent-room" &&
                toolContext.currentMessageId === "inbound-1" &&
                args.target === "adopted-thread" &&
                (args.threadId === undefined || args.threadId === "adopted-thread"),
            },
          },
        },
      };
      setActivePluginRegistry(
        createTestRegistry([{ pluginId: "workspace", plugin, source: "test", origin: "bundled" }]),
      );
      const config = {
        ...workspaceConfig,
        tools: { message: { crossContext: { allowWithinProvider: true } } },
      };
      setRuntimeConfigSnapshot(config);
      const source = {
        currentChannelProvider: "workspace",
        currentChannelId: "parent-room",
        currentMessagingTarget: "parent-room",
        currentMessageId: "inbound-1",
        currentChatType: "channel" as const,
      };
      const sessionKey = "agent:main:workspace:channel:parent-room";
      const capability = mintMessageActionTurnCapability({
        agentId: "main",
        runId: "adopted-thread-run",
        sessionKey,
        requesterAccountId: "default",
        toolContext: source,
      });
      const operation = createReplyOperation({
        sessionKey,
        sessionId: "adopted-thread-session",
        resetTriggered: false,
      });
      const mirror = vi.spyOn(transcriptRuntime, "appendAssistantMessageToSessionTranscript");
      if (mode === "mirror failed") {
        mirror.mockRejectedValue(new Error("synthetic transcript writer changed"));
      }
      try {
        const tool = createMessageTool({
          config,
          ...source,
          agentId: "main",
          runId: "adopted-thread-run",
          sessionId: operation.sessionId,
          agentSessionKey: sessionKey,
          agentAccountId: "default",
          messageActionTurnCapability: capability,
          sourceReplyDeliveryMode: "automatic",
          runMessageAction: (input) => runMessageAction({ ...input, skipQueue: true }),
          getScopedChannelsCommandSecretTargets: () => ({ targetIds: new Set<string>() }),
          resolveCommandSecretRefsViaGateway: async ({ config: resolvedConfig }) => ({
            resolvedConfig,
            diagnostics: [],
            targetStatesByPath: {},
            hadUnresolvedTargets: false,
          }),
        });
        const execution = tool.execute("adopted-thread-send", {
          action: "send",
          target: mode === "unrelated target" ? "other-room" : "adopted-thread",
          message: "Completed the request.",
          final: mode !== "progress",
          ...(mode === "dry run" ? { dryRun: true } : {}),
        });
        let delivered: true | undefined;
        if (mode === "failed send") {
          await expect(execution).rejects.toThrow("synthetic provider failure");
        } else {
          const result = await execution;
          delivered = readEmbeddedMessageDeliveryFact(
            readToolResultDetails(result)?.messageDelivery,
          )?.sourceReplyDelivered;
          expect(delivered === true).toBe(completes);
        }
        expect(operation.sourceReplyDelivered).toBe(completes);
        expect(handleAction).not.toHaveBeenCalled();
        expect(sendText).toHaveBeenCalledTimes(mode === "dry run" ? 0 : chunked ? 2 : 1);
        if (mode === "mirror failed") {
          expect(mirror).toHaveBeenCalled();
        }
        const assistant = buildEmbeddedRunnerAssistant({ content: [] });
        const attempt = makeEmbeddedRunnerAttempt({
          assistantTexts: [],
          lastAssistant: assistant,
          currentAttemptAssistant: assistant,
          sourceReplyDelivered: delivered,
          toolMetas: [{ toolName: "message", replaySafe: false }],
          itemLifecycle: { startedCount: 1, completedCount: 1, activeCount: 0 },
        });
        const finalization = resolveSettledTurnFinalizationRequest({
          runParams: {
            sessionId: operation.sessionId,
            runId: "adopted-thread-run",
            workspaceDir: "fixture-workspace",
            prompt: "Complete the request in its anchored thread.",
            timeoutMs: 60_000,
            trigger: "user",
            terminalReplyExpectation: "required",
          },
          attempt,
          activeErrorContext: { provider: "test", model: "test-model" },
          modelApi: "openai-responses",
          executionContract: undefined,
          payloadsWithToolMedia: [],
          hasTerminalToolPresentation: false,
          terminalState: resolveEmbeddedRunAttemptTerminalState({ attempt, assistant }),
          settledTurnFinalizationAvailable: true,
        });
        expect(finalization === null).toBe(completes);
        expect(sendText).toHaveBeenCalledTimes(mode === "dry run" ? 0 : chunked ? 2 : 1);
      } finally {
        mirror.mockRestore();
        operation.complete();
        revokeMessageActionTurnCapability(capability);
      }
    },
  );
});
