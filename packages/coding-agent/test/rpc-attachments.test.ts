/**
 * Tests that the RPC prompt/steer/follow_up commands accept media attachments
 * via the `attachments` field and merge the deprecated `images` alias.
 */
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Context,
	EventStream,
	getModel,
	type UserMessage,
} from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import type { AgentSessionRuntime } from "../src/core/agent-session-runtime.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRegistry } from "../src/core/model-registry.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { runRpcMode } from "../src/modes/rpc/rpc-mode.ts";
import { createTestResourceLoader } from "./utilities.ts";

const rpcIo = vi.hoisted(() => ({
	outputLines: [] as string[],
	lineHandler: undefined as ((line: string) => void) | undefined,
}));

vi.mock("../src/core/output-guard.js", () => ({
	flushRawStdout: vi.fn(async () => {}),
	takeOverStdout: vi.fn(),
	waitForRawStdoutBackpressure: vi.fn(async () => {}),
	writeRawStdout: (line: string) => {
		rpcIo.outputLines.push(line);
	},
}));

vi.mock("../src/modes/interactive/theme/theme.js", () => ({ theme: {} }));

vi.mock("../src/modes/rpc/jsonl.js", () => ({
	attachJsonlLineReader: vi.fn((_stream: NodeJS.ReadableStream, onLine: (line: string) => void) => {
		rpcIo.lineHandler = onLine;
		return () => {};
	}),
	serializeJsonLine: (value: unknown) => `${JSON.stringify(value)}\n`,
}));

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

const VIDEO = { type: "video" as const, data: "dmlk", mimeType: "video/mp4" };
const AUDIO = { type: "audio" as const, data: "YXVk", mimeType: "audio/wav" };
const IMAGE = { type: "image" as const, data: "aW1n", mimeType: "image/png" };

async function startRpc(): Promise<{
	lineHandler: (line: string) => void;
	capturedContexts: Context[];
	cleanup: () => Promise<void>;
}> {
	rpcIo.outputLines = [];
	rpcIo.lineHandler = undefined;

	const tempDir = join(tmpdir(), `pi-rpc-attach-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	mkdirSync(tempDir, { recursive: true });

	const model = getModel("anthropic", "claude-sonnet-4-5");
	if (!model) throw new Error("Test model not found");

	const capturedContexts: Context[] = [];
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model, systemPrompt: "Test", tools: [] },
		streamFn: (_model, context, _options) => {
			capturedContexts.push(context);
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				stream.push({ type: "start", partial: createAssistantMessage("") });
				stream.push({ type: "done", reason: "stop", message: createAssistantMessage("done") });
			});
			return stream;
		},
	});

	const sessionManager = SessionManager.inMemory();
	const settingsManager = SettingsManager.create(tempDir, tempDir);
	const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
	const modelRegistry = ModelRegistry.create(authStorage, tempDir);
	authStorage.setRuntimeApiKey("anthropic", "test-key");

	const session = new AgentSession({
		agent,
		sessionManager,
		settingsManager,
		cwd: tempDir,
		modelRegistry,
		resourceLoader: createTestResourceLoader(),
	});

	const runtimeHost = {
		session,
		newSession: vi.fn(async () => ({ cancelled: true })),
		switchSession: vi.fn(async () => ({ cancelled: true })),
		fork: vi.fn(async () => ({ cancelled: true, selectedText: "" })),
		dispose: vi.fn(async () => {}),
		setRebindSession: vi.fn(),
	} as unknown as AgentSessionRuntime;

	void runRpcMode(runtimeHost);
	await vi.waitFor(() => expect(rpcIo.lineHandler).toBeDefined());

	return {
		lineHandler: rpcIo.lineHandler!,
		capturedContexts,
		cleanup: async () => {
			try {
				if (session.isStreaming) await session.abort();
			} catch {
				// ignore test cleanup failures
			}
			session.dispose();
			if (existsSync(tempDir)) rmSync(tempDir, { recursive: true });
		},
	};
}

function lastUserMessage(contexts: Context[]): UserMessage {
	const messages = contexts[contexts.length - 1].messages;
	const user = [...messages].reverse().find((m) => m.role === "user");
	if (!user) throw new Error("No user message captured");
	return user as UserMessage;
}

describe("RPC media attachments", () => {
	afterEach(() => {
		rpcIo.outputLines = [];
		rpcIo.lineHandler = undefined;
	});

	it("delivers attachments from the prompt command in the user message", async () => {
		const { lineHandler, capturedContexts, cleanup } = await startRpc();
		try {
			lineHandler(
				JSON.stringify({ id: "p1", type: "prompt", message: "what happens here?", attachments: [VIDEO, AUDIO] }),
			);
			await vi.waitFor(() => expect(capturedContexts.length).toBeGreaterThan(0));

			const content = lastUserMessage(capturedContexts).content as Exclude<UserMessage["content"], string>;
			expect(content).toEqual([{ type: "text", text: "what happens here?" }, VIDEO, AUDIO]);
		} finally {
			await cleanup();
		}
	});

	it("merges the deprecated images alias before attachments", async () => {
		const { lineHandler, capturedContexts, cleanup } = await startRpc();
		try {
			lineHandler(
				JSON.stringify({ id: "p1", type: "prompt", message: "both fields", images: [IMAGE], attachments: [VIDEO] }),
			);
			await vi.waitFor(() => expect(capturedContexts.length).toBeGreaterThan(0));

			const content = lastUserMessage(capturedContexts).content as Exclude<UserMessage["content"], string>;
			expect(content).toEqual([{ type: "text", text: "both fields" }, IMAGE, VIDEO]);
		} finally {
			await cleanup();
		}
	});

	it("still accepts the images field alone", async () => {
		const { lineHandler, capturedContexts, cleanup } = await startRpc();
		try {
			lineHandler(JSON.stringify({ id: "p1", type: "prompt", message: "just images", images: [IMAGE] }));
			await vi.waitFor(() => expect(capturedContexts.length).toBeGreaterThan(0));

			const content = lastUserMessage(capturedContexts).content as Exclude<UserMessage["content"], string>;
			expect(content).toEqual([{ type: "text", text: "just images" }, IMAGE]);
		} finally {
			await cleanup();
		}
	});
});
