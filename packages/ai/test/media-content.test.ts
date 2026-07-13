/**
 * Tests for video/audio content support: capability-gated downgrade in
 * transformMessages and per-provider content-part mapping.
 */
import { describe, expect, it } from "vitest";
import { convertMessages as convertGoogleMessages } from "../src/api/google-shared.ts";
import { convertMessages as convertCompletionsMessages } from "../src/api/openai-completions.ts";
import { convertResponsesMessages } from "../src/api/openai-responses-shared.ts";
import { transformMessages } from "../src/api/transform-messages.ts";
import type { Api, Context, Model, OpenAICompletionsCompat, UserMessage } from "../src/types.ts";

const completionsCompat: Required<OpenAICompletionsCompat> = {
	supportsStore: true,
	supportsDeveloperRole: true,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	maxTokensField: "max_completion_tokens",
	requiresToolResultName: false,
	requiresAssistantAfterToolResult: false,
	requiresThinkingAsText: false,
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: "openai",
	openRouterRouting: {},
	vercelGatewayRouting: {},
	chatTemplateKwargs: {},
	zaiToolStream: false,
	supportsStrictMode: true,
	cacheControlFormat: "anthropic",
	sendSessionAffinityHeaders: false,
	sessionAffinityFormat: "openai",
	supportsLongCacheRetention: true,
};

function makeModel<TApi extends Api>(api: TApi, input: Model<TApi>["input"]): Model<TApi> {
	return {
		id: "test-model",
		name: "Test Model",
		api,
		provider: "openai",
		baseUrl: "https://example.invalid/v1",
		reasoning: false,
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 16000,
	};
}

function userMessage(content: UserMessage["content"]): UserMessage {
	return { role: "user", content, timestamp: 1 };
}

const VIDEO = { type: "video" as const, data: "dmlk", mimeType: "video/mp4" };
const AUDIO_WAV = { type: "audio" as const, data: "YXVk", mimeType: "audio/wav" };
const AUDIO_OGG = { type: "audio" as const, data: "YXVk", mimeType: "audio/ogg" };
const IMAGE = { type: "image" as const, data: "aW1n", mimeType: "image/png" };

describe("transformMessages media downgrade", () => {
	it("replaces unsupported video/audio with placeholders, keeps supported images", () => {
		const model = makeModel("openai-completions", ["text", "image"]);
		const messages = transformMessages(
			[userMessage([{ type: "text", text: "look" }, IMAGE, VIDEO, AUDIO_WAV])],
			model,
		);

		const content = (messages[0] as UserMessage).content as Exclude<UserMessage["content"], string>;
		expect(content).toEqual([
			{ type: "text", text: "look" },
			IMAGE,
			{ type: "text", text: "(video omitted: model does not support video)" },
			{ type: "text", text: "(audio omitted: model does not support audio)" },
		]);
	});

	it("passes video/audio through when the model declares support", () => {
		const model = makeModel("openai-completions", ["text", "image", "video", "audio"]);
		const messages = transformMessages([userMessage([{ type: "text", text: "look" }, VIDEO, AUDIO_WAV])], model);

		const content = (messages[0] as UserMessage).content as Exclude<UserMessage["content"], string>;
		expect(content).toEqual([{ type: "text", text: "look" }, VIDEO, AUDIO_WAV]);
	});

	it("collapses consecutive unsupported blocks of the same kind into one placeholder", () => {
		const model = makeModel("openai-completions", ["text"]);
		const messages = transformMessages([userMessage([VIDEO, VIDEO, AUDIO_WAV])], model);

		const content = (messages[0] as UserMessage).content as Exclude<UserMessage["content"], string>;
		expect(content).toEqual([
			{ type: "text", text: "(video omitted: model does not support video)" },
			{ type: "text", text: "(audio omitted: model does not support audio)" },
		]);
	});

	it("downgrades unsupported media in tool results with tool placeholders", () => {
		const model = makeModel("openai-completions", ["text"]);
		const messages = transformMessages(
			[
				{
					role: "toolResult",
					toolCallId: "call_1",
					toolName: "record",
					content: [{ type: "text", text: "done" }, VIDEO],
					isError: false,
					timestamp: 1,
				},
			],
			model,
		);

		expect(messages[0].content).toEqual([
			{ type: "text", text: "done" },
			{ type: "text", text: "(tool video omitted: model does not support video)" },
		]);
	});
});

describe("openai-completions media content parts", () => {
	function convert(input: Model<"openai-completions">["input"], content: UserMessage["content"]) {
		const model = makeModel("openai-completions", input);
		const context: Context = { messages: [userMessage(content)] };
		const params = convertCompletionsMessages(model, context, completionsCompat);
		const user = params.find((p) => p.role === "user");
		return user?.content as any[];
	}

	it("maps video blocks to video_url data-URI parts", () => {
		const parts = convert(["text", "video"], [{ type: "text", text: "clip?" }, VIDEO]);
		expect(parts).toContainEqual({
			type: "video_url",
			video_url: { url: "data:video/mp4;base64,dmlk" },
		});
	});

	it("maps wav/mp3 audio to the official input_audio part", () => {
		const parts = convert(["text", "audio"], [{ type: "text", text: "transcribe" }, AUDIO_WAV]);
		expect(parts).toContainEqual({
			type: "input_audio",
			input_audio: { data: "YXVk", format: "wav" },
		});
	});

	it("maps other audio mimeTypes to audio_url data-URI parts", () => {
		const parts = convert(["text", "audio"], [{ type: "text", text: "transcribe" }, AUDIO_OGG]);
		expect(parts).toContainEqual({
			type: "audio_url",
			audio_url: { url: "data:audio/ogg;base64,YXVk" },
		});
	});

	it("still maps image blocks to image_url parts", () => {
		const parts = convert(["text", "image"], [{ type: "text", text: "see" }, IMAGE]);
		expect(parts).toContainEqual({
			type: "image_url",
			image_url: { url: "data:image/png;base64,aW1n" },
		});
	});

	it("sends no media parts when the model lacks the capability", () => {
		const parts = convert(["text"], [{ type: "text", text: "hi" }, VIDEO, AUDIO_WAV]);
		expect(parts.every((p) => p.type === "text")).toBe(true);
	});

	it("forwards supported media from tool results as a user message", () => {
		const model = makeModel("openai-completions", ["text", "video"]);
		const context: Context = {
			messages: [
				{
					role: "toolResult",
					toolCallId: "call_1",
					toolName: "record",
					content: [{ type: "text", text: "recorded" }, VIDEO],
					isError: false,
					timestamp: 1,
				},
			],
		};
		const params = convertCompletionsMessages(model, context, completionsCompat);
		const user = params.find((p) => p.role === "user");
		expect(user).toBeDefined();
		expect(user!.content).toContainEqual({
			type: "video_url",
			video_url: { url: "data:video/mp4;base64,dmlk" },
		});
	});
});

describe("openai-responses media content parts", () => {
	function convert(input: Model<"openai-responses">["input"], content: UserMessage["content"]) {
		const model = makeModel("openai-responses", input);
		const context: Context = { messages: [userMessage(content)] };
		const items = convertResponsesMessages(model, context, new Set());
		const user = items.find((item) => "role" in item && item.role === "user") as { content: any[] } | undefined;
		return user?.content as any[];
	}

	it("maps video blocks to input_video parts", () => {
		const parts = convert(["text", "video"], [{ type: "text", text: "clip?" }, VIDEO]);
		expect(parts).toContainEqual({
			type: "input_video",
			video_url: "data:video/mp4;base64,dmlk",
		});
	});

	it("maps wav/mp3 audio to the official input_audio part", () => {
		const parts = convert(["text", "audio"], [{ type: "text", text: "transcribe" }, AUDIO_WAV]);
		expect(parts).toContainEqual({
			type: "input_audio",
			input_audio: { data: "YXVk", format: "wav" },
		});
	});

	it("downgrades unsupported media to placeholder text parts", () => {
		const parts = convert(["text", "image"], [{ type: "text", text: "hi" }, VIDEO]);
		expect(parts).toEqual([
			{ type: "input_text", text: "hi" },
			{ type: "input_text", text: "(video omitted: model does not support video)" },
		]);
	});
});

describe("google media content parts", () => {
	function convert(input: Model<"google-generative-ai">["input"], content: UserMessage["content"]) {
		const model = { ...makeModel("google-generative-ai", input), provider: "google" };
		const context: Context = { messages: [userMessage(content)] };
		const contents = convertGoogleMessages(model as Model<"google-generative-ai">, context);
		return contents.find((c) => c.role === "user")?.parts as any[];
	}

	it("maps video and audio blocks to inlineData parts", () => {
		const parts = convert(["text", "image", "video", "audio"], [{ type: "text", text: "look" }, VIDEO, AUDIO_WAV]);
		expect(parts).toContainEqual({ inlineData: { mimeType: "video/mp4", data: "dmlk" } });
		expect(parts).toContainEqual({ inlineData: { mimeType: "audio/wav", data: "YXVk" } });
	});

	it("downgrades unsupported media to placeholder text parts", () => {
		const parts = convert(["text", "image"], [{ type: "text", text: "look" }, VIDEO]);
		expect(parts).toEqual([{ text: "look" }, { text: "(video omitted: model does not support video)" }]);
	});
});
