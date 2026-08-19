export const QWEN_ADVANCED_PREFIX = '__VOICEBOX_QWEN_ADVANCED__:';

export const QWEN_ADVANCED_DEFAULTS = {
  temperature: 0.9,
  topP: 1.0,
  topK: 50,
  repetitionPenalty: 1.05,
} as const;

export interface QwenAdvancedControls {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
}

/**
 * Base Qwen voice cloning does not consume natural-language ``instruct`` text.
 * We reuse Voicebox's already-persisted optional field as an internal envelope
 * for Base-only sampling controls so retries/regenerations keep the same knobs
 * without a database migration. The backend strips this before model inference.
 */
export function encodeQwenAdvancedControls(controls: QwenAdvancedControls): string {
  return `${QWEN_ADVANCED_PREFIX}${JSON.stringify({
    temperature: controls.temperature,
    top_p: controls.topP,
    top_k: controls.topK,
    repetition_penalty: controls.repetitionPenalty,
  })}`;
}
