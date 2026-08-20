// @ts-expect-error Bun provides this module at test runtime; the app does not
// ship Bun's ambient types to browser production code.
import { describe, expect, test } from 'bun:test';
import {
  buildIndexTTSEngineOptions,
  getIndexTTSVisibleControls,
  INDEXTTS_DEFAULTS,
  INDEXTTS_EMOTION_DIMENSIONS,
  INDEXTTS_EMOTION_MODES,
  INDEXTTS_GENERATION_ENGINE_OPTION,
  INDEXTTS_HELP,
  INDEXTTS_NEUTRAL_VECTOR,
  INDEXTTS_PROFILE_ENGINE_OPTION,
  INDEXTTS_PRONUNCIATION_EXAMPLES,
  INDEXTTS_SUPPORTED_LANGUAGES,
  isIndexTTSProfileTypeCompatible,
} from './indexTts';

const base = {
  indexTtsEmotionMode: INDEXTTS_DEFAULTS.emotionMode,
  indexTtsEmotionText: '',
  indexTtsEmotionAudioAssetId: '',
  indexTtsEmoAlpha: INDEXTTS_DEFAULTS.emoAlpha,
  indexTtsUseRandom: INDEXTTS_DEFAULTS.useRandom,
  indexTtsDurationFactor: INDEXTTS_DEFAULTS.durationFactor,
  indexTtsHappy: 0,
  indexTtsAngry: 0,
  indexTtsSad: 0,
  indexTtsAfraid: 0,
  indexTtsDisgusted: 0,
  indexTtsMelancholic: 0,
  indexTtsSurprised: 0,
  indexTtsCalm: 0,
};

describe('IndexTTS form helpers', () => {
  test('uses documented defaults and upstream vector order', () => {
    expect(INDEXTTS_DEFAULTS).toMatchObject({
      emotionMode: 'natural',
      emoAlpha: 0.6,
      useRandom: false,
      durationFactor: 1,
    });
    expect(INDEXTTS_EMOTION_DIMENSIONS).toEqual([
      'happy',
      'angry',
      'sad',
      'afraid',
      'disgusted',
      'melancholic',
      'surprised',
      'calm',
    ]);
    expect(INDEXTTS_NEUTRAL_VECTOR).toEqual({
      happy: 0,
      angry: 0,
      sad: 0,
      afraid: 0,
      disgusted: 0,
      melancholic: 0,
      surprised: 0,
      calm: 0,
    });
  });

  test('serializes only the selected mutually exclusive emotion input', () => {
    expect(
      buildIndexTTSEngineOptions({
        ...base,
        indexTtsEmotionMode: 'instruction',
        indexTtsEmotionText: ' calm and grave ',
      }).indextts,
    ).toEqual({
      emotion_mode: 'instruction',
      emotion_text: 'calm and grave',
      emo_alpha: 0.6,
      use_random: false,
      duration_factor: 1,
    });
    expect(
      buildIndexTTSEngineOptions({
        ...base,
        indexTtsEmotionMode: 'vector',
        indexTtsCalm: 0.5,
        indexTtsSad: 0.2,
      }).indextts?.emotion_vector,
    ).toEqual({ ...INDEXTTS_NEUTRAL_VECTOR, calm: 0.5, sad: 0.2 });
  });

  test('drives mode-conditional controls', () => {
    expect(getIndexTTSVisibleControls('natural')).toEqual({
      instruction: false,
      vector: false,
      audio: false,
      strength: false,
      random: false,
    });
    expect(getIndexTTSVisibleControls('instruction').instruction).toBe(true);
    expect(getIndexTTSVisibleControls('vector').vector).toBe(true);
    expect(getIndexTTSVisibleControls('audio').audio).toBe(true);
    expect(getIndexTTSVisibleControls('auto').random).toBe(true);
    expect(getIndexTTSVisibleControls('audio').random).toBe(false);
  });

  test('provides generation/profile selectors and only official 2.5 languages', () => {
    expect(INDEXTTS_GENERATION_ENGINE_OPTION).toEqual({
      value: 'indextts',
      label: 'IndexTTS 2.5 (Expressive Clone)',
      engine: 'indextts',
    });
    expect(INDEXTTS_PROFILE_ENGINE_OPTION.value).toBe('indextts');
    expect(INDEXTTS_SUPPORTED_LANGUAGES).toEqual(['zh', 'en', 'ja', 'es', 'ar']);
    expect(isIndexTTSProfileTypeCompatible('cloned')).toBe(true);
    expect(isIndexTTSProfileTypeCompatible('preset')).toBe(false);
    expect(isIndexTTSProfileTypeCompatible('designed')).toBe(false);
  });

  test('defines every conditional mode and the required cautionary help', () => {
    expect(INDEXTTS_EMOTION_MODES.map((mode) => mode.value)).toEqual([
      'natural',
      'auto',
      'instruction',
      'vector',
      'audio',
    ]);
    expect(INDEXTTS_HELP.instruction).toContain('not a universal arbitrary instruction');
    expect(INDEXTTS_HELP.strength).toContain('0.6 or lower');
    expect(INDEXTTS_HELP.speed).toContain('shorter/faster');
    expect(INDEXTTS_HELP.speed).toContain('longer/slower');
    expect(INDEXTTS_HELP.random).toContain('reduce speaker-cloning fidelity');
  });

  test('syntax help includes official CMU, Pinyin, and Kana examples', () => {
    expect(INDEXTTS_PRONUNCIATION_EXAMPLES.map((example) => example.label)).toEqual([
      'English · CMU phonemes',
      'Chinese · numbered Pinyin',
      'Japanese · Kana',
    ]);
    expect(INDEXTTS_PRONUNCIATION_EXAMPLES.every((example) => example.text.includes('<'))).toBe(
      true,
    );
  });
});
