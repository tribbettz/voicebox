import type { EngineOptions, IndexTTSEmotionMode, IndexTTSEmotionVector } from '@/lib/api/types';

export const INDEXTTS_GENERATION_ENGINE_OPTION = {
  value: 'indextts',
  label: 'IndexTTS 2.5 (Expressive Clone)',
  engine: 'indextts',
} as const;

export const INDEXTTS_PROFILE_ENGINE_OPTION = {
  value: 'indextts',
  label: 'IndexTTS 2.5 (Expressive Clone)',
} as const;

export const INDEXTTS_SUPPORTED_LANGUAGES = ['zh', 'en', 'ja', 'es', 'ar'] as const;

export const INDEXTTS_EMOTION_MODES = [
  { value: 'natural', label: 'Natural / speaker reference' },
  { value: 'auto', label: 'Auto from script' },
  { value: 'instruction', label: 'Emotion instruction' },
  { value: 'vector', label: 'Emotion vector' },
  { value: 'audio', label: 'Emotion reference audio' },
] as const satisfies readonly { value: IndexTTSEmotionMode; label: string }[];

export const INDEXTTS_HELP = {
  mode: 'Natural follows the speaker reference. Auto analyzes the script. Instruction interprets natural language primarily as emotion/delivery guidance. Vector uses the eight official dimensions. Audio uses a separate emotion reference.',
  instruction:
    'Natural-language guidance for emotion and delivery, such as calm or restrained concern. This is not a universal arbitrary instruction-following prompt.',
  strength:
    'emo_alpha: 0 applies little or no requested guidance; 1 applies full guidance. Stronger is not automatically more natural. Around 0.6 or lower is the official suggested starting point for text-derived emotion.',
  speed:
    'IndexTTS duration_factor. Less than 1.0 produces shorter/faster speech; greater than 1.0 produces longer/slower speech. This is a duration factor, not a precise target duration.',
  random:
    'Introduces stochastic emotion variation. Upstream warns that enabling use_random may reduce speaker-cloning fidelity. Off by default; the Seed control still governs reproducibility.',
} as const;

export const INDEXTTS_PRONUNCIATION_EXAMPLES = [
  { label: 'English · CMU phonemes', text: 'He had a <minute|M IH1 . N AH0 T> to respond.' },
  { label: 'Chinese · numbered Pinyin', text: '他在银<行|XING2>里工作。' },
  { label: 'Japanese · Kana', text: '彼は料理が<上手|じょうず>です。' },
] as const;

export const INDEXTTS_EMOTION_DIMENSIONS = [
  'happy',
  'angry',
  'sad',
  'afraid',
  'disgusted',
  'melancholic',
  'surprised',
  'calm',
] as const;

export type IndexTTSEmotionDimension = (typeof INDEXTTS_EMOTION_DIMENSIONS)[number];

export function isIndexTTSProfileTypeCompatible(voiceType?: string): boolean {
  return !voiceType || voiceType === 'cloned';
}

export const INDEXTTS_NEUTRAL_VECTOR: IndexTTSEmotionVector = {
  happy: 0,
  angry: 0,
  sad: 0,
  afraid: 0,
  disgusted: 0,
  melancholic: 0,
  surprised: 0,
  calm: 0,
};

export const INDEXTTS_DEFAULTS = {
  emotionMode: 'natural' as IndexTTSEmotionMode,
  emotionText: '',
  emotionAudioAssetId: '',
  emotionAudioFilename: '',
  emoAlpha: 0.6,
  useRandom: false,
  durationFactor: 1,
  vector: INDEXTTS_NEUTRAL_VECTOR,
};

export interface IndexTTSFormState {
  indexTtsEmotionMode: IndexTTSEmotionMode;
  indexTtsEmotionText: string;
  indexTtsEmotionAudioAssetId: string;
  indexTtsEmoAlpha: number;
  indexTtsUseRandom: boolean;
  indexTtsDurationFactor: number;
  indexTtsHappy: number;
  indexTtsAngry: number;
  indexTtsSad: number;
  indexTtsAfraid: number;
  indexTtsDisgusted: number;
  indexTtsMelancholic: number;
  indexTtsSurprised: number;
  indexTtsCalm: number;
}

export function getIndexTTSVisibleControls(mode: IndexTTSEmotionMode) {
  return {
    instruction: mode === 'instruction',
    vector: mode === 'vector',
    audio: mode === 'audio',
    strength: mode !== 'natural',
    random: mode === 'auto' || mode === 'instruction' || mode === 'vector',
  };
}

export function buildIndexTTSEngineOptions(values: IndexTTSFormState): EngineOptions {
  const mode = values.indexTtsEmotionMode;
  const options: NonNullable<EngineOptions['indextts']> = {
    emotion_mode: mode,
    emo_alpha: values.indexTtsEmoAlpha,
    use_random: values.indexTtsUseRandom,
    duration_factor: values.indexTtsDurationFactor,
  };

  if (mode === 'instruction') {
    options.emotion_text = values.indexTtsEmotionText.trim();
  } else if (mode === 'vector') {
    options.emotion_vector = {
      happy: values.indexTtsHappy,
      angry: values.indexTtsAngry,
      sad: values.indexTtsSad,
      afraid: values.indexTtsAfraid,
      disgusted: values.indexTtsDisgusted,
      melancholic: values.indexTtsMelancholic,
      surprised: values.indexTtsSurprised,
      calm: values.indexTtsCalm,
    };
  } else if (mode === 'audio') {
    options.emotion_audio_asset_id = values.indexTtsEmotionAudioAssetId;
  }

  return { indextts: options };
}
