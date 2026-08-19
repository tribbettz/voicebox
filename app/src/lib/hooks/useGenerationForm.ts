import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig } from '@/lib/api/types';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import {
  encodeQwenAdvancedControls,
  QWEN_ADVANCED_DEFAULTS,
} from '@/lib/utils/qwenAdvanced';
import { useGenerationStore } from '@/stores/generationStore';
import { useUIStore } from '@/stores/uiStore';

const generationSchema = z.object({
  text: z.string().min(1, '').max(50000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().min(0).optional(),
  modelSize: z.enum(['1.7B', '0.6B', '1B', '3B']).optional(),
  instruct: z.string().max(500).optional(),
  engine: z
    .enum([
      'qwen',
      'qwen_custom_voice',
      'luxtts',
      'chatterbox',
      'chatterbox_turbo',
      'tada',
      'kokoro',
    ])
    .optional(),
  personality: z.boolean().optional(),
  qwenTemperature: z.number().min(0.1).max(1.5),
  qwenTopP: z.number().min(0.1).max(1),
  qwenTopK: z.number().int().min(1).max(100),
  qwenRepetitionPenalty: z.number().min(1).max(1.5),
  maxChunkChars: z.number().int().min(100).max(5000),
  crossfadeMs: z.number().int().min(0).max(500),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
  getEffectsChain?: () => EffectConfig[] | undefined;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);
  const { settings: genSettings } = useGenerationSettings();
  const defaultMaxChunkChars = genSettings?.max_chunk_chars ?? 800;
  const defaultCrossfadeMs = genSettings?.crossfade_ms ?? 50;
  const normalizeAudio = genSettings?.normalize_audio ?? true;
  const selectedEngine = useUIStore((state) => state.selectedEngine);
  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  useModelDownloadToast({
    modelName: downloadingModelName || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModelName,
  });

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: '1.7B',
      instruct: '',
      engine: (selectedEngine as GenerationFormValues['engine']) || 'qwen',
      personality: false,
      qwenTemperature: QWEN_ADVANCED_DEFAULTS.temperature,
      qwenTopP: QWEN_ADVANCED_DEFAULTS.topP,
      qwenTopK: QWEN_ADVANCED_DEFAULTS.topK,
      qwenRepetitionPenalty: QWEN_ADVANCED_DEFAULTS.repetitionPenalty,
      maxChunkChars: defaultMaxChunkChars,
      crossfadeMs: defaultCrossfadeMs,
      ...options.defaultValues,
    },
  });

  // Generation settings arrive asynchronously. Reflect the saved values in
  // Advanced Options unless the user has already edited those fields locally.
  useEffect(() => {
    if (!genSettings) return;
    if (!form.getFieldState('maxChunkChars').isDirty) {
      form.setValue('maxChunkChars', genSettings.max_chunk_chars);
    }
    if (!form.getFieldState('crossfadeMs').isDirty) {
      form.setValue('crossfadeMs', genSettings.crossfade_ms);
    }
  }, [form, genSettings]);

  async function handleSubmit(
    data: GenerationFormValues,
    selectedProfileId: string | null,
  ): Promise<void> {
    if (!selectedProfileId) {
      toast({
        title: 'No profile selected',
        description: 'Please select a voice profile from the cards above.',
        variant: 'destructive',
      });
      return;
    }

    try {
      const engine = data.engine || 'qwen';
      const modelName =
        engine === 'luxtts'
          ? 'luxtts'
          : engine === 'chatterbox'
            ? 'chatterbox-tts'
            : engine === 'chatterbox_turbo'
              ? 'chatterbox-turbo'
              : engine === 'tada'
                ? data.modelSize === '3B'
                  ? 'tada-3b-ml'
                  : 'tada-1b'
                : engine === 'kokoro'
                  ? 'kokoro'
                  : engine === 'qwen_custom_voice'
                    ? `qwen-custom-voice-${data.modelSize}`
                    : `qwen-tts-${data.modelSize}`;
      const displayName =
        engine === 'luxtts'
          ? 'LuxTTS'
          : engine === 'chatterbox'
            ? 'Chatterbox TTS'
            : engine === 'chatterbox_turbo'
              ? 'Chatterbox Turbo'
              : engine === 'tada'
                ? data.modelSize === '3B'
                  ? 'TADA 3B Multilingual'
                  : 'TADA 1B'
                : engine === 'kokoro'
                  ? 'Kokoro 82M'
                  : engine === 'qwen_custom_voice'
                    ? data.modelSize === '1.7B'
                      ? 'Qwen CustomVoice 1.7B'
                      : 'Qwen CustomVoice 0.6B'
                    : data.modelSize === '1.7B'
                      ? 'Qwen TTS 1.7B'
                      : 'Qwen TTS 0.6B';

      try {
        const modelStatus = await apiClient.getModelStatus();
        const model = modelStatus.models.find((m) => m.model_name === modelName);

        if (model && !model.downloaded) {
          setDownloadingModelName(modelName);
          setDownloadingDisplayName(displayName);
        }
      } catch (error) {
        console.error('Failed to check model status:', error);
      }

      const hasModelSizes =
        engine === 'qwen' || engine === 'qwen_custom_voice' || engine === 'tada';
      const effectsChain = options.getEffectsChain?.();

      const instruct =
        engine === 'qwen_custom_voice'
          ? data.instruct || undefined
          : engine === 'qwen'
            ? encodeQwenAdvancedControls({
                temperature: data.qwenTemperature,
                topP: data.qwenTopP,
                topK: data.qwenTopK,
                repetitionPenalty: data.qwenRepetitionPenalty,
              })
            : undefined;

      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        text: data.text,
        language: data.language,
        seed: data.seed,
        model_size: hasModelSizes ? data.modelSize : undefined,
        engine,
        instruct,
        personality: data.personality || undefined,
        max_chunk_chars: data.maxChunkChars,
        crossfade_ms: data.crossfadeMs,
        normalize: normalizeAudio,
        effects_chain: effectsChain?.length ? effectsChain : undefined,
      });

      addPendingGeneration(result.id);

      form.reset({
        text: '',
        language: data.language,
        seed: data.seed,
        modelSize: data.modelSize,
        instruct: '',
        engine: data.engine,
        personality: data.personality,
        qwenTemperature: data.qwenTemperature,
        qwenTopP: data.qwenTopP,
        qwenTopK: data.qwenTopK,
        qwenRepetitionPenalty: data.qwenRepetitionPenalty,
        maxChunkChars: data.maxChunkChars,
        crossfadeMs: data.crossfadeMs,
      });
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Failed to generate audio',
        variant: 'destructive',
      });
    } finally {
      setDownloadingModelName(null);
      setDownloadingDisplayName(null);
    }
  }

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
  };
}
