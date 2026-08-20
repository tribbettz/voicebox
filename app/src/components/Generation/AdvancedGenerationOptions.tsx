import { ChevronDown, CircleHelp, Code2, Loader2, RotateCcw, Upload, X } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { FormControl, FormField, FormItem, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';
import {
  getIndexTTSVisibleControls,
  INDEXTTS_EMOTION_DIMENSIONS,
  INDEXTTS_EMOTION_MODES,
  INDEXTTS_HELP,
  INDEXTTS_NEUTRAL_VECTOR,
  INDEXTTS_PRONUNCIATION_EXAMPLES,
  type IndexTTSEmotionDimension,
} from '@/lib/utils/indexTts';

interface AdvancedGenerationOptionsProps {
  form: UseFormReturn<GenerationFormValues>;
}

function FieldHelp({ label, text }: { label: string; text: string }) {
  const tooltipId = useId();

  return (
    <button
      type="button"
      className="group/help relative inline-flex shrink-0 rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      aria-label={`${label} help`}
      aria-describedby={tooltipId}
    >
      <CircleHelp className="h-3.5 w-3.5 text-muted-foreground/70" aria-hidden />
      <span
        id={tooltipId}
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-1/2 z-[9999] mb-2 w-64 max-w-[calc(100vw-2rem)] -translate-x-1/2 rounded-md border border-border bg-popover px-3 py-2 text-left text-xs leading-relaxed text-popover-foreground opacity-0 shadow-lg transition-opacity group-hover/help:visible group-hover/help:opacity-100 group-focus-visible/help:visible group-focus-visible/help:opacity-100"
      >
        {text}
      </span>
    </button>
  );
}

function NumberField({
  form,
  name,
  label,
  help,
  min,
  max,
  step,
  disabled = false,
}: {
  form: UseFormReturn<GenerationFormValues>;
  name:
    | 'seed'
    | 'qwenTemperature'
    | 'qwenTopP'
    | 'qwenTopK'
    | 'qwenRepetitionPenalty'
    | 'indexTtsEmoAlpha'
    | 'indexTtsDurationFactor'
    | 'indexTtsHappy'
    | 'indexTtsAngry'
    | 'indexTtsSad'
    | 'indexTtsAfraid'
    | 'indexTtsDisgusted'
    | 'indexTtsMelancholic'
    | 'indexTtsSurprised'
    | 'indexTtsCalm'
    | 'maxChunkChars'
    | 'crossfadeMs';
  label: string;
  help: string;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs font-medium text-foreground/90">
            <span>{label}</span>
            <FieldHelp label={label} text={help} />
          </div>
          <FormControl>
            <Input
              type="number"
              min={min}
              max={max}
              step={step}
              disabled={disabled}
              value={field.value ?? ''}
              onChange={(event) => {
                const raw = event.target.value;
                field.onChange(raw === '' ? undefined : Number(raw));
              }}
              className="h-8 text-xs bg-card"
            />
          </FormControl>
          <FormMessage className="text-xs" />
        </FormItem>
      )}
    />
  );
}

type IndexTTSVectorFieldName =
  | 'indexTtsHappy'
  | 'indexTtsAngry'
  | 'indexTtsSad'
  | 'indexTtsAfraid'
  | 'indexTtsDisgusted'
  | 'indexTtsMelancholic'
  | 'indexTtsSurprised'
  | 'indexTtsCalm';

const INDEXTTS_VECTOR_FIELDS: Record<
  IndexTTSEmotionDimension,
  { name: IndexTTSVectorFieldName; label: string; help: string }
> = {
  happy: { name: 'indexTtsHappy', label: 'Happy', help: 'Adds positive, joyful delivery.' },
  angry: { name: 'indexTtsAngry', label: 'Angry', help: 'Adds anger or forcefulness.' },
  sad: { name: 'indexTtsSad', label: 'Sad', help: 'Adds sadness to the delivery.' },
  afraid: { name: 'indexTtsAfraid', label: 'Afraid', help: 'Adds fear or apprehension.' },
  disgusted: { name: 'indexTtsDisgusted', label: 'Disgusted', help: 'Adds aversion or disgust.' },
  melancholic: {
    name: 'indexTtsMelancholic',
    label: 'Melancholic',
    help: 'Adds subdued, low, reflective emotion.',
  },
  surprised: {
    name: 'indexTtsSurprised',
    label: 'Surprised',
    help: 'Adds surprise or startled emphasis.',
  },
  calm: { name: 'indexTtsCalm', label: 'Calm', help: 'Adds composed, restrained delivery.' },
};

function IndexSliderField({
  form,
  name,
  label,
  help,
  min,
  max,
  step,
}: {
  form: UseFormReturn<GenerationFormValues>;
  name: 'indexTtsEmoAlpha' | 'indexTtsDurationFactor' | IndexTTSVectorFieldName;
  label: string;
  help: string;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <FormField
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem className="space-y-1.5">
          <div className="flex items-center justify-between gap-2 text-xs font-medium">
            <span className="flex items-center gap-1.5">
              {label}
              <FieldHelp label={label} text={help} />
            </span>
            <span className="tabular-nums text-muted-foreground">
              {Number(field.value).toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Slider
              min={min}
              max={max}
              step={step}
              value={[Number(field.value)]}
              onValueChange={([value]) => field.onChange(value)}
              className="flex-1"
              aria-label={label}
            />
            <Input
              type="number"
              min={min}
              max={max}
              step={step}
              value={field.value}
              onChange={(event) => field.onChange(Number(event.target.value))}
              className="h-8 w-20 text-xs tabular-nums"
              aria-label={`${label} value`}
            />
          </div>
          <FormMessage className="text-xs" />
        </FormItem>
      )}
    />
  );
}

export function AdvancedGenerationOptions({ form }: AdvancedGenerationOptionsProps) {
  const [syntaxOpen, setSyntaxOpen] = useState(false);
  const [uploadingEmotionAudio, setUploadingEmotionAudio] = useState(false);
  const emotionFileInput = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  const engine = form.watch('engine') || 'qwen';
  const qwenEnabled = engine === 'qwen';
  const indexTtsEnabled = engine === 'indextts';
  const emotionMode = form.watch('indexTtsEmotionMode');
  const visibleIndexControls = getIndexTTSVisibleControls(emotionMode);
  const vectorTotal = INDEXTTS_EMOTION_DIMENSIONS.reduce(
    (sum, dimension) => sum + Number(form.watch(INDEXTTS_VECTOR_FIELDS[dimension].name) || 0),
    0,
  );

  async function uploadEmotionAudio(file: File) {
    setUploadingEmotionAudio(true);
    form.clearErrors('indexTtsEmotionAudioAssetId');
    try {
      const asset = await apiClient.uploadEmotionAudio(file);
      form.setValue('indexTtsEmotionAudioAssetId', asset.id, { shouldDirty: true });
      form.setValue('indexTtsEmotionAudioFilename', asset.filename, { shouldDirty: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to upload emotion audio';
      form.setError('indexTtsEmotionAudioAssetId', { type: 'server', message });
      toast({ title: 'Emotion audio upload failed', description: message, variant: 'destructive' });
    } finally {
      setUploadingEmotionAudio(false);
      if (emotionFileInput.current) emotionFileInput.current.value = '';
    }
  }

  return (
    <>
      <details className="group/advanced mt-3 rounded-2xl border border-border/70 bg-card/40 px-3 py-2">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-medium text-foreground/90 select-none">
          <span>Advanced Options</span>
          <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open/advanced:rotate-180" />
        </summary>

        <div className="mt-3 space-y-4 border-t border-border/60 pt-3">
          {qwenEnabled && (
            <div>
              <div className="mb-2 flex items-center justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold">Qwen Base sampling</div>
                  <div className="text-[11px] text-muted-foreground">
                    Fine-tune variation and stability for Qwen cloned voices.
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <NumberField
                  form={form}
                  name="qwenTemperature"
                  label="Temperature"
                  help="Controls randomness. Lower values are steadier and more repeatable; higher values can sound more varied or expressive but may become less stable. Qwen's reference default is 0.9."
                  min={0.1}
                  max={1.5}
                  step={0.05}
                />
                <NumberField
                  form={form}
                  name="qwenTopP"
                  label="Top P"
                  help="Nucleus sampling cutoff. Lower values restrict generation to more likely choices; 1.0 leaves the full probability mass available."
                  min={0.1}
                  max={1}
                  step={0.05}
                />
                <NumberField
                  form={form}
                  name="qwenTopK"
                  label="Top K"
                  help="Limits each sampling step to the K most likely choices. Lower values are more conservative; higher values allow more variation. Qwen's reference default is 50."
                  min={1}
                  max={100}
                  step={1}
                />
                <NumberField
                  form={form}
                  name="qwenRepetitionPenalty"
                  label="Repetition penalty"
                  help="Discourages repeated token patterns. Slightly increasing this can help with loops or repeated phrasing; excessive values can make speech unnatural. Qwen's reference default is 1.05."
                  min={1}
                  max={1.5}
                  step={0.01}
                />
              </div>
            </div>
          )}

          {indexTtsEnabled && (
            <section className="space-y-4">
              <div>
                <div className="text-xs font-semibold">IndexTTS 2.5 emotion and delivery</div>
                <div className="text-[11px] text-muted-foreground">
                  Choose one emotion source independently from the cloned speaker reference.
                </div>
              </div>

              <FormField
                control={form.control}
                name="indexTtsEmotionMode"
                render={({ field }) => (
                  <FormItem className="space-y-1.5">
                    <div className="flex items-center gap-1.5 text-xs font-medium">
                      Emotion / Delivery Mode
                      <FieldHelp label="Emotion / Delivery Mode" text={INDEXTTS_HELP.mode} />
                    </div>
                    <Select
                      value={field.value}
                      onValueChange={(value) => {
                        field.onChange(value);
                        if (value === 'natural' || value === 'audio') {
                          form.setValue('indexTtsUseRandom', false, { shouldDirty: true });
                        }
                      }}
                    >
                      <FormControl>
                        <SelectTrigger className="h-8 bg-card text-xs">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {INDEXTTS_EMOTION_MODES.map((mode) => (
                          <SelectItem key={mode.value} value={mode.value}>
                            {mode.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {emotionMode === 'auto' && (
                      <p className="text-[11px] leading-relaxed text-muted-foreground">
                        IndexTTS analyzes the spoken script to infer delivery. Official guidance
                        suggests moderate strength, around 0.6 or lower, for natural output.
                      </p>
                    )}
                    {emotionMode === 'natural' && (
                      <p className="text-[11px] leading-relaxed text-muted-foreground">
                        Uses the cloned speaker recording as its natural emotion reference.
                      </p>
                    )}
                    <FormMessage className="text-xs" />
                  </FormItem>
                )}
              />

              {visibleIndexControls.instruction && (
                <FormField
                  control={form.control}
                  name="indexTtsEmotionText"
                  render={({ field }) => (
                    <FormItem className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-xs font-medium">
                        Emotion / delivery instruction
                        <FieldHelp
                          label="Emotion / delivery instruction"
                          text={INDEXTTS_HELP.instruction}
                        />
                      </div>
                      <FormControl>
                        <Textarea
                          {...field}
                          maxLength={500}
                          placeholder="calm, grave, measured, restrained concern"
                          className="min-h-20 resize-y bg-card text-xs"
                        />
                      </FormControl>
                      <FormMessage className="text-xs" />
                    </FormItem>
                  )}
                />
              )}

              {visibleIndexControls.vector && (
                <div className="space-y-3 rounded-xl border border-border/60 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="text-xs font-medium">Official 8-value emotion vector</div>
                      <div
                        className={`text-[11px] ${vectorTotal > 0.8 ? 'text-destructive' : 'text-muted-foreground'}`}
                      >
                        Total {vectorTotal.toFixed(2)} / 0.80 maximum
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-7 gap-1 text-xs"
                      onClick={() => {
                        for (const dimension of INDEXTTS_EMOTION_DIMENSIONS) {
                          form.setValue(
                            INDEXTTS_VECTOR_FIELDS[dimension].name,
                            INDEXTTS_NEUTRAL_VECTOR[dimension],
                            { shouldDirty: true },
                          );
                        }
                      }}
                    >
                      <RotateCcw className="h-3 w-3" /> Reset
                    </Button>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {INDEXTTS_EMOTION_DIMENSIONS.map((dimension) => {
                      const config = INDEXTTS_VECTOR_FIELDS[dimension];
                      return (
                        <IndexSliderField
                          key={dimension}
                          form={form}
                          name={config.name}
                          label={config.label}
                          help={`${config.help} Official values range from 0 to 1; the combined vector must total 0.8 or less.`}
                          min={0}
                          max={0.8}
                          step={0.05}
                        />
                      );
                    })}
                  </div>
                </div>
              )}

              {visibleIndexControls.audio && (
                <FormField
                  control={form.control}
                  name="indexTtsEmotionAudioAssetId"
                  render={() => (
                    <FormItem className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-xs font-medium">
                        Emotion reference audio
                        <FieldHelp
                          label="Emotion reference audio"
                          text="A separate 2–15 second recording supplies emotion/delivery while the profile sample supplies speaker identity. Uploads are normalized and retained for seven days so retry/regenerate can reuse them."
                        />
                      </div>
                      <input
                        ref={emotionFileInput}
                        type="file"
                        accept="audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/aac,audio/webm,audio/opus"
                        className="sr-only"
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          if (file) void uploadEmotionAudio(file);
                        }}
                      />
                      {form.watch('indexTtsEmotionAudioFilename') ? (
                        <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card px-3 py-2 text-xs">
                          <span className="truncate">
                            {form.watch('indexTtsEmotionAudioFilename')}
                          </span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0"
                            aria-label="Clear emotion reference audio"
                            onClick={() => {
                              form.setValue('indexTtsEmotionAudioAssetId', '', {
                                shouldDirty: true,
                              });
                              form.setValue('indexTtsEmotionAudioFilename', '', {
                                shouldDirty: true,
                              });
                            }}
                          >
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ) : (
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-8 gap-1.5 text-xs"
                          disabled={uploadingEmotionAudio}
                          onClick={() => emotionFileInput.current?.click()}
                        >
                          {uploadingEmotionAudio ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Upload className="h-3.5 w-3.5" />
                          )}
                          {uploadingEmotionAudio ? 'Uploading…' : 'Choose audio file'}
                        </Button>
                      )}
                      <FormMessage className="text-xs" />
                    </FormItem>
                  )}
                />
              )}

              <div className="grid gap-4 sm:grid-cols-2">
                {visibleIndexControls.strength && (
                  <IndexSliderField
                    form={form}
                    name="indexTtsEmoAlpha"
                    label="Emotion strength"
                    help={INDEXTTS_HELP.strength}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                )}
                <IndexSliderField
                  form={form}
                  name="indexTtsDurationFactor"
                  label="Speaking speed"
                  help={INDEXTTS_HELP.speed}
                  min={0.5}
                  max={2}
                  step={0.05}
                />
              </div>

              {visibleIndexControls.random && (
                <FormField
                  control={form.control}
                  name="indexTtsUseRandom"
                  render={({ field }) => (
                    <FormItem className="flex items-start gap-2 space-y-0">
                      <FormControl>
                        <Checkbox checked={field.value} onCheckedChange={field.onChange} />
                      </FormControl>
                      <div className="flex items-center gap-1.5 text-xs">
                        Random emotion sampling
                        <FieldHelp label="Random emotion sampling" text={INDEXTTS_HELP.random} />
                      </div>
                    </FormItem>
                  )}
                />
              )}
            </section>
          )}

          <div className="border-t border-border/60 pt-3">
            <div className="mb-2 text-xs font-semibold">Pacing and reproducibility</div>
            <div className="grid grid-cols-2 gap-3">
              <NumberField
                form={form}
                name="maxChunkChars"
                label="Max chunk characters"
                help="Long text is split at natural boundaries before generation. Smaller chunks usually improve pacing and sentence endings, but can make delivery less continuous."
                min={100}
                max={5000}
                step={50}
              />
              <NumberField
                form={form}
                name="crossfadeMs"
                label="Chunk crossfade (ms)"
                help="Short overlap used when joining automatically generated chunks. It smooths joins but does not apply across an explicit [pause ...] directive."
                min={0}
                max={500}
                step={10}
              />
              <NumberField
                form={form}
                name="seed"
                label="Seed"
                help={
                  indexTtsEnabled
                    ? 'Sets IndexTTS Python, NumPy, Torch, and CUDA sampling seeds. Reuse the same integer and options for a more repeatable result; leave blank for a fresh variation.'
                    : 'Use the same integer to make sampling more repeatable. Leave blank for a fresh random generation.'
                }
                min={0}
                step={1}
              />
            </div>
          </div>

          <button
            type="button"
            onClick={() => setSyntaxOpen(true)}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:underline"
          >
            <Code2 className="h-3.5 w-3.5" />
            View available text syntax
          </button>
        </div>
      </details>

      <Dialog open={syntaxOpen} onOpenChange={setSyntaxOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Generation text syntax</DialogTitle>
            <DialogDescription>
              These directives can be placed directly inside the text you generate.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-5 text-sm">
            <section>
              <h3 className="mb-2 font-semibold">VOICEBOX-NATIVE</h3>
              <p className="mb-2 text-xs text-muted-foreground">
                Voicebox parses these directives and inserts audio before the selected model output
                is joined.
              </p>
              <div className="space-y-2">
                <div className="rounded-lg border border-border bg-muted/30 p-3">
                  <code className="text-xs">[pause 1200]</code>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Inserts exactly 1,200 milliseconds of silence. Bare numbers are milliseconds.
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-muted/30 p-3">
                  <code className="text-xs">[pause 500ms]</code>
                  <span className="mx-2 text-muted-foreground">or</span>
                  <code className="text-xs">[pause 1.2s]</code>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Equivalent explicit unit forms. Individual pauses are capped at 30 seconds.
                  </p>
                </div>
              </div>
            </section>

            <section>
              <h3 className="mb-2 font-semibold">CHATTERBOX TURBO-NATIVE</h3>
              <p className="mb-2 text-xs text-muted-foreground">
                These are model-supported paralinguistic events. Other engines may read them aloud,
                ignore them, or behave unpredictably.
              </p>
              <div className="flex flex-wrap gap-2">
                {[
                  '[laugh]',
                  '[chuckle]',
                  '[gasp]',
                  '[cough]',
                  '[sigh]',
                  '[groan]',
                  '[sniff]',
                  '[shush]',
                  '[clear throat]',
                ].map((tag) => (
                  <code
                    key={tag}
                    className="rounded-md border border-border bg-muted/30 px-2 py-1 text-xs"
                  >
                    {tag}
                  </code>
                ))}
              </div>
            </section>

            <section>
              <h3 className="mb-2 font-semibold">INDEXTTS 2.5-NATIVE PRONUNCIATION</h3>
              <p className="mb-2 text-xs text-muted-foreground">
                Replace a word&apos;s pronunciation with <code>&lt;word|pronunciation&gt;</code>.
                These tags apply only when IndexTTS 2.5 is selected.
              </p>
              <div className="space-y-2">
                {INDEXTTS_PRONUNCIATION_EXAMPLES.map((example) => (
                  <div
                    key={example.label}
                    className="rounded-lg border border-border bg-muted/30 p-3"
                  >
                    <div className="text-xs font-medium">{example.label}</div>
                    <code className="mt-1 block text-xs">{example.text}</code>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-lg border border-border bg-muted/20 p-3">
              <h3 className="mb-1 text-xs font-semibold">Example</h3>
              <pre className="whitespace-pre-wrap text-xs text-muted-foreground">{`The entity operates under the historic designation.\n\n[pause 1200]\n\nSanta Claus.`}</pre>
            </section>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
