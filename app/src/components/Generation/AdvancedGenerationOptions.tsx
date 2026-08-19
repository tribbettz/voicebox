import { ChevronDown, CircleHelp, Code2 } from 'lucide-react';
import { useState } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { FormControl, FormField, FormItem, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

interface AdvancedGenerationOptionsProps {
  form: UseFormReturn<GenerationFormValues>;
}

function FieldHelp({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex shrink-0" tabIndex={0}>
      <CircleHelp className="h-3.5 w-3.5 text-muted-foreground/70" aria-hidden />
      <span className="pointer-events-none absolute bottom-full left-1/2 z-[9999] mb-2 hidden w-64 -translate-x-1/2 rounded-md border border-border bg-popover px-3 py-2 text-xs leading-relaxed text-popover-foreground shadow-lg group-hover:block group-focus:block">
        {text}
      </span>
    </span>
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
            <FieldHelp text={help} />
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

export function AdvancedGenerationOptions({ form }: AdvancedGenerationOptionsProps) {
  const [syntaxOpen, setSyntaxOpen] = useState(false);
  const engine = form.watch('engine') || 'qwen';
  const qwenEnabled = engine === 'qwen';

  return (
    <>
      <details className="group mt-3 rounded-2xl border border-border/70 bg-card/40 px-3 py-2">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-medium text-foreground/90 select-none">
          <span>Advanced Options</span>
          <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
        </summary>

        <div className="mt-3 space-y-4 border-t border-border/60 pt-3">
          <div>
            <div className="mb-2 flex items-center justify-between gap-3">
              <div>
                <div className="text-xs font-semibold">Qwen Base sampling</div>
                <div className="text-[11px] text-muted-foreground">
                  {qwenEnabled
                    ? 'Fine-tune variation and stability for Qwen cloned voices.'
                    : 'Available when Qwen TTS 1.7B or 0.6B is selected.'}
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
                disabled={!qwenEnabled}
              />
              <NumberField
                form={form}
                name="qwenTopP"
                label="Top P"
                help="Nucleus sampling cutoff. Lower values restrict generation to more likely choices; 1.0 leaves the full probability mass available."
                min={0.1}
                max={1}
                step={0.05}
                disabled={!qwenEnabled}
              />
              <NumberField
                form={form}
                name="qwenTopK"
                label="Top K"
                help="Limits each sampling step to the K most likely choices. Lower values are more conservative; higher values allow more variation. Qwen's reference default is 50."
                min={1}
                max={100}
                step={1}
                disabled={!qwenEnabled}
              />
              <NumberField
                form={form}
                name="qwenRepetitionPenalty"
                label="Repetition penalty"
                help="Discourages repeated token patterns. Slightly increasing this can help with loops or repeated phrasing; excessive values can make speech unnatural. Qwen's reference default is 1.05."
                min={1}
                max={1.5}
                step={0.01}
                disabled={!qwenEnabled}
              />
            </div>
          </div>

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
                help="Use the same integer to make sampling more repeatable. Leave blank for a fresh random generation."
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
              <h3 className="mb-2 font-semibold">All engines</h3>
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
              <h3 className="mb-2 font-semibold">Chatterbox Turbo only</h3>
              <p className="mb-2 text-xs text-muted-foreground">
                These are model-supported paralinguistic events. Other engines may read them aloud, ignore them, or behave unpredictably.
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
                  <code key={tag} className="rounded-md border border-border bg-muted/30 px-2 py-1 text-xs">
                    {tag}
                  </code>
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
