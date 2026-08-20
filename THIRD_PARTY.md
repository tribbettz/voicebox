# Optional third-party runtimes and models

## IndexTTS 2.5

Voicebox can optionally download and run [IndexTTS](https://github.com/index-tts/index-tts)
for expressive zero-shot voice cloning. The integration pins upstream source
commit `ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c` and the
[`IndexTeam/IndexTTS-2.5`](https://huggingface.co/IndexTeam/IndexTTS-2.5) model
snapshot `c39ce5ba981572cb187443877ff559dfb246ce63`.

IndexTTS source, model weights, and outputs are governed by the **bilibili Model
Use License Agreement**, not Voicebox's MIT license. Review the current
agreement in the [upstream repository](https://github.com/index-tts/index-tts/blob/ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c/LICENSE)
before downloading or using IndexTTS. The downloaded model directory retains
the upstream `LICENSE` file. Voicebox does not redistribute the weights.

IndexTTS also uses separately licensed auxiliary models from Meta/Facebook
(`w2v-bert-2.0`), FunASR (`campplus`), and NVIDIA (`BigVGAN`). Their pinned
files are downloaded directly from their official Hugging Face repositories;
users are responsible for reviewing the corresponding repository terms.
