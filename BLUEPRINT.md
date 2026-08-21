# Sorani Voice Studio  
## A–Z Blueprint for a Lean, Production-Grade Kurdish TTS Platform

**Version:** 21 August 2026  
**Primary model:** VoxCPM2  
**Honorable challenger:** CosyVoice3  
**Goal:** commercially usable, exceptionally natural Sorani speech, professional voice cloning, expressive delivery, streaming inference, and repeatable fine-tuning.

---

## 1. Final Architecture Decision

Build a **modular control plane with isolated GPU workers**, not a large microservice system.

### Core stack

| Layer | Recommended choice |
|---|---|
| TTS foundation | **VoxCPM2** |
| Research challenger | **CosyVoice3** |
| Production inference | **vLLM-Omni** |
| LoRA research/preview | Official VoxCPM Python runtime |
| Frontend | **Next.js 16.3, TypeScript, Tailwind CSS, shadcn/ui** |
| Backend/API | **FastAPI, Pydantic, SQLAlchemy, Alembic** |
| Workflow engine | **Temporal Cloud** |
| GPU training launcher | **SkyPilot Managed Jobs** |
| Database | **PostgreSQL**, optionally pgvector |
| Audio/model storage | **S3-compatible object storage** |
| Experiment tracking | **Weights & Biases** |
| Monitoring | **OpenTelemetry, Sentry, Prometheus/Grafana** |
| Generated-audio watermarking | **AudioSeal** |
| Local development | Docker Compose |
| Infrastructure | OpenTofu/Terraform |
| CI/CD | GitHub Actions |

VoxCPM2 remains the best core because it combines Apache-2.0 licensing, 2B parameters, 2.36 million hours of multilingual pretraining, controllable cloning, voice design, native 48-kHz output, streaming, official LoRA/full-SFT support, and Arabic/Turkish coverage. Sorani itself is not currently one of its 30 officially supported languages.

---

## 2. Brutal Data Reality

A **30-hour fine-tune may create an impressive demo**, but it does not establish a robust Sorani foundation model.

VoxCPM’s official guidance categorizes new-language adaptation as a **full-fine-tuning task requiring roughly 500+ hours**, whereas speaker or style adaptation can use much less data. Official estimates are approximately 20 GB VRAM for VoxCPM2 LoRA and 40 GB for full fine-tuning before additional distributed-training overhead.

Therefore:

| Stage | Realistic data target |
|---|---:|
| Architecture pilot | 30–50 clean hours |
| First viable Sorani foundation | 150–250 clean hours |
| Serious production foundation | **300–500+ clean hours** |
| Neutral premium voice | 3–6 studio hours |
| Flagship expressive voice | **12–20 studio hours** |
| Canonical zero-shot reference pack | 6–12 clips, 8–20 seconds each |

The pilot determines whether VoxCPM2 actually beats the existing F5-TTS model. It is not the final dataset.

---

## 3. System Topology

```text
┌──────────────────────────────────────────────────────────────────┐
│                          Web Browser                             │
│ Next.js studio: speakers, datasets, training, evaluation, TTS   │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS / SSE / streaming audio
┌─────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Control Plane                       │
│ Auth • RBAC • speaker rights • API • quotas • normalization     │
│ dataset/version management • model registry • audit logging     │
└──────────────┬──────────────────┬──────────────────┬──────────────┘
               │                  │                  │
       ┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
       │ PostgreSQL     │ │ S3/R2 Storage  │ │ Temporal Cloud │
       │ metadata/state │ │ audio/models   │ │ durable jobs   │
       └────────────────┘ └────────────────┘ └──────┬─────────┘
                                                    │
                       ┌────────────────────────────┼───────────────┐
                       │                            │               │
              ┌────────▼────────┐         ┌─────────▼───────┐ ┌────▼───────┐
              │ CPU Data Worker│         │ GPU Training    │ │ Evaluation │
              │ FFmpeg, VAD,   │         │ SkyPilot jobs   │ │ workers    │
              │ ASR, text QA   │         │ VoxCPM scripts  │ │ ASR/MOS/A-B│
              └─────────────────┘         └─────────────────┘ └────────────┘
                                                    │
┌───────────────────────────────────────────────────▼──────────────┐
│                       Model Registry                             │
│ base model → Sorani foundation → speaker adapters → deployments │
└───────────────────────────────────────────────────┬──────────────┘
                                                    │
┌───────────────────────────────────────────────────▼──────────────┐
│                 Internal Inference Service                      │
│ vLLM-Omni VoxCPM2 • official VoxCPM LoRA workers • AudioSeal    │
└──────────────────────────────────────────────────────────────────┘
```

The architecture deliberately contains only four meaningful deployable systems:

1. Web frontend.
2. Control API and workflow workers.
3. Training/data workers.
4. GPU inference service.

Do not split speakers, datasets, billing, evaluation and training into separate microservices initially.

---

## 4. Best Frontend

Use:

- **Next.js 16.3 App Router**
- TypeScript
- Tailwind CSS
- shadcn/ui
- TanStack Query
- React Hook Form and Zod
- WaveSurfer.js for waveform review
- AudioWorklet for gapless streamed playback
- next-intl for English/Sorani localization
- CSS logical properties for native RTL support

Next.js 16.3 is the current stable line as of August 2026. Production should stay on the latest Active-LTS security release rather than a canary build.

### Required screens

#### Dashboard

Show:

- active voices
- dataset hours
- training runs
- deployed model
- synthesis volume
- quality regressions
- GPU utilization
- failed jobs

#### Speaker Profiles

Each profile shows:

- speaker identity and dialect
- consent and commercial rights
- canonical references
- available styles
- recording progress
- voice adapter
- deployed model version
- similarity and naturalness scores
- usage history
- revoke/disable control

#### Dataset Studio

Provide:

- resumable uploads
- waveform and transcript
- raw versus normalized text
- automatic quality flags
- speaker verification
- style label
- accept, reject or retake
- keyboard-first review
- immutable dataset freezing

#### Training Studio

Provide controlled presets rather than dozens of raw parameters:

- Sorani pilot LoRA
- full Sorani foundation SFT
- premium-speaker LoRA
- style/domain adaptation
- continuation from checkpoint

Show:

- dataset version
- estimated GPU requirements
- selected base model
- current step
- validation loss
- evaluation samples
- cost guardrail
- checkpoint history

#### Evaluation Lab

Support blind:

- A/B comparisons
- pairwise preference
- MOS scoring
- pronunciation scoring
- speaker-similarity scoring
- emotion-adherence scoring
- F5 versus VoxCPM2 versus CosyVoice3

#### Voice Playground

Controls:

- voice
- text
- style
- emotional intensity
- speed
- seed
- reference clip
- pronunciation overrides
- streaming on/off
- WAV/FLAC/MP3 output

#### Deployments

Show:

- active model
- canary model
- voice adapters
- API keys
- latency
- rollback
- audit history

---

## 5. Best Backend

Use a **FastAPI modular monolith** because the entire ML stack is Python.

### Backend packages

- Python **3.11**
- FastAPI
- Pydantic
- SQLAlchemy 2
- Alembic
- PostgreSQL
- boto3 or an S3-compatible client
- Temporal Python SDK
- FFmpeg
- PyTorch
- Hugging Face Datasets
- official VoxCPM repository
- SkyPilot client

Python 3.10–3.11 is also the officially recommended VoxCPM training environment, so Python 3.11 avoids unnecessary ML dependency breakage.

### Contract strategy

FastAPI owns the OpenAPI specification.

Generate the TypeScript client automatically from OpenAPI. Do not manually duplicate request and response types between Python and TypeScript.

### No Redis initially

Temporal already provides durable task execution, while PostgreSQL owns product state and vLLM handles inference scheduling. Redis adds another failure surface without solving a necessary first-stage problem.

Temporal is appropriate because training, dataset processing and evaluation workflows must resume after container crashes, GPU failures or network interruptions.

---

## 6. GPU Infrastructure

Use **SkyPilot Managed Jobs** to launch unchanged Dockerized VoxCPM training jobs across RunPod, Lambda Cloud, AWS, GCP or later a private Kubernetes cluster.

SkyPilot provides job recovery across GPU failures and spot preemption, supports multiple GPU providers and can restart from checkpoints stored in cloud object storage.

### Recommended compute

| Workload | Recommended GPU |
|---|---|
| LoRA pilot | 1× 48-GB GPU |
| Single-speaker LoRA | 1× 24–48-GB GPU |
| Full Sorani SFT | 1× 80-GB GPU or multi-GPU |
| Development inference | 1× 24-GB GPU |
| Production inference | 1× L40S 48 GB per warm replica |
| Heavy evaluation/batch synthesis | spare 24–48-GB worker |

Although the official vLLM-Omni recipe says VoxCPM2 can run on a single 24-GB consumer GPU, 48 GB gives safer concurrency and room for production buffers.

### Every training job must

- run from an immutable Docker image
- receive an immutable dataset ID
- checkpoint to S3
- log to W&B
- record Git commit and image digest
- support restart from the latest checkpoint
- upload final weights and evaluation samples
- terminate its GPU automatically

---

## 7. Data Sources: Use, Reject or License

### Useful existing resources

#### Mozilla Common Voice 26 Central Kurdish

It contains:

- 194.64 recorded hours
- **137.9 validated hours**
- 170,972 clips
- 2,038 speakers
- CC0 licensing

Use it for **language breadth**, pronunciation diversity and initial foundation adaptation—not for cloning identifiable Common Voice speakers. Mozilla also forbids attempting to determine speaker identities and restricts re-hosting, so commercial generative use still deserves a dedicated legal review.

#### Central Kurdish TTS Dataset 1.0

This provides 2 hours and 18 minutes of aligned, high-quality single-speaker speech under CC-BY-4.0. It is useful as a small quality reference, pronunciation benchmark and supplementary training source.

#### Existing F5-TTS data

Retain the user’s existing clean F5 dataset as:

- the first pilot training set
- the principal quality baseline
- a source of difficult Sorani test sentences
- evidence for which spelling and phoneme patterns already work

### Research-only or legally unsuitable

The TTS4All Central Kurdish dataset contains more than 35 hours from three speakers, but its CC-BY-NC-ND license makes it unsuitable for the commercial production model. It may be used only where its license allows research evaluation.

RegaLabs-TTS genuinely exists and reports a 53-hour CosyVoice3 Sorani adaptation. However, the quality claims are self-reported, female cloning is untested, and the README adds attribution and use conditions beyond a plain upstream Apache statement. Treat it as a **benchmark and implementation reference**, not a trusted commercial foundation without independent technical and legal review.

Any dataset with no explicit speaker consent, provenance or commercial training license enters **quarantine**, not the training pool.

---

## 8. Sorani Data Plan

### Production foundation target

Build toward:

- **300–500+ clean hours**
- 40–100 speakers
- balanced male/female representation
- Slemani and Erbil speech
- formal and conversational Sorani
- news, documentary and narration
- code-switching with English and Arabic
- numbers, dates, money and proper names
- expressive but naturally acted material

### Recommended composition

| Category | Share |
|---|---:|
| Clean neutral multi-speaker speech | 45% |
| Conversational/warm speech | 20% |
| News/documentary/formal | 15% |
| Expressive/emotional | 10% |
| Names, numbers, code-switching, technical language | 10% |

### Flagship speaker recording

For every premium voice, record approximately:

| Style | Share |
|---|---:|
| Neutral general-purpose | 35% |
| Warm conversational | 20% |
| Authoritative/documentary | 15% |
| Energetic/promotional | 10% |
| Empathetic/soft | 8% |
| Happy | 4% |
| Sad/serious | 4% |
| Urgent/angry controlled | 2% |
| Whisper-like | 2% |

A premium voice should not merely read the same style for 15 hours. Emotional range must be intentionally designed into the recording script.

---

## 9. Speaker Profile Architecture

A speaker profile is a governed production asset, not simply an uploaded WAV file.

### Required profile fields

#### Identity

- internal UUID
- professional/display name
- preferred Sorani spelling
- dialect and regional accent
- age range and voice descriptors
- organization or owner

#### Rights

- signed consent document
- commercial-use permission
- model-training permission
- voice-cloning permission
- derivative-model permission
- territories
- expiry
- prohibited contexts
- attribution requirement
- revocation terms

#### Acoustic information

- microphone
- room
- sample rate and bit depth
- recording engineer
- canonical speaker embedding
- background-noise profile
- approved reference clips

#### Voice assets

- neutral reference
- warm reference
- energetic reference
- serious reference
- emotional references
- exact reference transcripts
- Sorani adapter ID
- production model ID
- pronunciation dictionary
- style presets

#### Quality

- naturalness
- speaker similarity
- pronunciation score
- emotion adherence
- long-form stability
- last review date

#### Security

- active, suspended or revoked status
- permitted API keys or organizations
- synthesis audit log
- watermark identifier

### Critical rights design

Do not permanently mix every premium actor into the shared Sorani foundation.

Use:

```text
VoxCPM2
   ↓
Shared Sorani Foundation
   ├── Speaker A LoRA
   ├── Speaker B LoRA
   ├── Speaker C LoRA
   └── Speaker D LoRA
```

This gives better separation, easier versioning and a more realistic revocation path.

True removal of one speaker from a fully fine-tuned shared model is not reliably guaranteed. Therefore, identifiable premium voice data should enter the shared foundation only when the speaker contract explicitly permits an enduring derivative foundation model.

---

## 10. Sorani Text Frontend

Do not build every Kurdish text-processing rule from nothing.

Bootstrap from:

- `ckb-textify`
- KLPT
- AsoSoft Library
- AsoSoft G2P
- `ckb-g2p`

`ckb-textify` and AsoSoft already provide Sorani normalization, number handling and G2P-related functionality. However, these are community tools, not automatically trustworthy production standards. Fork, pin and validate them against a native-linguist test suite.

Research on Central Kurdish also confirms that orthographic inconsistency materially harms downstream language performance, making normalization a core model component rather than a cosmetic preprocessing step.

### Processing pipeline

```text
Raw text
  → Unicode normalization
  → Arabic/Persian/Kurdish character folding
  → spelling standardization
  → sentence segmentation
  → number/date/time/currency expansion
  → abbreviation expansion
  → phone and identifier handling
  → foreign-name transliteration
  → pronunciation overrides
  → punctuation/prosody preparation
  → normalized spoken form
```

### Must cover

- `ي` versus `ی`
- `ك` versus `ک`
- `ه` versus `ە`
- Arabic, Persian and Western digits
- percentages
- IQD, USD and other currencies
- dates and time
- phone numbers
- abbreviations
- URLs and emails
- English names
- Arabic names
- acronyms
- punctuation
- optional diacritics
- alternative Sorani spellings

### G2P use

VoxCPM2 is tokenizer-free and does not require an IPA input pipeline.

Use G2P for:

- coverage analysis
- phoneme balancing
- pronunciation QA
- test-set construction
- lexicon generation
- detecting rare or missing phonemes

For production synthesis, prefer a versioned mapping:

```text
written form → approved spoken Sorani form
```

rather than forcing every sentence through an unproven G2P output.

---

## 11. Audio Ingestion and Curation

### Preserve two forms

#### Immutable archive

- original WAV/FLAC
- 48 kHz
- 24-bit
- unchanged
- SHA-256 hash
- full recording metadata

#### Training derivative

- mono
- model-required sample rate
- silence trimmed
- no clipping
- gentle loudness consistency
- exact transcript
- no destructive denoising

VoxCPM2’s training encoder uses 16-kHz audio while its decoder produces 48-kHz output. Preserve the high-resolution archive and derive model-specific training files rather than permanently downsampling the source.

### Processing sequence

1. Create signed upload.
2. Hash and archive raw audio.
3. Verify speaker and consent.
4. Detect format, clipping and corruption.
5. Run VAD.
6. Run diarization only for multi-speaker recordings.
7. Segment speech.
8. Generate or import transcript.
9. Normalize Sorani text.
10. Compare transcript using independent ASR.
11. Score noise, silence and duration.
12. Human-review every accepted premium clip.
13. Freeze an immutable dataset version.

### Clip requirements

The official VoxCPM guidance identifies **3–30 seconds** as the practical range and warns that trailing silence above roughly 0.5 seconds commonly causes generation that fails to stop. Exact audio/text matching is essential.

For premium recordings, target mostly **3–18-second** clips.

### Internal utterance record

```json
{
  "utterance_id": "uuid",
  "speaker_id": "uuid",
  "audio_uri": "s3://bucket/derived/example.wav",
  "raw_text": "original transcript",
  "normalized_text": "approved spoken text",
  "duration_seconds": 8.42,
  "style": "warm_conversational",
  "dialect": "slemani",
  "source": "studio",
  "consent_id": "uuid",
  "quality_status": "approved",
  "dataset_version": "ckb-foundation-v3"
}
```

At training launch, generate the official VoxCPM JSONL structure containing `audio`, `text`, optional same-speaker `ref_audio`, duration and dataset ID.

---

## 12. Training Strategy

### Stage 0 — Locked baselines

Before adapting anything, generate a permanent benchmark from:

1. Current F5-TTS model.
2. VoxCPM2 zero-shot.
3. CosyVoice3.
4. RegaLabs Sorani model where licensing permits testing.

Never change the benchmark sentences after model development begins.

### Stage 1 — Sorani pilot LoRA

Use:

- 30–50 hours
- multi-speaker Sorani
- speaker-disjoint validation
- official VoxCPM LoRA implementation
- at least two LoRA configurations
- one 48-GB GPU

Purpose:

- prove Sorani pronunciation
- test emotional control
- compare against F5
- identify missing frontend rules
- determine whether full SFT is justified

### Stage 2 — Production Sorani foundation

Move to full SFT only after the pilot passes.

Use:

- 300–500+ hours
- 40–100 speakers
- clean Sorani majority
- a small replay set from supported languages to reduce catastrophic forgetting
- explicit code-switch and style examples
- one 80-GB GPU or multi-GPU training
- regular Sorani evaluation checkpoints

The official VoxCPM pipeline supports both LoRA and full SFT through the same training script and supports multi-GPU execution with `torchrun`.

### Stage 3 — Premium voice adapters

Train a separate LoRA from the approved Sorani foundation.

For every target utterance, randomly select a different clean reference clip from the same speaker. This follows VoxCPM’s official `ref_audio` conditioning design.

Start with:

- 3–6 hours for a neutral commercial voice
- 12–20 hours for a flagship expressive voice
- held-out scripts and held-out recording sessions
- multiple style references

### Stage 4 — Style calibration

Do not assume text instructions alone will produce stable emotion.

For each voice preset, store:

- natural-language control instruction
- canonical reference clip
- reference transcript
- generation parameters
- tested seed range
- approved example output

Example preset:

```text
Style: warm_documentary
Instruction: calm, warm, assured documentary narration; measured pace
Reference: speaker-a/warm-documentary-03.wav
Speed: 0.96
CFG: approved production setting
```

### Stage 5 — Long-form tuning

Evaluate:

- 10-minute narration
- 30-minute audiobook chapter
- news article
- conversation
- repeated names
- punctuation-heavy text
- paragraphs containing English names

Do not approve a model based only on short demo sentences.

---

## 13. Model and Dataset Versioning

Every model must be reproducible from:

- base-model SHA
- training-code Git commit
- Docker image digest
- dataset-version SHA
- normalizer version
- G2P/lexicon version
- training configuration
- random seed
- GPU type
- checkpoint
- evaluation set version

### Model hierarchy

```text
openbmb/VoxCPM2
└── voxcpm2-ckb-foundation-v1
    ├── voice-lamo-v1
    ├── voice-a-v1
    ├── voice-b-v1
    └── voice-c-v1
```

### Model states

- draft
- training
- evaluating
- rejected
- approved
- canary
- production
- deprecated
- revoked

No checkpoint may reach production simply because training completed.

---

## 14. Inference and Voice Cloning

### Three product modes

#### Zero-shot preview

Input:

- 10–30 seconds of consented reference audio
- exact transcript where available
- target Sorani text

Use for testing, not automatically for public publishing.

#### Registered voice

Use:

- approved speaker profile
- precomputed reference cache
- canonical reference transcript
- approved style presets

vLLM-Omni supports precomputed VoxCPM2 custom voices and streaming through an OpenAI-compatible `/v1/audio/speech` interface.

#### Premium voice

Use:

- Sorani foundation
- speaker-specific LoRA or voice-specific deployment
- reference cache
- style preset
- pronunciation dictionary

The official VoxCPM runtime already supports LoRA loading, disabling, unloading and replacement.

### Important serving constraint

vLLM-Omni’s general VoxCPM2 inference and reference cloning are documented. Production-grade VoxCPM2 LoRA hot-swapping is not yet documented clearly enough to make it a launch dependency.

Therefore:

- use vLLM-Omni for foundation and registered-reference voices
- use official VoxCPM workers for LoRA voices initially
- run one dedicated deployment per approved flagship adapter if necessary
- consolidate only after adapter behavior is tested end to end

### External API

Expose an OpenAI-style endpoint:

```http
POST /v1/audio/speech
```

Request:

```json
{
  "model": "sorani-pro-v1",
  "input": "Normalized or raw Sorani text",
  "voice": "lamo",
  "style": "warm_documentary",
  "speed": 1.0,
  "seed": 42,
  "format": "wav",
  "stream": true
}
```

The FastAPI gateway resolves `voice` into:

- speaker profile
- rights status
- adapter or reference cache
- exact VoxCPM parameters
- pronunciation overrides
- watermark ID

### Never expose vLLM directly

Current vLLM documentation warns that API-key protection does not cover every inference-capable endpoint. Keep vLLM on a private network behind the FastAPI gateway or a hardened reverse proxy.

---

## 15. Streaming Audio

For the browser:

1. FastAPI validates the request.
2. Internal service calls vLLM-Omni.
3. PCM chunks stream to the browser.
4. AudioWorklet buffers and plays gaplessly.
5. Final WAV/FLAC can optionally be assembled and stored.

The vLLM-Omni VoxCPM2 example already includes gapless streaming with an AudioWorklet-based player. Reuse that approach rather than developing a custom audio scheduler.

Target production gates:

- P95 time to first audio below 500 ms
- real-time factor below 0.5
- no audible gaps
- no first-chunk clipping
- deterministic cancellation
- generation stops immediately when the user cancels

WebRTC and LiveKit are unnecessary for the initial TTS studio. Add them only for a future full-duplex voice-agent product.

---

## 16. Evaluation System

### Fixed test suites

#### Core Sorani

- common vocabulary
- difficult phonemes
- regional words
- minimal pairs
- spelling variants

#### Production text

- dates
- money
- phone numbers
- percentages
- news
- company names
- Kurdish personal names
- English and Arabic names
- technical vocabulary

#### Expressive speech

- warm
- happy
- serious
- urgent
- sad
- empathetic
- energetic
- whisper-like

#### Long-form

- paragraphs
- articles
- narration
- dialogue
- 10–30-minute generation

### Automated evaluation

Measure:

- normalized CER using an independent Sorani ASR
- speaker-embedding similarity
- duration and stop failures
- repetition and omission
- clipping
- silence ratio
- loudness
- long-form drift
- output watermark detection

Automated MOS or similarity scores are screening tools, not the final decision.

### Native-speaker evaluation

Use:

- 500–700 held-out sentences
- at least 20–30 native Sorani listeners
- blind randomized playback
- headphones recommendation
- pairwise preference plus 1–5 scores

Score separately:

1. Naturalness.
2. Sorani pronunciation.
3. Speaker similarity.
4. Emotion/style authenticity.
5. Text accuracy.
6. Long-form stability.

### Suggested production gates

- at least 55% blind preference over F5 with statistical confidence
- at least 95% native pronunciation acceptance
- under 1% severe repetition, omission or stop failures
- no meaningful speaker identity leakage between profiles
- style-adherence acceptance above 85%
- no quality regression after watermarking
- P95 TTFB under 500 ms on production hardware

If VoxCPM2 fails these gates, move the same dataset and evaluation protocol to **CosyVoice3** rather than changing multiple variables simultaneously.

CosyVoice3 remains the correct honorable challenger because it has full-stack training/deployment support, instruction-controlled emotion and speed, cross-lingual cloning and approximately 150-ms bi-streaming, although Sorani is not an upstream supported language.

---

## 17. Consent, Security and Abuse Prevention

### Voice enrollment

Require:

- signed agreement
- identity verification
- live consent phrase
- confirmation of training and cloning rights
- permitted-use selection
- human administrator approval

### Production restrictions

- no anonymous public clone-anyone endpoint
- no cloning from URLs
- no celebrity voice library without contracts
- no automatic activation after upload
- rate limits per voice
- organization-level access controls
- full synthesis audit history

### Data protection

- encrypt audio and models at rest
- use short-lived signed URLs
- isolate organizations
- never return raw canonical references
- log every training and synthesis action
- support consent expiry
- immediately block revoked voices

### Watermarking

Apply AudioSeal after synthesis and before delivery.

AudioSeal is MIT licensed, supports streaming, works with high-rate speech including 48-kHz audio, and can embed a 16-bit identifier. Use the identifier to map audio to a deployment or voice version.

Store:

- watermark payload
- model version
- speaker profile
- organization
- timestamp
- synthesis request ID

Watermarking helps provenance but does not eliminate abuse; the audit and consent system remains mandatory.

---

## 18. Database Model

### Core tables

```text
organizations
users
organization_members
api_keys

speakers
speaker_consents
speaker_references
speaker_styles
speaker_pronunciations

datasets
dataset_versions
utterances
utterance_reviews

training_runs
training_checkpoints
training_metrics

model_versions
voice_adapters
evaluation_runs
evaluation_scores

deployments
synthesis_jobs
generated_assets
audit_events
```

### Important relationships

```text
Speaker
  ├── many Consent records
  ├── many Reference clips
  ├── many Utterances
  ├── many Style presets
  └── many Voice adapters

Dataset
  └── immutable Dataset versions
       └── approved Utterances

Model version
  ├── one base model
  ├── one dataset version
  ├── many evaluation runs
  └── many voice adapters
```

Use PostgreSQL as the source of truth. Store large audio, manifests, checkpoints and outputs in object storage.

---

## 19. API Surface

### Speakers

```http
POST   /v1/speakers
GET    /v1/speakers
GET    /v1/speakers/{speaker_id}
PATCH  /v1/speakers/{speaker_id}
POST   /v1/speakers/{speaker_id}/consents
POST   /v1/speakers/{speaker_id}/references
POST   /v1/speakers/{speaker_id}/styles
POST   /v1/speakers/{speaker_id}/revoke
```

### Datasets

```http
POST   /v1/datasets
POST   /v1/datasets/{dataset_id}/uploads
POST   /v1/datasets/{dataset_id}/process
POST   /v1/datasets/{dataset_id}/freeze
GET    /v1/datasets/{dataset_id}/utterances
PATCH  /v1/utterances/{utterance_id}
```

### Training

```http
POST   /v1/training-runs
GET    /v1/training-runs/{run_id}
GET    /v1/training-runs/{run_id}/events
POST   /v1/training-runs/{run_id}/cancel
POST   /v1/training-runs/{run_id}/resume
```

Use Server-Sent Events for training progress and logs.

### Evaluation

```http
POST   /v1/evaluations
GET    /v1/evaluations/{evaluation_id}
POST   /v1/evaluations/{evaluation_id}/ratings
POST   /v1/evaluations/{evaluation_id}/approve
```

### Synthesis

```http
POST   /v1/audio/speech
POST   /v1/audio/preview-clone
GET    /v1/audio/jobs/{job_id}
DELETE /v1/audio/jobs/{job_id}
```

### Deployment

```http
POST   /v1/models/{model_id}/deploy
POST   /v1/deployments/{deployment_id}/canary
POST   /v1/deployments/{deployment_id}/promote
POST   /v1/deployments/{deployment_id}/rollback
```

---

## 20. Repository Structure

```text
sorani-voice/
├── apps/
│   └── web/                         # Next.js frontend
├── services/
│   ├── api/                         # FastAPI control plane
│   ├── workflows/                   # Temporal workflows
│   ├── data-worker/                 # audio/text processing
│   ├── evaluation-worker/           # automatic evaluation
│   └── inference-gateway/           # secure TTS gateway
├── ml/
│   ├── voxcpm/                      # pinned upstream integration
│   ├── training/                    # configs and launch code
│   ├── evaluation/                  # benchmark scripts
│   └── challenger-cosyvoice/        # controlled challenger
├── packages/
│   ├── ckb-frontend/                # Sorani normalization
│   ├── audio-processing/
│   ├── contracts/                   # generated API client
│   └── shared-config/
├── datasets/
│   ├── manifests/
│   ├── schemas/
│   └── benchmark/
├── infra/
│   ├── opentofu/
│   ├── skypilot/
│   └── docker/
├── tests/
│   ├── golden-text/
│   ├── audio-regression/
│   ├── api/
│   └── e2e/
└── docs/
    ├── architecture.md
    ├── data-policy.md
    ├── speaker-consent.md
    ├── evaluation-spec.md
    └── deployment-runbook.md
```

Pin the upstream VoxCPM repository to an exact commit. Keep modifications in a thin integration layer rather than heavily forking the training code.

---

## 21. Testing and CI

### On every pull request

Run:

- Ruff
- mypy or pyright
- pytest
- database migration checks
- frontend type checking
- ESLint
- unit tests
- golden Sorani normalizer tests
- API contract generation
- Playwright smoke tests
- Docker build

### On model changes

Run:

- 100-sentence Sorani regression set
- reference-cloning test
- three style presets
- stop/repetition test
- watermark detection
- latency smoke test

### Before deployment

Run:

- full benchmark
- human evaluation
- rights check
- security review
- canary synthesis
- rollback test

---

## 22. Deployment Recommendation

### Lean production configuration

- **Web:** Vercel
- **FastAPI and CPU workers:** managed container platform such as Cloud Run
- **Database:** managed PostgreSQL
- **Object storage:** Cloudflare R2 or S3
- **Workflow engine:** Temporal Cloud
- **Training:** SkyPilot-managed GPU jobs
- **Inference:** warm vLLM-Omni GPU service
- **Premium LoRA voices:** official VoxCPM workers until adapter serving is fully validated
- **Monitoring:** Sentry plus OpenTelemetry/Grafana

### Do not add initially

- Kubernetes
- Kafka
- separate vector database
- custom GPU scheduler
- custom vocoder
- custom audio codec
- custom annotation platform
- custom streaming engine
- separate service for every domain

Move to Kubernetes only when multiple continuously utilized GPU replicas make the added operational cost worthwhile.

---

## 23. Build Order

### Phase 1 — Truth baseline

- freeze the F5 benchmark
- deploy untouched VoxCPM2
- evaluate zero-shot Sorani
- evaluate CosyVoice3
- establish native-listener protocol

### Phase 2 — Data and speaker registry

- implement consent records
- upload and review audio
- build Sorani normalization
- import existing data
- create immutable dataset versions

### Phase 3 — Pilot

- prepare 30–50 hours
- train VoxCPM2 LoRA
- compare against F5
- fix frontend and data failures
- reject the architecture if it cannot beat F5

### Phase 4 — Foundation

- expand toward 300–500+ hours
- full-fine-tune VoxCPM2
- preserve multilingual/style ability
- create production benchmark

### Phase 5 — Premium voices

- record 12–20 hours per flagship voice
- train separate adapters
- build canonical reference and style packs
- complete rights and quality approval

### Phase 6 — Production serving

- deploy vLLM-Omni foundation
- deploy approved premium-voice workers
- add streaming
- watermark every output
- expose secured API

### Phase 7 — Scale

- autoscaling
- batch synthesis
- multiple regions
- cost reporting
- customer organizations
- billing
- optional WebRTC voice-agent mode

---

## 24. Final System Definition

The real high-end system is:

> **A VoxCPM2-based Sorani foundation trained on a large, rights-cleared multi-speaker corpus; professional voices stored as separable speaker adapters and curated reference/style packs; a deterministic Sorani text frontend; native-speaker evaluation against the existing F5 baseline; vLLM-Omni streaming inference behind a secured FastAPI gateway; and a Next.js studio managing consent, datasets, training, evaluation, deployments and audit history.**

The most important decisions are not the dashboard framework or number of services. They are:

1. **Do not mistake a 30-hour demo for a language foundation.**
2. **Keep premium speaker identities separable from the shared model.**
3. **Make Sorani normalization and pronunciation QA first-class components.**
4. **Require blind native evaluation before model promotion.**
5. **Reuse official VoxCPM training and vLLM serving instead of rebuilding them.**
6. **Do not expose unrestricted voice cloning.**
7. **Keep the launch architecture simple enough for one strong engineering team to operate.**