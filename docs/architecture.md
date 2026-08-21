# Hawa Sorani Voice Studio - System Architecture & Engineering Blueprint

## 1. Overview
Hawa Sorani Voice Studio is a production-grade text-to-speech platform specifically engineered for Central Kurdish (Sorani / کوردیی ناوەندی). It utilizes **VoxCPM2** as the foundation model with **CosyVoice3** as the honorable research challenger, backed by a modular FastAPI control plane, deterministic Kurdish text normalization (`ckb-frontend`), AudioSeal watermarking, SkyPilot GPU orchestration, and a modern Next.js 16.3 App Router studio interface.

## 2. Topology

```text
┌──────────────────────────────────────────────────────────────────┐
│                      Next.js 16.3 Web Studio                     │
│ Dashboard • Speakers • Datasets • Training • Evaluation • Studio  │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS / SSE / PCM Streaming
┌─────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Control Plane (v1)                    │
│ Auth • RBAC • Speaker Rights • Normalization • Audit Logging     │
└──────────────┬──────────────────┬──────────────────┬──────────────┘
               │                  │                  │
       ┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
       │ PostgreSQL DB  │ │ S3/R2 Storage  │ │ Temporal Cloud │
       │ 20+ entities   │ │ audio & models │ │ durable jobs   │
       └────────────────┘ └────────────────┘ └──────┬─────────┘
                                                     │
                                           ┌─────────▼────────┐
                                           │ SkyPilot GPU     │
                                           │ Training Runner  │
                                           │ (L40S / A100)    │
                                           └──────────────────┘
```

## 3. Audio & ML Pipeline
1. **Audio Ingestion**: 48 kHz 24-bit uncompressed master archive preserved alongside a 16 kHz VAD-trimmed derivative.
2. **Text Normalization**: Deterministic Unicode folding (`ی`/`ي`, `ک`/`ك`, `ە`/`ه`), number-to-words expansion, IQD/USD currency conversion, date/time formatting, and foreign tech name transliteration.
3. **Training Strategy**:
   - **Pilot LoRA (30–50 hrs)**: Quick validation against F5-TTS baseline on single 48GB GPU.
   - **Foundation SFT (300–500 hrs)**: Multi-speaker full fine-tuning with replay regularization.
   - **Speaker LoRAs (12–20 hrs)**: Separable voice adapters per premium voice actor.
4. **Watermarking**: Inaudible 16-bit AudioSeal watermark embedded in all generated speech for provenance and abuse protection.
