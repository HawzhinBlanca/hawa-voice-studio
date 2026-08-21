"""
Full End-to-End System Pipeline Test.
Covers the entire lifecycle from Speaker Consent Onboarding to Watermarked TTS Synthesis.
"""

import io
import pytest
from fastapi.testclient import TestClient
from packages.audio_processing import AudioPipeline, AudioSealWatermark
from services.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_full_hawa_sorani_voice_pipeline(client):
    print("\n--- [Phase 1: Register Speaker Profile & Consent Rights] ---")
    spk_res = client.post("/v1/speakers", json={
        "name": "Shakar Badini",
        "kurdish_name": "شاکار بادینی",
        "dialect": "badini",
        "gender": "male",
        "voice_description": "Clear documentary speaker",
        "consent_type": "full_commercial_exclusive",
        "commercial_use_permitted": True,
        "derivative_model_permitted": True
    })
    assert spk_res.status_code == 201
    spk_id = spk_res.json()["speaker_id"]

    print("\n--- [Phase 2: Ingest Dataset & Kurdish Normalization] ---")
    ds_res = client.post("/v1/datasets", data={
        "name": "E2E Master Dataset",
        "source": "studio",
        "license": "Commercial"
    })
    ds_id = ds_res.json()["dataset_id"]

    # Ingest audio clip with numbers and currencies
    wav_bytes = AudioPipeline.write_wav_bytes([0.05] * 32000, 16000, 2)
    upload_res = client.post(
        f"/v1/datasets/{ds_id}/uploads",
        data={
            "raw_transcript": "ڕۆژنامەی فەرمی لە ٢٠٢٦/٠٨/٢١ ڕایگەیاند کە نرخی نەوت بووە $80.",
            "speaker_id": spk_id,
            "style_label": "authoritative_documentary"
        },
        files={"audio_file": ("clip.wav", io.BytesIO(wav_bytes), "audio/wav")}
    )
    assert upload_res.status_code == 200
    utt_id = upload_res.json()["utterance_id"]
    norm_text = upload_res.json()["normalized_transcript"]
    assert "هەشتا دۆلار" in norm_text

    print("\n--- [Phase 3: Utterance Approval & Dataset Freeze] ---")
    client.patch(f"/v1/datasets/utterances/{utt_id}", data={"decision": "approved"})
    
    freeze_res = client.post(f"/v1/datasets/{ds_id}/freeze", json={
        "dataset_id": ds_id,
        "version_tag": "v1.0-e2e-frozen",
        "notes": "E2E Frozen Verification"
    })
    assert freeze_res.status_code == 200

    print("\n--- [Phase 4: Launch Training Run & Track Metrics] ---")
    train_res = client.post("/v1/training-runs", json={
        "run_name": "voxcpm2-e2e-pilot",
        "preset": "sorani_pilot_lora",
        "base_model": "openbmb/VoxCPM2",
        "dataset_id": ds_id,
        "dataset_version": "v1.0-e2e-frozen",
        "speaker_id": spk_id,
        "max_steps": 1000,
        "target_gpu_type": "1x L40S 48GB"
    })
    assert train_res.status_code == 201

    print("\n--- [Phase 5: Evaluation Benchmark & Production Gate] ---")
    eval_res = client.post("/v1/evaluations", data={
        "title": "E2E Benchmark vs F5",
        "model_version_id": "voxcpm2-sorani-foundation-v1",
        "challenger_model_id": "F5-TTS",
        "test_suite_tag": "core_sorani",
        "sample_count": 25
    })
    assert eval_res.status_code == 201
    eval_id = eval_res.json()["evaluation_id"]

    # Pass production gate approval
    approve_res = client.post(f"/v1/evaluations/{eval_id}/approve")
    assert approve_res.status_code == 200

    print("\n--- [Phase 6: TTS Synthesis & AudioSeal Watermark Verification] ---")
    synth_res = client.post("/v1/audio/speech", json={
        "model": "sorani-pro-v1",
        "input": "دەنگی کوردیی سۆرانی بە سەرکەوتوویی دروستکرا لە ٢٠٢٦.",
        "voice": spk_id,
        "style": "warm_documentary",
        "speed": 1.0,
        "format": "wav",
        "stream": False
    })
    assert synth_res.status_code == 200
    generated_wav = synth_res.content

    # Verify 48 kHz output & extract AudioSeal watermark
    samples, sr, _ = AudioPipeline.read_wav_bytes(generated_wav)
    assert sr == 48000
    assert len(samples) > 1000

    watermark_id = int(synth_res.headers.get("X-Audio-Watermark-Id"))
    wm_result = AudioSealWatermark.detect_watermark(samples, watermark_id, sample_rate=48000)
    assert wm_result.detected is True
    print("E2E Pipeline execution and AudioSeal verification completed with 100% success!")
