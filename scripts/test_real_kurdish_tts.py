"""
Real Sorani Kurdish TTS Inference using Pretrained F5-TTS Sorani Kurdish Engine & Vocos Vocoder.
Synthesizes high-fidelity intelligible human Kurdish speech from input text.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

# Add F5-TTS to path
f5_src = r"D:\Hawzhin Personal\sorani_f5tts_data_kit\training_pc\F5-TTS\src"
if f5_src not in sys.path:
    sys.path.insert(0, f5_src)

from f5_tts.model import CFM, DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text

# Import Hawa Studio normalizer & audio processing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from packages.ckb_frontend.normalizer import SoraniNormalizer
from packages.audio_processing.watermark import AudioSealWatermark
from packages.audio_processing.quality_analyzer import QualityAnalyzer


def run_kurdish_tts_inference(
    gen_text: str,
    output_wav: str = "output/farmer_goose_story_real.wav",
    ckpt_path: str = r"D:\Hawzhin Personal\sorani_f5tts_data_kit\training_pc\F5-TTS\ckpts\target_voice\model_last.pt",
    vocab_file: str = r"D:\Hawzhin Personal\sorani_f5tts_data_kit\training_pc\F5-TTS\data\target_voice_custom\vocab.txt",
    ref_audio: str = r"D:\Hawzhin Personal\sorani_f5tts_data_kit\data\target_voice\wavs\B7890RX_0001.wav",
    ref_text: str = "کە سوریا بووە گۆمی خوێن، سەدان هەزار سوری ئاوارەی تورکیا و هەرێمی کوردستان بوون. نەتانەوێ بزانن لە تورکیای دۆست و هاوپەیمانیان چۆن پێشوازییان لێ کرا.",
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0
):
    print("=" * 75)
    print(" HAWA SORANI VOICE STUDIO -- REAL PRODUCTION KURDISH TTS SYNTHESIS")
    print("=" * 75)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Normalize Input Text
    print("\n--- [Stage 1/5] Normalizing Kurdish Text ---")
    normalizer = SoraniNormalizer()
    normalized_gen_text = normalizer.normalize(gen_text)
    normalized_ref_text = normalizer.normalize(ref_text)
    print(f"Raw Input: '{gen_text}'")
    print(f"Normalized: '{normalized_gen_text}'")
    
    # 2. Load Model & Vocoder
    print("\n--- [Stage 2/5] Loading Pretrained Sorani F5-TTS Backbone & Vocos ---")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Vocab File: {vocab_file}")
    
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    
    ema_model = load_model(
        model_cls,
        model_cfg,
        ckpt_path,
        mel_spec_type="vocos",
        vocab_file=vocab_file,
        device=device
    )
    
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    print("Vocoder and Diffusion Transformer loaded successfully.")
    
    # 3. Preprocess Reference Audio
    print("\n--- [Stage 3/5] Audio Conditioning & Reference Processing ---")
    print(f"Reference Audio: {ref_audio}")
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, normalized_ref_text)
    
    # 4. Neural Flow-Matching Speech Generation
    print("\n--- [Stage 4/5] Neural Speech Synthesis via Flow Matching ODE ---")
    t0 = time.time()
    
    final_wave, final_sample_rate, spectragram = infer_process(
        ref_audio,
        ref_text,
        normalized_gen_text,
        ema_model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        target_rms=0.1,
        speed=speed,
        device=device
    )
    
    gen_duration = len(final_wave) / final_sample_rate
    elapsed = time.time() - t0
    rtf = elapsed / gen_duration
    print(f"Synthesis Complete! Generated {gen_duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.3f}x)")
    
    # 5. Audio Watermarking & Quality Scorecard
    print("\n--- [Stage 5/5] Audio Processing, Watermark & Quality Verification ---")
    
    # Convert to 48kHz if required or keep 24kHz studio master
    audio_float = final_wave.astype(float).tolist()
    
    watermarker = AudioSealWatermark()
    watermarked_audio_list = watermarker.embed_watermark(audio_float, payload_16bit=0x4853, sample_rate=final_sample_rate)
    watermarked_audio = np.array(watermarked_audio_list, dtype=np.float32)
    
    detection = watermarker.detect_watermark(watermarked_audio_list, candidate_payload=0x4853, sample_rate=final_sample_rate)
    
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, watermarked_audio, final_sample_rate)
    print(f"[OUTPUT SAVED] Studio WAV ({final_sample_rate}Hz): {output_wav}")
    
    # Export 320k MP3
    output_mp3 = output_wav.replace(".wav", ".mp3")
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_wav, "-codec:a", "libmp3lame", "-b:a", "320k", output_mp3],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        print(f"[OUTPUT SAVED] 320kbps Studio MP3: {output_mp3}")
    except Exception as e:
        print(f"(MP3 export note: {e})")
        
    quality = QualityAnalyzer.evaluate(watermarked_audio_list, sample_rate=final_sample_rate)
    
    print("\n" + "=" * 75)
    print(" KURDISH SPEECH SYNTHESIS SCORECARD")
    print("=" * 75)
    print(f"  Input Characters:        {len(gen_text)} characters")
    print(f"  Audio Duration:          {gen_duration:.2f} seconds")
    print(f"  Sample Rate:             {final_sample_rate:,} Hz")
    print(f"  SNR (Signal-to-Noise):   {quality.snr_db:.1f} dB")
    print(f"  Silence Ratio:           {quality.silence_ratio:.1%}")
    print(f"  Clipping Rate:           {quality.clipping_rate:.3%}")
    print(f"  Production Grade Gate:   {'PASS (High Quality)' if quality.is_acceptable else 'REVIEW REQUIRED'}")
    print(f"  AudioSeal Provenance:    {'VERIFIED (0x4853)' if detection.detected else 'UNVERIFIED'}")
    print("=" * 75)
    
    return {
        "status": "success",
        "output_wav": output_wav,
        "output_mp3": output_mp3,
        "duration_seconds": gen_duration,
        "quality": {
            "snr_db": quality.snr_db,
            "is_acceptable": quality.is_acceptable,
            "is_watermarked": detection.detected
        }
    }


if __name__ == "__main__":
    kurdish_story = (
        "جوتیارێک قازێکی هەبوو .. رۆژێک لە رۆژان ، قازەکەی هێلکەیەکی سادە زێڕی کرد .. "
        "لەو رۆژەەوە بەدواوە ، قازەکە هەموو رۆژێک هێلکەیەکی زێڕی تری دەکرد .. "
        "جوتیارەکە دەوڵەمەند .. بەڵام لەگەڵ ئەوەشدا ، زۆر چاوچنۆک بوو . "
    )
    
    run_kurdish_tts_inference(kurdish_story)
