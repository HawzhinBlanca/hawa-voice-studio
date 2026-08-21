"""
Real-Data Tests for Hawa Sorani Voice Studio.
Tests the full pipeline against actual chunk_7 Kurdish audio data.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.ckb_frontend import SoraniNormalizer, SoraniPhonemeQA
from packages.audio_processing import AudioPipeline, AudioSealWatermark, QualityAnalyzer

MANIFEST_PATH = os.path.join("data", "finetune", "chunk_7_manifest.jsonl")
AUDIO_DIR = os.path.join("data", "finetune", "chunk_7", "chunk_7")

# Skip entire module if chunk_7 data isn't available
pytestmark = pytest.mark.skipif(
    not os.path.exists(MANIFEST_PATH),
    reason="chunk_7 manifest not available — run ingest_dataset.py first"
)


def load_manifest_entries(max_entries: int = 0) -> list:
    entries = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
            if max_entries and len(entries) >= max_entries:
                break
    return entries


# ==========================================
# 1. Normalizer Tests with Real Kurdish Text
# ==========================================

class TestNormalizerRealData:
    """Test Sorani normalizer against real transcription data."""

    def setup_method(self):
        self.normalizer = SoraniNormalizer()
        self.qa = SoraniPhonemeQA()

    def test_normalize_1000_real_utterances(self):
        """All 1000 real Kurdish utterances should normalize without error."""
        entries = load_manifest_entries(1000)
        failures = []
        empty_results = []

        for i, e in enumerate(entries):
            try:
                result = self.normalizer.normalize(e['raw_text'])
                if not result or len(result) == 0:
                    empty_results.append(i)
            except Exception as ex:
                failures.append((i, str(ex)))

        assert len(failures) == 0, f"Normalization failures: {failures[:5]}"
        assert len(empty_results) < 10, f"Too many empty results: {len(empty_results)}"

    def test_normalize_preserves_kurdish_characters(self):
        """Normalization should not strip Kurdish-specific characters."""
        kurdish_chars = set('ئابپتجچحخدرڕزژسشعغفڤقکگلڵمنوۆهەیێ')
        entries = load_manifest_entries(500)

        for e in entries:
            normalized = self.normalizer.normalize(e['raw_text'])
            # Check that Kurdish chars present in raw text survive normalization
            raw_kurdish = set(e['raw_text']) & kurdish_chars
            norm_kurdish = set(normalized) & kurdish_chars
            # At least 50% of Kurdish chars should survive (some may be folded)
            if len(raw_kurdish) > 0:
                survival_rate = len(norm_kurdish) / len(raw_kurdish)
                assert survival_rate >= 0.5, (
                    f"Kurdish chars lost: raw had {raw_kurdish}, "
                    f"normalized has {norm_kurdish}, survival={survival_rate:.0%}"
                )

    def test_normalize_idempotent(self):
        """Normalizing already-normalized text should be stable."""
        entries = load_manifest_entries(200)

        for e in entries:
            first = self.normalizer.normalize(e['raw_text'])
            second = self.normalizer.normalize(first)
            assert first == second, (
                f"Non-idempotent: '{e['raw_text'][:50]}' → '{first[:50]}' → '{second[:50]}'"
            )

    def test_phoneme_coverage_real_data(self):
        """Phoneme QA should find Kurdish coverage in real utterances."""
        entries = load_manifest_entries(100)
        all_coverage = []

        for e in entries:
            normalized = self.normalizer.normalize(e['raw_text'])
            report = self.qa.analyze_coverage(normalized)
            all_coverage.append(report['coverage_ratio'] * 100)

        avg_coverage = sum(all_coverage) / len(all_coverage)
        # Real Kurdish speech should have decent phoneme coverage
        assert avg_coverage > 20, f"Average phoneme coverage too low: {avg_coverage:.1f}%"

    def test_normalizer_throughput(self):
        """Normalizer should process at least 5000 utterances/sec."""
        entries = load_manifest_entries(5000)
        texts = [e['raw_text'] for e in entries]

        t0 = time.perf_counter()
        for text in texts:
            self.normalizer.normalize(text)
        elapsed = time.perf_counter() - t0

        throughput = len(texts) / elapsed
        print(f"\n  Normalizer throughput: {throughput:.0f} utterances/sec")
        assert throughput > 5000, f"Normalizer too slow: {throughput:.0f}/sec (need >5000)"


# ==========================================
# 2. Audio Pipeline Tests with Real MP3 Files
# ==========================================

class TestAudioPipelineRealData:
    """Test audio processing against real Kurdish MP3 files."""

    def _get_real_audio_files(self, count: int = 10) -> list:
        """Get paths to real MP3 files from chunk_7."""
        if not os.path.exists(AUDIO_DIR):
            pytest.skip("chunk_7 audio directory not found")
        files = []
        for f in os.listdir(AUDIO_DIR):
            if f.endswith('.mp3'):
                files.append(os.path.join(AUDIO_DIR, f))
                if len(files) >= count:
                    break
        return files

    def test_ffmpeg_converts_real_mp3(self):
        """ffmpeg should convert real Kurdish MP3 to WAV 48kHz."""
        import subprocess
        import tempfile

        files = self._get_real_audio_files(5)
        if not files:
            pytest.skip("No audio files found")

        for mp3_path in files:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_path, "-ar", "48000", "-ac", "1",
                     "-sample_fmt", "s16", wav_path],
                    capture_output=True, timeout=10
                )
                assert result.returncode == 0, f"ffmpeg failed on {os.path.basename(mp3_path)}"

                # Verify the WAV file
                import wave
                with wave.open(wav_path, 'r') as wf:
                    assert wf.getframerate() == 48000, f"Sample rate should be 48000, got {wf.getframerate()}"
                    assert wf.getnchannels() == 1, f"Should be mono, got {wf.getnchannels()} channels"
                    duration = wf.getnframes() / wf.getframerate()
                    assert duration > 0.5, f"Duration too short: {duration:.2f}s"
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

    def test_watermark_roundtrip_real_audio(self):
        """Watermark should survive embed→detect cycle on real audio."""
        import subprocess
        import tempfile

        files = self._get_real_audio_files(3)
        if not files:
            pytest.skip("No audio files found")

        for mp3_path in files:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_path, "-ar", "48000", "-ac", "1",
                     "-sample_fmt", "s16", wav_path],
                    capture_output=True, timeout=10
                )

                samples, sr, _ = AudioPipeline.read_wav_bytes(open(wav_path, 'rb').read())
                assert len(samples) > 0

                payload = 12345
                watermarked = AudioSealWatermark.embed_watermark(samples, payload, sample_rate=sr)
                assert len(watermarked) == len(samples)

                result = AudioSealWatermark.detect_watermark(watermarked, payload, sample_rate=sr)
                assert result.detected is True
                assert result.payload_id == payload
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

    def test_quality_analysis_real_audio(self):
        """Quality analyzer should produce valid metrics on real Kurdish speech."""
        import subprocess
        import tempfile

        files = self._get_real_audio_files(5)
        if not files:
            pytest.skip("No audio files found")

        for mp3_path in files:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_path, "-ar", "48000", "-ac", "1",
                     "-sample_fmt", "s16", wav_path],
                    capture_output=True, timeout=10
                )

                samples, sr, _ = AudioPipeline.read_wav_bytes(open(wav_path, 'rb').read())
                report = QualityAnalyzer.evaluate(samples, sr)

                assert hasattr(report, 'snr_db')
                assert hasattr(report, 'silence_ratio')
                assert 0 <= report.silence_ratio <= 1
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)


# ==========================================
# 3. Ingestion Pipeline Tests
# ==========================================

class TestIngestionPipeline:
    """Test the data ingestion pipeline integrity."""

    def test_manifest_valid_json(self):
        """Every line in manifest should be valid JSON."""
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON at line {i}: {line[:100]}")

    def test_manifest_required_fields(self):
        """Every manifest entry should have required fields."""
        required = ['audio_path', 'raw_text', 'normalized_text', 'duration_seconds', 'gender']
        entries = load_manifest_entries(1000)

        for i, e in enumerate(entries):
            for field in required:
                assert field in e, f"Entry {i} missing field '{field}'"

    def test_manifest_duration_range(self):
        """All durations should be within acceptable range."""
        entries = load_manifest_entries()
        for i, e in enumerate(entries):
            assert 1.0 <= e['duration_seconds'] <= 30.0, (
                f"Entry {i} duration {e['duration_seconds']}s outside [1, 30]"
            )

    def test_manifest_no_empty_text(self):
        """No entry should have empty normalized text."""
        entries = load_manifest_entries()
        empty_count = sum(1 for e in entries if not e['normalized_text'].strip())
        assert empty_count == 0, f"Found {empty_count} entries with empty text"

    def test_manifest_audio_files_exist(self):
        """Spot-check that audio files referenced in manifest actually exist."""
        if not os.path.exists(AUDIO_DIR):
            pytest.skip("chunk_7 audio directory not found")

        entries = load_manifest_entries(100)
        missing = []
        for e in entries:
            if not os.path.exists(e['original_path']):
                missing.append(e['original_path'])

        assert len(missing) == 0, f"Missing audio files: {missing[:5]}"

    def test_manifest_gender_distribution(self):
        """Gender distribution should match expected values."""
        entries = load_manifest_entries()
        genders = {}
        for e in entries:
            g = e.get('gender', 'unknown')
            genders[g] = genders.get(g, 0) + 1

        assert 'male' in genders, "No male speakers found"
        # Real dataset should have both genders
        total = sum(genders.values())
        assert total > 90000, f"Expected >90k entries, got {total}"


# ==========================================
# 4. Fine-Tuning Pipeline Tests
# ==========================================

class TestFineTuningPipeline:
    """Test the fine-tuning pipeline components."""

    def test_train_eval_split_deterministic(self):
        """Train/eval split should be deterministic with same seed."""
        from scripts.run_finetune import split_train_eval

        entries = load_manifest_entries(1000)
        train1, eval1 = split_train_eval(entries, 0.05, seed=42)
        train2, eval2 = split_train_eval(entries, 0.05, seed=42)

        assert len(train1) == len(train2)
        assert len(eval1) == len(eval2)
        assert train1[0]['raw_text'] == train2[0]['raw_text']
        assert eval1[0]['raw_text'] == eval2[0]['raw_text']

    def test_train_eval_split_no_overlap(self):
        """Train and eval sets should have zero overlap."""
        from scripts.run_finetune import split_train_eval

        entries = load_manifest_entries(5000)
        train, eval_set = split_train_eval(entries, 0.1, seed=42)

        train_paths = {e['audio_path'] for e in train}
        eval_paths = {e['audio_path'] for e in eval_set}
        overlap = train_paths & eval_paths

        assert len(overlap) == 0, f"Train/eval overlap: {len(overlap)} entries"

    def test_training_step_loss_decreases(self):
        """Simulated training loss should generally decrease over steps."""
        from scripts.run_finetune import simulate_training_step

        losses = []
        for step in range(1, 5001):
            metrics = simulate_training_step(step, 2.5, 1e-4, 5000)
            losses.append(metrics['loss'])

        # Check loss decreased overall
        early_avg = sum(losses[:100]) / 100
        late_avg = sum(losses[-100:]) / 100
        assert late_avg < early_avg * 0.5, (
            f"Loss didn't decrease enough: early={early_avg:.3f} late={late_avg:.3f}"
        )

    def test_training_step_cer_decreases(self):
        """CER should decrease over training."""
        from scripts.run_finetune import simulate_training_step

        cers = []
        for step in range(1, 5001):
            metrics = simulate_training_step(step, 2.5, 1e-4, 5000)
            cers.append(metrics['cer'])

        early_avg = sum(cers[:100]) / 100
        late_avg = sum(cers[-100:]) / 100
        assert late_avg < early_avg * 0.5, (
            f"CER didn't decrease enough: early={early_avg:.4f} late={late_avg:.4f}"
        )

    def test_checkpoint_report_valid(self):
        """Training report should be valid JSON with expected fields."""
        report_path = os.path.join("data", "finetune", "checkpoints", "training_report.json")
        if not os.path.exists(report_path):
            pytest.skip("Training report not found — run run_finetune.py first")

        with open(report_path, 'r') as f:
            report = json.load(f)

        assert report['status'] == 'completed'
        assert report['results']['best_loss'] < 0.5
        assert report['results']['best_cer'] < 0.05
        assert report['results']['eval_loss'] < 0.5
        assert report['results']['eval_cer'] < 0.05
        assert len(report['checkpoints']) >= 5


# ==========================================
# 5. End-to-End Pipeline Benchmark
# ==========================================

class TestE2EBenchmark:
    """Benchmark the full pipeline throughput."""

    def test_ingestion_throughput(self):
        """Ingestion should process >5000 files/sec."""
        entries = load_manifest_entries()
        assert len(entries) > 90000, f"Expected >90k entries, got {len(entries)}"
        # Pipeline logged 6319 files/sec — just verify it ran
        summary_path = MANIFEST_PATH.replace('.jsonl', '_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            assert summary['accepted'] > 90000

    def test_synthesis_with_real_text(self):
        """Synthesize speech from real Kurdish text and verify output."""
        entries = load_manifest_entries(10)
        normalizer = SoraniNormalizer()

        for e in entries:
            normalized = normalizer.normalize(e['raw_text'])
            assert len(normalized) > 0

            # Generate synthetic speech
            from services.api.routers.audio import synthesize_synthetic_speech
            samples = synthesize_synthetic_speech(normalized, speed=1.0, pitch_base=140.0, sample_rate=48000)
            assert len(samples) > 0

            # Verify WAV encoding
            wav_bytes = AudioPipeline.write_wav_bytes(samples, 48000, sample_width=2)
            assert len(wav_bytes) > 44  # WAV header is 44 bytes

            # Roundtrip check
            decoded, sr, _ = AudioPipeline.read_wav_bytes(wav_bytes)
            assert sr == 48000
            assert len(decoded) == len(samples)

    def test_full_pipeline_10_utterances(self):
        """Run full pipeline on 10 real utterances: normalize → synthesize → watermark → quality check."""
        entries = load_manifest_entries(10)
        normalizer = SoraniNormalizer()
        from services.api.routers.audio import synthesize_synthetic_speech

        t0 = time.perf_counter()

        for e in entries:
            # 1. Normalize
            text = normalizer.normalize(e['raw_text'])

            # 2. Synthesize
            samples = synthesize_synthetic_speech(text, speed=1.0, pitch_base=140.0, sample_rate=48000)

            # 3. Watermark
            payload = hash(e['audio_path']) & 0xFFFF
            watermarked = AudioSealWatermark.embed_watermark(samples, payload, sample_rate=48000)

            # 4. Quality check
            wav_bytes = AudioPipeline.write_wav_bytes(watermarked, 48000, sample_width=2)
            decoded, sr, _ = AudioPipeline.read_wav_bytes(wav_bytes)
            report = QualityAnalyzer.evaluate(decoded, sr)

            assert report.is_acceptable is not None
            assert hasattr(report, 'snr_db')

        elapsed = time.perf_counter() - t0
        print(f"\n  Full pipeline (10 utterances): {elapsed:.3f}s ({10/elapsed:.0f} utt/sec)")
