# Kurdish Voice Data Governance & Licensing Policy

## 1. Commercial Clearance Guidelines
- **Mozilla Common Voice 26 (Central Kurdish)**: 137.9 validated hours under CC0. Permitted for broad phonetic foundation training; prohibited for identifiable individual voice cloning.
- **Central Kurdish TTS 1.0**: CC-BY-4.0 cleared audio used for pronunciation benchmarks and supplemental training.
- **Flagship Studio Data**: 100% rights-cleared, contracted with voice actors for commercial synthesis and adapter creation.
- **Quarantine Policy**: Any dataset lacking written consent, provenance, or commercial licensing enters strict quarantine and is excluded from training pools.

## 2. Audio Processing Standards
- **VAD Trimming**: Trailing silence must strictly stay below 0.5s to prevent VoxCPM2 generation runaway loops.
- **Sample Rate**: Master archive at 48,000 Hz / 24-bit; training derivatives at 16,000 Hz mono.
