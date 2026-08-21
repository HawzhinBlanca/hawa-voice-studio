# Voice Talent Consent & Identity Agreement Template

## Legal Voice Licensing Framework
1. **Grant of Rights**: Explicit authorization for training generative neural acoustic models and LoRA adapters.
2. **Separable Architecture**: Premium actors are isolated into separable LoRA adapters rather than permanently merged into the shared foundation weights, enabling true revocation compliance.
3. **Prohibited Contexts**: Defamatory content, political impersonation without consent, adult entertainment, and fraudulent synthetic media.
4. **Immediate Revocation**: Calling `/v1/speakers/{id}/revoke` locks the voice profile, terminates running adapters, and blocks all subsequent synthesis jobs.
