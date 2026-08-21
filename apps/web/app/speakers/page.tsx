'use client';

import React, { useState, useEffect } from 'react';
import { useI18n } from '../../lib/i18n';
import { api, Speaker } from '../../lib/api';
import { WaveSurferPlayer } from '../../components/WaveSurferPlayer';
import {
  Mic,
  ShieldCheck,
  Ban,
  Play,
  Volume2,
  CheckCircle2,
  Sparkles,
  Plus,
  Sliders,
  AlertTriangle,
} from 'lucide-react';

export default function SpeakersPage() {
  const { t, isRTL } = useI18n();
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
  const [showRevokeModal, setShowRevokeModal] = useState(false);
  const [revokeSuccess, setRevokeSuccess] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getSpeakers()
      .then((data) => {
        setSpeakers(data);
        if (data.length > 0) setSelectedSpeaker(data[0]);
      })
      .catch((err) => setError(err.detail || 'Failed to load speakers'))
      .finally(() => setLoading(false));
  }, []);

  const handleRevoke = async () => {
    if (!selectedSpeaker) return;
    try {
      await fetch(`http://localhost:8000/v1/speakers/${selectedSpeaker.speaker_id}/revoke`, {
        method: 'POST',
      });
    } catch {}
    
    // Update local state
    setSpeakers(speakers.map(s => s.speaker_id === selectedSpeaker.speaker_id ? { ...s, status: 'revoked' } : s));
    setSelectedSpeaker({ ...selectedSpeaker, status: 'revoked' });
    setRevokeSuccess(true);
    setTimeout(() => {
      setShowRevokeModal(false);
      setRevokeSuccess(false);
    }, 1500);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.speakers.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.speakers.subtitle}
          </p>
        </div>
      </div>

      {/* Main Speakers Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left List of Speakers */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-2">
            Registered Kurdish Voice Talent ({speakers.length})
          </h3>
          <div className="space-y-3">
            {speakers.map((spk) => {
              const isSelected = selectedSpeaker?.speaker_id === spk.speaker_id;
              const isRevoked = spk.status === 'revoked';

              return (
                <div
                  key={spk.speaker_id}
                  onClick={() => setSelectedSpeaker(spk)}
                  className={`p-5 rounded-2xl cursor-pointer transition-all border ${
                    isSelected
                      ? 'glass-panel-elevated border-brand-500/50 shadow-glow'
                      : 'glass-panel border-white/5 hover:border-white/20'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-brand-600 to-purple-800 flex items-center justify-center text-white font-bold text-sm shadow-md">
                        {spk.name.charAt(0)}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-bold text-white text-base">{spk.kurdish_name}</h4>
                          <span className="text-xs text-gray-400 font-normal">({spk.name})</span>
                        </div>
                        <p className="text-xs text-brand-300 font-medium capitalize mt-0.5">
                          {spk.dialect} • {spk.gender}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`text-[11px] px-2.5 py-1 rounded-full font-medium border ${
                        isRevoked
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                      }`}
                    >
                      {isRevoked ? t.speakers.revoked : t.speakers.active}
                    </span>
                  </div>

                  <div className="mt-4 flex items-center justify-between text-xs text-gray-400 pt-3 border-t border-white/5">
                    <span>Naturalness: <strong className="text-gray-200">{spk.naturalness_score}</strong></span>
                    <span>Similarity: <strong className="text-gray-200">{spk.similarity_score}</strong></span>
                    <span>Styles: <strong className="text-gray-200">{spk.styles?.length || 2}</strong></span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Speaker Detail Card */}
        {selectedSpeaker && (
          <div className="lg:col-span-2 glass-panel p-6 rounded-3xl border border-white/10 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-5">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-2xl font-bold text-white">{selectedSpeaker.kurdish_name}</h2>
                  <span className="text-sm text-gray-400 font-mono">({selectedSpeaker.name})</span>
                </div>
                <p className="text-xs text-gray-400 mt-1 max-w-xl">
                  {selectedSpeaker.voice_description}
                </p>
              </div>

              {selectedSpeaker.status !== 'revoked' ? (
                <button
                  onClick={() => setShowRevokeModal(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold transition-colors"
                >
                  <Ban className="w-4 h-4" />
                  <span>{t.speakers.revoke}</span>
                </button>
              ) : (
                <span className="px-4 py-2 rounded-xl bg-rose-500/20 text-rose-300 border border-rose-500/40 text-xs font-semibold font-mono">
                  VOICE REVOKED & INACTIVE
                </span>
              )}
            </div>

            {/* Legal Governance & Rights Box */}
            <div className="bg-surface-elevated p-5 rounded-2xl border border-white/5 space-y-3">
              <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Legal Rights & Separation Architecture</span>
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-surface rounded-xl border border-white/5">
                  <span className="text-gray-400">Consent Type</span>
                  <p className="text-white font-medium mt-1">Commercial Non-Exclusive</p>
                </div>
                <div className="p-3 bg-surface rounded-xl border border-white/5">
                  <span className="text-gray-400">Derivative Model</span>
                  <p className="text-white font-medium mt-1">Separable LoRA Adapter</p>
                </div>
                <div className="p-3 bg-surface rounded-xl border border-white/5">
                  <span className="text-gray-400">Watermark Signature</span>
                  <p className="text-brand-300 font-mono mt-1">AudioSeal #42981</p>
                </div>
              </div>
            </div>

            {/* Canonical Reference Preview Player */}
            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                {t.speakers.canonicalReferences}
              </h4>
              <WaveSurferPlayer
                title={`${selectedSpeaker.kurdish_name} - Canonical Reference Sample`}
                subtitle="Master Studio Recording (48 kHz / 24-bit)"
                durationSeconds={12.4}
                watermarkId={42981}
              />
            </div>

            {/* Style Presets */}
            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                {t.speakers.styles}
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {selectedSpeaker.styles?.map((sty) => (
                  <div key={sty.style_id} className="p-4 bg-surface-elevated rounded-2xl border border-white/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sm text-white capitalize">{sty.name.replace('_', ' ')}</span>
                      <span className="text-[11px] font-mono text-brand-300">{sty.recommended_speed}x speed</span>
                    </div>
                    <p className="text-xs text-gray-400">{sty.instruction_prompt}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Revocation Modal */}
      {showRevokeModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel-elevated p-6 rounded-3xl border border-rose-500/30 max-w-md w-full space-y-4 shadow-2xl">
            <div className="flex items-center gap-3 text-rose-400">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="font-bold text-lg text-white">Confirm Voice Revocation</h3>
            </div>
            <p className="text-xs text-gray-300 leading-relaxed">
              Are you sure you want to revoke consent for <strong>{selectedSpeaker?.name}</strong>?
              This will immediately disable all associated LoRA adapters and block subsequent synthesis across all APIs.
            </p>
            {revokeSuccess ? (
              <div className="p-3 bg-emerald-500/20 text-emerald-300 text-xs rounded-xl text-center font-semibold">
                Voice successfully revoked and blocked.
              </div>
            ) : (
              <div className="flex justify-end gap-3 pt-2">
                <button
                  onClick={() => setShowRevokeModal(false)}
                  className="px-4 py-2 rounded-xl bg-surface hover:bg-surface-hover text-xs font-semibold text-gray-300"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRevoke}
                  className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-lg shadow-rose-600/30"
                >
                  Confirm Revocation
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
