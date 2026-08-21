'use client';

import React, { useState, useEffect } from 'react';
import { useI18n } from '../../lib/i18n';
import { api, Dataset } from '../../lib/api';
import { WaveSurferPlayer } from '../../components/WaveSurferPlayer';
import {
  Database,
  Upload,
  CheckCircle,
  XCircle,
  RotateCcw,
  Lock,
  FileAudio,
  Sparkles,
  AlertCircle,
  FileCheck,
} from 'lucide-react';

export default function DatasetsPage() {
  const { t } = useI18n();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  
  // Sample utterances in review queue
  const [utterances, setUtterances] = useState([
    {
      id: 'utt-01',
      rawText: 'لە ٢٠٢٦/٠٨/٢١دا سەردانی سلێمانیم کرد و $150م خەرج کرد.',
      normalizedText: 'لە بیست و یەکی ئابی ساڵی دوو هەزار و بیست و شەشدا سەردانی سلێمانیم کرد و سەد و پەنجا دۆلارم خەرج کرد.',
      duration: 6.82,
      snr: 28.5,
      silenceRatio: 0.04,
      status: 'pending_review',
      speaker: 'Lamo Slemani',
    },
    {
      id: 'utt-02',
      rawText: 'ڕۆڵەی دڵسۆزی گەل بۆ هەمیشە دەمێنێتەوە.',
      normalizedText: 'ڕۆڵەی دڵسۆزی گەل بۆ هەمیشە دەمێنێتەوە.',
      duration: 4.15,
      snr: 32.1,
      silenceRatio: 0.03,
      status: 'approved',
      speaker: 'Lamo Slemani',
    },
    {
      id: 'utt-03',
      rawText: 'نرخیIQD لە بازاڕی شاری هەولێر جێگیرە.',
      normalizedText: 'نرخی دیناری عێراقی لە بازاڕی شاری هەولێر جێگیرە.',
      duration: 5.20,
      snr: 18.2,
      silenceRatio: 0.12,
      status: 'pending_review',
      speaker: 'Heja Erbil',
    }
  ]);

  const [currentReviewIdx, setCurrentReviewIdx] = useState(0);
  const currentUtt = utterances[currentReviewIdx];
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api.getDatasets()
      .then((data) => {
        setDatasets(data);
        if (data.length > 0) setSelectedDataset(data[0]);
      })
      .catch((err) => setError(err.detail || 'Failed to load datasets'))
      .finally(() => setLoading(false));
  }, []);

  const handleReviewDecision = (decision: 'approved' | 'rejected' | 'retake_requested') => {
    const updated = [...utterances];
    updated[currentReviewIdx].status = decision;
    setUtterances(updated);
    if (currentReviewIdx < utterances.length - 1) {
      setCurrentReviewIdx(currentReviewIdx + 1);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.datasets.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.datasets.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-white/10 text-xs font-semibold text-gray-200 transition-colors">
            <Lock className="w-4 h-4 text-brand-400" />
            <span>{t.datasets.freezeDataset}</span>
          </button>
          <button className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white text-xs font-semibold shadow-glow transition-all">
            <Upload className="w-4 h-4" />
            <span>{t.datasets.uploadAudio}</span>
          </button>
        </div>
      </div>

      {/* Dataset Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {datasets.map((ds) => {
          const isSelected = selectedDataset?.dataset_id === ds.dataset_id;
          return (
            <div
              key={ds.dataset_id}
              onClick={() => setSelectedDataset(ds)}
              className={`p-6 rounded-3xl cursor-pointer transition-all border ${
                isSelected
                  ? 'glass-panel-elevated border-brand-500/50 shadow-glow'
                  : 'glass-panel border-white/10 hover:border-white/20'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-2xl bg-brand-500/10 text-brand-300 border border-brand-500/20">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white text-base">{ds.name}</h3>
                    <span className="text-xs text-gray-400 font-mono">{ds.license}</span>
                  </div>
                </div>
                {ds.is_frozen && (
                  <span className="px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 text-xs font-mono border border-purple-500/30 flex items-center gap-1.5">
                    <Lock className="w-3 h-3" />
                    <span>{ds.current_version}</span>
                  </span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-white/5 text-xs">
                <div>
                  <span className="text-gray-400">Total Hours</span>
                  <div className="font-bold text-white text-sm mt-0.5">{ds.total_hours}h</div>
                </div>
                <div>
                  <span className="text-gray-400">Approved</span>
                  <div className="font-bold text-emerald-400 text-sm mt-0.5">{ds.approved_hours}h</div>
                </div>
                <div>
                  <span className="text-gray-400">Utterances</span>
                  <div className="font-bold text-gray-200 text-sm mt-0.5">{ds.utterance_count.toLocaleString()}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Review Queue & Normalization QA Panel */}
      {currentUtt && (
        <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-6">
          <div className="flex items-center justify-between border-b border-white/10 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <FileCheck className="w-5 h-5 text-brand-400" />
                <span>Utterance Review Queue ({currentReviewIdx + 1} of {utterances.length})</span>
              </h3>
              <p className="text-xs text-gray-400">Speaker: {currentUtt.speaker}</p>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-xs px-3 py-1 rounded-full font-medium border ${
                currentUtt.status === 'approved' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' :
                currentUtt.status === 'rejected' ? 'bg-rose-500/20 text-rose-300 border-rose-500/30' :
                'bg-amber-500/20 text-amber-300 border-amber-500/30'
              }`}>
                {currentUtt.status.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>

          {/* Raw vs Normalized Text Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-surface-elevated p-5 rounded-2xl border border-white/5 space-y-2">
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{t.datasets.rawText}</span>
              <p className="text-base text-gray-200 font-medium leading-relaxed">{currentUtt.rawText}</p>
            </div>
            <div className="bg-surface-elevated p-5 rounded-2xl border border-brand-500/20 space-y-2">
              <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-brand-400" />
                <span>{t.datasets.normalizedText}</span>
              </span>
              <p className="text-base text-white font-medium leading-relaxed">{currentUtt.normalizedText}</p>
            </div>
          </div>

          {/* Waveform Inspection */}
          <WaveSurferPlayer
            title={`Derivative Audio (16 kHz Mono VAD-Trimmed)`}
            subtitle={`Duration: ${currentUtt.duration}s • SNR: ${currentUtt.snr} dB • Silence Ratio: ${(currentUtt.silenceRatio * 100).toFixed(1)}%`}
            durationSeconds={currentUtt.duration}
          />

          {/* Review Decision Buttons */}
          <div className="flex items-center justify-between pt-2">
            <div className="text-xs text-gray-400 font-mono">
              Shortcuts: <kbd className="px-2 py-1 bg-surface-elevated rounded border border-white/10 text-gray-300">A</kbd> Accept • <kbd className="px-2 py-1 bg-surface-elevated rounded border border-white/10 text-gray-300">R</kbd> Reject • <kbd className="px-2 py-1 bg-surface-elevated rounded border border-white/10 text-gray-300">T</kbd> Retake
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => handleReviewDecision('retake_requested')}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-semibold transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>{t.datasets.retake}</span>
              </button>
              <button
                onClick={() => handleReviewDecision('rejected')}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold transition-colors"
              >
                <XCircle className="w-4 h-4" />
                <span>{t.datasets.reject}</span>
              </button>
              <button
                onClick={() => handleReviewDecision('approved')}
                className="flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-lg shadow-emerald-600/20 transition-colors"
              >
                <CheckCircle className="w-4 h-4" />
                <span>{t.datasets.accept}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
