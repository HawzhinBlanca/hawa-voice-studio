'use client';

import React, { useState } from 'react';
import { useI18n } from '../../lib/i18n';
import { WaveSurferPlayer } from '../../components/WaveSurferPlayer';
import {
  Award,
  CheckCircle,
  ThumbsUp,
  Sliders,
  ShieldAlert,
  Sparkles,
  BarChart3,
  TrendingUp,
} from 'lucide-react';

export default function EvaluationPage() {
  const { t } = useI18n();
  const [selectedModelPref, setSelectedModelPref] = useState<'A' | 'B' | null>(null);
  const [naturalnessMos, setNaturalnessMos] = useState(5);
  const [pronunciationScore, setPronunciationScore] = useState(5);
  const [similarityScore, setSimilarityScore] = useState(5);
  const [submitted, setSubmitted] = useState(false);

  const testSentence = "ڕۆڵەی دڵسۆزی گەلەکەمان بۆ هەمیشە لە دڵماندا بە زیندوویی دەمێنێتەوە.";

  const handleSubmitRating = () => {
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 2000);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.evaluation.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.evaluation.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-mono">
          <span>Gate Requirement: ≥55% Win Rate</span>
        </div>
      </div>

      {/* Model Benchmark Radar Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="glass-panel-elevated p-6 rounded-3xl border border-brand-500/40 space-y-4 shadow-glow">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-white text-base flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-brand-400" />
              <span>VoxCPM2 Sorani</span>
            </h3>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono">
              Production Candidate
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between text-gray-300">
              <span>Naturalness MOS:</span>
              <span className="font-bold text-emerald-400 font-mono">4.75 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Pronunciation Accuracy:</span>
              <span className="font-bold text-emerald-400 font-mono">4.88 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Speaker Similarity:</span>
              <span className="font-bold text-emerald-400 font-mono">4.82 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Win Rate vs F5:</span>
              <span className="font-bold text-purple-300 font-mono">78.5%</span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-white text-base">CosyVoice3</h3>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 font-mono">
              Challenger
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between text-gray-300">
              <span>Naturalness MOS:</span>
              <span className="font-bold text-gray-200 font-mono">4.62 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Pronunciation Accuracy:</span>
              <span className="font-bold text-gray-200 font-mono">4.70 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Speaker Similarity:</span>
              <span className="font-bold text-gray-200 font-mono">4.68 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Win Rate vs F5:</span>
              <span className="font-bold text-purple-300 font-mono">68.2%</span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-white text-base">F5-TTS</h3>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-white/10 text-gray-400 font-mono">
              Baseline
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between text-gray-300">
              <span>Naturalness MOS:</span>
              <span className="font-bold text-gray-400 font-mono">4.15 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Pronunciation Accuracy:</span>
              <span className="font-bold text-gray-400 font-mono">4.30 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Speaker Similarity:</span>
              <span className="font-bold text-gray-400 font-mono">4.25 / 5.0</span>
            </div>
            <div className="flex justify-between text-gray-300">
              <span>Win Rate vs F5:</span>
              <span className="font-bold text-gray-400 font-mono">50.0%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Blind A/B Interactive Listening Test */}
      <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-6">
        <div className="border-b border-white/10 pb-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-brand-400" />
            <span>Blind A/B Native Listener Test</span>
          </h3>
          <p className="text-xs text-gray-400 mt-1">
            Listen to both anonymized samples and rate pronunciation accuracy and naturalness.
          </p>
        </div>

        {/* Test Sentence Card */}
        <div className="p-4 bg-surface-elevated rounded-2xl border border-white/5 text-center space-y-1">
          <span className="text-xs text-gray-400 uppercase font-mono">Target Sorani Sentence</span>
          <p className="text-lg font-bold text-white">{testSentence}</p>
        </div>

        {/* Anonymized Samples Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className={`p-5 rounded-2xl border transition-all ${
            selectedModelPref === 'A' ? 'glass-panel-elevated border-brand-500/50' : 'bg-surface-elevated border-white/5'
          }`}>
            <WaveSurferPlayer title="Sample Option A (Blind)" durationSeconds={4.8} />
            <button
              onClick={() => setSelectedModelPref('A')}
              className={`w-full mt-3 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 border transition-all ${
                selectedModelPref === 'A'
                  ? 'bg-brand-600 text-white border-brand-500 shadow-md'
                  : 'bg-surface hover:bg-surface-hover text-gray-300 border-white/10'
              }`}
            >
              <ThumbsUp className="w-3.5 h-3.5" />
              <span>Select Sample A as Better</span>
            </button>
          </div>

          <div className={`p-5 rounded-2xl border transition-all ${
            selectedModelPref === 'B' ? 'glass-panel-elevated border-brand-500/50' : 'bg-surface-elevated border-white/5'
          }`}>
            <WaveSurferPlayer title="Sample Option B (Blind)" durationSeconds={4.6} />
            <button
              onClick={() => setSelectedModelPref('B')}
              className={`w-full mt-3 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 border transition-all ${
                selectedModelPref === 'B'
                  ? 'bg-brand-600 text-white border-brand-500 shadow-md'
                  : 'bg-surface hover:bg-surface-hover text-gray-300 border-white/10'
              }`}
            >
              <ThumbsUp className="w-3.5 h-3.5" />
              <span>Select Sample B as Better</span>
            </button>
          </div>
        </div>

        {/* MOS Rating Sliders */}
        <div className="bg-surface-elevated p-5 rounded-2xl border border-white/5 space-y-4">
          <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
            Detailed 1–5 MOS Ratings for Preferred Option
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Naturalness MOS</span>
                <span className="font-bold text-brand-300 font-mono">{naturalnessMos} / 5</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="0.5"
                value={naturalnessMos}
                onChange={(e) => setNaturalnessMos(parseFloat(e.target.value))}
                className="w-full accent-brand-500"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Pronunciation Accuracy</span>
                <span className="font-bold text-brand-300 font-mono">{pronunciationScore} / 5</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="0.5"
                value={pronunciationScore}
                onChange={(e) => setPronunciationScore(parseFloat(e.target.value))}
                className="w-full accent-brand-500"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Speaker Similarity</span>
                <span className="font-bold text-brand-300 font-mono">{similarityScore} / 5</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="0.5"
                value={similarityScore}
                onChange={(e) => setSimilarityScore(parseFloat(e.target.value))}
                className="w-full accent-brand-500"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={handleSubmitRating}
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white text-xs font-semibold shadow-glow transition-all active:scale-95"
          >
            <CheckCircle className="w-4 h-4" />
            <span>{submitted ? 'Rating Recorded!' : 'Submit Blind Evaluation'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
