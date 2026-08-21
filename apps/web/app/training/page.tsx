'use client';

import React, { useState, useEffect } from 'react';
import { useI18n } from '../../lib/i18n';
import { api, TrainingRun } from '../../lib/api';
import {
  Cpu,
  Play,
  CheckCircle2,
  Clock,
  Layers,
  Activity,
  DollarSign,
  TrendingDown,
  Sparkles,
  Server,
  Zap,
} from 'lucide-react';

export default function TrainingPage() {
  const { t } = useI18n();
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('sorani_pilot_lora');
  const [isLaunching, setIsLaunching] = useState(false);

  useEffect(() => {
    api.getTrainingRuns().then(setRuns);
  }, []);

  const presets = [
    {
      id: 'sorani_pilot_lora',
      title: 'Sorani Pilot LoRA',
      description: 'Quick multi-speaker pilot (30–50 hrs) to validate Sorani phonemes and benchmark against F5 baseline.',
      gpu: '1x L40S 48GB (SkyPilot Spot)',
      vram: '20 GB VRAM',
      recommendedSteps: 20000,
      estimatedCost: '$45 - $60',
    },
    {
      id: 'full_sorani_foundation_sft',
      title: 'Full Sorani Foundation SFT',
      description: 'Supervised full fine-tuning (300–500 hrs) with multi-language replay regularization.',
      gpu: '8x A100 80GB (Distributed Torchrun)',
      vram: '40 GB VRAM per GPU',
      recommendedSteps: 50000,
      estimatedCost: '$280 - $350',
    },
    {
      id: 'premium_speaker_lora',
      title: 'Premium Speaker LoRA Adapter',
      description: 'Flagship single-voice adapter (12–20 studio hrs) with multi-style reference conditioning.',
      gpu: '1x L40S 48GB (SkyPilot Spot)',
      vram: '20 GB VRAM',
      recommendedSteps: 10000,
      estimatedCost: '$25 - $35',
    }
  ];

  const handleLaunch = () => {
    setIsLaunching(true);
    setTimeout(() => {
      setIsLaunching(false);
      const newRun: TrainingRun = {
        run_id: `run-${Date.now().toString().slice(-4)}`,
        run_name: `voxcpm2-${selectedPreset.replace(/_/g, '-')}`,
        preset: selectedPreset,
        base_model: "openbmb/VoxCPM2",
        dataset_version: "v2.1-frozen",
        status: "running",
        current_step: 100,
        total_steps: 20000,
        current_loss: 2.34,
        best_loss: 2.34,
        gpu_type: "1x L40S 48GB (SkyPilot Spot)",
        estimated_cost_spent: 1.25,
      };
      setRuns([newRun, ...runs]);
    }, 1200);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.training.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.training.subtitle}
          </p>
        </div>
      </div>

      {/* Preset Selector */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-2">
          Select Controlled Training Preset
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {presets.map((p) => {
            const isSelected = selectedPreset === p.id;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedPreset(p.id)}
                className={`p-6 rounded-3xl cursor-pointer transition-all border flex flex-col justify-between ${
                  isSelected
                    ? 'glass-panel-elevated border-brand-500/50 shadow-glow'
                    : 'glass-panel border-white/10 hover:border-white/20'
                }`}
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-base">{p.title}</span>
                    {isSelected && (
                      <span className="p-1 rounded-full bg-brand-500/20 text-brand-400 border border-brand-500/30">
                        <CheckCircle2 className="w-4 h-4" />
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">{p.description}</p>
                </div>

                <div className="mt-6 pt-4 border-t border-white/5 space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between text-gray-300">
                    <span>Compute:</span>
                    <span className="text-brand-300">{p.gpu}</span>
                  </div>
                  <div className="flex items-center justify-between text-gray-300">
                    <span>Cost Guardrail:</span>
                    <span className="text-emerald-400">{p.estimatedCost}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={handleLaunch}
            disabled={isLaunching}
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white text-sm font-semibold shadow-glow transition-all active:scale-95 disabled:opacity-50"
          >
            <Play className="w-4 h-4 fill-white" />
            <span>{isLaunching ? 'Dispatching SkyPilot Job...' : t.training.launchJob}</span>
          </button>
        </div>
      </div>

      {/* Active Runs Table */}
      <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-6">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-brand-400" />
          <span>Active & Historic Training Runs</span>
        </h3>

        <div className="space-y-4">
          {runs.map((run) => {
            const progress = (run.current_step / run.total_steps) * 100;
            return (
              <div key={run.run_id} className="bg-surface-elevated p-5 rounded-2xl border border-white/5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-white text-sm">{run.run_name}</span>
                      <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-mono border border-brand-500/30">
                        {run.preset}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 font-mono mt-1">Base: {run.base_model} • Dataset: {run.dataset_version}</p>
                  </div>

                  <div className="flex items-center gap-4 text-xs font-mono">
                    <div className="text-right">
                      <span className="text-gray-400">Step</span>
                      <div className="text-white font-bold">{run.current_step.toLocaleString()} / {run.total_steps.toLocaleString()}</div>
                    </div>
                    <div className="text-right">
                      <span className="text-gray-400">Loss</span>
                      <div className="text-emerald-400 font-bold">{run.current_loss.toFixed(4)}</div>
                    </div>
                    <div className="text-right">
                      <span className="text-gray-400">Cost</span>
                      <div className="text-brand-300 font-bold">${run.estimated_cost_spent.toFixed(2)}</div>
                    </div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[11px] text-gray-400 font-mono">
                    <span>Progress ({progress.toFixed(1)}%)</span>
                    <span>SkyPilot Spot Auto-Recovery Enabled</span>
                  </div>
                  <div className="w-full h-2.5 bg-surface rounded-full overflow-hidden p-0.5 border border-white/5">
                    <div
                      className="h-full bg-gradient-to-r from-brand-600 via-purple-500 to-brand-accent rounded-full transition-all duration-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
