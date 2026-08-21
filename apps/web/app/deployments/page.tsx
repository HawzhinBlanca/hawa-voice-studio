'use client';

import React, { useState, useEffect } from 'react';
import { useI18n } from '../../lib/i18n';
import { api, Deployment } from '../../lib/api';
import {
  Server,
  Key,
  RotateCcw,
  TrendingUp,
  ShieldCheck,
  CheckCircle,
  Zap,
  Activity,
  Layers,
  Copy,
} from 'lucide-react';

export default function DeploymentsPage() {
  const { t } = useI18n();
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [trafficSplit, setTrafficSplit] = useState(25);
  const [copiedKey, setCopiedKey] = useState(false);
  const [rollbackSuccess, setRollbackSuccess] = useState(false);

  useEffect(() => {
    api.getDeployments().then(setDeployments);
  }, []);

  const handleRollback = async () => {
    setRollbackSuccess(true);
    setTimeout(() => setRollbackSuccess(false), 2500);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.deployments.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.deployments.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRollback}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            <span>{rollbackSuccess ? 'Rolled Back!' : t.deployments.rollback}</span>
          </button>
        </div>
      </div>

      {/* Production & Canary Cluster Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Active Production Model */}
        <div className="glass-panel-elevated p-6 rounded-3xl border border-brand-500/40 space-y-4 shadow-glow">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div>
              <span className="text-xs text-brand-300 font-mono font-semibold uppercase">{t.deployments.activeModel}</span>
              <h3 className="text-lg font-bold text-white mt-0.5">VoxCPM2-Sorani-Foundation-v1.4</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-mono border border-emerald-500/30">
              Active Production
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3 bg-surface rounded-xl border border-white/5">
              <span className="text-gray-400">P95 Latency</span>
              <div className="text-emerald-400 font-bold text-sm mt-1">285 ms</div>
            </div>
            <div className="p-3 bg-surface rounded-xl border border-white/5">
              <span className="text-gray-400">RTF Factor</span>
              <div className="text-purple-300 font-bold text-sm mt-1">0.24</div>
            </div>
            <div className="p-3 bg-surface rounded-xl border border-white/5">
              <span className="text-gray-400">Replicas</span>
              <div className="text-brand-300 font-bold text-sm mt-1">2x Warm L40S</div>
            </div>
          </div>
        </div>

        {/* Canary Traffic Routing Control */}
        <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-4">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div>
              <span className="text-xs text-purple-300 font-mono font-semibold uppercase">{t.deployments.canaryModel}</span>
              <h3 className="text-lg font-bold text-white mt-0.5">VoxCPM2-Sorani-LoRA-Pilot-v2</h3>
            </div>
            <span className="px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 text-xs font-mono border border-purple-500/30">
              Canary Split: {trafficSplit}%
            </span>
          </div>

          {/* Traffic Split Slider */}
          <div className="space-y-2 pt-2">
            <div className="flex justify-between text-xs text-gray-300">
              <span>Canary Traffic Allocation</span>
              <span className="font-mono text-purple-300">{trafficSplit}% Canary / {100 - trafficSplit}% Stable</span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={trafficSplit}
              onChange={(e) => setTrafficSplit(parseInt(e.target.value))}
              className="w-full accent-purple-500"
            />
          </div>

          <div className="flex justify-end pt-2">
            <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold shadow-md transition-colors">
              <CheckCircle className="w-4 h-4" />
              <span>{t.deployments.promote}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Secured API Key Gateway */}
      <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Key className="w-4 h-4 text-brand-400" />
              <span>Production OpenAI-Compatible API Key</span>
            </h3>
            <p className="text-xs text-gray-400">Endpoint: POST https://api.hawa.ai/v1/audio/speech</p>
          </div>

          <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-surface-elevated hover:bg-surface-hover text-xs text-gray-300 border border-white/10">
            <span>{t.deployments.generateKey}</span>
          </button>
        </div>

        <div className="flex items-center justify-between p-3.5 bg-surface-elevated rounded-2xl border border-white/5 font-mono text-xs text-gray-300">
          <span>hawa_live_98a72f10b89cd42e18fa90bc74</span>
          <button
            onClick={() => {
              navigator.clipboard.writeText("hawa_live_98a72f10b89cd42e18fa90bc74");
              setCopiedKey(true);
              setTimeout(() => setCopiedKey(false), 1500);
            }}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surface hover:bg-surface-hover text-brand-300 border border-white/10 transition-colors"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>{copiedKey ? 'Copied!' : 'Copy'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
