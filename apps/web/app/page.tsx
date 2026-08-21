'use client';

import React from 'react';
import Link from 'next/link';
import { useI18n } from '../lib/i18n';
import {
  Activity,
  Mic,
  Database,
  Cpu,
  Award,
  Radio,
  Zap,
  TrendingUp,
  Clock,
  ShieldCheck,
  ArrowUpRight,
} from 'lucide-react';

export default function DashboardPage() {
  const { t, isRTL } = useI18n();

  const stats = [
    {
      label: t.dashboard.activeVoices,
      value: "8",
      subtext: "4 Slemani, 3 Erbil, 1 Badini",
      icon: Mic,
      gradient: "from-brand-500 to-purple-600",
    },
    {
      label: t.dashboard.datasetHours,
      value: "320.5 hrs",
      subtext: "285.2 hrs verified & frozen",
      icon: Database,
      gradient: "from-emerald-500 to-teal-600",
    },
    {
      label: t.dashboard.trainingRuns,
      value: "1 Active",
      subtext: "Pilot LoRA at step 8,500",
      icon: Cpu,
      gradient: "from-amber-500 to-orange-600",
    },
    {
      label: t.dashboard.avgLatency,
      value: "285 ms",
      subtext: "P95 TTFB (Target: <500ms)",
      icon: Zap,
      gradient: "from-blue-500 to-indigo-600",
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Hero Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-3xl border border-white/10 shadow-glow">
        <div>
          <div className="flex items-center gap-2 text-brand-400 text-xs font-semibold uppercase tracking-wider mb-1">
            <ShieldCheck className="w-4 h-4" />
            <span>{t.dashboard.statusHealthy}</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white">
            {t.dashboard.title}
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            {t.dashboard.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/playground"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white text-sm font-semibold shadow-glow transition-all active:scale-95"
          >
            <Radio className="w-4 h-4" />
            <span>{t.nav.playground}</span>
            <ArrowUpRight className={`w-4 h-4 ${isRTL ? 'rotate-180' : ''}`} />
          </Link>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div
              key={idx}
              className="glass-panel p-5 rounded-2xl border border-white/10 hover:border-brand-500/30 transition-all group relative overflow-hidden"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-400">{stat.label}</span>
                <div className={`p-2 rounded-xl bg-gradient-to-br ${stat.gradient} text-white shadow-md`}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <div className="mt-4">
                <div className="text-2xl font-bold text-white tracking-tight">{stat.value}</div>
                <p className="text-xs text-gray-400 mt-1">{stat.subtext}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Central Content Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Architecture Status */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-3xl border border-white/10 space-y-6">
          <div className="flex items-center justify-between border-b border-white/10 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-brand-400" />
                <span>Foundation Architecture: VoxCPM2</span>
              </h3>
              <p className="text-xs text-gray-400">Production Inference & LoRA Worker Pipeline</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-brand-500/20 text-brand-300 text-xs font-mono border border-brand-500/30">
              Apache-2.0
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-surface-elevated p-4 rounded-2xl border border-white/5 space-y-1">
              <span className="text-xs text-gray-400">Active Checkpoint</span>
              <div className="text-sm font-semibold text-white font-mono">voxcpm2-ckb-v1.4</div>
              <span className="text-[11px] text-emerald-400">CER: 2.4% (ASR Verified)</span>
            </div>
            <div className="bg-surface-elevated p-4 rounded-2xl border border-white/5 space-y-1">
              <span className="text-xs text-gray-400">Audio Resolution</span>
              <div className="text-sm font-semibold text-white font-mono">48 kHz / 24-bit</div>
              <span className="text-[11px] text-purple-300">Native High Fidelity</span>
            </div>
            <div className="bg-surface-elevated p-4 rounded-2xl border border-white/5 space-y-1">
              <span className="text-xs text-gray-400">Watermark Security</span>
              <div className="text-sm font-semibold text-white font-mono">AudioSeal 16-bit</div>
              <span className="text-[11px] text-brand-accent">100% Provenance Trace</span>
            </div>
          </div>

          {/* Synthesis Volume Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-gray-300 font-medium">
              <span>Sorani Language Foundation Target (500 hrs)</span>
              <span className="font-mono text-brand-300">320.5 / 500 hrs (64%)</span>
            </div>
            <div className="w-full h-3 bg-surface-elevated rounded-full overflow-hidden p-0.5 border border-white/5">
              <div className="h-full bg-gradient-to-r from-brand-600 via-purple-500 to-brand-accent rounded-full transition-all duration-1000" style={{ width: '64%' }} />
            </div>
          </div>
        </div>

        {/* Quick Actions & Live GPU Load */}
        <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-5">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span>Infrastructure Health</span>
          </h3>

          <div className="space-y-4 text-xs">
            <div className="flex justify-between items-center bg-surface-elevated p-3 rounded-xl">
              <span className="text-gray-300">GPU Cluster (SkyPilot)</span>
              <span className="font-mono text-emerald-400 font-semibold">1x L40S 48GB (94% load)</span>
            </div>

            <div className="flex justify-between items-center bg-surface-elevated p-3 rounded-xl">
              <span className="text-gray-300">Temporal Workflow Engine</span>
              <span className="font-mono text-purple-300 font-semibold">Connected</span>
            </div>

            <div className="flex justify-between items-center bg-surface-elevated p-3 rounded-xl">
              <span className="text-gray-300">Audio Storage (Cloudflare R2)</span>
              <span className="font-mono text-brand-300 font-semibold">48.2 GB stored</span>
            </div>
          </div>

          <div className="pt-2 border-t border-white/10">
            <Link
              href="/training"
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-white/10 text-xs font-semibold text-gray-200 transition-colors"
            >
              <TrendingUp className="w-4 h-4 text-brand-400" />
              <span>Inspect Active Training Jobs</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
