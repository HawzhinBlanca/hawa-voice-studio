'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useI18n } from '../lib/i18n';
import {
  Activity,
  Mic,
  Database,
  Cpu,
  Award,
  Radio,
  Server,
  Globe,
  Sparkles,
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const { t, lang, setLang } = useI18n();

  const navItems = [
    { href: '/', label: t.nav.dashboard, icon: Activity },
    { href: '/speakers', label: t.nav.speakers, icon: Mic },
    { href: '/datasets', label: t.nav.datasets, icon: Database },
    { href: '/training', label: t.nav.training, icon: Cpu },
    { href: '/evaluation', label: t.nav.evaluation, icon: Award },
    { href: '/playground', label: t.nav.playground, icon: Radio },
    { href: '/deployments', label: t.nav.deployments, icon: Server },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-white/10 px-6 py-3.5 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo and Brand */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-brand-accent p-0.5 shadow-glow transition-transform group-hover:scale-105">
            <div className="w-full h-full bg-surface rounded-[10px] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-brand-300" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-purple-200 to-brand-400 bg-clip-text text-transparent">
                {t.appName}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-mono border border-brand-500/30">
                VoxCPM2 48kHz
              </span>
            </div>
            <p className="text-xs text-gray-400 font-light hidden sm:block">
              {t.tagline}
            </p>
          </div>
        </Link>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-surface-elevated/70 p-1.5 rounded-2xl border border-white/5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-brand-600 to-brand-700 text-white shadow-md'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Status Pill & Language Switcher */}
        <div className="flex items-center gap-3">
          {/* Health Status Indicator */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono">vLLM 48kHz Active</span>
          </div>

          {/* Language Toggle */}
          <button
            onClick={() => setLang(lang === 'en' ? 'ckb' : 'en')}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-white/10 text-xs font-medium text-gray-300 transition-colors"
          >
            <Globe className="w-3.5 h-3.5 text-brand-400" />
            <span>{lang === 'en' ? 'کوردی (Sorani)' : 'English'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
