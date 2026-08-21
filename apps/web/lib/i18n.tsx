'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

export type Language = 'en' | 'ckb';

export const translations = {
  en: {
    appName: "Hawa Voice Studio",
    tagline: "Production Central Kurdish (Sorani) TTS Platform",
    nav: {
      dashboard: "Dashboard",
      speakers: "Speakers",
      datasets: "Datasets",
      training: "Training",
      evaluation: "Evaluation",
      playground: "Playground",
      deployments: "Deployments",
    },
    dashboard: {
      title: "Control Center",
      subtitle: "Overview of Kurdish speech synthesis models, data pipelines, and infrastructure.",
      activeVoices: "Active Voices",
      datasetHours: "Dataset Hours",
      trainingRuns: "Training Runs",
      deployedModel: "Active Foundation",
      synthesisVolume: "Generated Audio",
      avgLatency: "P95 Latency",
      gpuLoad: "GPU Utilization",
      recentActivity: "Recent System Events",
      statusHealthy: "Systems Operational",
    },
    speakers: {
      title: "Speaker Profiles & Governance",
      subtitle: "Manage consented voice talent, recording progress, style presets, and revocation.",
      newSpeaker: "Register Speaker",
      name: "Speaker Name",
      dialect: "Dialect",
      consentStatus: "Consent Status",
      adapters: "LoRA Adapters",
      actions: "Actions",
      revoke: "Revoke Voice",
      active: "Active",
      revoked: "Revoked",
      canonicalReferences: "Canonical References",
      styles: "Available Styles",
    },
    datasets: {
      title: "Dataset Studio",
      subtitle: "Upload, normalize, inspect waveforms, and freeze immutable Kurdish audio datasets.",
      uploadAudio: "Upload Utterance",
      freezeDataset: "Freeze Version",
      rawText: "Raw Kurdish Transcript",
      normalizedText: "Normalized Spoken Form",
      duration: "Duration",
      qualityStatus: "Quality Status",
      accept: "Accept",
      reject: "Reject",
      retake: "Retake",
      approvedHours: "Approved Hours",
    },
    training: {
      title: "Training Studio",
      subtitle: "Launch VoxCPM2 LoRA pilots, full Sorani SFT, and speaker adapter training on SkyPilot GPUs.",
      launchJob: "Launch Training Job",
      preset: "Preset",
      baseModel: "Base Model",
      targetGpu: "GPU Target",
      steps: "Steps",
      loss: "Validation Loss",
      cer: "CER Score",
      checkpoints: "Checkpoints",
      running: "Training Running",
      completed: "Completed",
    },
    evaluation: {
      title: "Evaluation Lab",
      subtitle: "Blind A/B comparative tests, native speaker MOS scoring, and production gate verification.",
      newEvaluation: "Run Benchmark",
      blindAB: "Blind A/B Listening Test",
      f5Baseline: "F5-TTS Baseline",
      voxcpm: "VoxCPM2 Sorani",
      cosyvoice: "CosyVoice3 Challenger",
      naturalness: "Naturalness MOS",
      pronunciation: "Pronunciation Score",
      similarity: "Speaker Similarity",
      winRate: "Win Rate vs F5",
      approveProduction: "Approve for Production",
    },
    playground: {
      title: "Voice Playground",
      subtitle: "Interactive speech synthesis with low-latency streaming and AudioSeal watermarking.",
      inputPlaceholder: "دەقی سۆرانی لێرە بنووسە... بۆ نموونە: بەخێربێن بۆ ستۆدیۆی دەنگی هەوا",
      voice: "Voice",
      style: "Style Preset",
      speed: "Speed",
      streaming: "Real-time Streaming",
      watermark: "AudioSeal Watermark",
      generate: "Synthesize Speech",
      download: "Download WAV",
      chars: "Characters",
      latency: "TTFB",
    },
    deployments: {
      title: "Deployments & Serving",
      subtitle: "Manage production inference replicas, canary traffic routing, and instant rollback.",
      activeModel: "Active Production Model",
      canaryModel: "Canary Model",
      trafficSplit: "Traffic Allocation",
      promote: "Promote to 100%",
      rollback: "Instant Rollback",
      apiKeys: "API Keys",
      generateKey: "Generate API Key",
    }
  },
  ckb: {
    appName: "ستۆدیۆی دەنگی هەوا",
    tagline: "پلاتفۆرمی دەنگسازیی پێشکەوتووی کوردیی سۆرانی",
    nav: {
      dashboard: "داشبۆرد",
      speakers: "دەنگبێژان",
      datasets: "داتاسێتەکان",
      training: "ڕاهێنان",
      evaluation: "هەڵسەنگاندن",
      playground: "تاقیگەی دەنگ",
      deployments: "بڵاوکردنەوە",
    },
    dashboard: {
      title: "ناوەندی کۆنتڕۆڵ",
      subtitle: "پوختەی مۆدێلەکان، داتا، و ژێرخانی تەکنیکی دەنگی کوردی.",
      activeVoices: "دەنگە چالاکەکان",
      datasetHours: "کاتژمێری داتاسێت",
      trainingRuns: "ڕاهێنانەکان",
      deployedModel: "مۆدێلی بنەڕەتی",
      synthesisVolume: "دەنگی دروستکراو",
      avgLatency: "خێرایی وەڵامدانەوە",
      gpuLoad: "بەکارهێنانی GPU",
      recentActivity: "دوایین ڕووداوەکان",
      statusHealthy: "سیستەم لەوپەڕی کاراییە",
    },
    speakers: {
      title: "پڕۆفایل و مافی دەنگبێژان",
      subtitle: "بەڕێوەبردنی ڕەزامەندیی یاسایی، نموونەی دەنگەکان، و هەڵوەشاندنەوە.",
      newSpeaker: "تۆمارکردنی دەنگبێژ",
      name: "ناوی دەنگبێژ",
      dialect: "شێوەزار",
      consentStatus: "ڕەزامەندی یاسایی",
      adapters: "ئاداپتەرەکانی LoRA",
      actions: "کردارەکان",
      revoke: "هەڵوەشاندنەوەی دەنگ",
      active: "چالاک",
      revoked: "هەڵوەشاوە",
      canonicalReferences: "نموونە ستانداردەکان",
      styles: "شێوازە بەردەستەکان",
    },
    datasets: {
      title: "ستۆدیۆی داتاسێت",
      subtitle: "بارکردنی دەنگ، نۆرماڵایزکردنی دەق، و بەستنی ڤێرژنی چەسپاو بۆ ڕاهێنان.",
      uploadAudio: "بارکردنی دەنگ و دەق",
      freezeDataset: "بەستنی ڤێرژن",
      rawText: "دەقی خاو",
      normalizedText: "دەقی نۆرماڵکراوی بێژراو",
      duration: "ماوە",
      qualityStatus: "دۆخی کوالیتی",
      accept: "پەسەندکردن",
      reject: "ڕەتکردنەوە",
      retake: "دووبارەکردنەوە",
      approvedHours: "کاتژمێری پەسەندکراو",
    },
    training: {
      title: "ستۆدیۆی ڕاهێنان",
      subtitle: "دەستپێکردنی ڕاهێنانی VoxCPM2 بە شێوازی LoRA و Full SFT لەسەر GPU.",
      launchJob: "دەستپێکردنی ڕاهێنان",
      preset: "شێوازی ڕاهێنان",
      baseModel: "مۆدێلی بنەڕەتی",
      targetGpu: "جۆری GPU",
      steps: "هەنگاوەکان",
      loss: "ڕێژەی هەڵە (Loss)",
      cer: "نمرەی CER",
      checkpoints: "چێکپۆینتەکان",
      running: "ڕاهێنان لە کاردایە",
      completed: "تەواوبوو",
    },
    evaluation: {
      title: "تاقیگەی هەڵسەنگاندن",
      subtitle: "تاقیکردنەوەی کوێر A/B، نمرەدانانی خەڵکی ڕەسەن (MOS)، و مەرجەکانی بڵاوکردنەوە.",
      newEvaluation: "ئەنجامدانی تاقیکردنەوە",
      blindAB: "تاقیکردنەوەی کوێری دەنگ A/B",
      f5Baseline: "مۆدێلی پێشووی F5-TTS",
      voxcpm: "VoxCPM2ی کوردی",
      cosyvoice: "CosyVoice3 ڕکابەر",
      naturalness: "سروشتیبوون (MOS)",
      pronunciation: "دروستیی دەربڕین",
      similarity: "لێکچوونی دەنگ",
      winRate: "ڕێژەی سەرکەوتن بەسەر F5",
      approveProduction: "پەسەندکردن بۆ بەکارهێنانی گشتی",
    },
    playground: {
      title: "تاقیگەی دروستکردنی دەنگ",
      subtitle: "دروستکردنی دەنگی ڕاستەوخۆ بە کوالیتی بەرز و واتەرمارکی پارێزراوی AudioSeal.",
      inputPlaceholder: "دەقی سۆرانی لێرە بنووسە... بۆ نموونە: بەخێربێن بۆ ستۆدیۆی دەنگی هەوا لە ساڵی ٢٠٢٦",
      voice: "دەنگ",
      style: "شێوازی دەربڕین",
      speed: "خێرایی",
      streaming: "پەخشی ڕاستەوخۆ (Streaming)",
      watermark: "واتەرمارکی AudioSeal",
      generate: "دروستکردنی دەنگ",
      download: "داگرتنی WAV",
      chars: "پیتەکان",
      latency: "خێرایی دەستپێک",
    },
    deployments: {
      title: "بڵاوکردنەوە و بەردەستکردن",
      subtitle: "کۆنتڕۆڵکردنی مۆدێلی چالاک، دابەشکردنی ترافیک (Canary)، و گەڕانەوەی خێرا.",
      activeModel: "مۆدێلی سەرەکیی چالاک",
      canaryModel: "مۆدێلی کاتی (Canary)",
      trafficSplit: "دابەشکردنی ترافیک",
      promote: "کردنە سەرەکی (100%)",
      rollback: "گەڕانەوەی خێرا (Rollback)",
      apiKeys: "کلیلی API",
      generateKey: "دروستکردنی کلیل",
    }
  }
};

interface I18nContextType {
  lang: Language;
  setLang: (lang: Language) => void;
  t: typeof translations.en;
  isRTL: boolean;
}

const I18nContext = createContext<I18nContextType | null>(null);

export const I18nProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lang, setLang] = useState<Language>('ckb');

  const isRTL = lang === 'ckb';
  const t = translations[lang];

  useEffect(() => {
    document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
    document.documentElement.lang = lang;
  }, [lang, isRTL]);

  return (
    <I18nContext.Provider value={{ lang, setLang, t, isRTL }}>
      <div dir={isRTL ? 'rtl' : 'ltr'}>
        {children}
      </div>
    </I18nContext.Provider>
  );
};

export const useI18n = () => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
};
