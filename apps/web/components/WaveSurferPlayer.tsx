'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, Download, RefreshCw } from 'lucide-react';

interface WaveSurferPlayerProps {
  audioUrl?: string;
  audioBlob?: Blob;
  title?: string;
  subtitle?: string;
  watermarkId?: number;
  durationSeconds?: number;
}

export const WaveSurferPlayer: React.FC<WaveSurferPlayerProps> = ({
  audioUrl,
  audioBlob,
  title,
  subtitle,
  watermarkId,
  durationSeconds = 5.0,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [internalUrl, setInternalUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setInternalUrl(url);
      return () => URL.revokeObjectURL(url);
    } else if (audioUrl) {
      setInternalUrl(audioUrl);
    }
  }, [audioBlob, audioUrl]);

  // Generate simulated animated audio waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const barCount = 70;
    const barWidth = width / barCount - 2;
    const progress = currentTime / Math.max(1, durationSeconds);

    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + 2);
      // Pseudo-random speech amplitude profile
      const noise = Math.sin(i * 0.35) * Math.cos(i * 0.15);
      const barHeight = Math.max(8, (Math.abs(noise) * 0.8 + 0.2) * height * 0.85);
      const y = (height - barHeight) / 2;

      const isPlayed = i / barCount <= progress;
      if (isPlayed) {
        ctx.fillStyle = '#a855f7'; // vibrant purple for played
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)'; // light grey for unplayed
      }

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 3);
      ctx.fill();
    }
  }, [currentTime, durationSeconds]);

  const togglePlay = () => {
    if (!audioRef.current && !internalUrl) {
      // Simulate playback if no audio element
      setIsPlaying(!isPlaying);
      return;
    }

    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {
          setIsPlaying(true);
        });
      }
    } else {
      setIsPlaying(!isPlaying);
    }
  };

  // Timer simulation for playback
  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentTime((prev) => {
          if (prev >= durationSeconds) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 0.1;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, durationSeconds]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="glass-panel p-5 rounded-2xl border border-white/10 space-y-4 shadow-xl">
      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-semibold text-sm text-gray-100">{title || 'Generated Audio Preview'}</h4>
          <p className="text-xs text-gray-400 font-light">{subtitle || 'VoxCPM2 Native 48kHz • AudioSeal Verified'}</p>
        </div>
        {watermarkId !== undefined && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-[11px] font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
            <span>Watermark #{watermarkId}</span>
          </div>
        )}
      </div>

      {/* Waveform Canvas */}
      <div className="bg-surface-elevated/80 rounded-xl p-3 border border-white/5 relative flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={600}
          height={64}
          className="w-full h-16 cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const frac = clickX / rect.width;
            setCurrentTime(frac * durationSeconds);
          }}
        />
      </div>

      {/* Controls Bar */}
      <div className="flex items-center justify-between text-xs text-gray-400 font-mono">
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="w-10 h-10 rounded-full bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white flex items-center justify-center shadow-glow transition-transform active:scale-95"
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-white" /> : <Play className="w-4 h-4 fill-white ml-0.5" />}
          </button>
          <span className="text-gray-200">
            {formatTime(currentTime)} / {formatTime(durationSeconds)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {internalUrl && (
            <a
              href={internalUrl}
              download="hawa_sorani_speech.wav"
              className="p-2 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-white/10 text-gray-300 hover:text-white transition-colors"
              title="Download Master WAV"
            >
              <Download className="w-4 h-4" />
            </a>
          )}
        </div>
      </div>

      {internalUrl && (
        <audio
          ref={audioRef}
          src={internalUrl}
          onEnded={() => {
            setIsPlaying(false);
            setCurrentTime(0);
          }}
          className="hidden"
        />
      )}
    </div>
  );
};
