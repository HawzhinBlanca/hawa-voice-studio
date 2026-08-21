import './globals.css';
import { I18nProvider } from '../lib/i18n';
import { Navbar } from '../components/Navbar';

export const metadata = {
  title: 'Hawa Sorani Voice Studio | Production Kurdish TTS',
  description: 'Production-grade Central Kurdish (Sorani) Voice Studio powered by VoxCPM2, AudioSeal, and Next.js.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ckb" dir="rtl" className="dark">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-background text-gray-100 antialiased selection:bg-brand-500/30 selection:text-white">
        <I18nProvider>
          <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 space-y-8">
              {children}
            </main>
            <footer className="border-t border-white/5 py-6 text-center text-xs text-gray-500 font-mono">
              Hawa Sorani Voice Studio • August 2026 Release • OpenBMB VoxCPM2 Foundation
            </footer>
          </div>
        </I18nProvider>
      </body>
    </html>
  );
}
