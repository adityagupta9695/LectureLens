import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Rocket, Search, Mic, Box, FileText, Download, PlayCircle, Loader2, BookOpen, GraduationCap } from 'lucide-react';

type AppState = 'input' | 'loading' | 'dashboard';

export default function App() {
  const [appState, setAppState] = useState<AppState>('input');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState('');

  useEffect(() => {
    // Polling removed as backend process is now synchronous and awaited
  }, [appState]);

  const handleStart = async (url: string) => {
    setVideoUrl(url);
    
    // Immediately set loading and start spinner
    setAppState('loading');
    setLoadingProgress(50); // Fake progress to indicate activity
    
    try {
      // Send request to backend to start process
      const res = await fetch('http://localhost:8000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });
      if (res.ok) {
        setLoadingProgress(100);
        setAppState('dashboard');
      } else {
        const err = await res.json();
        alert("Failed to start process: " + err.detail);
        setAppState('input');
      }
    } catch (error) {
      console.error(error);
      alert("Network error: Could not reach backend processor.");
      setAppState('input');
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans overflow-x-hidden selection:bg-white selection:text-black">
      <AnimatePresence mode="wait">
        {appState === 'input' && (
          <InputPhase onStart={handleStart} key="input" />
        )}
        {appState === 'loading' && (
          <LoadingPhase progress={loadingProgress} key="loading" />
        )}
        {appState === 'dashboard' && (
          <DashboardPhase videoUrl={videoUrl} key="dashboard" />
        )}
      </AnimatePresence>
    </div>
  );
}

function InputPhase({ onStart }: { onStart: (url: string) => void }) {
  const [url, setUrl] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    setIsSubmitting(true);
    await onStart(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
      className="flex flex-col items-center justify-center min-h-screen p-4"
    >
      <div className="mb-12 text-center">
        <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-4">
          Lecture<span className="italic">Lens</span>
        </h1>
        <p className="text-lg md:text-xl tracking-widest uppercase font-light border-b border-white pb-2 inline-block">
          Multimodal Educational Tutor
        </p>
      </div>

      <form 
        onSubmit={handleSubmit}
        className="flex flex-col md:flex-row w-full max-w-3xl gap-4 items-center"
      >
        <div className="relative w-full">
          <PlayCircle className="absolute left-4 top-1/2 -translate-y-1/2 w-6 h-6 text-white" />
          <input
            type="url"
            required
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={isSubmitting}
            placeholder="Enter YouTube Lecture URL..."
            className="w-full bg-black border border-white text-white px-14 py-4 md:text-lg focus:outline-none focus:ring-1 focus:ring-white transition-all shadow-[0_0_15px_rgba(255,255,255,0.1)] focus:shadow-[0_0_25px_rgba(255,255,255,0.3)] disabled:opacity-50"
          />
        </div>
        <button
          type="submit"
          disabled={isSubmitting}
          className="flex items-center justify-center gap-2 bg-white text-black border border-white px-8 py-4 md:text-lg font-bold hover:bg-black hover:text-white transition-all whitespace-nowrap group min-w-[140px] disabled:opacity-50 disabled:hover:bg-white disabled:hover:text-black"
        >
          {isSubmitting ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Rocket className="w-5 h-5 group-hover:-translate-y-1 group-hover:translate-x-1 transition-transform" />
          )}
          Start
        </button>
      </form>
    </motion.div>
  );
}

function LoadingPhase({ progress }: { progress: number }) {
  const displayProgress = Math.min(100, Math.floor(progress));
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center justify-center min-h-screen"
    >
      <div className="relative flex items-center justify-center w-48 h-48 mb-8">
        <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="#ffffff"
            strokeWidth="1"
            className="opacity-20"
          />
          <motion.circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2"
            strokeDasharray="283"
            strokeDashoffset={283 - (283 * progress) / 100}
            className="transition-all duration-75 ease-linear"
          />
        </svg>
        <div className="text-4xl font-bold tracking-tighter">
          {displayProgress}%
        </div>
      </div>
      
      <motion.p 
        animate={{ opacity: [0.5, 1, 0.5] }} 
        transition={{ repeat: Infinity, duration: 1.5 }}
        className="text-xl tracking-widest uppercase font-light"
      >
        Ingesting Multimodal Data...
      </motion.p>
    </motion.div>
  );
}

interface SearchResult {
  id?: string | number;
  text?: string;
  spoken?: string;
  visual?: string;
  metadata?: {
    start_sec?: number;
    end_sec?: number;
    video_id?: string;
    content_type?: string;
    text_en?: string;
    text_hi?: string;
  };
}

function DashboardPhase({ videoUrl }: { videoUrl: string }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeFilter, setActiveFilter] = useState<'asr' | 'objects' | 'ocr' | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPptLoading, setIsPptLoading] = useState(false);
  const [isDeckLoading, setIsDeckLoading] = useState(false);

  const fetchResults = async (filterType: 'asr' | 'objects' | 'ocr') => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    setActiveFilter(filterType);
    
    try {
      const url = `http://localhost:8000/api/search?query=${encodeURIComponent(searchQuery)}&filter_type=${filterType}`;
      const response = await fetch(url);
      
      if (response.ok) {
        const data = await response.json();
        setResults(Array.isArray(data) ? data : data.results || []);
      } else {
        console.error("Search failed with status:", response.status);
        setResults([]);
      }
    } catch (error) {
      console.error("Failed to fetch search results:", error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (seconds?: number) => {
    if (seconds === undefined) return "00:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const getYoutubeLink = (startSec?: number) => {
    if (!videoUrl) return "#";
    try {
      const urlObj = new URL(videoUrl);
      if (startSec !== undefined) {
        if (urlObj.hostname.includes('youtube.com')) {
          urlObj.searchParams.set('t', `${Math.floor(startSec)}s`);
        } else if (urlObj.hostname.includes('youtu.be')) {
          urlObj.searchParams.set('t', `${Math.floor(startSec)}`);
        } else {
          urlObj.searchParams.set('t', `${Math.floor(startSec)}s`);
        }
      }
      return urlObj.toString();
    } catch {
      return videoUrl;
    }
  };

  const formatVisualText = (text: string) => {
    if (!text) return null;
    const parts = text.split(/(\$[^$]+\$)/g);
    return parts.map((part, index) => {
      if (part.startsWith('$') && part.endsWith('$')) {
        return <span key={index} className="text-yellow-400 font-mono tracking-tight">{part}</span>;
      }
      return <span key={index}>{part}</span>;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      fetchResults(activeFilter || 'asr');
    }
  };

  const handleGeneratePPT = async () => {
    if (results.length === 0) { alert('Run a search first to get results.'); return; }
    setIsPptLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/export/ppt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Export failed'); }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'StudyGuide.pptx';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert('PPT export failed: ' + err.message);
    } finally {
      setIsPptLoading(false);
    }
  };

  const handleExportNotion = () => {
    if (results.length === 0) { alert('Run a search first to get results.'); return; }
    const fmt = (s?: number) => {
      if (s === undefined) return '??:??';
      const m = Math.floor(s / 60), sec = Math.floor(s % 60);
      return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
    };
    const lines: string[] = ['# LectureLens Study Guide', ''];
    results.forEach(r => {
      const ts = `${fmt(r.metadata?.start_sec)} – ${fmt(r.metadata?.end_sec)}`;
      const type = r.metadata?.content_type ?? '';
      const tag = type === 'asr' ? '🎙️ Spoken' : type === 'visual_ocr' ? '📺 Screen Text' : type === 'visual_objects' ? '🔍 Objects' : '';
      lines.push(`## ${ts}  ${tag}`);
      lines.push(`> ${r.text || '(no text)'}`);
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'StudyGuide.md';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleGenerateStudyDeck = async () => {
    setIsDeckLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/export/presentation', {
        method: 'POST',
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Export failed'); }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'LectureLens_StudyDeck.pptx';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert('Study Deck export failed: ' + err.message);
    } finally {
      setIsDeckLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="min-h-screen p-6 md:p-10 flex flex-col max-w-7xl mx-auto"
    >
      {/* Header Section */}
      <header className="mb-8">
        <div className="flex flex-col md:flex-row gap-6 justify-between items-start md:items-center mb-8 pb-4 border-b border-white">
          <h2 className="text-3xl font-bold tracking-tighter">
            Stream<span className="italic">Stamper</span>
          </h2>
        </div>

        {/* Search Input right above buttons */}
        <div className="mb-4">
          <div className="relative w-full">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter search query..."
              className="w-full bg-black border border-white text-white px-12 py-3 focus:outline-none focus:ring-1 focus:ring-white transition-shadow shadow-[0_0_10px_rgba(255,255,255,0.05)] focus:shadow-[0_0_20px_rgba(255,255,255,0.2)]"
            />
          </div>
        </div>

        {/* Action Buttons Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <button 
            onClick={() => fetchResults('asr')}
            disabled={isLoading}
            className={`flex items-center justify-center gap-2 border border-white px-4 py-3 transition-colors ${activeFilter === 'asr' ? 'bg-white text-black' : 'bg-black text-white hover:bg-white/10'} disabled:opacity-50`}
          >
            <Mic className="w-4 h-4" />
            <span className="text-sm font-semibold tracking-wide">Search Spoken (ASR)</span>
          </button>
          <button 
            onClick={() => fetchResults('objects')}
            disabled={isLoading}
            className={`flex items-center justify-center gap-2 border border-white px-4 py-3 transition-colors ${activeFilter === 'objects' ? 'bg-white text-black' : 'bg-black text-white hover:bg-white/10'} disabled:opacity-50`}
          >
            <Box className="w-4 h-4" />
            <span className="text-sm font-semibold tracking-wide">Search Objects (YOLO)</span>
          </button>
          <button 
            onClick={() => fetchResults('ocr')}
            disabled={isLoading}
            className={`flex items-center justify-center gap-2 border border-white px-4 py-3 transition-colors ${activeFilter === 'ocr' ? 'bg-white text-black' : 'bg-black text-white hover:bg-white/10'} disabled:opacity-50`}
          >
            <FileText className="w-4 h-4" />
            <span className="text-sm font-semibold tracking-wide">Search Written (OCR)</span>
          </button>
          <button 
            onClick={handleGeneratePPT}
            disabled={isPptLoading || results.length === 0}
            className="flex items-center justify-center gap-2 border border-blue-500 bg-black hover:bg-blue-500/20 text-white px-4 py-3 transition-colors disabled:opacity-50">
            {isPptLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            <span className="text-sm font-semibold tracking-wide pt-0.5">{isPptLoading ? 'Generating...' : 'Export Results PPT'}</span>
          </button>
          <button 
            onClick={handleExportNotion}
            disabled={results.length === 0}
            className="flex items-center justify-center gap-2 border border-green-500 bg-black hover:bg-green-500/20 text-white px-4 py-3 transition-colors disabled:opacity-50">
            <BookOpen className="w-4 h-4" />
            <span className="text-sm font-semibold tracking-wide pt-0.5">Export to Notion</span>
          </button>
          <button 
            onClick={handleGenerateStudyDeck}
            disabled={isDeckLoading}
            className="flex items-center justify-center gap-2 border border-purple-500 bg-black hover:bg-purple-500/20 text-white px-4 py-3 transition-colors disabled:opacity-50">
            {isDeckLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GraduationCap className="w-4 h-4" />}
            <span className="text-sm font-semibold tracking-wide pt-0.5">{isDeckLoading ? 'Building...' : 'Generate Study Deck'}</span>
          </button>
        </div>
      </header>

      {/* Main Content Table Area */}
      <main className="flex-1 border border-white overflow-hidden flex flex-col relative">
        {isLoading && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
            <Loader2 className="w-8 h-8 animate-spin text-white" />
          </div>
        )}

        <div className="grid grid-cols-12 gap-4 border-b border-white p-4 font-bold tracking-widest uppercase text-xs">
          <div className="col-span-3 lg:col-span-2">Timestamp</div>
          <div className="col-span-4 lg:col-span-5 border-l border-white pl-4">Spoken Words</div>
          <div className="col-span-5 lg:col-span-5 border-l border-white pl-4">Visual Content</div>
        </div>

        <div className="overflow-y-auto flex-1 p-4 space-y-4">
          {results.length === 0 && !isLoading && (
            <div className="text-center text-white/50 py-10 uppercase tracking-widest text-sm">
              {activeFilter ? "No results found." : "Run a search to see results."}
            </div>
          )}
          
          {results.map((row, index) => (
            <div key={row.id || index} className="grid grid-cols-12 gap-4 p-4 border border-white hover:shadow-[0_0_15px_rgba(255,255,255,0.1)] transition-shadow group">
              <div className="col-span-3 lg:col-span-2">
                <a 
                  href={getYoutubeLink(row.metadata?.start_sec)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 hover:underline font-mono text-sm transition-colors cursor-pointer text-left block"
                >
                  {formatTime(row.metadata?.start_sec)}
                  {row.metadata?.end_sec ? ` - ${formatTime(row.metadata.end_sec)}` : ''}
                </a>
              </div>
              <div className="col-span-4 lg:col-span-5 leading-relaxed text-sm md:text-base pr-4">
                {activeFilter === 'asr' ? row.text : ''}
              </div>
              <div className="col-span-5 lg:col-span-5 leading-relaxed text-sm md:text-base border-l border-white pl-4 bg-black/50">
                {formatVisualText(activeFilter !== 'asr' ? (row.text || '') : '')}
              </div>
            </div>
          ))}
        </div>
      </main>
      
      <footer className="mt-8 text-center text-xs tracking-widest uppercase text-white/50 pb-4">
        LectureLens Indexing Engine v1.0.0
      </footer>
    </motion.div>
  );
}
