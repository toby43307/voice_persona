import React from 'react';
import { AppView } from '../types';

interface HomeProps {
  onNavigate: (view: AppView) => void;
}

export const Home: React.FC<HomeProps> = ({ onNavigate }) => {
  return (
    <div className="min-h-screen bg-[#050505] flex items-center justify-center relative overflow-hidden">
      {/* Background Ambience */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-900 rounded-full blur-[100px]"></div>
      </div>

      <div className="relative z-10 w-full max-w-lg p-8">
        <div className="bg-[#0f0f13] border border-gray-800 rounded-2xl p-8 text-center glow-box">
          
          <div className="mb-10 space-y-2">
            <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-200 glow-text tracking-wide">
              DATA HAMMER GROUP
            </h1>
            <p className="text-blue-400 text-lg tracking-[0.2em] font-light">
              TALKING SYSTEM
            </p>
          </div>

          <div className="space-y-4">
            <button 
              className="w-full py-3 px-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white rounded-lg font-medium transition-all duration-200 shadow-lg shadow-blue-900/30 flex items-center justify-center opacity-80 hover:opacity-100 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled
            >
              训练模型 (Train Model)
            </button>
            
            <button 
              className="w-full py-3 px-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white rounded-lg font-medium transition-all duration-200 shadow-lg shadow-blue-900/30 flex items-center justify-center opacity-80 hover:opacity-100 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled
            >
               视频生成 (Video Generation)
            </button>
            
            <button 
              onClick={() => onNavigate(AppView.CONVERSATION)}
              className="w-full py-3 px-6 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-400 hover:to-cyan-400 text-white rounded-lg font-bold transition-all duration-200 shadow-lg shadow-cyan-900/40 transform hover:scale-[1.02] flex items-center justify-center"
            >
              人机对话 (Interaction)
            </button>
          </div>

          <div className="mt-12 text-gray-600 text-xs tracking-wider">
            &copy; 2025 DATA HAMMER LAB
          </div>
        </div>
      </div>
    </div>
  );
};
