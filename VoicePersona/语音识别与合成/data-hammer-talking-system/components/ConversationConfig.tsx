import React, { useState, useEffect, useRef } from 'react';
import { AppView, ConversationConfig, ConnectionState } from '../types';
import { liveService } from '../services/geminiLive';

interface ConversationConfigProps {
  onBack: () => void;
}

export const ConversationInterface: React.FC<ConversationConfigProps> = ({ onBack }) => {
  const [config, setConfig] = useState<ConversationConfig>({
    modelName: 'Gemini-2.5-Flash',
    modelPath: '',
    refAudioPath: '',
    gpuIndex: 'GPU 0',
    voiceCloneModel: 'Voice Clone A',
    apiProvider: 'Gemini Live API',
  });

  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [volume, setVolume] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleStart = async () => {
    if (connectionState === ConnectionState.CONNECTED) {
      liveService.stopSession();
      setConnectionState(ConnectionState.DISCONNECTED);
      setVolume(0);
      return;
    }

    setConnectionState(ConnectionState.CONNECTING);
    
    await liveService.startSession({
      onOpen: () => setConnectionState(ConnectionState.CONNECTED),
      onClose: () => {
          setConnectionState(ConnectionState.DISCONNECTED);
          setVolume(0);
      },
      onError: (e) => {
        alert("Error connecting: " + e.message);
        setConnectionState(ConnectionState.ERROR);
        setVolume(0);
      },
      onVolumeChange: (vol) => {
        setVolume(Math.min(vol * 5, 1)); // Amplify for visuals
      }
    });
  };

  // Simple visualizer effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      if (connectionState === ConnectionState.CONNECTED) {
        // Draw Audio Waveform simulation
        ctx.beginPath();
        ctx.strokeStyle = '#3b82f6'; // Blue-500
        ctx.lineWidth = 2;
        
        for (let i = 0; i < canvas.width; i++) {
            const amplitude = volume * 50;
            const frequency = 0.1;
            const y = centerY + Math.sin(i * frequency + Date.now() * 0.01) * amplitude * Math.sin(i * 0.05);
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        ctx.stroke();

        // Glowing Circle Center
        const radius = 20 + volume * 30;
        const gradient = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, radius);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();

      } else {
        // Static Placeholder Text
        ctx.fillStyle = '#333';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText("Waiting for connection...", centerX, centerY);
      }

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(animationId);
  }, [volume, connectionState]);

  return (
    <div className="min-h-screen bg-[#f3f4f6] flex flex-col items-center py-10 px-4">
      <h1 className="text-2xl font-bold text-gray-700 mb-8">实时对话系统</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-sm w-full max-w-5xl flex flex-col md:flex-row gap-8">
        
        {/* Left Side: Video/Visualizer Area */}
        <div className="flex-1">
          <div className="bg-black rounded-lg aspect-video w-full flex items-center justify-center overflow-hidden relative shadow-inner">
             <canvas 
                ref={canvasRef} 
                width={640} 
                height={360} 
                className="w-full h-full object-cover"
             />
             
             {/* Fake Player Controls overlay to match screenshot */}
             <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-black/80 to-transparent flex items-center px-4 gap-4 opacity-70 hover:opacity-100 transition-opacity">
                <div className="text-white text-xs">▶ 0:00</div>
                <div className="flex-1 h-1 bg-gray-600 rounded-full"></div>
                <div className="text-white text-xs">🔊</div>
             </div>
          </div>
        </div>

        {/* Right Side: Configuration Form */}
        <div className="w-full md:w-[400px] flex flex-col gap-4">
          
          {/* Model Name */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">模型名称</label>
            <select 
              value={config.modelName}
              onChange={(e) => setConfig({...config, modelName: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 bg-white"
            >
              <option>Gemini-2.5-Flash</option>
              <option>Gemini-Pro-Preview</option>
              <option>Model 1</option>
            </select>
          </div>

          {/* Model Path */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">模型目录地址</label>
            <input 
              type="text" 
              placeholder="请输入模型目录路径"
              value={config.modelPath}
              onChange={(e) => setConfig({...config, modelPath: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 placeholder-gray-400"
            />
          </div>

          {/* Reference Audio Path */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">参考音频地址</label>
            <input 
              type="text" 
              placeholder="请输入参考音频路径"
              value={config.refAudioPath}
              onChange={(e) => setConfig({...config, refAudioPath: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 placeholder-gray-400"
            />
          </div>

          {/* GPU Selection */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">GPU选择</label>
            <select 
              value={config.gpuIndex}
              onChange={(e) => setConfig({...config, gpuIndex: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 bg-white"
            >
              <option>GPU 0</option>
              <option>GPU 1</option>
            </select>
          </div>

          {/* Voice Clone Model */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">语音克隆模型名称</label>
            <select 
              value={config.voiceCloneModel}
              onChange={(e) => setConfig({...config, voiceCloneModel: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 bg-white"
            >
              <option>Voice Clone A</option>
              <option>Voice Clone B</option>
            </select>
          </div>

          {/* API Selection */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-600">对话API选择</label>
            <select 
              value={config.apiProvider}
              onChange={(e) => setConfig({...config, apiProvider: e.target.value})}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 bg-white"
            >
              <option>Gemini Live API</option>
              <option>OpenAI API</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="mt-4 flex gap-2">
            <button 
                onClick={onBack}
                className="flex-1 py-2 px-4 border border-gray-300 text-gray-600 rounded hover:bg-gray-50 text-sm transition-colors"
            >
                返回
            </button>
            <button 
                onClick={handleStart}
                disabled={connectionState === ConnectionState.CONNECTING}
                className={`flex-1 py-2 px-4 text-white rounded text-sm transition-colors flex items-center justify-center gap-2
                    ${connectionState === ConnectionState.CONNECTED 
                        ? 'bg-red-500 hover:bg-red-600' 
                        : 'bg-[#3b82f6] hover:bg-blue-600'
                    } ${connectionState === ConnectionState.CONNECTING ? 'opacity-70 cursor-wait' : ''}`}
            >
                {connectionState === ConnectionState.CONNECTING && (
                    <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                )}
                {connectionState === ConnectionState.CONNECTED ? '■ 停止对话' : '▶ 开始对话'}
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};
