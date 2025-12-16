import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';

// Audio Context and Processing Variables
let audioContext: AudioContext | null = null;
let inputNode: GainNode | null = null;
let outputNode: GainNode | null = null;
let scriptProcessor: ScriptProcessorNode | null = null;
let sourceStream: MediaStreamAudioSourceNode | null = null;
let nextStartTime = 0;
const sources = new Set<AudioBufferSourceNode>();

// Helper to encode PCM data for the API
function encodePCM(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Helper to decode Base64 to PCM
function decodeBase64(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Helper to create AudioBuffer from PCM data
async function createAudioBuffer(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length;
  const buffer = ctx.createBuffer(1, frameCount, sampleRate);
  const channelData = buffer.getChannelData(0);
  
  for (let i = 0; i < frameCount; i++) {
    channelData[i] = dataInt16[i] / 32768.0;
  }
  return buffer;
}

// Create Blob for sending to API
function createPCM16Blob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encodePCM(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

export interface LiveSessionCallbacks {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
  onVolumeChange?: (volume: number) => void; // For visualizer
}

export class GeminiLiveService {
  private ai: GoogleGenAI;
  private sessionPromise: Promise<any> | null = null;
  private isActive = false;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async startSession(callbacks: LiveSessionCallbacks) {
    try {
      this.isActive = true;
      
      // 1. Setup Audio Contexts
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000, // Gemini Output Rate
      });
      
      // Input context (usually microphone is 16k or 44.1k/48k, we will downsample/resample if needed via context)
      // Note: We use the same context or a separate one. Let's use a separate 16k context for input to match Gemini requirement easily.
      const inputContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000, 
      });

      outputNode = audioContext.createGain();
      outputNode.connect(audioContext.destination);

      // 2. Get Microphone Stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 3. Connect to Gemini Live
      this.sessionPromise = this.ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
            console.log('Gemini Live Session Opened');
            callbacks.onOpen?.();

            // Start Audio Streaming Pipeline
            sourceStream = inputContext.createMediaStreamSource(stream);
            scriptProcessor = inputContext.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              if (!this.isActive) return;
              
              const inputData = e.inputBuffer.getChannelData(0);
              
              // Calculate volume for visualizer
              let sum = 0;
              for (let i = 0; i < inputData.length; i++) {
                  sum += inputData[i] * inputData[i];
              }
              const rms = Math.sqrt(sum / inputData.length);
              callbacks.onVolumeChange?.(rms);

              const pcmBlob = createPCM16Blob(inputData);
              
              this.sessionPromise?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };

            sourceStream.connect(scriptProcessor);
            scriptProcessor.connect(inputContext.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
             // Handle interruptions
            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              this.stopAllAudio();
              nextStartTime = 0;
            }

            // Handle Audio Output
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio && audioContext) {
              nextStartTime = Math.max(nextStartTime, audioContext.currentTime);
              
              const pcmData = decodeBase64(base64Audio);
              const audioBuffer = await createAudioBuffer(pcmData, audioContext);
              
              const source = audioContext.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode!);
              
              source.addEventListener('ended', () => {
                sources.delete(source);
              });
              
              source.start(nextStartTime);
              nextStartTime += audioBuffer.duration;
              sources.add(source);
              
              // Fake volume for visualizer based on output
               callbacks.onVolumeChange?.(0.5); // Simplified for visualizer feedback during playback
            }
          },
          onclose: () => {
            console.log('Session Closed');
            this.cleanup();
            callbacks.onClose?.();
          },
          onerror: (e) => {
            console.error('Session Error', e);
            this.cleanup();
            callbacks.onError?.(new Error('Connection error'));
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: 'You are a helpful AI assistant in a real-time voice conversation system.',
        },
      });

    } catch (error) {
      console.error("Failed to start session:", error);
      callbacks.onError?.(error instanceof Error ? error : new Error('Unknown error'));
      this.cleanup();
    }
  }

  private stopAllAudio() {
    for (const source of sources) {
      source.stop();
    }
    sources.clear();
  }

  stopSession() {
    this.isActive = false;
    // We cannot explicitly close the session via the SDK client easily if the close method isn't exposed on the session object
    // but disconnecting the input stream effectively stops the interaction.
    // The LiveClient doesn't have a direct 'disconnect' on the `ai.live` property, 
    // but the `connect` returns a Promise<LiveSession>.
    // For now, we clean up local resources.
    this.cleanup();
  }

  private cleanup() {
    this.isActive = false;
    
    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (sourceStream) {
      sourceStream.disconnect();
      sourceStream = null;
    }
    
    // Stop tracks
    navigator.mediaDevices.getUserMedia({audio: true}).then(stream => {
        stream.getTracks().forEach(track => track.stop());
    }).catch(() => {}); // Ignore if already stopped
    
    this.stopAllAudio();
    
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
      audioContext = null;
    }
  }
}

export const liveService = new GeminiLiveService();
