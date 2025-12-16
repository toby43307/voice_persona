export enum AppView {
  HOME = 'HOME',
  CONVERSATION = 'CONVERSATION',
}

export interface ConversationConfig {
  modelName: string;
  modelPath: string;
  refAudioPath: string;
  gpuIndex: string;
  voiceCloneModel: string;
  apiProvider: string;
}

export enum ConnectionState {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR',
}
