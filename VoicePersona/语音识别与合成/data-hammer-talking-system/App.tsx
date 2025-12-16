import React, { useState } from 'react';
import { Home } from './components/Home';
import { ConversationInterface } from './components/ConversationConfig';
import { AppView } from './types';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<AppView>(AppView.HOME);

  const renderView = () => {
    switch (currentView) {
      case AppView.HOME:
        return <Home onNavigate={setCurrentView} />;
      case AppView.CONVERSATION:
        return <ConversationInterface onBack={() => setCurrentView(AppView.HOME)} />;
      default:
        return <Home onNavigate={setCurrentView} />;
    }
  };

  return (
    <main className="w-full h-full">
      {renderView()}
    </main>
  );
};

export default App;
