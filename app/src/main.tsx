import { QueryClientProvider } from '@tanstack/react-query';
// import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './i18n';
import './index.css';
import { queryClient } from './lib/queryClient';

function installWebKitAudioConstraintGuard() {
  const mediaDevices = navigator.mediaDevices;
  if (!mediaDevices?.getUserMedia) return;

  const nativeGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices);
  mediaDevices.getUserMedia = (constraints?: MediaStreamConstraints) => {
    if (constraints?.audio && typeof constraints.audio === 'object') {
      return nativeGetUserMedia({ ...constraints, audio: true });
    }
    return nativeGetUserMedia(constraints);
  };
}

installWebKitAudioConstraintGuard();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      {/* <ReactQueryDevtools initialIsOpen={false} /> */}
    </QueryClientProvider>
  </React.StrictMode>,
);
