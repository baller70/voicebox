import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';
import { convertToWav } from '@/lib/utils/audio';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
  onRecordingChunk?: (blob: Blob, duration?: number) => void | Promise<void>;
}

export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
  onRecordingChunk,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const chunkTimerRef = useRef<number | null>(null);
  const chunkReadActiveRef = useRef(false);
  const startTimeRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);
  const recordingActiveRef = useRef(false);
  const stopRecordingRef = useRef<(() => Promise<void>) | null>(null);
  const useNativeMicrophone = platform.metadata.isTauri;

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      chunksRef.current = [];
      cancelledRef.current = false;
      setDuration(0);

      if (useNativeMicrophone) {
        // Ask WKWebView for microphone permission before opening the native
        // CoreAudio stream. cpal does not trigger macOS's TCC prompt itself,
        // so it otherwise reports an opaque CoreAudio error on first use.
        let permissionError: unknown = null;
        const mediaDevices = typeof navigator !== 'undefined' ? navigator.mediaDevices : null;
        if (mediaDevices?.getUserMedia) {
          try {
            const permissionStream = await mediaDevices.getUserMedia({ audio: true });
            permissionStream.getTracks().forEach((track) => track.stop());
          } catch (err) {
            permissionError = err;
            console.warn('WebView microphone permission request failed:', err);
          }
        }

        try {
          // Native capture treats 0 as unlimited. Dictation passes undefined
          // and must run until the user stops the hotkey; voice/profile
          // recorders pass an explicit cap when they need one.
          await platform.audio.startMicrophoneCapture(maxDurationSeconds ?? 0);
        } catch (err) {
          if (permissionError instanceof DOMException && permissionError.name === 'NotAllowedError') {
            throw new Error(
              'Microphone permission was denied. Allow Voicebox in System Settings > Privacy & Security > Microphone.',
            );
          }
          throw err;
        }
        setIsRecording(true);
        recordingActiveRef.current = true;
        startTimeRef.current = Date.now();
        timerRef.current = window.setInterval(() => {
          if (!startTimeRef.current) return;
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          setDuration(elapsed);
          if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
            void stopRecordingRef.current?.();
          }
        }, 100);
        if (onRecordingChunk && platform.audio.readMicrophoneCaptureChunk) {
          chunkTimerRef.current = window.setInterval(() => {
            if (chunkReadActiveRef.current || !startTimeRef.current) return;
            chunkReadActiveRef.current = true;
            platform.audio
              .readMicrophoneCaptureChunk?.(2500)
              .then((blob) => {
                if (!blob || !blob.size || !startTimeRef.current) return;
                const elapsed = (Date.now() - startTimeRef.current) / 1000;
                void onRecordingChunk(blob, elapsed);
              })
              .catch((err) => {
                console.warn('Progressive microphone chunk failed:', err);
              })
              .finally(() => {
                chunkReadActiveRef.current = false;
              });
          }, 4500);
        }
        return;
      }

      // Check if getUserMedia is available for the web-only fallback.
      if (typeof navigator === 'undefined') {
        throw new Error('Navigator API is not available.');
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error(
            'Microphone access is not available. Please use a secure context and allow microphone access.',
          );
        }
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      streamRef.current = stream;

      // Let WKWebView choose the recorder container/codec. Passing an
      // explicit WebM/Opus preference can fail on some macOS WebKit builds.
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Snapshot the cancellation flag and recorded duration immediately —
        // cancelRecording() clears chunks and sets cancelledRef synchronously
        // before this async handler runs, so we must check it first.
        const wasCancelled = cancelledRef.current;
        const recordedDuration = startTimeRef.current
          ? (Date.now() - startTimeRef.current) / 1000
          : undefined;

        const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        // Stop all tracks now that we have the data
        streamRef.current?.getTracks().forEach((track) => {
          track.stop();
        });
        streamRef.current = null;

        // Don't fire completion callback if the recording was cancelled
        if (wasCancelled) return;

        // Convert to WAV format to avoid needing ffmpeg on backend
        try {
          const wavBlob = await convertToWav(webmBlob);
          onRecordingComplete?.(wavBlob, recordedDuration);
        } catch (err) {
          console.error('Error converting audio to WAV:', err);
          // Fallback to original blob if conversion fails
          onRecordingComplete?.(webmBlob, recordedDuration);
        }
      };

      mediaRecorder.onerror = (event) => {
        setError('Recording error occurred');
        console.error('MediaRecorder error:', event);
      };

      // WebKit's MediaRecorder drops the WebM EBML header from chunks when
      // started with a timeslice, so concatenated blobs fail to parse in
      // both AudioContext and ffmpeg. Starting with no timeslice produces
      // exactly one dataavailable on stop() with a valid container.
      mediaRecorder.start();
      setIsRecording(true);
      recordingActiveRef.current = true;
      startTimeRef.current = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (startTimeRef.current) {
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration when the caller opts in — dictation
          // sessions pass undefined and run until the user releases the
          // chord or hits stop; voice-clone sample recorders pass 29s to
          // keep reference clips short.
          if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
              mediaRecorderRef.current.stop();
              setIsRecording(false);
              recordingActiveRef.current = false;
              if (timerRef.current !== null) {
                clearInterval(timerRef.current);
                timerRef.current = null;
              }
            }
          }
        }
      }, 100);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to access microphone. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
      recordingActiveRef.current = false;
    }
  }, [maxDurationSeconds, onRecordingChunk, onRecordingComplete, platform, useNativeMicrophone]);

  const stopRecording = useCallback(async () => {
    if (!isRecording) return;

    if (useNativeMicrophone) {
      setIsRecording(false);
      recordingActiveRef.current = false;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (chunkTimerRef.current !== null) {
        clearInterval(chunkTimerRef.current);
        chunkTimerRef.current = null;
      }
      try {
        for (let attempt = 0; chunkReadActiveRef.current && attempt < 20; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 50));
        }
        if (onRecordingChunk && platform.audio.readMicrophoneCaptureChunk && startTimeRef.current) {
          const chunk = await platform.audio.readMicrophoneCaptureChunk(300);
          if (chunk?.size) {
            const elapsed = (Date.now() - startTimeRef.current) / 1000;
            await onRecordingChunk(chunk, elapsed);
          }
        }
        const blob = await platform.audio.stopMicrophoneCapture();
        const recordedDuration = startTimeRef.current
          ? (Date.now() - startTimeRef.current) / 1000
          : undefined;
        onRecordingComplete?.(blob, recordedDuration);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to stop microphone capture.');
      }
      return;
    }

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      recordingActiveRef.current = false;

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [isRecording, onRecordingChunk, onRecordingComplete, platform, useNativeMicrophone]);

  useEffect(() => {
    stopRecordingRef.current = stopRecording;
  }, [stopRecording]);

  const cancelRecording = useCallback(async () => {
    if (useNativeMicrophone) {
      cancelledRef.current = true;
      if (isRecording) {
        await platform.audio.stopMicrophoneCapture().catch(() => {});
      }
      setIsRecording(false);
      recordingActiveRef.current = false;
      setDuration(0);
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (chunkTimerRef.current !== null) {
        clearInterval(chunkTimerRef.current);
        chunkTimerRef.current = null;
      }
      return;
    }

    if (mediaRecorderRef.current) {
      cancelledRef.current = true; // Must be set before stop() triggers onstop
      chunksRef.current = [];
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      recordingActiveRef.current = false;
      setDuration(0);
    }

    // Stop all tracks
    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [isRecording, platform, useNativeMicrophone]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      if (chunkTimerRef.current !== null) {
        clearInterval(chunkTimerRef.current);
      }
      if (recordingActiveRef.current && useNativeMicrophone) {
        void platform.audio.stopMicrophoneCapture().catch(() => {});
      }
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, [platform, useNativeMicrophone]);

  return {
    isRecording,
    duration,
    error,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
