import { invoke } from '@tauri-apps/api/core';
import { useEffect } from 'react';
import { useDictationReadiness } from '@/lib/hooks/useDictationReadiness';
import { useCaptureSettings } from '@/lib/hooks/useSettings';
import { usePlatform } from '@/platform/PlatformContext';

/**
 * Spawn (or quiet) the global hotkey monitor based on the saved
 * `capture_settings.hotkey_enabled` flag and the recording readiness gates,
 * and keep its bindings in sync with the user's chord choices.
 *
 * Boot sequence:
 *  - hotkey_enabled = false OR a model gate is missing → call
 *    `disable_hotkey` (no-op if monitor was never spawned). Crucially, we do
 *    *not* call `enable_hotkey` when users haven't opted in, AND the chord
 *    physically can't fire when models aren't downloaded — preventing
 *    the "stuck pill" failure mode where dictation triggers but has nowhere
 *    to land.
 *  - hotkey_enabled = true AND model gates green → call `enable_hotkey` with
 *    the saved chords. This creates the CGEventTap and, on macOS, requests
 *    Input Monitoring on first opt-in. We intentionally do this even when the
 *    current Input Monitoring check is false; otherwise a reset/stale TCC
 *    grant can deadlock the app in "missing permission" without ever asking
 *    macOS to add Voicebox back to the list.
 *
 * Call once from the main app shell.
 */
export function useChordSync() {
  const platform = usePlatform();
  const { settings } = useCaptureSettings();
  const { stt, llm } = useDictationReadiness();
  const enabled = settings?.hotkey_enabled;
  const pushKeys = settings?.chord_push_to_talk_keys;
  const toggleKeys = settings?.chord_toggle_to_talk_keys;

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    if (enabled === undefined || !pushKeys || !toggleKeys) return;
    const modelsReady = Boolean(stt?.ready && llm?.ready);
    const shouldArm = enabled && modelsReady;
    const command = shouldArm ? 'enable_hotkey' : 'disable_hotkey';
    const args = shouldArm ? { pushToTalk: pushKeys, toggleToTalk: toggleKeys } : {};
    invoke(command, args).catch((err) => {
      console.warn(`[chord-sync] ${command} failed:`, err);
    });
  }, [
    platform.metadata.isTauri,
    enabled,
    stt?.ready,
    llm?.ready,
    // Stringify so a referentially-new array with the same content
    // doesn't fire a redundant invoke on every settings refetch.
    pushKeys?.join(','),
    toggleKeys?.join(','),
  ]);
}
