// Compact transport that replaces the browser's native <audio controls>. The media element keeps
// owning playback and is the single time state: the transport, the waveform scrubber and the
// section rail all read and write `currentTime` on it.

export const PLAYBACK_RATES = Object.freeze([0.5, 0.75, 1, 1.25, 1.5, 2]);

export function formatTransportTime(value) {
  const total = Number.isFinite(value) && value > 0 ? value : 0;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = Math.floor(total % 60);
  const mm = hours ? String(minutes).padStart(2, "0") : String(minutes);
  return `${hours ? `${hours}:` : ""}${mm}:${String(seconds).padStart(2, "0")}`;
}

export function transportTimeText(currentTime, duration) {
  const known = Number.isFinite(duration) && duration > 0;
  return `${formatTransportTime(currentTime)} / ${known ? formatTransportTime(duration) : "--:--"}`;
}

/** Clamps a requested rate to the offered set, falling back to normal speed. */
export function normalizePlaybackRate(value) {
  const rate = Number(value);
  return PLAYBACK_RATES.includes(rate) ? rate : 1;
}

/**
 * Builds the transport for `media` and inserts it right after the element. `doc` is the owning
 * document. Returns a handle whose `sync()` re-reads the media state and `dispose()` removes it.
 */
export function setupAnalysisPlayer(media, doc = media.ownerDocument) {
  media.removeAttribute("controls");
  media.controls = false;

  const transport = doc.createElement("div");
  transport.className = "analysis-transport";
  transport.setAttribute("role", "group");
  transport.setAttribute("aria-label", "Playback");

  const playButton = doc.createElement("button");
  playButton.type = "button";
  playButton.className = "transport-button transport-play";

  const time = doc.createElement("span");
  time.className = "transport-time";
  time.setAttribute("aria-live", "off");

  const muteButton = doc.createElement("button");
  muteButton.type = "button";
  muteButton.className = "transport-button transport-mute";

  const volume = doc.createElement("input");
  volume.type = "range";
  volume.className = "transport-volume";
  volume.min = "0";
  volume.max = "1";
  volume.step = "0.05";
  volume.setAttribute("aria-label", "Volume");

  const rateLabel = doc.createElement("label");
  rateLabel.className = "transport-rate";
  rateLabel.textContent = "Speed ";
  const rate = doc.createElement("select");
  rate.setAttribute("aria-label", "Playback speed");
  for (const value of PLAYBACK_RATES) {
    const option = doc.createElement("option");
    option.value = String(value);
    option.textContent = `${value}×`;
    rate.append(option);
  }
  rateLabel.append(rate);

  transport.append(playButton, time, muteButton, volume, rateLabel);
  media.after(transport);

  const hasSource = () => Boolean(media.currentSrc || media.getAttribute("src"));

  function sync() {
    const playing = !media.paused && !media.ended;
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
    playButton.setAttribute("aria-pressed", String(playing));
    const loaded = hasSource();
    playButton.disabled = !loaded;
    time.textContent = transportTimeText(Number(media.currentTime) || 0, Number(media.duration));
    const muted = media.muted || media.volume === 0;
    muteButton.textContent = muted ? "Unmute" : "Mute";
    muteButton.setAttribute("aria-label", muted ? "Unmute" : "Mute");
    muteButton.setAttribute("aria-pressed", String(muted));
    volume.value = String(media.muted ? 0 : media.volume);
    rate.value = String(normalizePlaybackRate(media.playbackRate));
  }

  async function togglePlayback() {
    if (!hasSource()) return;
    if (media.paused || media.ended) {
      try {
        await media.play();
      } catch {
        // Autoplay policy or a decode error: the state below reflects that nothing plays.
      }
    } else {
      media.pause();
    }
    sync();
  }

  const listeners = [
    [playButton, "click", () => void togglePlayback()],
    [
      muteButton,
      "click",
      () => {
        if (media.muted || media.volume === 0) {
          media.muted = false;
          if (media.volume === 0) media.volume = 1;
        } else {
          media.muted = true;
        }
        sync();
      },
    ],
    [
      volume,
      "input",
      () => {
        const level = Math.min(1, Math.max(0, Number(volume.value) || 0));
        media.volume = level;
        media.muted = level === 0;
        sync();
      },
    ],
    [
      rate,
      "change",
      () => {
        const next = normalizePlaybackRate(rate.value);
        media.playbackRate = next;
        media.defaultPlaybackRate = next;
        sync();
      },
    ],
  ];
  const mediaEvents = [
    "play",
    "pause",
    "ended",
    "timeupdate",
    "seeked",
    "durationchange",
    "loadedmetadata",
    "emptied",
    "volumechange",
    "ratechange",
  ];
  for (const [target, type, handler] of listeners) target.addEventListener(type, handler);
  for (const type of mediaEvents) media.addEventListener(type, sync);
  sync();

  return {
    element: transport,
    sync,
    togglePlayback,
    dispose() {
      for (const [target, type, handler] of listeners) target.removeEventListener(type, handler);
      for (const type of mediaEvents) media.removeEventListener(type, sync);
      transport.remove();
    },
  };
}

/** Space and K toggle playback from the waveform scrubber, as in common media players. */
export function isPlaybackToggleKey(event) {
  return (event.key === " " || event.key === "k" || event.key === "K") && !event.altKey && !event.ctrlKey && !event.metaKey;
}

function setupPage() {
  for (const [mediaSelector, scrubberSelector] of [
    ["#audio-player", "#waveform"],
    ["#song-audio-player", "#song-timeline"],
  ]) {
    const media = document.querySelector(mediaSelector);
    if (!media) continue;
    const player = setupAnalysisPlayer(media, document);
    document.querySelector(scrubberSelector)?.addEventListener("keydown", (event) => {
      if (!isPlaybackToggleKey(event)) return;
      event.preventDefault();
      void player.togglePlayback();
    });
  }
}

if (typeof document !== "undefined" && typeof window !== "undefined") setupPage();
