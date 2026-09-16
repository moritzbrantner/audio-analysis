const recordButton = document.querySelector("#record-audio");
const recordingStatus = document.querySelector("#recording-status");
const fileInput = document.querySelector("#file-input");
const chooseFile = document.querySelector("#choose-file");
const inputPanel = document.querySelector("#input-panel");
const inputError = document.querySelector("#input-error");
const exampleButtons = Array.from(document.querySelectorAll("[data-example]"));

const preferredMimeTypes = [
  "audio/webm;codecs=opus",
  "audio/ogg;codecs=opus",
  "audio/mp4",
  "audio/webm",
  "audio/ogg",
];

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let finalizing = false;

if (recordButton && recordingStatus && fileInput && chooseFile && inputPanel) {
  const supported = Boolean(navigator.mediaDevices?.getUserMedia && typeof MediaRecorder === "function");

  if (!supported) {
    recordButton.disabled = true;
    recordingStatus.textContent =
      "Voice recording is not supported by this browser. You can still choose an audio file.";
  } else {
    recordButton.addEventListener("click", () => {
      if (mediaRecorder?.state === "recording") stopRecording();
      else void startRecording();
    });

    const busyObserver = new MutationObserver(syncRecordAvailability);
    busyObserver.observe(inputPanel, { attributes: true, attributeFilter: ["aria-busy"] });
    syncRecordAvailability();
  }
}

async function startRecording() {
  if (finalizing || inputPanel.getAttribute("aria-busy") === "true") return;

  clearCaptureError();
  recordButton.disabled = true;
  recordingStatus.textContent = "Requesting microphone access…";

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = selectRecordingMimeType();
    mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
    recordedChunks = [];

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    });

    mediaRecorder.addEventListener("error", (event) => {
      showCaptureError(`Recording failed: ${event.error?.message ?? "unknown MediaRecorder error"}`);
    });

    mediaRecorder.addEventListener("stop", finishRecording, { once: true });
    mediaRecorder.start();

    setCompetingInputsDisabled(true);
    recordButton.disabled = false;
    recordButton.textContent = "Stop recording";
    recordButton.setAttribute("aria-pressed", "true");
    recordingStatus.textContent = "Recording… Stop when you are ready to analyze this audio.";
  } catch (error) {
    stopMediaStream();
    mediaRecorder = null;
    setCompetingInputsDisabled(false);
    recordButton.textContent = "Record voice";
    recordButton.setAttribute("aria-pressed", "false");
    syncRecordAvailability();
    showCaptureError(captureErrorMessage(error));
  }
}

function stopRecording() {
  if (mediaRecorder?.state !== "recording" || finalizing) return;
  finalizing = true;
  recordButton.disabled = true;
  recordingStatus.textContent = "Finishing recording…";
  mediaRecorder.stop();
}

function finishRecording() {
  const recorder = mediaRecorder;
  const chunks = recordedChunks;

  mediaRecorder = null;
  recordedChunks = [];
  stopMediaStream();
  setCompetingInputsDisabled(false);
  recordButton.textContent = "Record voice";
  recordButton.setAttribute("aria-pressed", "false");
  finalizing = false;

  if (!chunks.length) {
    syncRecordAvailability();
    showCaptureError("The microphone recording was empty. Try recording again.");
    return;
  }

  const mimeType = recorder?.mimeType || chunks[0]?.type || "audio/webm";
  const blob = new Blob(chunks, { type: mimeType });
  if (!blob.size) {
    syncRecordAvailability();
    showCaptureError("The microphone recording was empty. Try recording again.");
    return;
  }

  const file = new File([blob], recordingFileName(mimeType), {
    type: mimeType,
    lastModified: Date.now(),
  });

  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    recordingStatus.textContent = "Recording complete. Analyzing it locally…";
    fileInput.dispatchEvent(new Event("change", { bubbles: true }));
  } catch (error) {
    showCaptureError(`Could not pass the recording to the audio inspector: ${captureErrorMessage(error)}`);
  } finally {
    syncRecordAvailability();
  }
}

function selectRecordingMimeType() {
  if (typeof MediaRecorder.isTypeSupported !== "function") return null;
  return preferredMimeTypes.find((type) => MediaRecorder.isTypeSupported(type)) ?? null;
}

function recordingFileName(mimeType) {
  const extension = mimeType.includes("mp4")
    ? "m4a"
    : mimeType.includes("ogg")
      ? "ogg"
      : mimeType.includes("webm")
        ? "webm"
        : "audio";
  const timestamp = new Date().toISOString().replaceAll(":", "-");
  return `voice-recording-${timestamp}.${extension}`;
}

function setCompetingInputsDisabled(disabled) {
  fileInput.disabled = disabled;
  chooseFile.disabled = disabled;
  for (const button of exampleButtons) button.disabled = disabled;
}

function syncRecordAvailability() {
  if (!recordButton || mediaRecorder?.state === "recording" || finalizing) return;
  recordButton.disabled = inputPanel?.getAttribute("aria-busy") === "true";
}

function stopMediaStream() {
  for (const track of mediaStream?.getTracks?.() ?? []) track.stop();
  mediaStream = null;
}

function clearCaptureError() {
  if (!inputError) return;
  inputError.hidden = true;
  inputError.textContent = "";
}

function showCaptureError(message) {
  if (!inputError) return;
  inputError.textContent = message;
  inputError.hidden = false;
  recordingStatus.textContent = "Microphone capture stopped. Your existing file and example options are still available.";
}

function captureErrorMessage(error) {
  if (error?.name === "NotAllowedError") {
    return "Microphone access was denied. Allow microphone access for this site, or choose an audio file instead.";
  }
  if (error?.name === "NotFoundError") {
    return "No microphone was found. Connect a microphone, or choose an audio file instead.";
  }
  return error instanceof Error ? error.message : String(error);
}
