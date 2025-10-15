#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dépendances:
#   pip install -U ollama rapidfuzz
#   # TTS (optionnel): pip install -U chatterbox-tts torchaudio
#   # Piper (recommandé pour vitesse): pip install -U piper-tts piper-phonemize onnxruntime
#   # (NVIDIA optionnel pour Piper):  pip install -U onnxruntime-gpu
#   # Fallback TTS (Linux): sudo apt-get install -y espeak-ng || sudo apt-get install -y espeak

import os, time, threading, queue, tempfile, sys, warnings, logging, shutil, subprocess, re, json, pathlib

# --- Forcer CPU pour éviter OOM / chargements CUDA accidentels ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
PATCH_TORCH_LOAD_CPU = True

import numpy as np
import sounddevice as sd
import soundfile as sf

# --- Couper le bruit de fond des bibliothèques ---
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning, module=r"perth\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"diffusers\..*")
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"transformers\.generation\.configuration_utils")
warnings.filterwarnings("ignore", message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated")
warnings.filterwarnings("ignore", message=r"We detected that you are passing `past_key_values`")

# -- MUTE Chatterbox mel/tokens warning ---------------------------------------
MEL_WARN_SUBSTR = "Reference mel length is not equal to 2 * reference token length"
class _SilenceMelWarn(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return MEL_WARN_SUBSTR not in record.getMessage()
        except Exception:
            return True
logging.getLogger().addFilter(_SilenceMelWarn())
# ------------------------------------------------------------------------------

from faster_whisper import WhisperModel
from ollama import Client as OllamaClient

# Rabaisse le niveau de log
for name in ["chatterbox", "diffusers", "transformers"]:
    logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger("chatterbox.models.t3.inference.alignment_stream_analyzer").setLevel(logging.ERROR)

# ===================== CONFIG =====================
WAKE_WORD = "jarvis"
WAKE_ALIASES = (
    "jervis", "je revis", "je revisse", "jar vise", "j'en revise",
    "jarvy", "jarviz", "jarviss", "jarvice",
    "ok jarvis", "dis jarvis", "hey jarvis",
)
WAKE_WINDOW_S = 1.8

FW_MODEL   = os.getenv("FW_MODEL", "small")
FW_DEVICE  = os.getenv("FW_DEVICE", "cpu")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")

# STT
FW_PROMPT     = os.getenv("FW_PROMPT", "Transcris strictement en français (fr). Noms propres: Jarvis, capitale, Paris, France, météo, heure, système.")
FW_BEAM_WAKE  = int(os.getenv("FW_BEAM_WAKE", "1"))
FW_BEAM_CMD   = int(os.getenv("FW_BEAM_CMD",  "1"))
FW_VAD_CMD    = int(os.getenv("FW_VAD_CMD",   "0"))   # 0 = OFF (on coupe au silence micro)

# Capture audio à faible latence (fin de phrase)
REC_MAX_S       = float(os.getenv("REC_MAX_S", "6.0"))
REC_CHUNK_S     = float(os.getenv("REC_CHUNK_S", "0.2"))
REC_END_SIL_S   = float(os.getenv("REC_END_SIL_S", "0.6"))

# Ollama
LLM_ID       = os.getenv("LLM_ID", "jarvis-hermes2pro-fr")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = os.getenv("OLLAMA_TIMEOUT", "120")  # "120" ou "none"
_ollama_timeout = None if OLLAMA_TIMEOUT.strip().lower() in ("none", "null") else float(OLLAMA_TIMEOUT)

# Wake config
REQUIRE_WAKE     = int(os.getenv("REQUIRE_WAKE", "1"))
WAKE_FUZZY       = int(os.getenv("WAKE_FUZZY", "1"))
WAKE_FUZZY_SCORE = int(os.getenv("WAKE_FUZZY_SCORE", "80"))

# ===================== TTS (config) =====================
# Sélecteur de moteur: "chatterbox" (par défaut) ou "piper"
TTS_ENGINE = os.getenv("TTS_ENGINE", "chatterbox").lower()

# Chatterbox (réglages + voix de référence)
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
TTS_LANG   = os.getenv("TTS_LANG", "fr")
DEFAULT_TTS_CFG_WEIGHT   = float(os.getenv("TTS_CFG_WEIGHT", "0.3"))
DEFAULT_TTS_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.4"))
DEFAULT_TTS_REF_WAV = os.getenv("TTS_REF_WAV", "/home/fungraphic/Jarvis_2.0/voix/jarvis_fr_film_mono24k.wav")
TTS_CLONE_ONCE   = int(os.getenv("TTS_CLONE_ONCE", "1"))
TTS_REF_TRIM_S   = float(os.getenv("TTS_REF_TRIM_S", "3.5"))
TTS_REF_PAD_S    = float(os.getenv("TTS_REF_PAD_S",  "0.12"))
TTS_REF_FRAME_S  = float(os.getenv("TTS_REF_FRAME_S","0.02"))

# Piper (chemins & sliders)
PIPER_VOICE     = os.getenv("PIPER_VOICE", "")  # ex: /home/.../fr_FR-lessac-medium.onnx  (et fr_FR-lessac-medium.onnx.json à côté)
PIPER_LENGTH    = float(os.getenv("PIPER_LENGTH", "0.9"))   # <1.0 = plus court/rapide
PIPER_NOISE     = float(os.getenv("PIPER_NOISE",  "0.667"))
PIPER_NOISE_W   = float(os.getenv("PIPER_NOISE_W","0.8"))
PIPER_SENT_SIL  = float(os.getenv("PIPER_SENT_SIL","0.10"))
PIPER_CUDA      = int(os.getenv("PIPER_CUDA", "0"))

# Audio out préf (éviter HDMI)
AUDIO_OUT_PREF = os.getenv("AUDIO_OUT", "analog")

# ===================== ÉTAT GLOBAL =====================
busy = threading.Event()
stop_main = threading.Event()
q_log = queue.Queue()

def log(msg: str):
    print(msg, flush=True)
    try: q_log.put_nowait(msg)
    except Exception: pass

# ===================== AUDIO (I/O + samplerate) =====================
def pick_devices_and_rates(prefer_output_name_substr: str | None = None):
    devices = sd.query_devices()
    in_idx, out_idx = None, None

    # Input
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            in_idx = i; break

    # Output
    if prefer_output_name_substr:
        pref = prefer_output_name_substr.lower()
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and pref in d["name"].lower():
                out_idx = i; break
    if out_idx is None:
        for key in ("analog", "speaker", "alc", "built-in", "usb"):
            for i, d in enumerate(devices):
                if d.get("max_output_channels", 0) > 0 and key in d["name"].lower():
                    out_idx = i; break
            if out_idx is not None:
                break
    if out_idx is None:
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0:
                out_idx = i; break

    if in_idx is None or out_idx is None:
        raise RuntimeError("Aucun device audio valide (entrée ou sortie).")

    sd.default.device = (in_idx, out_idx)

    in_dev  = sd.query_devices(in_idx,  kind='input')
    out_dev = sd.query_devices(out_idx, kind='output')
    sr_in  = int(in_dev.get('default_samplerate') or 48000)
    sr_out = int(out_dev.get('default_samplerate') or 48000)

    sd.default.samplerate = sr_in
    sd.default.channels   = 1
    # Latence basse I/O
    sd.default.latency = ("low", "low")
    sd.default.blocksize = 0  # 0 = min auto

    print(f"[AUDIO] Input  : {in_idx} '{in_dev['name']}' @ {sr_in} Hz")
    print(f"[AUDIO] Output : {out_idx} '{out_dev['name']}' @ {sr_out} Hz")
    print(f"[AUDIO] Using input samplerate: {sr_in} Hz")
    return in_idx, out_idx, sr_in, sr_out

def record_until_silence(max_s=REC_MAX_S, chunk_s=REC_CHUNK_S, end_silence_s=REC_END_SIL_S) -> np.ndarray:
    """Enregistre par petits blocs jusqu'à détection de fin de phrase (silence prolongé)."""
    sr = int(sd.default.samplerate)
    buf = []
    speech_seen = False
    silent_tail = 0.0
    t = 0.0
    while t < max_s and not stop_main.is_set():
        n = int(chunk_s * sr)
        chunk = sd.rec(n, samplerate=sr, channels=1, dtype="int16", blocking=True)
        sd.wait()
        x = chunk.reshape(-1)
        buf.append(x)
        if not is_silence(x):
            speech_seen = True
            silent_tail = 0.0
        else:
            silent_tail += chunk_s
            if speech_seen and silent_tail >= end_silence_s:
                break
        t += chunk_s
    return np.concatenate(buf) if buf else np.zeros(0, dtype=np.int16)

def record_seconds(seconds: float) -> np.ndarray:
    # conservé pour le wake (fenêtre courte)
    sr = int(sd.default.samplerate)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   dtype="int16", blocking=True)
    sd.wait()
    return audio.reshape(-1)

def is_silence(audio_i16: np.ndarray, abs_thr: int = 120, rms_thr: float = 60.0) -> bool:
    if audio_i16.size == 0:
        return True
    a = audio_i16.astype(np.float32)
    if np.max(np.abs(a)) < abs_thr:
        return True
    rms = np.sqrt(np.mean(a * a))
    return rms < rms_thr

def write_wav_tmp(audio_i16: np.ndarray, boost: float = 2.0) -> str:
    x = (audio_i16.astype(np.float32) / 32768.0) * float(boost)
    x = np.clip(x, -1.0, 1.0)
    path = tempfile.mktemp(prefix="jarvis_", suffix=".wav")
    sf.write(path, x, int(sd.default.samplerate))
    return path

# ===================== STT (Faster-Whisper) =====================
def init_faster_whisper():
    log(f"[STT] Chargement Faster-Whisper: model={FW_MODEL} device={FW_DEVICE} compute={FW_COMPUTE}")
    return WhisperModel(FW_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE)

def transcribe_path(
    model: WhisperModel,
    wav_path: str,
    language: str | None,
    beam_size: int = 1,
    initial_prompt: str | None = None,
    use_vad: bool = True,
) -> str:
    try:
        segments, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=use_vad,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=250),
        )
        return "".join([s.text for s in segments]).strip()
    except ValueError as e:
        if "max() arg is an empty sequence" in str(e):
            return ""
        raise

# ===================== TTS =====================
def _resample_linear(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or y.size == 0:
        return y
    n_to = int(round(y.size * sr_to / sr_from))
    if n_to <= 0:
        return y
    xi = np.linspace(0.0, 1.0, num=n_to, endpoint=False, dtype=np.float64)
    x  = np.linspace(0.0, 1.0, num=y.size, endpoint=False, dtype=np.float64)
    return np.interp(xi, x, y.astype(np.float64)).astype(np.float32)

def _say_fallback(text: str, lang: str) -> bool:
    """Essaye espeak-ng / espeak / say (macOS). Retourne True si OK."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        try:
            args = [exe, "-s", "150"]
            if lang:
                args += ["-v", lang]
            args += [text]
            subprocess.run(args, check=False)
            return True
        except Exception:
            pass
    exe = shutil.which("say")  # macOS
    if exe:
        try:
            subprocess.run([exe, text], check=False)
            return True
        except Exception:
            pass
    return False

def _best_window_trim(y: np.ndarray, sr: int, window_s: float, frame_s: float, pad_s: float):
    """Prend la fenêtre de durée window_s avec énergie moyenne max. Ajoute pad de chaque côté."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if y.size == 0:
        return y
    frame = max(1, int(frame_s * sr))
    win = max(1, int(window_s / frame_s))
    n_frames = len(y) // frame
    if n_frames == 0:
        return y
    e = np.abs(y[:n_frames*frame]).reshape(n_frames, frame).mean(axis=1)
    if len(e) <= win:
        start_f = 0
    else:
        c = np.convolve(e, np.ones(win, dtype=np.float32), mode="valid")
        start_f = int(np.argmax(c))
    start = max(0, start_f * frame - int(pad_s * sr))
    end = min(len(y), (start_f + win) * frame + int(pad_s * sr))
    return y[start:end]

class TTS:
    def __init__(self, lang_id: str = "fr"):
        self.model = None
        self.lang_id = lang_id
        self.ref_wav_src = DEFAULT_TTS_REF_WAV
        self.ref_wav = None  # chemin vers ref trim/cache
        self.cfg_weight = DEFAULT_TTS_CFG_WEIGHT
        self.exaggeration = DEFAULT_TTS_EXAGGERATION

        # Patch torch.load → CPU
        if PATCH_TORCH_LOAD_CPU:
            try:
                import torch
                _orig = torch.load
                def _patched_load(f, map_location=None, **kw):
                    if map_location is None:
                        map_location = "cpu"
                    return _orig(f, map_location=map_location, **kw)
                torch.load = _patched_load
                log("[TTS] torch.load patché (map_location='cpu').")
            except Exception as e:
                log(f"[TTS] Patch torch.load impossible: {e}")

        # Trim/cache référence si fournie
        if self.ref_wav_src and os.path.exists(self.ref_wav_src):
            try:
                y, sr = sf.read(self.ref_wav_src, always_2d=False)
                y = _best_window_trim(y, sr, TTS_REF_TRIM_S, TTS_REF_FRAME_S, TTS_REF_PAD_S)
                peak = np.max(np.abs(y)) or 1.0
                y = (y / peak * 0.98).astype(np.float32)
                self.ref_wav = tempfile.mktemp(prefix="jarvis_ref_", suffix=".wav")
                sf.write(self.ref_wav, y, sr)
                log(f"[TTS] Référence trim/cachée: {self.ref_wav}")
            except Exception as e:
                log(f"[TTS] Trim ref échoué ({e}), utilisation brute.")
                self.ref_wav = self.ref_wav_src
        else:
            self.ref_wav = self.ref_wav_src if self.ref_wav_src else None

        # Import “lazy”
        try:
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _MTL
                self.model = _MTL.from_pretrained(device="cpu")  # CPU forcé
                log(f"[TTS] Engine=Chatterbox Multilingual (cpu) [{self.lang_id}].")
            except Exception:
                from chatterbox.tts import ChatterboxTTS as _MONO
                self.model = _MONO.from_pretrained(device="cpu")
                log(f"[TTS] Engine=Chatterbox (mono) (cpu).")
        except Exception as e:
            log(f"[TTS] Échec init Chatterbox (CPU): {e}")
            self.model = None

    def speak(self, text: str):
        if not text:
            return
        if self.model is None:
            if _say_fallback(text, self.lang_id):
                return
            log(f"JARVIS: {text}")
            return
        try:
            log(f"JARVIS: {text}")
            kwargs = {"exaggeration": self.exaggeration, "cfg_weight": self.cfg_weight}
            if self.ref_wav and os.path.exists(self.ref_wav):
                kwargs["audio_prompt_path"] = self.ref_wav
            if hasattr(self.model, "generate") and "Multilingual" in type(self.model).__name__:
                kwargs["language_id"] = self.lang_id
            wav = self.model.generate(text, **kwargs)

            y  = wav.squeeze().detach().cpu().numpy().astype(np.float32)
            sr_model = int(getattr(self.model, "sr", 48000))
            out_dev  = sd.query_devices(sd.default.device[1], kind='output')
            sr_out   = int(out_dev.get('default_samplerate') or 48000)
            if sr_model != sr_out:
                y = _resample_linear(y, sr_model, sr_out)
            sd.play(y, sr_out, blocking=True)
        except Exception as e:
            log(f"[TTS] Erreur lecture (fallback system): {e}")
            if not _say_fallback(text, self.lang_id):
                log(f"JARVIS: {text}")

    @property
    def clone_once(self) -> bool:
        return bool(self.ref_wav) and bool(TTS_CLONE_ONCE)

# ===================== TTS (Piper, streaming rapide) =====================

def _resolve_piper_voice(spec: str | None) -> str | None:
    """Accepte un chemin avec/sans suffixes. Retourne le .onnx dont le .onnx.json compagnon existe."""
    if not spec:
        return None
    p = pathlib.Path(spec).expanduser()
    def ok(onnx_path: pathlib.Path) -> bool:
        return onnx_path.exists() and (onnx_path.with_suffix(onnx_path.suffix + ".json")).exists()
    s = str(p)
    # <voice>.onnx.json fourni
    if s.endswith(".onnx.json"):
        onnx = pathlib.Path(s[:-5])  # retire .json
        return str(onnx) if ok(onnx) else None
    # <voice>.onnx fourni
    if s.endswith(".onnx") and ok(p):
        return str(p)
    # Sans suffixe → tente .onnx
    if ok(p.with_suffix(".onnx")):
        return str(p.with_suffix(".onnx"))
    # Cherche dans dossiers connus
    name = p.name
    search_dirs = [
        p.parent if p.parent.exists() else pathlib.Path.cwd(),
        pathlib.Path.home()/".local/share/piper-voices",
        pathlib.Path("/usr/share/piper/voices"),
        pathlib.Path("/usr/share/piper-voices"),
        pathlib.Path("/opt/piper/voices"),
    ]
    for d in search_dirs:
        try:
            for cand in d.glob(f"{name}*.onnx"):
                if ok(cand):
                    return str(cand)
        except Exception:
            pass
    return None

class PiperTTS:
    """Synthèse vocale ultra-rapide via Piper en streaming (PCM brut vers la carte son)."""
    def __init__(self,
                 voice_path: str,
                 length_scale: float = 0.9,
                 noise_scale: float  = 0.667,
                 noise_w: float      = 0.8,
                 sentence_silence: float | None = 0.10,
                 use_cuda: bool = False):
        self.voice = os.path.abspath(voice_path)
        conf = self.voice + ".json" if not self.voice.endswith(".onnx.json") else self.voice
        if not os.path.exists(conf):
            raise FileNotFoundError(f"Config Piper introuvable: {conf}")
        with open(conf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.sample_rate = int(meta.get("audio", {}).get("sample_rate", 22050))

        self.length_scale = float(length_scale)
        self.noise_scale  = float(noise_scale)
        self.noise_w      = float(noise_w)
        self.sentence_silence = sentence_silence
        self.use_cuda = bool(use_cuda)

        self.exe = shutil.which("piper")
        if not self.exe:
            raise FileNotFoundError("Binaire 'piper' introuvable dans PATH.")

        log(f"[TTS] Engine=Piper | voice={self.voice} | sr={self.sample_rate} | cuda={self.use_cuda}")

    def speak(self, text: str):
        if not text:
            return
        log(f"JARVIS: {text}")
        # --output-raw (-f -) → flux PCM 16-bit mono sur stdout (officiel). Cf. docs/discussions Piper.  # :contentReference[oaicite:2]{index=2}
        args = [
            self.exe, "-m", self.voice,
            "--output-raw", "-f", "-",              # stdout: PCM 16-bit mono
            "--length-scale", str(self.length_scale),
            "--noise-scale",  str(self.noise_scale),
            "--noise-w",      str(self.noise_w),
        ]
        if self.sentence_silence is not None:
            # Contrôle des pauses inter-phrases.  # :contentReference[oaicite:3]{index=3}
            args += ["--sentence-silence", str(self.sentence_silence)]
        if self.use_cuda:
            args += ["--cuda"]

        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                proc.stdin.write(text.encode("utf-8") + b"\n")
                proc.stdin.close()
            except Exception:
                pass

            with sd.RawOutputStream(
                samplerate=self.sample_rate, channels=1, dtype="int16",
                blocksize=0, dither_off=True
            ) as stream:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    stream.write(chunk)
            proc.wait(timeout=5)
        except Exception as e:
            log(f"[TTS] Piper streaming erreur: {e}")
            # Secours : génération WAV puis lecture blocante
            tmp = tempfile.mktemp(prefix="piper_", suffix=".wav")
            try:
                subprocess.run(
                    [self.exe, "-m", self.voice, "-f", tmp,
                     "--length-scale", str(self.length_scale),
                     "--noise-scale",  str(self.noise_scale),
                     "--noise-w",      str(self.noise_w)],
                    input=text.encode("utf-8"), check=False
                )
                y, sr = sf.read(tmp, dtype="float32", always_2d=False)
                sd.play(y, sr, blocking=True)
            finally:
                try: os.remove(tmp)
                except Exception: pass

    @property
    def clone_once(self) -> bool:
        return False  # Piper n'a pas de coût de clonage par phrase → on peut streamer par phrases

# ===================== LLM via OLLAMA =====================
ollama_client = None  # global

def init_llm():
    global ollama_client
    try:
        if _ollama_timeout is None:
            ollama_client = OllamaClient(host=OLLAMA_HOST, timeout=None)
        else:
            ollama_client = OllamaClient(host=OLLAMA_HOST, timeout=_ollama_timeout)
    except TypeError:
        ollama_client = OllamaClient(host=OLLAMA_HOST)
    log(f"[LLM] Connexion Ollama: host={OLLAMA_HOST} model={LLM_ID} (timeout={OLLAMA_TIMEOUT})")
    return None, None

def generate_full(question: str) -> str:
    msgs = [
        {"role": "system", "content": "Tu es JARVIS, assistant direct et concis. Réponds en français."},
        {"role": "user",   "content": question},
    ]
    try:
        resp = ollama_client.chat(model=LLM_ID, messages=msgs, stream=False)
        return (resp.get("message", {}) or {}).get("content", "") or ""
    except Exception as e:
        log(f"[LLM] Erreur Ollama: {e}")
        return ""

def generate_stream(question: str):
    msgs = [
        {"role": "system", "content": "Tu es JARVIS, assistant direct et concis. Réponds en français."},
        {"role": "user",   "content": question},
    ]
    try:
        for chunk in ollama_client.chat(model=LLM_ID, messages=msgs, stream=True):
            yield (chunk.get("message", {}) or {}).get("content", "") or ""
    except Exception as e:
        log(f"[LLM] Erreur Ollama (stream): {e}")
        yield ""

# ===================== LOGIQUE =====================
HAS_RAPIDFUZZ = False
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    pass

_SENTENCE_END = re.compile(r'([\.!?…]+)(\s|$)')

def _split_on_sentence_boundaries(buff: str):
    out = []
    start = 0
    for m in _SENTENCE_END.finditer(buff):
        end = m.end()
        out.append(buff[start:end].strip())
        start = end
    rest = buff[start:]
    return [s for s in out if s], rest

def is_wake(text: str) -> bool:
    t = text.lower().strip()
    if not t: return False
    if WAKE_WORD in t: return True
    if any(a in t for a in WAKE_ALIASES): return True
    if WAKE_FUZZY and HAS_RAPIDFUZZ:
        try:
            score = fuzz.token_set_ratio(t, WAKE_WORD)
            log(f"[DBG] Wake fuzzy score={score:.1f} for '{t}' ~ '{WAKE_WORD}'")
            if score >= WAKE_FUZZY_SCORE: return True
        except Exception:
            pass
    return False

def handle_command(stt_model: WhisperModel, tts, tok, mdl):
    busy.set()
    try:
        tts.speak("Oui Stéphane?")
        audio = record_until_silence()
        if is_silence(audio):
            log("[STT] Silence détecté (commande).")
            tts.speak("Je n'ai pas entendu, répétez s'il vous plaît.")
            return
        wav = write_wav_tmp(audio, boost=2.0)
        user_text = transcribe_path(
            stt_model, wav, language="fr",
            beam_size=FW_BEAM_CMD, initial_prompt=FW_PROMPT, use_vad=bool(FW_VAD_CMD)
        ).lower()
        log(f"[STT] Vous: {user_text}")
        if not user_text:
            tts.speak("Je n'ai pas compris."); return
        if "arrête-toi" in user_text or "éteins-toi" in user_text:
            tts.speak("D'accord, je me mets en veille."); stop_main.set(); return

        if tts.clone_once:
            full = generate_full(user_text).strip()
            if not full:
                tts.speak("Désolé, je n'ai rien reçu du modèle."); return
            tts.speak(full)
        else:
            buffer = ""
            first_spoken = False
            for tok_txt in generate_stream(user_text):
                if not tok_txt:
                    continue
                buffer += tok_txt
                sentences, buffer = _split_on_sentence_boundaries(buffer)
                for s in sentences:
                    if s:
                        tts.speak(s); first_spoken = True
            rest = buffer.strip()
            if rest:
                tts.speak(rest)
            elif not first_spoken:
                tts.speak("Désolé, je n'ai rien reçu du modèle.")
    finally:
        busy.clear()

def wake_loop(stt_model: WhisperModel, tts, tok, mdl):
    try:
        tts.speak("Système en ligne. Je vous écoute.")
    except Exception:
        pass
    log(f"[BG] Écoute du mot-clé '{WAKE_WORD}' (REQUIRE_WAKE={REQUIRE_WAKE}). Ctrl+C pour quitter.")
    while not stop_main.is_set():
        if busy.is_set():
            time.sleep(0.05); continue
        audio = record_seconds(WAKE_WINDOW_S)
        if is_silence(audio):
            time.sleep(0.05); continue
        wav = write_wav_tmp(audio, boost=1.8)
        try:
            text = transcribe_path(
                stt_model, wav, language="fr",
                beam_size=FW_BEAM_WAKE, initial_prompt=FW_PROMPT, use_vad=True
            ).lower()
            if text: log(f"[BG] Entendu: {text}")
            if REQUIRE_WAKE:
                if is_wake(text):
                    log(f"[BG] Wake '{WAKE_WORD}' détecté.")
                    handle_command(stt_model, tts, tok, mdl)
            else:
                handle_command(stt_model, tts, tok, mdl)
        except Exception as e:
            log(f"[BG] STT wake erreur: {e}")
        time.sleep(0.05)

def main():
    pick_devices_and_rates(AUDIO_OUT_PREF)
    stt_model = init_faster_whisper()

    # --- Sélection du moteur TTS ---
    tts_engine = TTS_ENGINE  # déjà lower()
    if tts_engine == "piper":
        voice = _resolve_piper_voice(PIPER_VOICE)
        if not shutil.which("piper"):
            log("[TTS] Piper sélectionné mais binaire 'piper' introuvable → retour Chatterbox. Installe piper et redémarre. (ex: pacman/apt/pipx)")  # info explicite
            tts = TTS(lang_id=TTS_LANG)
        elif not voice:
            log(f"[TTS] Piper sélectionné mais voix introuvable: '{PIPER_VOICE}'. Attendu: <voice>.onnx + <voice>.onnx.json côte-à-côte. → retour Chatterbox.")  # :contentReference[oaicite:4]{index=4}
            tts = TTS(lang_id=TTS_LANG)
        else:
            tts = PiperTTS(
                voice_path=voice,
                length_scale=PIPER_LENGTH,
                noise_scale=PIPER_NOISE,
                noise_w=PIPER_NOISE_W,
                sentence_silence=PIPER_SENT_SIL,
                use_cuda=bool(PIPER_CUDA),
            )
            os.environ["TTS_CLONE_ONCE"] = "0"  # sans effet ici (PiperTTS.clone_once=False), mais clarifie l'intention.
    else:
        log("[TTS] Engine=Chatterbox (par config).")
        tts = TTS(lang_id=TTS_LANG)

    tok, mdl = init_llm()
    try:
        wake_loop(stt_model, tts, tok, mdl)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tts.speak("Au revoir.")
        except Exception:
            pass
        stop_main.set()

if __name__ == "__main__":
    main()
