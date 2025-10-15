#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, json, time, subprocess, importlib.util, threading, queue
from typing import List, Dict, Any, Optional

import streamlit as st
import requests

st.set_page_config(page_title="Jarvis — Streamlit UI (style shadcn) v3", layout="wide")

# -------------- Persistence --------------
CONFIG_DIR = os.path.expanduser("~/.jarvis")
CONFIG_PATH = os.path.join(CONFIG_DIR, "ui_config.json")

DEFAULT_CFG: Dict[str, Any] = {
    "whisper": {"model": "small", "lang": "fr", "vad": True, "device": "cpu", "compute": "int8"},
    "piper": {"base_dir": os.path.expanduser("~/.jarvis/voices"), "voice": "", "speaker_id": 0, "speed": 1.0, "sr": 22050},
    "ollama": {"host": "http://127.0.0.1:11434", "model": "qwen2.5:latest", "temperature": 0.7, "num_ctx": 4096, "stream": False},
    "mcp": {"servers": ["npx -y @modelcontextprotocol/server-filesystem"]},
    "jarvis": {"path": "./jarvis.py", "audio_out": "analog"}  # + ajout: chemin jarvis + préférence sortie audio
}

def load_cfg() -> Dict[str, Any]:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # shallow-merge with defaults
                cfg = DEFAULT_CFG.copy()
                for k, v in (data or {}).items():
                    if isinstance(v, dict):
                        cfg[k] = {**cfg.get(k, {}), **v}
                    else:
                        cfg[k] = v
                return cfg
    except Exception as e:
        st.warning(f"Lecture config échouée: {e}")
    return DEFAULT_CFG.copy()

def save_cfg(cfg: Dict[str, Any]):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        st.success(f"Configuration enregistrée : {CONFIG_PATH}")
    except Exception as e:
        st.error(f"Échec sauvegarde config: {e}")

CFG = load_cfg()

# -------------- Helpers --------------
def fetch_ollama_models(host: str) -> List[str]:
    """List models via /api/tags; return [] on failure."""
    try:
        url = host.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json() or {}
        models = []
        for m in data.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                models.append(name)
        return sorted(set(models))
    except Exception:
        return []

def is_ollama_up(host: str) -> bool:
    try:
        url = host.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=3)
        return r.ok
    except Exception:
        return False

def start_ollama_daemon(log_path: Optional[str] = None) -> bool:
    """
    Try to start 'ollama serve' detached. Returns True if Popen succeeded.
    The server may need a few seconds to become ready.
    """
    try:
        kwargs = {}
        if log_path:
            out = open(log_path, "ab", buffering=0)
            kwargs["stdout"] = out
            kwargs["stderr"] = subprocess.STDOUT
        # Start in a new process group to avoid blocking
        subprocess.Popen(["ollama", "serve"], start_new_session=True, **kwargs)
        return True
    except Exception as e:
        st.error(f"Impossible de lancer 'ollama serve': {e}")
        return False

def wait_ollama_ready(host: str, seconds: int = 10) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if is_ollama_up(host):
            return True
        time.sleep(0.6)
    return False

def pull_model(model: str) -> bool:
    try:
        proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, check=True)
        st.text(proc.stdout or "pull: ok")
        if proc.stderr:
            st.caption(proc.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Échec 'ollama pull {model}': {e.stderr or e}")
        return False

def warmup_model(host: str, model: str) -> bool:
    """Trigger a tiny /api/generate to load weights into memory."""
    try:
        url = host.rstrip("/") + "/api/generate"
        payload = {"model": model, "prompt": "ok", "stream": False}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        _ = r.json()
        return True
    except Exception as e:
        st.error(f"Warmup échoué: {e}")
        return False

def save_uploaded(file, dest_dir: str) -> Optional[str]:
    try:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        return path
    except Exception as e:
        st.error(f"Enregistrement échoué: {e}")
        return None

# -------------- Jarvis runtime (local) --------------
if "jarvis_mod" not in st.session_state:
    st.session_state.jarvis_mod = None
if "jarvis_thread" not in st.session_state:
    st.session_state.jarvis_thread = None
if "jarvis_running" not in st.session_state:
    st.session_state.jarvis_running = False
if "jarvis_logbuf" not in st.session_state:
    st.session_state.jarvis_logbuf = []

def _load_jarvis_module(path: str):
    spec = importlib.util.spec_from_file_location("jarvis", path)
    if not spec or not spec.loader:
        raise RuntimeError("Impossible de charger jarvis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # charge ton jarvis.py
    return mod

def start_jarvis(path: str, audio_out_pref: str = "analog"):
    if st.session_state.jarvis_running:
        st.warning("Jarvis tourne déjà.")
        return
    if not os.path.exists(path):
        st.error(f"jarvis.py introuvable: {path}")
        return
    try:
        jarvis = _load_jarvis_module(path)
        st.session_state.jarvis_mod = jarvis
        # passer la préférence de sortie audio à jarvis via env
        os.environ["AUDIO_OUT"] = audio_out_pref
        def _run():
            try:
                jarvis.main()  # lance la boucle wake+STT+TTS locale
            except Exception as e:
                try:
                    jarvis.q_log.put_nowait(f"[UI] Exception: {e}")
                except Exception:
                    pass
            finally:
                st.session_state.jarvis_running = False
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        st.session_state.jarvis_thread = t
        st.session_state.jarvis_running = True
        st.success("Jarvis démarré (audio local).")
    except Exception as e:
        st.error(f"Erreur au démarrage de Jarvis: {e}")

def stop_jarvis():
    if not st.session_state.jarvis_running or not st.session_state.jarvis_mod:
        st.info("Jarvis n'est pas en cours.")
        return
    try:
        st.session_state.jarvis_mod.stop_main.set()
        st.info("Arrêt demandé.")
    except Exception as e:
        st.error(f"Stop error: {e}")

def drain_jarvis_logs(max_keep: int = 800):
    jm = st.session_state.jarvis_mod
    if not jm or not hasattr(jm, "q_log"):
        return
    q = jm.q_log
    pulled = []
    try:
        for _ in range(500):
            try:
                pulled.append(q.get_nowait())
            except queue.Empty:
                break
    except Exception:
        pass
    if pulled:
        st.session_state.jarvis_logbuf += pulled
        st.session_state.jarvis_logbuf = st.session_state.jarvis_logbuf[-max_keep:]

# -------------- CSS --------------
st.markdown("""
<style>
:root{
  --bg: #0b0f14; --panel: #0e141b; --muted: #121923; --border: #1b2633;
  --primary: #44f1ff; --primary-2: #00c2d1; --fg: #e6f1ff; --muted-fg:#7b8ba1; --accent:#101926;
}
.main .block-container{ padding-top: 0rem; }
body{
  background: radial-gradient(900px 450px at 50% 28%, rgba(0, 194, 209, 0.08), transparent 55%),
              radial-gradient(700px 400px at 70% 20%, rgba(68, 241, 255, 0.05), transparent 60%),
              linear-gradient(180deg, #0b0f14 0%, #0a0e13 100%);
  color: var(--fg);
}
body::before{
  content:""; position:fixed; inset:0;
  background:
     linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px) 0 0/ 60px 60px,
     linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px) 0 0/ 60px 60px;
  mask-image: radial-gradient(circle at 50% 28%, black 0%, transparent 70%);
  pointer-events:none;
}
.topbar{ position: sticky; top: 0; z-index: 2; backdrop-filter: blur(6px);
  background: linear-gradient(180deg, rgba(11,15,20,0.85) 0%, rgba(11,15,20,0.35) 100%);
  border-bottom: 1px solid var(--border); padding: 10px 14px; margin-bottom: 8px; }
.status-row{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
.status-left{ display:flex; gap:16px; align-items:center; }
.pill{ display:flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
  background: linear-gradient(180deg, var(--panel), var(--muted)); border: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px; color: var(--fg); }
.dot{ width:8px; height:8px; border-radius:999px; background: var(--primary);
  box-shadow: 0 0 12px rgba(68,241,255,0.9), 0 0 30px rgba(68,241,255,0.35); animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse{0%{transform:scale(.9);opacity:.8}50%{transform:scale(1.2);opacity:1}100%{transform:scale(.9);opacity:.8}}
.card{ background: linear-gradient(180deg, var(--panel), var(--muted)); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px; }
.card h3{ margin: 0 0 8px 0; color: var(--fg); }
.muted{ color: var(--muted-fg); }

.visualizer{ position:relative; width:100%; max-width: 420px; height: 320px;
  display:flex; align-items:center; justify-content:center; overflow:hidden; border-radius:12px;
  background: radial-gradient(600px 600px at 50% 50%, rgba(68,241,255,0.05), transparent 70%);
  border: 1px dashed rgba(255,255,255,0.08); margin: 0 auto; }
.ring{ position:absolute; border:1px solid rgba(68,241,255,0.25); border-radius:999px; animation: ringPulse 2.4s ease-in-out infinite; }
.r1{ width:40%; height:40%; animation-delay: 0s; }
.r2{ width:62%; height:62%; animation-delay: .18s; }
.r3{ width:84%; height:84%; animation-delay: .36s; }
@keyframes ringPulse{ 0%{ box-shadow: 0 0 0 0 rgba(68,241,255,0.25); }
  50%{ box-shadow: 0 0 16px 0 rgba(68,241,255,0.50), inset 0 0 18px rgba(68,241,255,0.08); }
  100%{ box-shadow: 0 0 0 0 rgba(68,241,255,0.25); } }

.chat-wrap{ display:flex; flex-direction:column; height:520px; }
.msgs{ flex:1; overflow:auto; display:flex; flex-direction:column; gap:10px; padding-right:6px; }
.bubble{ max-width: 92%; padding:10px 12px; border-radius: 12px; border: 1px solid var(--border); }
.user{ align-self:flex-end; background: rgba(68,241,255,0.08); }
.assistant{ align-self:flex-start; background: rgba(255,255,255,0.04); }
.input-row{ display:flex; gap:8px; align-items:center; margin-top:8px; }
.input-row input{ flex:1; background: var(--accent); color: var(--fg); border:1px solid var(--border); border-radius: 10px; padding:10px 12px; }
.input-row button{ background: linear-gradient(180deg, #0b485b, #083947);
  border:1px solid var(--primary-2); color: white; border-radius: 10px; padding:10px 14px; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# -------------- Top bar --------------
st.markdown("""
<div class="topbar">
  <div class="status-row">
    <div class="status-left">
      <div class="pill"><div class="dot"></div><span>WHISPER</span><span class="muted">• Ready</span></div>
      <div class="pill"><div class="dot"></div><span>PIPER</span><span class="muted">• Ready</span></div>
      <div class="pill"><div class="dot"></div><span>OLLAMA</span><span class="muted">• {status}</span></div>
      <div class="pill"><div class="dot"></div><span>MCP</span><span class="muted">• cfg</span></div>
    </div>
    <div></div>
  </div>
</div>
""".format(status=("Up" if is_ollama_up(CFG["ollama"]["host"]) else "Down")), unsafe_allow_html=True)

# -------------- Tabs --------------
tab_interface, tab_settings = st.tabs(["🎛️ Interface", "⚙️ Settings"])

with tab_interface:
    c1, c2 = st.columns([5,7])
    with c1:
        st.markdown('<div class="card"><h3>Radar Vocal</h3>', unsafe_allow_html=True)
        st.markdown('<div class="visualizer"><div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div></div>', unsafe_allow_html=True)
        st.markdown('<p class="muted" style="margin-top:8px;">Affichage compact. (Le backend micro/FFT est côté jarvis.py)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>Chat</h3><div class="chat-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="msgs">', unsafe_allow_html=True)
        msgs = st.session_state.setdefault("messages", [])
        for m in msgs:
            role = m.get("role","assistant")
            content = m.get("content","")
            cls = "bubble assistant" if role!="user" else "bubble user"
            st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        cols = st.columns([1,10,2])
        with cols[1]:
            # FIX: label non vide + masqué => supprime le warning Streamlit
            user_text = st.text_input("Message", key="chat_input",
                                      placeholder="MODE CHAT ACTIVÉ - Commence la conversation",
                                      label_visibility="collapsed")
        with cols[2]:
            if st.button("Envoyer", use_container_width=True):
                if user_text.strip():
                    msgs.append({"role":"user","content": user_text.strip()})
                    msgs.append({"role":"assistant","content": "Réponse simulée (brancher Ollama)."})
                    st.session_state["messages"] = msgs
                    st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

with tab_settings:
    # Ajout d'un sous-onglet Jarvis pour audio local
    t0, t1, t2, t3, t4, t5 = st.tabs(["Jarvis", "Whisper", "Piper", "Ollama", "MCP", "💾 Sauvegarde"])

    with t0:
        st.write("**Jarvis (audio local)**")
        CFG["jarvis"]["path"] = st.text_input("Chemin jarvis.py", value=CFG["jarvis"]["path"])
        CFG["jarvis"]["audio_out"] = st.text_input("Préférence sortie audio (AUDIO_OUT)", value=CFG["jarvis"]["audio_out"])

        colJ1, colJ2, colJ3 = st.columns(3)
        with colJ1:
            if st.button("▶️ Démarrer Jarvis"):
                start_jarvis(CFG["jarvis"]["path"], CFG["jarvis"]["audio_out"])
        with colJ2:
            if st.button("⏹️ Arrêter Jarvis"):
                stop_jarvis()
        with colJ3:
            if st.button("♻️ Recharger module"):
                # forcer rechargement au prochain start
                st.session_state.jarvis_mod = None
                st.success("Module Jarvis sera rechargé au prochain démarrage.")

        # Logs live
        if st.session_state.jarvis_running:
            drain_jarvis_logs()
        st.code("\n".join(st.session_state.get("jarvis_logbuf", [])) or "—", language="bash")

        # Astuce devices
        st.caption("Astuce: si l’audio sort au mauvais endroit, mets AUDIO_OUT sur une sous-chaîne du nom du périphérique (ex: 'analog', 'hdmi').")

    with t1:
        CFG["whisper"]["model"] = st.selectbox(
            "Modèle Fast Whisper", ["tiny","base","small","medium","large-v3"],
            index=["tiny","base","small","medium","large-v3"].index(CFG["whisper"]["model"])
        )
        CFG["whisper"]["lang"]  = st.selectbox("Langue", ["fr","en","auto"], index=["fr","en","auto"].index(CFG["whisper"]["lang"]))
        CFG["whisper"]["vad"]   = st.checkbox("VAD activé", value=CFG["whisper"]["vad"])
        CFG["whisper"]["device"]= st.selectbox("Device", ["cpu","cuda"], index=0 if CFG["whisper"]["device"]=="cpu" else 1)
        CFG["whisper"]["compute"]= st.selectbox(
            "Compute", ["int8","int8_float16","float16","float32"],
            index=["int8","int8_float16","float16","float32"].index(CFG["whisper"]["compute"])
        )

    with t2:
        st.write("**Sélection de la voix Piper**")
        CFG["piper"]["base_dir"] = st.text_input("Dossier des voix (local serveur)", value=CFG["piper"]["base_dir"])
        # Uploaders
        up1, up2 = st.columns(2)
        with up1:
            up_onnx = st.file_uploader("Voice ONNX", type=["onnx"], accept_multiple_files=False, key="up_onnx")
            if up_onnx:
                saved = save_uploaded(up_onnx, CFG["piper"]["base_dir"])
                if saved:
                    st.success(f"Enregistré : {saved}")
        with up2:
            up_json = st.file_uploader("Voice JSON (optional)", type=["json"], accept_multiple_files=False, key="up_json")
            if up_json:
                saved = save_uploaded(up_json, CFG["piper"]["base_dir"])
                if saved:
                    st.success(f"Enregistré : {saved}")
        # Directory listing
        voice_files = []
        try:
            if os.path.isdir(CFG["piper"]["base_dir"]):
                voice_files = [f for f in os.listdir(CFG["piper"]["base_dir"]) if f.endswith(".onnx")]
        except Exception as e:
            st.warning(f"Dossier inaccessible: {e}")
        sel_list = [""] + voice_files
        idx_sel = sel_list.index(CFG["piper"].get("voice","")) if CFG["piper"].get("voice","") in sel_list else 0
        sel = st.selectbox("Choisir une voix", sel_list, index=idx_sel)
        CFG["piper"]["voice"] = sel
        CFG["piper"]["speaker_id"] = st.number_input("Speaker ID", 0, 999, int(CFG["piper"]["speaker_id"]), 1)
        CFG["piper"]["speed"] = st.number_input("Vitesse (length scale)", 0.5, 2.0, float(CFG["piper"]["speed"]), 0.05)
        CFG["piper"]["sr"] = st.number_input("Sample rate", 8000, 48000, int(CFG["piper"]["sr"]), 1000)

    with t3:
        st.write("**Ollama**")
        CFG["ollama"]["host"] = st.text_input("Host", value=CFG["ollama"]["host"])
        # List models from /api/tags
        models = fetch_ollama_models(CFG["ollama"]["host"])
        if models:
            idx = models.index(CFG["ollama"]["model"]) if CFG["ollama"]["model"] in models else 0
            CFG["ollama"]["model"] = st.selectbox("Modèle local installé", models, index=idx)
        else:
            CFG["ollama"]["model"] = st.text_input("Modèle (texte libre)", value=CFG["ollama"]["model"])
            st.warning("Aucun modèle listé — Ollama indisponible ? (ou vide)")

        CFG["ollama"]["temperature"] = st.slider("Température", 0.0, 1.5, float(CFG["ollama"]["temperature"]), 0.05)
        CFG["ollama"]["num_ctx"] = st.number_input("Contexte (num_ctx)", 512, 16384, int(CFG["ollama"]["num_ctx"]), 512)
        CFG["ollama"]["stream"] = st.toggle("Streaming tokens", value=bool(CFG["ollama"]["stream"]))

        st.markdown("---")
        colA, colB, colC, colD = st.columns(4)
        with colA:
            if st.button("Tester connexion"):
                st.info("Connexion à /api/tags…")
                ok = is_ollama_up(CFG["ollama"]["host"])
                st.success("OK") if ok else st.error("KO")
        with colB:
            if st.button("Démarrer Ollama (serve)"):
                log_path = os.path.join(CONFIG_DIR, "ollama_serve.log")
                ok = start_ollama_daemon(log_path=log_path)
                if ok:
                    st.info("Démarré. Attente de disponibilité…")
                    if wait_ollama_ready(CFG["ollama"]["host"], seconds=12):
                        st.success("Ollama prêt.")
                    else:
                        st.warning("Toujours indisponible. Vérifie les logs: " + log_path)
        with colC:
            if st.button("Pull modèle sélectionné"):
                model = CFG["ollama"]["model"]
                if not model:
                    st.error("Aucun modèle sélectionné.")
                else:
                    pull_model(model)
        with colD:
            if st.button("Warm-up modèle"):
                model = CFG["ollama"]["model"]
                if not model:
                    st.error("Aucun modèle sélectionné.")
                else:
                    ok = warmup_model(CFG["ollama"]["host"], model)
                    st.success("Chargé") if ok else st.error("Échec warm-up")

    with t4:
        servers_str = "\n".join(CFG["mcp"].get("servers", []))
        new_servers = st.text_area("Serveurs MCP", value=servers_str, height=120)
        CFG["mcp"]["servers"] = [s.strip() for s in new_servers.splitlines() if s.strip()]

    with t5:
        csa, csb = st.columns(2)
        with csa:
            if st.button("💾 Enregistrer la configuration"):
                save_cfg(CFG)
        with csb:
            if st.button("↩️ Recharger depuis disque"):
                st.cache_data.clear()
                st.rerun()

st.caption(f"Config: {CONFIG_PATH}")
