#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, subprocess, importlib.util, threading, queue, html, re, uuid, shlex, asyncio, copy, math
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import streamlit as st
import requests
import streamlit.components.v1 as components  # <- pour l'iframe HTML du radar

st.set_page_config(page_title="Jarvis — Streamlit UI (style shadcn) v3", layout="wide")

# -------------- Persistence --------------
CONFIG_DIR = os.path.expanduser("~/.jarvis")
CONFIG_PATH = os.path.join(CONFIG_DIR, "ui_config.json")

DEFAULT_CFG: Dict[str, Any] = {
    "whisper": {
        "model": "small",
        "lang": "fr",
        "vad": True,
        "device": "cpu",
        "compute": "int8",
        "prompt": "Transcris strictement en français (fr). Noms propres: Jarvis, capitale, Paris, France, météo, heure, système.",
    },
    "piper": {
        "base_dir": os.path.expanduser("~/.jarvis/voices"),
        "voice": "",
        "speaker_id": 0,
        "speed": 0.9,
        "noise": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.10,
        "use_cuda": False,
    },
    "ollama": {
        "host": "http://127.0.0.1:11434",
        "model": "qwen2.5:latest",
        "temperature": 0.7,
        "num_ctx": 4096,
        "stream": False,
    },
    # --- MCP: configuration simplifiée (via MCPJungle Gateway) ---
    "mcp": {
        "gateway": {
            "enabled": True,
            "base_url": "http://127.0.0.1:8080",  # ou un endpoint de groupe, ex: http://127.0.0.1:8080/v0/groups/<name>/mcp
            "auth_header": ""                      # ex: "Bearer xxxxx"
        }
    },
    "jarvis": {
        "path": "./jarvis.py",
        "audio_out": "analog",
        "tts_engine": "piper",
        "tts_lang": "fr",
        "wake_word": "jarvis",
        "wake_aliases": "",
        "require_wake": True,
        "wake_fuzzy": True,
        "wake_fuzzy_score": 80,
    },
}

MCP_LOG_DIR = os.path.join(CONFIG_DIR, "mcp-logs")
os.makedirs(MCP_LOG_DIR, exist_ok=True)

def _normalize_mcp_gateway(cfg: Dict[str, Any]) -> None:
    """
    Normalise la section MCP pour la nouvelle topologie 'gateway'.
    - Écrase les anciens champs 'servers', 'proxy', 'jungle' s'ils existent.
    """
    mcp = cfg.setdefault("mcp", {})
    gw = mcp.get("gateway") or {}
    mcp["gateway"] = {
        "enabled": bool(gw.get("enabled", True)),
        "base_url": str(gw.get("base_url", "http://127.0.0.1:8080")),
        "auth_header": str(gw.get("auth_header", "")),
    }
    # purge legacy clés si présentes
    for legacy in ("servers", "proxy", "jungle"):
        if legacy in mcp:
            try:
                del mcp[legacy]
            except Exception:
                pass

def load_cfg() -> Dict[str, Any]:
    """Charge la config depuis le fichier ou retourne la config par défaut."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                cfg = copy.deepcopy(DEFAULT_CFG)
                # merge shallow par section
                for k, v in (data or {}).items():
                    if isinstance(v, dict):
                        cfg[k] = {**cfg.get(k, {}), **copy.deepcopy(v)}
                    else:
                        cfg[k] = v
                _normalize_mcp_gateway(cfg)
                return cfg
    except Exception as e:
        st.warning(f"Lecture config échouée: {e}")
    cfg = copy.deepcopy(DEFAULT_CFG)
    _normalize_mcp_gateway(cfg)
    return cfg

def save_cfg(cfg: Dict[str, Any]):
    """Sauvegarde la config (avec validation minimale)."""
    try:
        if not isinstance(cfg, dict):
            st.error("Config invalide: doit être un dictionnaire.")
            return
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        st.success(f"Configuration enregistrée : {CONFIG_PATH}")
    except Exception as e:
        st.error(f"Échec sauvegarde config: {e}")

CFG = load_cfg()

# -------------- Helpers --------------
def _maybe_join_voice(base_dir: str, voice: str) -> str:
    voice = (voice or "").strip()
    if not voice:
        return ""
    if os.path.isabs(voice):
        return os.path.expanduser(voice)
    return os.path.join(os.path.expanduser(base_dir or ""), voice)

def _apply_env_from_cfg(cfg: Dict[str, Any]):
    """Applique les variables d'environnement depuis la config."""
    whisper = cfg.get("whisper", {})
    jarvis_cfg = cfg.get("jarvis", {})
    piper_cfg = cfg.get("piper", {})
    ollama_cfg = cfg.get("ollama", {})
    env_map = {
        "FW_MODEL": whisper.get("model"),
        "FW_DEVICE": whisper.get("device"),
        "FW_COMPUTE": whisper.get("compute"),
        "FW_LANGUAGE": whisper.get("lang"),
        "FW_PROMPT": whisper.get("prompt"),
        "FW_VAD_CMD": "1" if whisper.get("vad") else "0",
        "FW_VAD_WAKE": "1" if whisper.get("vad") else "0",
        "WAKE_WORD": jarvis_cfg.get("wake_word"),
        "WAKE_ALIASES": jarvis_cfg.get("wake_aliases"),
        "REQUIRE_WAKE": "1" if jarvis_cfg.get("require_wake", True) else "0",
        "WAKE_FUZZY": "1" if jarvis_cfg.get("wake_fuzzy", True) else "0",
        "WAKE_FUZZY_SCORE": str(jarvis_cfg.get("wake_fuzzy_score", 80)),
        "TTS_ENGINE": jarvis_cfg.get("tts_engine"),
        "TTS_LANG": jarvis_cfg.get("tts_lang"),
        "AUDIO_OUT": jarvis_cfg.get("audio_out"),
        "PIPER_VOICE": _maybe_join_voice(piper_cfg.get("base_dir", ""), piper_cfg.get("voice", "")),
        "PIPER_LENGTH": str(piper_cfg.get("speed", 0.9)),
        "PIPER_NOISE": str(piper_cfg.get("noise", 0.667)),
        "PIPER_NOISE_W": str(piper_cfg.get("noise_w", 0.8)),
        "PIPER_SENT_SIL": str(piper_cfg.get("sentence_silence", 0.10)),
        "PIPER_CUDA": "1" if piper_cfg.get("use_cuda") else "0",
        "PIPER_SPEAKER_ID": str(piper_cfg.get("speaker_id")) if piper_cfg.get("speaker_id") is not None else None,
        "OLLAMA_HOST": ollama_cfg.get("host"),
        "LLM_ID": ollama_cfg.get("model"),
        "OLLAMA_TEMPERATURE": str(ollama_cfg.get("temperature", 0.7)),
        "OLLAMA_NUM_CTX": str(ollama_cfg.get("num_ctx", 4096)),
        "OLLAMA_STREAM": "1" if ollama_cfg.get("stream") else "0",
        # compat: plus de liste de serveurs locale => passe un tableau vide
        "MCP_SERVERS_JSON": "[]",
    }
    for key, value in env_map.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        value_str = str(value).strip()
        if value_str == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value_str

def fetch_ollama_models(host: str) -> List[str]:
    """Récupère la liste des modèles Ollama disponibles."""
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
    """Vérifie si Ollama est accessible."""
    try:
        url = host.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=3)
        return r.ok
    except Exception:
        return False

def _build_ollama_chat_messages(messages):
    """Construit la liste des messages au format Ollama."""
    mapped = []
    for m in messages[-20:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role not in ("user", "assistant", "system"):
            role = "assistant" if role != "user" else "user"
        mapped.append({"role": role, "content": content})
    if mapped and mapped[0]["role"] == "assistant":
        mapped = [{"role": "system", "content": "You are Jarvis."}] + mapped
    return mapped

def ollama_reply(cfg, messages) -> str:
    """Obtient une réponse d'Ollama."""
    host = cfg["ollama"]["host"]
    model = cfg["ollama"]["model"]
    options = {
        "temperature": float(cfg["ollama"].get("temperature", 0.7)),
        "num_ctx": int(cfg["ollama"].get("num_ctx", 4096)),
    }
    if not is_ollama_up(host):
        return "Ollama est indisponible (API /api/tags KO)."
    try:
        url = host.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": _build_ollama_chat_messages(messages),
            "stream": False,
            "options": options,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json() or {}
        return (data.get("message") or {}).get("content") or ""
    except Exception:
        try:
            last_user = next((m["content"] for m in reversed(messages) if m.get("role")=="user"), "")
            url = host.rstrip("/") + "/api/generate"
            payload = {"model": model, "prompt": last_user, "stream": False, "options": options}
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json() or {}
            return data.get("response") or data.get("text") or ""
        except Exception as e2:
            return f"Erreur Ollama: {e2}"

def start_ollama_daemon(log_path: Optional[str] = None) -> bool:
    """Lance le daemon Ollama."""
    try:
        kwargs = {}
        if log_path:
            out = open(log_path, "ab", buffering=0)
            kwargs["stdout"] = out
            kwargs["stderr"] = subprocess.STDOUT
        subprocess.Popen(["ollama", "serve"], start_new_session=True, **kwargs)
        return True
    except Exception as e:
        st.error(f"Impossible de lancer 'ollama serve': {e}")
        return False

def wait_ollama_ready(host: str, seconds: int = 10) -> bool:
    """Attend que Ollama soit prêt."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        if is_ollama_up(host):
            return True
        time.sleep(0.6)
    return False

def pull_model(model: str) -> bool:
    """Télécharge un modèle Ollama."""
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
    """Charge un modèle en mémoire."""
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
    """Sauvegarde un fichier uploadé."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        return path
    except Exception as e:
        st.error(f"Enregistrement échoué: {e}")
        return None

# -------------- MCP Gateway (HTTP/SSE) --------------
def _mcp_import_ok() -> bool:
    """Vérifie si le SDK MCP est disponible."""
    try:
        import importlib.util
        return importlib.util.find_spec("mcp") is not None
    except Exception:
        return False

def _fmt_call_result(result: Any) -> str:
    """Convertit un CallToolResult en texte lisible."""
    try:
        from mcp import types
    except Exception:
        return json.dumps(getattr(result, "__dict__", str(result)), ensure_ascii=False, indent=2)
    parts: List[str] = []
    sc = getattr(result, "structuredContent", None)
    if sc not in (None, {}):
        try:
            parts.append(json.dumps(sc, ensure_ascii=False, indent=2))
        except Exception:
            parts.append(str(sc))
    for c in getattr(result, "content", []) or []:
        try:
            from mcp import types as _t
        except Exception:
            _t = None
        if _t and isinstance(c, _t.TextContent):
            parts.append(c.text)
        elif _t and isinstance(c, _t.EmbeddedResource):
            res = c.resource
            uri = getattr(res, "uri", "")
            if hasattr(res, "text"):
                parts.append(f"[resource {uri}]\n{res.text}")
            else:
                parts.append(f"[resource {uri}] (binaire)")
        else:
            parts.append(str(c))
    if getattr(result, "isError", False) and not parts:
        parts.append("[MCP] Tool execution failed.")
    out = "\n".join([p for p in parts if p]) or "(vide)"
    return out

def _mcp_gateway_cfg() -> Dict[str, Any]:
    return (CFG.get("mcp", {}).get("gateway") or {})

def mcp_list_tools_via_gateway() -> List[str]:
    """Liste les tools du gateway MCPJungle. Préfère Streamable HTTP (/mcp), fallback SSE (/sse)."""
    gw = _mcp_gateway_cfg()
    base = (gw.get("base_url", "") or "").rstrip("/")
    headers = {}
    if gw.get("auth_header"):
        headers["Authorization"] = gw["auth_header"]

    # Streamable HTTP d'abord
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client  # SDK Python récent
        async def _go():
            async with streamablehttp_client(f"{base}/mcp", headers=headers) as (r, w, _):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    resp = await sess.list_tools()
                    return [t.name for t in (resp.tools or [])]
        return asyncio.run(_go())
    except Exception:
        # Fallback SSE
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        async def _go_sse():
            async with sse_client(f"{base}/sse", headers=headers) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    resp = await sess.list_tools()
                    return [t.name for t in (resp.tools or [])]
        return asyncio.run(_go_sse())

def mcp_call_tool_via_gateway(tool: str, arguments: Dict[str, Any], timeout_s: int = 60) -> str:
    """Appelle un tool via le gateway MCPJungle."""
    gw = _mcp_gateway_cfg()
    base = (gw.get("base_url", "") or "").rstrip("/")
    headers = {}
    if gw.get("auth_header"):
        headers["Authorization"] = gw["auth_header"]

    # Streamable HTTP
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async def _go():
            async with streamablehttp_client(f"{base}/mcp", headers=headers) as (r, w, _):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    try:
                        res = await asyncio.wait_for(sess.call_tool(tool, arguments or {}), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        return "[MCP] Timeout d'exécution du tool."
                    return _fmt_call_result(res)
        return asyncio.run(_go())
    except Exception:
        # Fallback SSE
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        async def _go_sse():
            async with sse_client(f"{base}/sse", headers=headers) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    try:
                        res = await asyncio.wait_for(sess.call_tool(tool, arguments or {}), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        return "[MCP] Timeout d'exécution du tool."
                    return _fmt_call_result(res)
        return asyncio.run(_go_sse())

# -------------- Jarvis runtime (local) --------------
if "jarvis_mod" not in st.session_state:
    st.session_state.jarvis_mod = None
if "jarvis_thread" not in st.session_state:
    st.session_state.jarvis_thread = None
if "jarvis_running" not in st.session_state:
    st.session_state.jarvis_running = False
if "jarvis_logbuf" not in st.session_state:
    st.session_state.jarvis_logbuf = []
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "chat"
if "last_assistant_ts" not in st.session_state:
    st.session_state.last_assistant_ts = 0.0
if "assistant_speaking" not in st.session_state:
    st.session_state.assistant_speaking = False
if "assistant_vu_level" not in st.session_state:
    st.session_state.assistant_vu_level = 0.0
if "assistant_vu_history" not in st.session_state:
    st.session_state.assistant_vu_history = []
if "mode_toggle" not in st.session_state:
    st.session_state.mode_toggle = False
if "auto_refresh_chat" not in st.session_state:
    st.session_state.auto_refresh_chat = False  # désactivé par défaut

def _set_interaction_mode(mode: str):
    """Définit le mode d'interaction (vocal ou chat)."""
    mode = "vocal" if mode == "vocal" else "chat"
    if mode != st.session_state.interaction_mode and mode == "vocal":
        st.session_state["chat_input"] = ""
    st.session_state.interaction_mode = mode
    st.session_state.mode_toggle = mode == "vocal"

def _sync_mode_from_toggle():
    """Synchronise le mode depuis le toggle."""
    desired = "vocal" if st.session_state.get("mode_toggle") else "chat"
    _set_interaction_mode(desired)

_set_interaction_mode(st.session_state.interaction_mode)

def _load_jarvis_module(path: str):
    """Charge le module jarvis.py."""
    spec = importlib.util.spec_from_file_location("jarvis", path)
    if not spec or not spec.loader:
        raise RuntimeError("Impossible de charger jarvis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def start_jarvis(cfg: Dict[str, Any]):
    """Démarre Jarvis en arrière-plan."""
    if st.session_state.jarvis_running:
        st.warning("Jarvis tourne déjà.")
        return
    jarvis_cfg = cfg.get("jarvis", {})
    path = jarvis_cfg.get("path", "./jarvis.py")
    if not os.path.exists(path):
        st.error(f"jarvis.py introuvable: {path}")
        return
    try:
        _apply_env_from_cfg(cfg)
        jarvis = _load_jarvis_module(path)
        st.session_state.jarvis_mod = jarvis
        if hasattr(jarvis, "q_log"):
            try:
                jarvis.q_log.put_nowait("[UI] Paramètres synchronisés depuis Streamlit.")
            except Exception:
                pass
        def _run():
            try:
                jarvis.main()
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
    """Arrête Jarvis."""
    if not st.session_state.jarvis_running or not st.session_state.jarvis_mod:
        st.info("Jarvis n'est pas en cours.")
        return
    try:
        st.session_state.jarvis_mod.stop_main.set()
        st.info("Arrêt demandé.")
    except Exception as e:
        st.error(f"Stop error: {e}")

def drain_jarvis_logs(max_keep: int = 800) -> List[str]:
    """Vide la queue de logs de Jarvis."""
    jm = st.session_state.jarvis_mod
    if not jm or not hasattr(jm, "q_log"):
        return []
    q = jm.q_log
    pulled = []
    try:
        while True:
            try:
                pulled.append(q.get_nowait())
            except queue.Empty:
                break
    except Exception:
        pass
    if pulled:
        displayable = [line for line in pulled if not str(line).startswith("[VU]")]
        if displayable:
            st.session_state.jarvis_logbuf += displayable
            st.session_state.jarvis_logbuf = st.session_state.jarvis_logbuf[-max_keep:]
    return pulled

def _update_speaking_state_from_logs(logs: List[str]):
    """Met à jour l'état parlant depuis les logs."""
    if not logs:
        return
    now = time.time()
    history = list(st.session_state.get("assistant_vu_history", []))
    last_ts = float(st.session_state.get("last_assistant_ts", 0.0))
    speaking = bool(st.session_state.get("assistant_speaking", False))
    level = float(st.session_state.get("assistant_vu_level", 0.0))

    for raw in logs:
        line = (raw or "").strip()
        if not line:
            continue
        if line.startswith("[VU]"):
            try:
                level = max(0.0, min(1.0, float(line[4:]) if len(line) > 4 else 0.0))
            except ValueError:
                continue
            history.append((now, level))
            cutoff = now - 3.0
            history = [(ts, val) for ts, val in history if ts >= cutoff]
            if level > 0.03:
                last_ts = now
                speaking = True
            elif now - last_ts > 0.6:
                speaking = False
            continue
        if line.startswith("JARVIS:"):
            last_ts = now
            speaking = True

    st.session_state.assistant_vu_history = history[-240:]
    st.session_state.assistant_vu_level = level
    st.session_state.last_assistant_ts = last_ts
    st.session_state.assistant_speaking = speaking

new_logs = drain_jarvis_logs() if st.session_state.jarvis_running else []
if new_logs:
    _update_speaking_state_from_logs(new_logs)

# ---- parsing des logs + mise à jour messages ----
def _ingest_chat_from_logs(logs: List[str]):
    """Ingère les messages chat depuis les logs."""
    if not logs:
        return
    msgs = st.session_state.setdefault("messages", [])
    changed = False
    for raw in logs:
        line = (raw or "").strip()
        if not line:
            continue
        if line.startswith("[VU]"):
            continue
        m_user = re.match(r'^\[(?:STT|VOICE)\]\s*(?:Vous|You)\s*:\s*(.+)$', line, flags=re.IGNORECASE)
        if m_user:
            content = m_user.group(1).strip()
            if content:
                msgs.append({"role": "user", "content": content})
                changed = True
            continue
        m_ass = re.match(r'^(?:JARVIS|Assistant)\s*:\s*(.+)$', line, flags=re.IGNORECASE)
        if m_ass:
            content = m_ass.group(1).strip()
            if content:
                msgs.append({"role": "assistant", "content": content})
                changed = True
            continue
    if changed:
        st.session_state["messages"] = msgs[-200:]
        st.rerun()

_ingest_chat_from_logs(new_logs)

# -------------- ORT providers (affichage dans topbar) --------------
def _ort_providers() -> Tuple[List[str], str]:
    try:
        import onnxruntime as ort
        provs = list(ort.get_available_providers())
        label = "CPU"
        if "TensorrtExecutionProvider" in provs:
            label = "TensorRT"
        elif "CUDAExecutionProvider" in provs:
            label = "CUDA"
        return provs, label
    except Exception as e:
        return [], f"ERR: {getattr(e, '__class__', type(e)).__name__}"

# -------------- CSS --------------
st.markdown("""
<style>
:root{
  --bg: #0b0f14; --panel: #0e141b; --muted: #121923; --border: #1b2633;
  --primary: #44f1ff; --primary-2: #00c2d1; --fg: #e6f1ff; --muted-fg:#7b8ba1; --accent:#101926;
  --ok: #2ecc71; --bad: #e74c3c;
}
.main .block-container{ padding-top: 0rem; }
body{
  background: radial-gradient(900px 450px at 50% 28%, rgba(0, 194, 209, 0.08), transparent 55%),
              radial-gradient(700px 400px at 70% 20%, rgba(68,241,255,0.05), transparent 60%),
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
  font-family: ui-monospace; font-size: 12px; color: var(--fg); }
.dot{ width:8px; height:8px; border-radius:999px; opacity:.95 }
.dot.ok{ background: var(--ok); box-shadow: 0 0 12px rgba(46,204,113,.9), 0 0 30px rgba(46,204,113,.35); animation: pulse 1.6s ease-in-out infinite; }
.dot.bad{ background: var(--bad); box-shadow: 0 0 12px rgba(231,76,60,.9), 0 0 30px rgba(231,76,60,.35); }
@keyframes pulse{0%{transform:scale(.9);opacity:.8}50%{transform:scale(1.2);opacity:1}100%{transform:scale(.9);opacity:.8}}
.card{ background: linear-gradient(180deg, var(--panel), var(--muted)); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }
.card h3{ margin: 0 0 8px 0; color: var(--fg); }
.muted{ color: var(--muted-fg); }
.chat-card{ display:flex; flex-direction:column; }
.chat-card-body{ display:flex; flex-direction:column; gap:14px; }
.chat-wrap{ display:flex; flex-direction:column; height:520px; }
.msgs{ flex:1; overflow:auto; display:flex; flex-direction:column; gap:10px; padding-right:6px; min-width:0; }
.bubble{ max-width: 92%; padding:10px 12px; border-radius: 12px; border: 1px solid var(--border); white-space:pre-wrap; overflow-wrap:anywhere; }
.user{ align-self:flex-end; background: rgba(68,241,255,0.08); }
.assistant{ align-self:flex-start; background: rgba(255,255,255,0.04); }
.chat-toggle{ margin-top:8px; }
.chat-toggle [data-testid="stToggle"]{ width:100%; background: var(--accent); padding:10px 12px; border-radius: 12px; border:1px solid var(--border); }
.chat-toggle [data-testid="stToggle"] label{ color: var(--fg); font-weight:600; }
.chat-toggle [data-testid="stToggle"] [data-testid="stTickBar"]{ background: var(--primary); }
.chat-input input{ background: var(--accent) !important; color: var(--fg) !important; border:1px solid var(--border); border-radius: 10px; padding:10px 12px; }
.chat-send button{ background: linear-gradient(180deg, #0b485b, #083947); border:1px solid var(--primary-2); color: white; border-radius: 10px; padding:10px 14px; }
</style>
""", unsafe_allow_html=True)

# -------------- Top bar --------------
def _pill_html(name: str, ok: bool, label: str) -> str:
    cls = "dot ok" if ok else "dot bad"
    safe_label = html.escape(label)
    return f'<div class="pill"><div class="{cls}"></div><span>{name}</span><span class="muted">• {safe_label}</span></div>'

ollama_up = is_ollama_up(CFG["ollama"]["host"])
ollama_status = "Up" if ollama_up else "Down"

# Statut MCP: simple (activé ?), libellé = host
gw = CFG.get("mcp", {}).get("gateway", {})
mcp_ok = bool(gw.get("enabled", False))
mcp_host_label = gw.get("base_url", "") or "off"
pill_mcp = _pill_html("MCP", mcp_ok, mcp_host_label)

jarvis_running = bool(st.session_state.get("jarvis_running"))
whisper_ok = jarvis_running
whisper_label = "Ready" if whisper_ok else "Off"

voice_path = _maybe_join_voice(CFG["piper"].get("base_dir", ""), CFG["piper"].get("voice", ""))
piper_ok = jarvis_running and CFG["jarvis"].get("tts_engine") == "piper" and voice_path and os.path.exists(voice_path)
piper_label = "Ready" if piper_ok else "Off"

# ORT (ONNX Runtime) status
ort_provs, ort_label = _ort_providers()
ort_ok = bool(ort_provs)
pill_ort = _pill_html("ONNXRT", ort_ok, ort_label)

pill_whisper = _pill_html("WHISPER", whisper_ok, whisper_label)
pill_piper   = _pill_html("PIPER",   piper_ok,   piper_label)
pill_ollama  = _pill_html("OLLAMA",  ollama_up,  ollama_status)

st.markdown(
    f"""
<div class="topbar">
  <div class="status-row">
    <div class="status-left">
      {pill_whisper}
      {pill_piper}
      {pill_ollama}
      {pill_mcp}
      {pill_ort}
    </div>
    <div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Helper pour afficher le radar (dans un composant HTML dédié) ----------
def render_radar(speaking: bool, mode: str, vu_history: List[Tuple[float, float]]):
    svg_cls = "speaking" if speaking else ""
    center_cls = "speaking" if speaking else ""
    label = "Mode vocal" if mode == "vocal" else "Mode chat"

    now = time.time()
    history_payload: List[Dict[str, float]] = []
    for entry in vu_history[-240:]:
        try:
            ts, val = entry
        except (TypeError, ValueError):
            continue
        try:
            stamp = float(ts)
            level_val = float(val)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(stamp) and math.isfinite(level_val)):
            continue
        level_val = max(0.0, min(1.0, level_val))
        dt = max(0.0, now - stamp)
        history_payload.append({"dt": dt, "level": level_val})

    current_level = max(0.0, min(1.0, float(st.session_state.get("assistant_vu_level", 0.0))))
    initial_level = history_payload[-1]["level"] if history_payload else current_level
    history_json = json.dumps(history_payload, ensure_ascii=False)

    html_snippet = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        body {{
          margin: 0; padding: 0;
          background: transparent;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .wrap {{
          background: linear-gradient(180deg, #0e141b, #121923);
          border: 1px solid #1b2633;
          border-radius: 14px;
          padding: 14px;
        }}
        .title {{ color: #e6f1ff; margin: 0 0 8px 0; font-weight: 600; }}
        .radar-container {{
          position: relative; width: min(100%, 400px); aspect-ratio: 1;
          display: flex; align-items: center; justify-content: center; margin: 0 auto;
          filter: drop-shadow(0 0 20px rgba(68,241,255,.20));
        }}
        svg {{ width: 100%; height: 100%; }}
        .radar-center-circle {{ fill: #0d7a8f; transition: fill .3s ease; }}
        .radar-center-circle.speaking {{ fill: #00d4e8; filter: drop-shadow(0 0 15px rgba(68,241,255,.6)); }}
        .radar-dot {{ fill: #00d4e8; opacity: .9; }}
        .wave {{
          fill: none;
          stroke: #00d4e8;
          stroke-width: 2;
          opacity: 0;
          transition: opacity .25s ease;
        }}

        .radar-container[data-speaking="false"] .wave {{
          opacity: 0 !important;
        }}

        @media (prefers-reduced-motion: reduce) {{
          .wave {{
            transition: none;
          }}
        }}

        .radar-status {{
          text-align:center; margin-top:8px; color:#44f1ff; font-size:13px;
          letter-spacing:1px; text-transform:uppercase;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h3 class="title">Radar Vocal</h3>
        <div class="radar-container" data-speaking="{str(bool(speaking)).lower()}" data-level="{initial_level:.3f}">
          <svg class="{svg_cls}" viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet">
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            <circle cx="150" cy="150" r="140" fill="none" stroke="#00d4e8" stroke-width="1" opacity="0.3"/>
            <circle cx="150" cy="150" r="105" fill="none" stroke="#00d4e8" stroke-width="1" opacity="0.25"/>
            <circle cx="150" cy="150" r="70"  fill="none" stroke="#00d4e8" stroke-width="1.5" opacity="0.35"/>
            <circle cx="150" cy="150" r="35"  fill="none" stroke="#00d4e8" stroke-width="1" opacity="0.3"/>

            <line x1="10"  y1="150" x2="290" y2="150" stroke="#00d4e8" stroke-width="1" opacity="0.2"/>
            <line x1="150" y1="10"  x2="150" y2="290" stroke="#00d4e8" stroke-width="1" opacity="0.2"/>

            <g id="waves">
              <circle class="wave wave-1" cx="150" cy="150" r="35" filter="url(#glow)"/>
              <circle class="wave wave-2" cx="150" cy="150" r="35" filter="url(#glow)"/>
              <circle class="wave wave-3" cx="150" cy="150" r="35" filter="url(#glow)"/>
            </g>

            <circle class="radar-center-circle {center_cls}" cx="150" cy="150" r="28"/>

            <circle class="radar-dot" cx="150" cy="20"  r="5"/>
            <circle class="radar-dot" cx="195" cy="37"  r="5"/>
            <circle class="radar-dot" cx="250" cy="80"  r="5"/>
            <circle class="radar-dot" cx="275" cy="135" r="5"/>
            <circle class="radar-dot" cx="280" cy="150" r="5"/>
            <circle class="radar-dot" cx="275" cy="220" r="5"/>
            <circle class="radar-dot" cx="220" cy="270" r="5"/>
            <circle class="radar-dot" cx="150" cy="280" r="5"/>
            <circle class="radar-dot" cx="105" cy="263" r="5"/>
            <circle class="radar-dot" cx="25"  cy="220" r="5"/>
            <circle class="radar-dot" cx="20"  cy="150" r="5"/>
            <circle class="radar-dot" cx="105" cy="37"  r="5"/>
          </svg>
        </div>
        <div class="radar-status">{label}</div>
      </div>
      <script>
        (() => {{
          const container = document.querySelector('.radar-container');
          if (!container) {{ return; }}
          const svg = container.querySelector('svg');
          if (!svg) {{ return; }}
          const waves = Array.from(svg.querySelectorAll('.wave'));
          const centerCircle = svg.querySelector('.radar-center-circle');
          if (!waves.length || !centerCircle) {{ return; }}

          const MIN_RADIUS = 35;
          const MAX_RADIUS = 140;
          const BASE_CYCLE_MS = 1800;
          const OFFSET_MS = BASE_CYCLE_MS / waves.length;

          const payload = {history_json};
          const nowBase = performance.now();
          const initialLevel = Math.min(Math.max(parseFloat(container.dataset.level || '0') || 0, 0), 1);

          let samples = Array.isArray(payload)
            ? payload
                .map((entry) => {{
                  if (!entry) {{ return null; }}
                  const lvl = Number(entry.level);
                  const dt = Number(entry.dt);
                  if (!Number.isFinite(lvl) || !Number.isFinite(dt)) {{ return null; }}
                  const level = Math.min(Math.max(lvl, 0), 1);
                  const age = Math.max(dt, 0);
                  return {{ level, time: nowBase - age * 1000 }};
                }})
                .filter((item) => item !== null)
                .sort((a, b) => a.time - b.time)
            : [];

          if (!samples.length) {{
            samples.push({{ level: initialLevel, time: nowBase }});
          }}

          let smoothedLevel = samples[samples.length - 1]?.level ?? initialLevel;
          let speaking = svg.classList.contains('speaking');
          let lastToggleTs = nowBase;

          const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
          let prefersReducedMotion = reducedMotionQuery.matches;

          const handleReducedMotion = (event) => {{
            prefersReducedMotion = event.matches;
            if (prefersReducedMotion) {{
              container.setAttribute('data-speaking', 'false');
              waves.forEach((wave) => {{
                wave.setAttribute('r', MIN_RADIUS.toString());
                wave.style.opacity = '0';
              }});
              centerCircle.setAttribute('r', '28');
            }} else {{
              lastToggleTs = performance.now();
            }}
          }};

          if (reducedMotionQuery.addEventListener) {{
            reducedMotionQuery.addEventListener('change', handleReducedMotion);
          }} else if (reducedMotionQuery.addListener) {{
            reducedMotionQuery.addListener(handleReducedMotion);
          }}

          handleReducedMotion(reducedMotionQuery);

          const levelAt = (timeMs) => {{
            if (!samples.length) {{
              return svg.classList.contains('speaking') ? 0.2 : 0;
            }}
            if (samples.length === 1) {{
              return samples[0].level;
            }}
            let prev = samples[0];
            if (timeMs <= prev.time) {{
              return prev.level;
            }}
            for (let i = 1; i < samples.length; i += 1) {{
              const sample = samples[i];
              if (timeMs <= sample.time) {{
                const span = Math.max(sample.time - prev.time, 1);
                const ratio = Math.min(Math.max((timeMs - prev.time) / span, 0), 1);
                return prev.level + (sample.level - prev.level) * ratio;
              }}
              prev = sample;
            }}
            return prev.level;
          }};

          const animate = (now) => {{
            if (prefersReducedMotion) {{
              requestAnimationFrame(animate);
              return;
            }}

            samples = samples.filter((sample) => now - sample.time <= 2600);
            if (!samples.length) {{
              samples.push({{ level: 0, time: now }});
            }}

            const rawLevel = Math.min(Math.max(levelAt(now), 0), 1);
            smoothedLevel = smoothedLevel * 0.65 + rawLevel * 0.35;

            const attrSpeaking = svg.classList.contains('speaking');
            const speakingNow = attrSpeaking || smoothedLevel > 0.04;
            if (speakingNow !== speaking) {{
              speaking = speakingNow;
              container.setAttribute('data-speaking', String(speaking));
              if (speaking && !attrSpeaking) {{
                svg.classList.add('speaking');
              }} else if (!speaking && attrSpeaking) {{
                svg.classList.remove('speaking');
              }}
              if (speaking) {{
                lastToggleTs = now;
              }} else {{
                waves.forEach((wave) => {{
                  wave.setAttribute('r', MIN_RADIUS.toString());
                  wave.style.opacity = '0';
                }});
              }}
            }}

            const visualLevel = smoothedLevel > 0.015 ? smoothedLevel : (speakingNow ? 0.12 : 0);
            container.dataset.level = visualLevel.toFixed(3);

            const cycleMs = speakingNow ? BASE_CYCLE_MS * (1 - Math.min(visualLevel, 0.6) * 0.35) : BASE_CYCLE_MS * 1.1;

            waves.forEach((wave, index) => {{
              const elapsed = now - lastToggleTs - index * OFFSET_MS;
              if (!speakingNow || elapsed < 0) {{
                wave.setAttribute('r', MIN_RADIUS.toString());
                wave.style.opacity = '0';
                return;
              }}

              const progress = (elapsed % cycleMs) / cycleMs;
              const maxRadius = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * visualLevel;
              const radius = MIN_RADIUS + (maxRadius - MIN_RADIUS) * progress;
              const opacity = Math.max(0, (1 - progress) * (0.35 + visualLevel * 0.55));

              wave.setAttribute('r', radius.toFixed(2));
              wave.style.opacity = opacity.toFixed(2);
            }});

            const centerRadius = 28 + visualLevel * 10;
            centerCircle.setAttribute('r', centerRadius.toFixed(2));

            requestAnimationFrame(animate);
          }};

          requestAnimationFrame(animate);
        }})();
      </script>
    </body>
    </html>
    """
    components.html(html_snippet, height=480, scrolling=False)

# -------------- Tabs --------------
tab_interface, tab_settings = st.tabs(["🎛️ Interface", "⚙️ Settings"])

# -------------------- Interface (Chat) --------------------
with tab_interface:
    now_ts = time.time()
    last_ts = st.session_state.get("last_assistant_ts", 0.0)
    recent_assistant = last_ts and (now_ts - last_ts) < 6
    speaking = bool(recent_assistant)
    st.session_state.assistant_speaking = speaking

    mode = st.session_state.get("interaction_mode", "chat")

    c1, c2 = st.columns([5,7])
    with c1:
        # Radar
        render_radar(
            speaking=speaking,
            mode=mode,
            vu_history=st.session_state.get("assistant_vu_history", []),
        )

    with c2:
        msgs = st.session_state.setdefault("messages", [])

        def _render_messages(messages: List[Dict[str, str]]) -> str:
            if not messages:
                return (
                    '<div class="msgs"><div class="bubble assistant muted">'
                    "Aucun échange pour le moment."
                    "</div></div>"
                )
            bubbles = []
            for msg in messages:
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                cls = "bubble assistant" if role != "user" else "bubble user"
                safe_content = html.escape(str(content)).replace("\n", "<br />")
                bubbles.append(f'<div class="{cls}">{safe_content}</div>')
            return f"<div class='msgs'>{''.join(bubbles)}</div>"

        chat_slot = st.empty()
        chat_slot.markdown(
            f'''
            <div class="card chat-card">
              <div class="chat-card-body">
                <h3>Chat</h3>
                <div class="chat-wrap">
                  {_render_messages(msgs)}
                </div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # --- Commandes
        outer_cols = st.columns([3, 12])
        with outer_cols[0]:
            st.markdown('<div class="chat-toggle">', unsafe_allow_html=True)
            st.toggle(
                "🎙️ Mode vocal",
                key="mode_toggle",
                value=st.session_state.mode_toggle,
                help="Active le mode vocal pour répondre par la voix",
                on_change=_sync_mode_from_toggle,
            )

            st.divider()
            st.write("🔎 **MCP Gateway**")
            gw_enabled = CFG["mcp"]["gateway"].get("enabled", True)
            CFG["mcp"]["gateway"]["enabled"] = st.toggle(
                "Activer le gateway MCPJungle",
                value=bool(gw_enabled),
                help="Expose tous tes tools via un endpoint unique."
            )
            st.caption("Raccourci chat: `/tool <nom> {json}`  → appelle le tool via le gateway.")

            st.divider()
            st.write("🔄 **Rafraîchissement**")
            st.session_state.auto_refresh_chat = st.toggle(
                "Auto-refresh pendant la parole",
                value=bool(st.session_state.get("auto_refresh_chat", False)),
                help="Évite que les boutons deviennent difficilement cliquables."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        mode = st.session_state.interaction_mode
        placeholder_text = (
            "MODE VOCAL ACTIVÉ - Utilise ton micro"
            if mode == "vocal"
            else "MODE CHAT ACTIVÉ - Commence la conversation (astuce: /tool nom {json})"
        )

        with outer_cols[1]:
            with st.form("chat_form", clear_on_submit=True, border=False):
                inner_cols = st.columns([8, 3])
                with inner_cols[0]:
                    user_text = st.text_area(
                        "Message",
                        key="chat_input",
                        placeholder=placeholder_text,
                        height=80,
                        label_visibility="collapsed",
                        disabled=(mode == "vocal"),
                    )
                with inner_cols[1]:
                    submitted = st.form_submit_button(
                        "Envoyer" if mode == "chat" else "🎤 Écoute",
                        use_container_width=True,
                        disabled=(mode != "chat"),
                    )

            if submitted and mode == "chat":
                clean_text = (user_text or "").strip()
                if clean_text:
                    used_gateway = False
                    # --- Raccourci: /tool <nom> {json}
                    m_tool = re.match(r"^/tool\s+([A-Za-z0-9_.:\-]+)\s*(.*)$", clean_text)
                    if m_tool and CFG["mcp"]["gateway"].get("enabled", True):
                        tool_name = m_tool.group(1)
                        arg_str = (m_tool.group(2) or "").strip()
                        try:
                            if arg_str.startswith("{"):
                                args = json.loads(arg_str)
                            else:
                                args = {}
                                for kv in shlex.split(arg_str):
                                    if "=" in kv:
                                        k, v = kv.split("=", 1)
                                        args[k] = v
                        except Exception as e:
                            args = {}
                            msgs.append({"role": "assistant", "content": f"[MCP] Args invalides: {e}"})
                        msgs.append({"role": "user", "content": clean_text})
                        out = mcp_call_tool_via_gateway(tool_name, args, timeout_s=60)
                        msgs.append({"role": "assistant", "content": out or "(vide)"})
                        used_gateway = True

                    # --- Sinon: LLM pur
                    if not used_gateway:
                        msgs.append({"role": "user", "content": clean_text})
                        reply = ollama_reply(CFG, msgs) or "Réponse vide d'Ollama (vérifie le modèle)."
                        msgs.append({"role": "assistant", "content": reply})

                    st.session_state["messages"] = msgs[-200:]
                    st.session_state.last_assistant_ts = time.time()
                    st.session_state.assistant_speaking = True
                    st.rerun()

        if mode == "vocal":
            st.info("Mode vocal actif : Jarvis répondra via la voix lorsque le backend est connecté.")

        st.caption(f"Messages en mémoire: {len(st.session_state.get('messages', []))}")

# -------------------- Settings --------------------
with tab_settings:
    t0, t1, t2, t3, t4, t5 = st.tabs(["Jarvis", "Whisper", "Piper", "Ollama", "MCP", "💾 Sauvegarde"])

    with t0:
        st.write("**Jarvis (audio local)**")
        current_path = CFG["jarvis"].get("path", "./jarvis.py")
        jarvis_path = st.text_input("Chemin jarvis.py", value=current_path)
        CFG["jarvis"]["path"] = jarvis_path.strip() or current_path
        CFG["jarvis"]["audio_out"] = st.text_input(
            "Préférence sortie audio (AUDIO_OUT)",
            value=CFG["jarvis"].get("audio_out", "analog"),
            help="Chaîne partielle du nom du périphérique (ex: analog, speaker, hdmi). Laisser vide pour auto.",
        ).strip()

        ctts1, ctts2 = st.columns(2)
        with ctts1:
            tts_options = ["piper", "fallback"]
            current_engine = CFG["jarvis"].get("tts_engine", "piper")
            idx_engine = tts_options.index(current_engine) if current_engine in tts_options else 0
            CFG["jarvis"]["tts_engine"] = st.selectbox("Moteur TTS", tts_options, index=idx_engine)
        with ctts2:
            CFG["jarvis"]["tts_lang"] = st.text_input(
                "Langue TTS (fallback)", value=CFG["jarvis"].get("tts_lang", "fr")
            ).strip() or "fr"

        st.markdown("---")
        wake_col1, wake_col2 = st.columns(2)
        with wake_col1:
            CFG["jarvis"]["wake_word"] = st.text_input(
                "Mot d'activation principal", value=CFG["jarvis"].get("wake_word", "jarvis")
            ).strip() or "jarvis"
            CFG["jarvis"]["require_wake"] = st.checkbox(
                "Nécessite le mot-clé pour écouter",
                value=bool(CFG["jarvis"].get("require_wake", True)),
            )
        with wake_col2:
            CFG["jarvis"]["wake_fuzzy"] = st.checkbox(
                "Tolérance phonétique (RapidFuzz)",
                value=bool(CFG["jarvis"].get("wake_fuzzy", True)),
            )
            CFG["jarvis"]["wake_fuzzy_score"] = st.slider(
                "Score fuzzy minimal",
                50,
                100,
                int(CFG["jarvis"].get("wake_fuzzy_score", 80)),
            )

        CFG["jarvis"]["wake_aliases"] = st.text_area(
            "Alias supplémentaires (un par ligne ou séparés par des virgules)",
            value=CFG["jarvis"].get("wake_aliases", ""),
            height=80,
        ).strip()

        colJ1, colJ2, colJ3 = st.columns(3)
        with colJ1:
            if st.button("▶️ Démarrer Jarvis"):
                start_jarvis(CFG)
        with colJ2:
            if st.button("⏹️ Arrêter Jarvis"):
                stop_jarvis()
        with colJ3:
            if st.button("♻️ Recharger module"):
                st.session_state.jarvis_mod = None
                st.success("Module Jarvis sera rechargé au prochain démarrage.")

        if st.session_state.jarvis_running:
            drain_jarvis_logs()
        st.code("\n".join(st.session_state.get("jarvis_logbuf", [])) or "—", language="bash")

        st.caption("Astuce: si l’audio sort au mauvais endroit, mets AUDIO_OUT sur une sous-chaîne du nom du périphérique (ex: 'analog', 'hdmi').")

    with t1:
        model_options = ["tiny", "base", "small", "medium", "large-v3"]
        current_model = CFG["whisper"].get("model", "small")
        if current_model not in model_options:
            current_model = "small"
        CFG["whisper"]["model"] = st.selectbox(
            "Modèle Fast Whisper", model_options, index=model_options.index(current_model)
        )

        lang_options = ["fr", "en", "auto"]
        current_lang = CFG["whisper"].get("lang", "fr")
        if current_lang not in lang_options:
            current_lang = "fr"
        CFG["whisper"]["lang"] = st.selectbox("Langue", lang_options, index=lang_options.index(current_lang))

        CFG["whisper"]["vad"] = st.checkbox("VAD activé", value=bool(CFG["whisper"].get("vad", True)))

        device_options = ["cpu", "cuda"]
        current_device = CFG["whisper"].get("device", "cpu")
        if current_device not in device_options:
            current_device = "cpu"
        CFG["whisper"]["device"] = st.selectbox(
            "Device", device_options, index=device_options.index(current_device)
        )

        compute_options = ["int8", "int8_float16", "float16", "float32"]
        current_compute = CFG["whisper"].get("compute", "int8")
        if current_compute not in compute_options:
            current_compute = "int8"
        CFG["whisper"]["compute"] = st.selectbox(
            "Compute", compute_options,
            index=compute_options.index(current_compute) if current_compute in compute_options else 0
        )

        CFG["whisper"]["prompt"] = st.text_area(
            "Prompt initial (contexte)",
            value=CFG["whisper"].get("prompt", ""),
            height=100,
        ).strip()

    with t2:
        st.write("**Sélection de la voix Piper**")
        CFG["piper"]["base_dir"] = st.text_input(
            "Dossier des voix (local serveur)",
            value=str(CFG["piper"].get("base_dir", "")),
        ).strip()
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
        colP1, colP2 = st.columns(2)
        with colP1:
            CFG["piper"]["speaker_id"] = st.number_input("Speaker ID", min_value=0, max_value=999, value=int(CFG["piper"].get("speaker_id", 0)), step=1)
            CFG["piper"]["use_cuda"] = st.toggle("Activer CUDA (--cuda)", value=bool(CFG["piper"].get("use_cuda", False)))
        with colP2:
            CFG["piper"]["speed"] = st.number_input("Vitesse (length scale)", min_value=0.5, max_value=2.0, value=float(CFG["piper"].get("speed", 0.9)), step=0.05)

        CFG["piper"]["noise"] = st.slider("Noise scale", 0.0, 2.0, float(CFG["piper"].get("noise", 0.667)), 0.01)
        CFG["piper"]["noise_w"] = st.slider("Noise width", 0.0, 2.0, float(CFG["piper"].get("noise_w", 0.8)), 0.01)
        CFG["piper"]["sentence_silence"] = st.slider("Silence inter-phrases (s)", 0.0, 1.0, float(CFG["piper"].get("sentence_silence", 0.10)), 0.01)

    with t3:
        st.write("**Ollama**")
        CFG["ollama"]["host"] = st.text_input("Host", value=CFG["ollama"]["host"])
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
        st.write("**MCP (via MCPJungle Gateway)**")
        mcp_installed = _mcp_import_ok()
        if not mcp_installed:
            st.info("Installe le SDK MCP: `pip install mcp` (ou `pip install modelcontextprotocol`).")

        CFG["mcp"]["gateway"]["enabled"] = st.toggle(
            "Activer MCPJungle",
            value=bool(CFG["mcp"]["gateway"].get("enabled", True))
        )
        CFG["mcp"]["gateway"]["base_url"]  = st.text_input(
            "Gateway URL", value=CFG["mcp"]["gateway"].get("base_url","http://127.0.0.1:8080")
        )
        CFG["mcp"]["gateway"]["auth_header"] = st.text_input(
            "Authorization (optionnel)", value=CFG["mcp"]["gateway"].get("auth_header",""), placeholder="Bearer xxx"
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("🔍 Lister tools (Gateway)"):
                try:
                    tools = mcp_list_tools_via_gateway()
                    st.success(", ".join(tools) or "(aucun)")
                except Exception as e:
                    st.error(f"list_tools: {e}")
        with colB:
            with st.popover("▶️ Appeler un tool (Gateway)"):
                tname = st.text_input("Nom du tool", placeholder="ex: time__get_current_time")
                targ  = st.text_area("Arguments (JSON)", placeholder='{"foo":"bar"}', height=120)
                if st.button("Exécuter"):
                    try:
                        args = json.loads(targ) if targ.strip() else {}
                        out = mcp_call_tool_via_gateway(tname.strip(), args, timeout_s=60)
                        st.code(out or "(vide)")
                    except Exception as e:
                        st.error(f"call_tool: {e}")

        st.caption("Raccourci chat:  `/tool <nom> {json}`  — se connecte au gateway MCPJungle.")

    with t5:
        csa, csb = st.columns(2)
        with csa:
            if st.button("💾 Enregistrer la configuration"):
                save_cfg(CFG)
        with csb:
            if st.button("↩️ Recharger depuis disque"):
                st.cache_data.clear()
                st.rerun()

# --- Auto-refresh quand Jarvis tourne en mode vocal (désactivable) ---
if (
    st.session_state.get("jarvis_running")
    and st.session_state.get("interaction_mode") == "vocal"
    and st.session_state.get("auto_refresh_chat", False)
    and (time.time() - st.session_state.get("last_assistant_ts", 0.0)) < 10
):
    time.sleep(0.3)
    st.rerun()
