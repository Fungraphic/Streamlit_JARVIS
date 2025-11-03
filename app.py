#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, subprocess, importlib.util, threading, queue, html, re, uuid, shlex, asyncio, copy, math, sys
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import streamlit as st
import requests
import streamlit.components.v1 as components  # <- pour l'iframe HTML du radar

# Import du style JARVIS HUD
from jarvis_ui_style import inject_jarvis_css, render_chat_messages

st.set_page_config(page_title="J.A.R.V.I.S. - Assistant Vocal Local", layout="wide", initial_sidebar_state="expanded")

# Injecter le CSS JARVIS HUD
st.markdown(inject_jarvis_css(), unsafe_allow_html=True)

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
    # --- LLM / Prompt système ---
    "llm": {
        "system_prompt": (
            "Tu es JARVIS, un assistant francophone, concis et utile.\n\n"
            "═══ RÈGLE D'OR : RÉPONDS DIRECTEMENT SANS OUTIL PAR DÉFAUT ═══\n\n"
            "Tu as accès à des outils (documentation, web, conversions). N'utilise un outil QUE si strictement nécessaire.\n\n"
            "❌ NE PAS UTILISER D'OUTIL pour :\n"
            "• Salutations : \"Bonjour\" → Réponds : \"Bonjour ! Comment puis-je vous aider ?\"\n"
            "• Questions sur toi : \"Qui es-tu ?\" → Réponds : \"Je suis JARVIS, votre assistant.\"\n"
            "• Explications générales : \"C'est quoi Python ?\" → Utilise tes connaissances\n"
            "• Conversations courantes : \"Comment vas-tu ?\", \"Merci\", etc.\n\n"
            "✅ UTILISER UN OUTIL uniquement pour :\n"
            "• Heure/timezone : \"Quelle heure à Tokyo ?\" → outil 'get_current_time' ou 'convert_time'\n"
            "• Documentation précise : \"Doc Python asyncio\" → outil 'get-library-docs' avec library exacte\n"
            "• Recherche web : \"Cherche les news\" → outil 'fetch'\n"
            "• Code Python : \"Analyse ce code\" → outil 'analyze_python_file'\n\n"
            "EXEMPLES CONCRETS :\n"
            "Q: \"Bonjour\" → R: \"Bonjour ! Comment puis-je vous aider ?\" [PAS D'OUTIL]\n"
            "Q: \"Explique asyncio\" → R: \"asyncio est un module Python pour...\" [PAS D'OUTIL]\n"
            "Q: \"Quelle heure à New York ?\" → R: [OUTIL: convert_time avec timezone]\n"
            "Q: \"Doc React hooks\" → R: [OUTIL: get-library-docs avec react]\n\n"
            "Toujours répondre en français, de façon brève et précise.\n"
        ),
        # Héritage / options d'agent
        "agent_enabled": False,        # (legacy) heuristique bloc JSON
        "agent_max_rounds": 3,         # nombre max de tours pour l'agent (tool-calling inclus)
        "selector_json": False,        # (legacy) sélecteur JSON manuel
        "use_ollama_tools": True       # <-- clé principale : activer tool-calling natif Ollama
    },
    # --- MCP: configuration via Docker MCP Toolkit ---
    "mcp": {
        "docker": {
            "enabled": True,  # ← Activé par défaut si Docker MCP disponible
            "docker_cmd": "docker",  # Commande docker (peut être chemin absolu)
            "auto_web": False,  # <- OFF par défaut pour forcer le modèle à choisir les tools
            "auto_web_topk": 5,
            "chat_shortcuts": True  # /web /fetch /tool
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

def _normalize_mcp_docker(cfg: Dict[str, Any]) -> None:
    """
    Normalise la section MCP pour Docker MCP Toolkit.
    Purge les anciennes clés (gateway/servers/proxy/jungle).
    """
    mcp = cfg.setdefault("mcp", {})
    docker = mcp.get("docker") or {}
    mcp["docker"] = {
        "enabled": bool(docker.get("enabled", True)),  # Activé par défaut
        "docker_cmd": str(docker.get("docker_cmd", "docker")),
        "auto_web": bool(docker.get("auto_web", False)),
        "auto_web_topk": int(docker.get("auto_web_topk", 5)),
        "chat_shortcuts": bool(docker.get("chat_shortcuts", True)),
    }
    for legacy in ("gateway", "servers", "proxy", "jungle"):
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
                for k, v in (data or {}).items():
                    if isinstance(v, dict):
                        cfg[k] = {**cfg.get(k, {}), **copy.deepcopy(v)}
                    else:
                        cfg[k] = v
                _normalize_mcp_docker(cfg)
                return cfg
    except Exception as e:
        st.warning(f"Lecture config échouée: {e}")
    cfg = copy.deepcopy(DEFAULT_CFG)
    _normalize_mcp_docker(cfg)
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
    docker_cfg = (cfg.get("mcp", {}).get("docker") or {})

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
        "MCP_SERVERS_JSON": "[]",  # Legacy compat
        "MCP_DOCKER_ENABLED": "1" if docker_cfg.get("enabled", True) else "0",
        "MCP_DOCKER_CMD": docker_cfg.get("docker_cmd", "docker"),
        "MCP_AUTO_WEB": "1" if docker_cfg.get("auto_web", False) else "0",
        "MCP_AUTO_WEB_TOPK": str(docker_cfg.get("auto_web_topk", 5)),
        "MCP_CHAT_SHORTCUTS": "1" if docker_cfg.get("chat_shortcuts", True) else "0",
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

def _send_mic_state(enabled: bool):
    """Essaye plusieurs voies: file de commandes q_cmd, fonctions utilitaires, attribut, puis trace."""
    jm = st.session_state.get("jarvis_mod")
    state = "ON" if enabled else "OFF"
    if jm is not None:
        # 1) File de contrôle (recommandé)
        for name in ("q_cmd", "q_control", "control_q", "q_in"):
            q = getattr(jm, name, None)
            if q is not None:
                try:
                    q.put_nowait({"type": "mic", "enabled": bool(enabled)})
                    break
                except Exception:
                    pass
        else:
            # 2) Fonctions directes si exposées par jarvis.py
            for fn_name in ("set_listen_enabled", "set_mic_enabled", "toggle_listen"):
                fn = getattr(jm, fn_name, None)
                if callable(fn):
                    try:
                        fn(bool(enabled))
                        break
                    except Exception:
                        pass
            else:
                # 3) Attribut simple
                if hasattr(jm, "listen_enabled"):
                    try:
                        setattr(jm, "listen_enabled", bool(enabled))
                    except Exception:
                        pass
        # Trace UI -> logs Jarvis
        try:
            if hasattr(jm, "q_log"):
                jm.q_log.put_nowait(f"[UI] MIC {state} demandé (toggle)")
        except Exception:
            pass

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
    """Construit la liste des messages au format Ollama en préfixant le prompt système (si fourni)."""
    sys_prompt = ""
    try:
        sys_prompt = (CFG.get("llm", {}) or {}).get("system_prompt", "") or ""
        sys_prompt = str(sys_prompt).strip()
    except Exception:
        sys_prompt = ""
    mapped = []
    for m in messages[-20:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role not in ("user", "assistant", "system", "tool"):
            role = "assistant" if role != "user" else "user"
        obj = {"role": role, "content": content}
        for k in ("tool_calls", "thinking", "tool_name"):
            if k in m:
                obj[k] = m[k]
        mapped.append(obj)
    if sys_prompt:
        mapped = [{"role": "system", "content": sys_prompt}] + mapped
    elif mapped and mapped[0]["role"] == "assistant":
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
        # Fallback /api/generate
        try:
            last_user = next((m["content"] for m in reversed(messages) if m.get("role")=="user"), "")
            sys_prompt = (CFG.get("llm", {}) or {}).get("system_prompt", "") or ""
            base_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + last_user
            url = host.rstrip("/") + "/api/generate"
            payload = {"model": model, "prompt": base_prompt, "stream": False, "options": options}
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json() or {}
            return data.get("response") or data.get("text") or ""
        except Exception as e2:
            return f"Erreur Ollama: {e2}"

# --- Streaming Ollama (tokens en direct) ---
def ollama_stream_chat(cfg: Dict[str, Any], messages: List[Dict[str, Any]], on_delta):
    """Stream /api/chat en direct (sans tools). Appelle on_delta(text_chunk) à chaque token/delta."""
    host = cfg["ollama"]["host"].rstrip("/")
    model = cfg["ollama"]["model"]
    options = {
        "temperature": float(cfg["ollama"].get("temperature", 0.7)),
        "num_ctx": int(cfg["ollama"].get("num_ctx", 4096)),
    }
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": _build_ollama_chat_messages(messages),
        "stream": True,
        "options": options,
    }
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        buf = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            chunk = (((obj.get("message") or {}).get("content")) or "")
            if chunk:
                buf.append(chunk)
                on_delta(chunk)
        return "".join(buf)

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

def _mcp_docker_cfg() -> Dict[str, Any]:
    return (CFG.get("mcp", {}).get("docker") or {})

# --- Liste complète des tools (avec schema) via Docker MCP ---
def mcp_list_tools_full_via_gateway() -> List[Dict[str, Any]]:
    """
    Récupère name/description/inputSchema pour chaque tool via Docker MCP Gateway (stdio).
    Retourne une liste vide si Docker MCP n'est pas accessible.
    """
    docker_cfg = _mcp_docker_cfg()
    if not docker_cfg.get("enabled", True):
        return []

    docker_cmd = docker_cfg.get("docker_cmd", "docker")

    # Utiliser Docker MCP Gateway via stdio
    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters

        async def _go():
            # Lancer docker mcp gateway run en subprocess stdio
            server_params = StdioServerParameters(
                command=docker_cmd,
                args=["mcp", "gateway", "run"],
                env=None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    return [dict(
                        name=t.name,
                        description=(t.description or ""),
                        inputSchema=(getattr(t, "inputSchema", None) or {})
                    ) for t in (resp.tools or [])]

        # Utiliser get_event_loop ou créer un nouveau selon le contexte
        try:
            loop = asyncio.get_running_loop()
            # Si un loop existe déjà, créer une tâche synchrone via thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _go())
                return future.result(timeout=65)
        except RuntimeError:
            # Pas de loop actif, utiliser asyncio.run normalement
            return asyncio.run(_go())
    except Exception as e:
        # Docker MCP non accessible : retourner liste vide (pas une erreur fatale)
        import sys
        print(f"[MCP] Docker MCP Gateway non accessible: {e}", file=sys.stderr)
        return []

def mcp_call_tool_via_gateway(tool: str, arguments: Dict[str, Any], timeout_s: int = 60) -> str:
    """Appelle un tool via Docker MCP Gateway (stdio). Mode chat uniquement."""
    docker_cfg = _mcp_docker_cfg()
    docker_cmd = docker_cfg.get("docker_cmd", "docker")

    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters

        async def _go():
            server_params = StdioServerParameters(
                command=docker_cmd,
                args=["mcp", "gateway", "run"],
                env=None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    try:
                        res = await asyncio.wait_for(session.call_tool(tool, arguments or {}), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        return "[MCP] Timeout d'exécution du tool."
                    return _fmt_call_result(res)

        return asyncio.run(_go())
    except Exception as e:
        return f"[MCP] Erreur Docker MCP: {e}"

# ----- Récupération "raw" d'un tool (structuredContent + textes) via Docker MCP -----
def _mcp_call_tool_raw(tool: str, arguments: Dict[str, Any], timeout_s: int = 60) -> Tuple[Any, List[str]]:
    docker_cfg = _mcp_docker_cfg()
    docker_cmd = docker_cfg.get("docker_cmd", "docker")

    try:
        from mcp import ClientSession, types as _t
        from mcp.client.stdio import stdio_client, StdioServerParameters

        async def _go():
            server_params = StdioServerParameters(
                command=docker_cmd,
                args=["mcp", "gateway", "run"],
                env=None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await asyncio.wait_for(session.call_tool(tool, arguments or {}), timeout=timeout_s)
                    sc = getattr(res, "structuredContent", None)
                    texts: List[str] = []
                    for c in getattr(res, "content", []) or []:
                        if isinstance(c, _t.TextContent):
                            texts.append(c.text)
                        elif isinstance(c, _t.EmbeddedResource):
                            uri = getattr(c.resource, "uri", "")
                            if uri:
                                texts.append(uri)
                    return sc, texts

        return asyncio.run(_go())

    except Exception:
        return None, []

# ----- Parcours générique et fix placeholders (nouveau helper) -----
def _walk_find_result_items(obj) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if any(k in obj for k in ("url", "href", "link")):
            items.append(obj)
        for v in obj.values():
            items += _walk_find_result_items(v)
    elif isinstance(obj, list):
        for it in obj:
            items += _walk_find_result_items(it)
    return items

def _fix_placeholder_urls(items: List[Dict[str, Any]], url_candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Si un item a 'url' (ou href/link) = '\1'/'\2'/... ou vide, on remappe :
      - si des indices différents sont présents -> mapping par indice
      - sinon -> mapping séquentiel (1er item -> 1ère URL, etc.)
    """
    placeholders = []
    for it in items:
        url = it.get("url") or it.get("href") or it.get("link") or ""
        m = re.fullmatch(r"\s*\\([0-9]+)\s*", str(url))
        placeholders.append(int(m.group(1)) if m else None)

    if any(p is not None for p in placeholders):
        uniq = {p for p in placeholders if p is not None}
        fixed = []
        if len(uniq) > 1:
            # mapping par indice
            for i, it in enumerate(items):
                new = dict(it)
                url = new.get("url") or new.get("href") or new.get("link") or ""
                m = re.fullmatch(r"\s*\\([0-9]+)\s*", str(url))
                if m:
                    idx = int(m.group(1))
                    val = url_candidates[idx-1] if 1 <= idx <= len(url_candidates) else ""
                    if "url" in new:
                        new["url"] = val
                    elif "href" in new:
                        new["href"] = val
                    else:
                        new["link"] = val
                fixed.append(new)
            return fixed
        else:
            # mapping séquentiel
            fixed = []
            for i, it in enumerate(items):
                new = dict(it)
                url = new.get("url") or new.get("href") or new.get("link") or ""
                if (not url) or re.fullmatch(r"\s*\\[0-9]+\s*", str(url)):
                    val = url_candidates[i] if i < len(url_candidates) else ""
                    if "url" in new:
                        new["url"] = val
                    elif "href" in new:
                        new["href"] = val
                    else:
                        new["link"] = val
                fixed.append(new)
            return fixed
    return items

def _format_ddg_items(items: List[Dict[str, Any]], max_items: int) -> str:
    out_lines = []
    for i, it in enumerate(items[:max_items], 1):
        title = it.get("title") or it.get("name") or it.get("text") or it.get("heading") \
                or (it.get("url") or it.get("href") or it.get("link") or "")
        url = it.get("url") or it.get("href") or it.get("link") or ""
        summary = it.get("snippet") or it.get("summary") or it.get("description") or it.get("body") or ""
        block = f"{i}. {title}\n   URL: {url}\n   Summary: {summary}".rstrip()
        out_lines.append(block)
    return "\n\n".join(out_lines) if out_lines else "(aucun résultat)"

# ----- Helpers URLs (mis à jour) -----
def _extract_http_urls_from_texts(texts: List[str]) -> List[str]:
    urls: List[str] = []
    for t in texts or []:
        for u in re.findall(r"https?://[^\s<>\]\"')]+", t):
            urls.append(u)
    # dédup en conservant l'ordre
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

def _replace_backref_urls(raw: str, urls: List[str]) -> str:
    indices = re.findall(r"URL:\s*\\([0-9]+)", raw, flags=re.IGNORECASE)
    if not indices:
        raw = re.sub(r"URL:\s*[\x00-\x1f]", "URL: (non fourni)", raw, flags=re.IGNORECASE)
        return raw

    if len(set(indices)) > 1:
        def repl(m):
            idx = int(m.group(1))
            return f"URL: {urls[idx-1]}" if 1 <= idx <= len(urls) else "URL: (non fourni)"
        raw = re.sub(r"URL:\s*\\([0-9]+)", repl, raw, flags=re.IGNORECASE)
    else:
        it = iter(urls)
        def repl_seq(_):
            try:
                return f"URL: {next(it)}"
            except StopIteration:
                return "URL: (non fourni)"
        raw = re.sub(r"URL:\s*\\[0-9]+", repl_seq, raw, flags=re.IGNORECASE)

    raw = re.sub(r"URL:\s*\\[0-9]+", "URL: (non fourni)", raw, flags=re.IGNORECASE)
    raw = re.sub(r"URL:\s*[\x00-\x1f]", "URL: (non fourni)", raw, flags=re.IGNORECASE)
    return raw

def _cleanup_found_header(raw: str) -> str:
    return re.sub(r"^\s*Found\s+\d+\s+search\s+results:\s*\n?", "", raw, flags=re.IGNORECASE|re.MULTILINE)

def ddg_search(query: str, topk: int = 5) -> str:
    try:
        sc, texts = _mcp_call_tool_raw("ddg__search", {"query": query, "max_results": int(topk)}, timeout_s=60)
        url_candidates = _extract_http_urls_from_texts(texts)

        # 1) JSON structuré exploitable
        if isinstance(sc, (dict, list)):
            items = _walk_find_result_items(sc)
            if items:
                items = _fix_placeholder_urls(items, url_candidates)

                def _is_ph(u: str) -> bool:
                    return (not u) or bool(re.fullmatch(r"\s*\\[0-9]+\s*", str(u)))

                # Si on a encore des placeholders (ou aucun candidat), fallback texte
                still_placeholders = any(
                    _is_ph(it.get("url") or it.get("href") or it.get("link") or "")
                    for it in items
                )
                if still_placeholders or not url_candidates:
                    raw = mcp_call_tool_via_gateway(
                        "ddg__search", {"query": query, "max_results": int(topk)}, timeout_s=60
                    ) or "(vide)"
                    raw = _replace_backref_urls(raw, url_candidates)
                    raw = _cleanup_found_header(raw)
                    return raw

                return _format_ddg_items(items, max_items=int(topk))

        # 2) Fallback texte brut + remplacement
        raw = "\n".join(texts) if texts else mcp_call_tool_via_gateway(
            "ddg__search", {"query": query, "max_results": int(topk)}, timeout_s=60
        )
        raw = raw or "(vide)"
        raw = _replace_backref_urls(raw, url_candidates)
        raw = _cleanup_found_header(raw)
        return raw

    except Exception as e:
        return f"[MCP/ddg__search] {e}"

def ddg_fetch(url: str) -> str:
    """Fetch contenu via le Gateway (tool: ddg__fetch_content)."""
    try:
        out = mcp_call_tool_via_gateway("ddg__fetch_content", {"url": url}, timeout_s=60)
        return out or "(vide)"
    except Exception as e:
        return f"[MCP/ddg__fetch_content] {e}"

# ---------- NOUVEAU : Augmenter la réponse vocale avec les résultats web identiques au chat ----------
def _augment_vocal_msg_if_needed(messages: List[Dict[str, str]]) -> None:
    """En mode vocal, si la dernière réponse assistant ne contient pas d'URL formatées,
    on ajoute un bloc '🔎 Résultats...' identique au mode chat via ddg_search()."""
    try:
        if st.session_state.get("interaction_mode") != "vocal":
            return

        docker = CFG.get("mcp", {}).get("docker", {})
        if not docker.get("enabled", True):
            return
        if not messages:
            return

        # Trouver dernier assistant et dernier user avant lui
        last_ass_idx = next((i for i in range(len(messages)-1, -1, -1)
                             if messages[i].get("role") == "assistant"), None)
        if last_ass_idx is None:
            return
        last_user_idx = next((i for i in range(last_ass_idx-1, -1, -1)
                              if messages[i].get("role") == "user"), None)
        if last_user_idx is None:
            return

        ass_text = messages[last_ass_idx].get("content", "") or ""
        # Si déjà des URLs formatées, on ne touche rien
        if re.search(r'\bURL:\s*https?://', ass_text):
            return

        # Anti-doublon (clé = question + longueur réponse)
        cache = st.session_state.setdefault("_vocal_aug_cache", set())
        user_q = (messages[last_user_idx].get("content") or "").strip()
        if not user_q:
            return
        cache_key = f"{user_q}|{len(ass_text)}"
        if cache_key in cache:
            return

        # Générer le bloc "pretty" identique au chat
        pretty = ddg_search(user_q, topk=int(docker.get("auto_web_topk", 5)))
        if pretty and pretty.strip():
            messages[last_ass_idx]["content"] = (ass_text.rstrip() +
                                                 "\n\n" +
                                                 f"🔎 Résultats pour « {user_q} »\n\n{pretty}").strip()
            cache.add(cache_key)
    except Exception as e:
        st.session_state.setdefault("agent_trace", []).append(
            {"role": "system", "content": f"[Vocal augment error] {e}"}
        )

# ---------- Agent (legacy) : détection JSON dans le texte ----------
_TOOL_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE | re.MULTILINE)
_BRACED_JSON_RE = re.compile(r"\{[\s\S]*\}")

def _extract_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """Extrait {\"tool\": \"...\", \"args\": {...}} depuis 'text' si présent."""
    if not text:
        return None
    candidate = None
    m = _TOOL_JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
    if not candidate:
        for mm in _BRACED_JSON_RE.finditer(text):
            blob = mm.group(0).strip()
            if 2 <= len(blob) <= 4000:
                candidate = blob
                break
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "tool" in data and isinstance(data.get("args", {}), dict):
            return data
    except Exception:
        return None
    return None

def maybe_run_tool_and_answer(cfg: Dict[str, Any], msgs: List[Dict[str, str]], llm_first_reply: str) -> Optional[str]:
    """
    (Legacy) Si 'llm_first_reply' contient un JSON d'appel d'outil, exécute l'outil via MCP,
    insère le résultat, puis relance le LLM. Retourne la réponse finale si un outil a été utilisé.
    """
    if not (cfg.get("llm", {}).get("agent_enabled", False)):
        return None
    max_rounds = int(cfg.get("llm", {}).get("agent_max_rounds", 2))
    used_any_tool = False
    current_reply = llm_first_reply

    for _ in range(max_rounds):
        call = _extract_tool_json(current_reply or "")
        if not call:
            break
        tool_name = str(call.get("tool", "")).strip()
        args = call.get("args", {}) or {}
        if not tool_name:
            break
        used_any_tool = True
        msgs.append({"role": "system", "content": f"[Agent] Appel outil demandé : {tool_name} {json.dumps(args, ensure_ascii=False)}"})
        out = mcp_call_tool_via_gateway(tool_name, args, timeout_s=90)
        msgs.append({"role": "system", "content": f"[Résultat outil: {tool_name}]\n{out}"})
        msgs.append({"role": "system", "content": "À partir du résultat d'outil ci-dessus, réponds brièvement en français et cite les URLs si pertinentes."})
        current_reply = ollama_reply(cfg, msgs)

    if used_any_tool:
        msgs.append({"role": "assistant", "content": current_reply or "(réponse vide)"})
        return current_reply or ""
    return None

# ---------- NOUVEAU : Tool-calling natif Ollama (boucle d'agent) ----------
def build_ollama_tools_from_mcp() -> List[Dict[str, Any]]:
    """
    Transforme les tools MCP (name/description/inputSchema) en liste Ollama 'tools'.
    """
    tools = []
    for t in mcp_list_tools_full_via_gateway() or []:
        name = t.get("name") or ""
        if not name:
            continue
        desc = t.get("description") or ""
        schema = t.get("inputSchema") or {}
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema if isinstance(schema, dict) else {}
            }
        }
        tools.append(tool)
    return tools

def chat_with_tools_agent(cfg: Dict[str, Any], user_text: str, ui_msgs: List[Dict[str, str]]) -> Optional[str]:
    """
    Boucle d'agent Tool-Calling:
      - Passe la liste des tools à Ollama
      - Laisse le modèle décider d'appeler 0..N tools
      - Exécute chaque tool_call via MCP puis renvoie les résultats (role='tool')
      - Reboucle jusqu'à absence de tool_calls ou max_rounds atteint
    Retourne la réponse finale (texte) ou None si échec.
    NOTE: ui_msgs sert de buffer de traces (role='system'), invisible côté UI.
    """
    if not cfg.get("llm", {}).get("use_ollama_tools", True):
        return None
    host = cfg["ollama"]["host"]
    model = cfg["ollama"]["model"]
    if not is_ollama_up(host):
        return None

    tools = build_ollama_tools_from_mcp()
    if not tools:
        return None  # pas de tools disponibles

    max_rounds = int(cfg.get("llm", {}).get("agent_max_rounds", 3))
    options = {
        "temperature": float(cfg["ollama"].get("temperature", 0.7)),
        "num_ctx": int(cfg["ollama"].get("num_ctx", 4096)),
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": (cfg.get("llm", {}).get("system_prompt") or "You are Jarvis.")},
        {"role": "user", "content": user_text},
    ]

    url = host.rstrip("/") + "/api/chat"
    final_answer = None

    for _ in range(max_rounds):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": options,
        }
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json() or {}
        except Exception as e:
            ui_msgs.append({"role": "system", "content": f"[Agent] Erreur appel Ollama: {e}"})
            return None

        msg = (data.get("message") or {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if tool_calls:
            for call in tool_calls:
                fn = (call.get("function") or {})
                tname = str(fn.get("name") or "").strip()
                targs = fn.get("arguments") or {}
                ui_msgs.append({"role": "system", "content": f"[Agent] Appel tool: {tname} args={json.dumps(targs, ensure_ascii=False)}"})
                try:
                    result = mcp_call_tool_via_gateway(tname, targs, timeout_s=90)
                except Exception as e:
                    result = f"[MCP] Erreur exécution tool {tname}: {e}"
                messages.append({"role": "tool", "tool_name": tname, "content": str(result)})
                ui_msgs.append({"role": "system", "content": f"[Résultat tool {tname}]\n{(str(result) or '(vide)')[:8000]}"})
            continue

        final_answer = content or ""
        break

    if final_answer is None:
        final_answer = content or ""

    return final_answer or None

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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "username" not in st.session_state:
    st.session_state.username = "User"

def _set_interaction_mode(mode: str):
    mode = "vocal" if mode == "vocal" else "chat"
    if mode != st.session_state.interaction_mode and mode == "vocal":
        st.session_state["chat_input"] = ""
    st.session_state.interaction_mode = mode
    st.session_state.mode_toggle = mode == "vocal"
    # ⬇️ NOUVEAU : pilote le micro backend en même temps
    _send_mic_state(mode == "vocal")

def _sync_mode_from_toggle():
    desired = "vocal" if st.session_state.get("mode_toggle") else "chat"
    _set_interaction_mode(desired)

def _on_mode_toggle():
    # Le widget est la vérité : on pousse l'état vers interaction_mode
    is_vocal = bool(st.session_state.get("mode_toggle"))
    _set_interaction_mode("vocal" if is_vocal else "chat")
    # Si tu as implémenté la commande micro :
    try:
        _send_mic_state(is_vocal)   # optionnel mais recommandé
    except Exception:
        pass

def _load_jarvis_module(path: str):
    spec = importlib.util.spec_from_file_location("jarvis", path)
    if not spec or not spec.loader:
        raise RuntimeError("Impossible de charger jarvis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def start_jarvis(cfg: Dict[str, Any]):
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
        # ⬇️ NOUVEAU : applique l'état micro selon le mode actuel
        _send_mic_state(st.session_state.get("interaction_mode", "chat") == "vocal")
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

def drain_jarvis_logs(max_keep: int = 800) -> List[str]:
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

# ---------- Ingestion chat : NOUVELLE VERSION ----------
CHAT_PREFIX_USER = re.compile(r'^\[(?:STT|VOICE)\]\s*(?:Vous|You)\s*:\s*(.+)$', re.IGNORECASE)
CHAT_PREFIX_ASSISTANT = re.compile(r'^(?:JARVIS|Assistant)\s*:\s*(.+)$', re.IGNORECASE)
NON_CHAT_PREFIX = re.compile(r'^\[(?:VU|AUDIO|TTS|LLM|BG|INFO|WARN|ERR|DEBUG|MCP)\]', re.IGNORECASE)

def _append_assistant_line(messages: List[Dict[str, str]], text: str) -> None:
    text = "" if text is None else str(text)
    if messages and messages[-1].get("role") == "assistant":
        prev = messages[-1]["content"]
        sep = "\n" if (prev and text and not prev.endswith("\n")) else ""
        messages[-1]["content"] = (prev + sep + text).rstrip()
    else:
        messages.append({"role": "assistant", "content": text.rstrip()})

def _ingest_chat_from_logs(logs: List[str], *, rerun: bool = True):
    if not logs:
        return
    msgs = st.session_state.setdefault("messages", [])
    assembling = bool(st.session_state.get("_assembling_assistant", False))
    changed = False

    for raw in logs:
        line = (raw or "").rstrip("\n")
        if line == "":
            if assembling:
                _append_assistant_line(msgs, "")
                changed = True
            continue

        m_user = CHAT_PREFIX_USER.match(line)
        if m_user:
            assembling = False
            content = m_user.group(1).strip()
            if content:
                msgs.append({"role": "user", "content": content})
                changed = True
            continue

        m_ass = CHAT_PREFIX_ASSISTANT.match(line)
        if m_ass:
            assembling = True
            _append_assistant_line(msgs, m_ass.group(1).strip())
            changed = True
            continue

        if NON_CHAT_PREFIX.match(line):
            assembling = False
            continue

        if assembling:
            _append_assistant_line(msgs, line)
            changed = True

    st.session_state["_assembling_assistant"] = assembling
    if changed:
        # Ajout "pretty" en mode vocal si la réponse est juste un résumé
        if st.session_state.get("interaction_mode") == "vocal":
            _augment_vocal_msg_if_needed(msgs)

        st.session_state["messages"] = msgs[-200:]
        if rerun:
            st.rerun()

# ---------- Orchestration message (chat & voix) ----------
def process_user_text(clean_text: str, origin: str = "chat", already_appended: bool = False):
    """
    Orchestrateur des réponses.
    - already_appended=True si le message utilisateur a déjà été poussé dans st.session_state['messages']
      (pour afficher tout de suite la bulle utilisateur + un placeholder '…' avant les appels réseau).
    - Les contextes (DDG, traces agent) sont injectés comme role='system' (non affichés dans l'UI).
    """
    msgs = st.session_state.setdefault("messages", [])
    agent_trace = st.session_state.setdefault("agent_trace", [])
    docker = CFG.get("mcp", {}).get("docker", {})
    used_gateway = False

    # Ajouter le message user si pas déjà ajouté par le handler d'UI
    if not already_appended:
        msgs.append({"role": "user", "content": clean_text})

    # (0) Tool-calling natif (si dispo ET mode chat uniquement)
    # En mode vocal, on désactive les outils MCP pour éviter les conflits asyncio
    if CFG.get("llm", {}).get("use_ollama_tools", True) and origin == "chat":
        final = chat_with_tools_agent(CFG, clean_text, agent_trace)
        if isinstance(final, str) and final.strip():
            msgs.append({"role": "assistant", "content": final})
            st.session_state["messages"] = msgs[-200:]
            st.session_state.last_assistant_ts = time.time()
            st.session_state.assistant_speaking = True
            return

    # (1) Raccourcis (/web, /fetch, /tool) — AFFICHAGE DIRECT des résultats (pas de résumé)
    if origin == "chat" and docker.get("chat_shortcuts", True) and docker.get("enabled", True):
        m_ddg  = re.match(r"^/(?:web|ddg)\s+(.+)$", clean_text, flags=re.IGNORECASE)
        m_fetch = re.match(r"^/fetch\s+(\S+)$", clean_text, flags=re.IGNORECASE)
        m_tool  = re.match(r"^/tool\s+([A-Za-z0-9_.:\-]+)\s*(.*)$", clean_text)

        if m_ddg:
            query = m_ddg.group(1).strip()
            pretty = ddg_search(query, topk=int(docker.get('auto_web_topk',5)))
            msgs.append({"role": "assistant", "content": f"🔎 Résultats pour « {query} »\n\n{pretty}"})
            st.session_state["messages"] = msgs[-200:]
            st.session_state.last_assistant_ts = time.time()
            st.session_state.assistant_speaking = True
            return

        elif m_fetch:
            url = m_fetch.group(1).strip()
            out = ddg_fetch(url)
            msgs.append({"role": "assistant", "content": out})
            st.session_state["messages"] = msgs[-200:]
            st.session_state.last_assistant_ts = time.time()
            st.session_state.assistant_speaking = True
            return

        elif m_tool:
            tool_name = m_tool.group(1)
            arg_str   = (m_tool.group(2) or "").strip()
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
                st.session_state["messages"] = msgs[-200:]
                st.session_state.last_assistant_ts = time.time()
                st.session_state.assistant_speaking = True
                return

            out = mcp_call_tool_via_gateway(tool_name, args, timeout_s=60)
            msgs.append({"role": "assistant", "content": out or "(vide)"})
            st.session_state["messages"] = msgs[-200:]
            st.session_state.last_assistant_ts = time.time()
            st.session_state.assistant_speaking = True
            return

    # (2) Auto-web (si activé) — AFFICHAGE DIRECT 'pretty'
    if docker.get("enabled", True) and docker.get("auto_web", False) and not used_gateway:
        pretty = ddg_search(clean_text, topk=int(docker.get("auto_web_topk", 5)))
        msgs.append({"role": "assistant", "content": f"🔎 Résultats (auto) pour « {clean_text} »\n\n{pretty}"})
        st.session_state["messages"] = msgs[-200:]
        st.session_state.last_assistant_ts = time.time()
        st.session_state.assistant_speaking = True
        return

    # (3) LLM pur
    if not used_gateway:
        reply = ollama_reply(CFG, msgs)
        final = maybe_run_tool_and_answer(CFG, msgs, reply)
        if final is None:
            msgs.append({"role": "assistant", "content": reply or "Réponse vide d'Ollama (vérifie le modèle)."})

    st.session_state["messages"] = msgs[-200:]
    st.session_state.last_assistant_ts = time.time()
    st.session_state.assistant_speaking = True

new_logs = drain_jarvis_logs() if st.session_state.jarvis_running else []
if new_logs:
    _update_speaking_state_from_logs(new_logs)
    _ingest_chat_from_logs(new_logs, rerun=False)

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

# ---------- Helper pour afficher le radar ----------
# Fonction render_radar supprimée - remplacée par image statique

# =============================================================================
# LAYOUT JARVIS - NO TABS, 2 COLUMNS
# =============================================================================

# Header - Espace réduit de 75% (10px → 2.5px)
st.markdown("""
<div style="text-align: center; padding: 2.5px 0;">
    <h1 style="font-size: 48px; margin: 0;">J.A.R.V.I.S.</h1>
    <p style="color: var(--jarvis-muted); font-size: 14px; letter-spacing: 0.2em; text-transform: uppercase;">
        Just A Rather Very Intelligent System
    </p>
</div>
""", unsafe_allow_html=True)

# Two-column layout: Radar (left) + Chat (right)
col_radar, col_chat = st.columns([1, 3])

with col_radar:
    # Image JARVIS (remplace le radar vocal)
    st.image("assets/jarvis.png", width="stretch")

    # Controls (toggles + clear button)
    st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(45,212,255,.25), transparent);'>", unsafe_allow_html=True)

    st.toggle("🎙️ Mode Vocal", key="mode_toggle", on_change=_on_mode_toggle)

    docker_enabled_state = CFG["mcp"]["docker"].get("enabled", True)
    st.toggle("🐳 Docker MCP", key="docker_toggle", value=docker_enabled_state)
    CFG["mcp"]["docker"]["enabled"] = bool(st.session_state.get("docker_toggle", docker_enabled_state))

    st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(45,212,255,.25), transparent);'>", unsafe_allow_html=True)

    if st.button("🗑️ Effacer l'historique", width="stretch"):
        st.session_state.messages = []
        st.rerun()

with col_chat:
    # Chat zone
    chat_container = st.container()

    with chat_container:
        # Render messages with JARVIS style
        if st.session_state.messages:
            st.markdown(
                render_chat_messages(
                    st.session_state.messages[-50:],
                    username=st.session_state.username
                ),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                render_chat_messages([], username=st.session_state.username),
                unsafe_allow_html=True
            )

    # Input zone at bottom
    st.markdown("<br>", unsafe_allow_html=True)

    mode = st.session_state.interaction_mode
    placeholder_text = (
        "🎙️ MODE VOCAL ACTIF"
        if mode == "vocal"
        else "Écris un message (/tool nom {json}, /web requête, /fetch URL)"
    )

    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([10, 1])

        with cols[0]:
            user_input = st.text_area(
                "Message",
                placeholder=placeholder_text,
                height=45,
                label_visibility="collapsed",
                disabled=(mode == "vocal"),
                key="user_input"
            )

        with cols[1]:
            submitted = st.form_submit_button(
                "📨",
                width="stretch",
                help="Envoyer le message",
                disabled=(mode == "vocal")
            )

    if submitted and user_input and user_input.strip() and mode == "chat":
        clean_text = user_input.strip()

        # Ajouter le message utilisateur
        st.session_state["messages"].append({"role": "user", "content": clean_text})

        # Process the message AVANT tout rerun
        process_user_text(clean_text, origin="chat", already_appended=True)

        # Remove placeholder "…" (ajouté par process_user_text si nécessaire)
        msgs_now = st.session_state.get("messages", [])
        for i in range(len(msgs_now)-1, -1, -1):
            if msgs_now[i].get("role") == "assistant" and msgs_now[i].get("content") == "…":
                msgs_now.pop(i)
                break
        st.session_state["messages"] = msgs_now

        st.rerun()

    # Poll logs if Jarvis is running
    if st.session_state.get("jarvis_running"):
        mode = st.session_state.get("interaction_mode", "chat")
        idle_timeout = 120.0 if mode == "vocal" else 6.0
        poll_interval = 0.10 if mode == "vocal" else 0.15
        last_activity = time.time()

        while True:
            logs = drain_jarvis_logs()
            if logs:
                _update_speaking_state_from_logs(logs)
                _ingest_chat_from_logs(logs, rerun=False)
                last_activity = time.time()
                st.rerun()

            assembling = bool(st.session_state.get("_assembling_assistant", False))
            if (time.time() - last_activity > idle_timeout) and not assembling:
                break

            time.sleep(poll_interval)

# =============================================================================
# SIDEBAR - SETTINGS (like in demo)
# =============================================================================

@st.cache_data(ttl=60)
def get_ollama_models(host: str):
    """Wrapper around fetch_ollama_models with caching"""
    return fetch_ollama_models(host)

@st.cache_data
def get_piper_voices(base_dir: str = "~/.jarvis/voices"):
    """Get list of Piper voices"""
    try:
        voice_dir = Path(base_dir).expanduser()
        if voice_dir.exists():
            voices = list(voice_dir.glob("*.onnx"))
            return [str(v) for v in voices] if voices else ["~/.jarvis/voices/fr_FR-siwis-medium.onnx"]
        return ["~/.jarvis/voices/fr_FR-siwis-medium.onnx"]
    except:
        return ["~/.jarvis/voices/fr_FR-siwis-medium.onnx"]

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ---- Jarvis Backend ----
    with st.expander("🎙️ Jarvis (Backend)", expanded=False):
        current_path = CFG["jarvis"].get("path", "./jarvis.py")
        jarvis_path = st.text_input("Chemin jarvis.py", value=current_path)
        CFG["jarvis"]["path"] = jarvis_path.strip() or current_path

        CFG["jarvis"]["audio_out"] = st.text_input(
            "Préférence sortie audio (AUDIO_OUT)",
            value=CFG["jarvis"].get("audio_out", "analog"),
            help="Chaîne partielle du nom du périphérique (ex: analog, speaker, hdmi). Laisser vide pour auto.",
        ).strip()

        tts_options = ["piper", "fallback"]
        current_engine = CFG["jarvis"].get("tts_engine", "piper")
        idx_engine = tts_options.index(current_engine) if current_engine in tts_options else 0
        CFG["jarvis"]["tts_engine"] = st.selectbox("Moteur TTS", tts_options, index=idx_engine)

        CFG["jarvis"]["tts_lang"] = st.text_input(
            "Langue TTS (fallback)", value=CFG["jarvis"].get("tts_lang", "fr")
        ).strip() or "fr"

        CFG["jarvis"]["wake_word"] = st.text_input(
            "Mot d'activation",
            value=CFG["jarvis"].get("wake_word", "jarvis"),
        ).strip() or "jarvis"

        CFG["jarvis"]["require_wake"] = st.checkbox(
            "Nécessite mot d'activation",
            value=bool(CFG["jarvis"].get("require_wake", True)),
        )

        CFG["jarvis"]["wake_fuzzy_score"] = st.slider(
            "Score fuzzy matching",
            0, 100,
            CFG["jarvis"].get("wake_fuzzy_score", 80),
            5,
            help="Tolérance pour la détection du mot (80 = bon équilibre)"
        )

        CFG["jarvis"]["wake_aliases"] = st.text_area(
            "Alias supplémentaires (un par ligne ou séparés par des virgules)",
            value=CFG["jarvis"].get("wake_aliases", ""),
            height=80,
        ).strip()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Démarrer"):
                start_jarvis(CFG)
        with col2:
            if st.button("⏹️ Arrêter"):
                stop_jarvis()
        with col3:
            if st.button("♻️ Recharger"):
                st.session_state.jarvis_mod = None
                st.success("Module rechargé.")

        if st.session_state.jarvis_running:
            drain_jarvis_logs()
        st.code("\n".join(st.session_state.get("jarvis_logbuf", [])) or "—", language="bash")

    # ---- Whisper (STT) ----
    with st.expander("🎤 Whisper (STT)", expanded=False):
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        current_whisper = CFG["whisper"]["model"]
        whisper_idx = whisper_models.index(current_whisper) if current_whisper in whisper_models else 2

        CFG["whisper"]["model"] = st.selectbox(
            "Modèle Whisper",
            whisper_models,
            index=whisper_idx
        )

        CFG["whisper"]["lang"] = st.text_input(
            "Langue",
            value=CFG["whisper"]["lang"]
        )

        CFG["whisper"]["vad"] = st.checkbox(
            "VAD (Voice Activity Detection)",
            value=CFG["whisper"]["vad"]
        )

        device_options = ["cpu", "cuda"]
        device_idx = device_options.index(CFG["whisper"]["device"]) if CFG["whisper"]["device"] in device_options else 0
        CFG["whisper"]["device"] = st.selectbox(
            "Device",
            device_options,
            index=device_idx
        )

        compute_options = ["int8", "float16", "float32"]
        compute_idx = compute_options.index(CFG["whisper"]["compute"]) if CFG["whisper"]["compute"] in compute_options else 0
        CFG["whisper"]["compute"] = st.selectbox(
            "Compute Type",
            compute_options,
            index=compute_idx
        )

    # ---- Piper (TTS) ----
    with st.expander("🗣️ Piper (TTS)", expanded=False):
        voices = get_piper_voices()
        current_voice = CFG["piper"]["voice"]
        voice_idx = voices.index(current_voice) if current_voice in voices else 0

        CFG["piper"]["voice"] = st.selectbox(
            "Voix Piper",
            voices,
            index=voice_idx,
            help="Fichiers .onnx dans ~/.jarvis/voices"
        )

        CFG["piper"]["speed"] = st.slider(
            "Vitesse (length_scale)",
            0.5, 1.5,
            CFG["piper"]["speed"],
            0.05
        )

        CFG["piper"]["noise"] = st.slider(
            "Noise Scale",
            0.0, 1.0,
            CFG["piper"]["noise"],
            0.01
        )

        CFG["piper"]["noise_w"] = st.slider(
            "Noise W",
            0.0, 1.0,
            CFG["piper"]["noise_w"],
            0.01
        )

        CFG["piper"]["sentence_silence"] = st.slider(
            "Silence entre phrases (s)",
            0.0, 0.5,
            CFG["piper"]["sentence_silence"],
            0.01
        )

        CFG["piper"]["use_cuda"] = st.checkbox(
            "Utiliser CUDA (GPU)",
            value=CFG["piper"]["use_cuda"],
            help="⚠️ Nécessite onnxruntime-gpu"
        )

    # ---- Ollama (LLM) ----
    with st.expander("🧠 Ollama (LLM)", expanded=False):
        CFG["ollama"]["host"] = st.text_input(
            "Host Ollama",
            value=CFG["ollama"]["host"]
        )

        # Get models list
        models = get_ollama_models(CFG["ollama"]["host"])
        current_model = CFG["ollama"]["model"]

        # Add current model if not in list
        if current_model not in models:
            models.append(current_model)

        model_idx = models.index(current_model) if current_model in models else 0

        CFG["ollama"]["model"] = st.selectbox(
            "Modèle LLM",
            models,
            index=model_idx,
            help="Modèles installés sur Ollama"
        )

        # Refresh button
        if st.button("🔄 Rafraîchir les modèles"):
            st.cache_data.clear()
            st.rerun()

        CFG["ollama"]["temperature"] = st.slider(
            "Température",
            0.0, 2.0,
            CFG["ollama"]["temperature"],
            0.05,
            help="Créativité: 0 = déterministe, 2 = très créatif"
        )

        CFG["ollama"]["num_ctx"] = st.number_input(
            "Context Window",
            min_value=512,
            max_value=32768,
            value=CFG["ollama"]["num_ctx"],
            step=512,
            help="Nombre de tokens de contexte"
        )

        CFG["ollama"]["stream"] = st.checkbox(
            "Mode Streaming",
            value=CFG["ollama"]["stream"],
            help="Afficher les tokens au fur et à mesure"
        )

    # ---- Docker MCP Toolkit ----
    with st.expander("🐳 Docker MCP Toolkit", expanded=False):
        st.markdown("""
        **Docker MCP Toolkit** : Système MCP natif avec isolation par conteneurs.
        - Serveurs installés : `context7`, `fetch`, `mcp-python-refactoring`, `time`
        - Communication via `docker mcp gateway run` (stdio)
        """)

        CFG["mcp"]["docker"]["enabled"] = st.checkbox(
            "Docker MCP activé",
            value=CFG["mcp"]["docker"]["enabled"],
            help="Active le Docker MCP Toolkit pour accéder aux serveurs MCP containerisés"
        )

        CFG["mcp"]["docker"]["docker_cmd"] = st.text_input(
            "Commande Docker",
            value=CFG["mcp"]["docker"]["docker_cmd"],
            help="Chemin vers la commande docker (par défaut: 'docker')"
        )

        # Afficher la liste des serveurs MCP installés
        if st.button("🔍 Lister les serveurs MCP"):
            try:
                import subprocess
                result = subprocess.run(
                    ["docker", "mcp", "server", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    st.success(f"**Serveurs installés:**\n```\n{result.stdout}\n```")
                else:
                    st.error(f"Erreur: {result.stderr}")
            except Exception as e:
                st.error(f"Docker MCP non disponible: {e}")

        CFG["mcp"]["docker"]["auto_web"] = st.checkbox(
            "Auto Web Search",
            value=CFG["mcp"]["docker"]["auto_web"],
            help="Activer la recherche web automatique"
        )

        CFG["mcp"]["docker"]["chat_shortcuts"] = st.checkbox(
            "Raccourcis chat (/web, /fetch, /tool)",
            value=CFG["mcp"]["docker"]["chat_shortcuts"],
            help="Active les commandes rapides dans le chat"
        )

    st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(45,212,255,.25), transparent);'>", unsafe_allow_html=True)

    # Save button
    if st.button("💾 Sauvegarder la config", width="stretch"):
        save_cfg(CFG)
        st.info(f"📊 Modèle actuel: {CFG['ollama']['model']}")

    st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(45,212,255,.25), transparent);'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 20px; opacity: 0.6;">
        <p style="font-size: 12px; letter-spacing: 0.1em;">
            J.A.R.V.I.S. v2.0<br>
            Architecture Hybride CPU/GPU<br>
            © 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 40px 20px; opacity: 0.4;">
    <p style="font-size: 12px; letter-spacing: 0.15em; text-transform: uppercase;">
        Powered by Ollama · Whisper · Piper TTS
    </p>
</div>
""", unsafe_allow_html=True)
