# Jarvis — Streamlit UI (shadcn-like)

![Interface – Jarvis UI](docs/interface.png)

> _Assistant vocal + chat local avec Whisper/Faster-Whisper, Piper TTS, Ollama et outils MCP (DuckDuckGo, fetch, etc.)._

---

## ✨ Fonctionnalités

- **Interface moderne** (Streamlit) avec topbar d’état (WHISPER, PIPER, OLLAMA, MCP, ONNXRT).
- **Radar vocal** animé (SVG) rendu via un composant **HTML/JS** isolé. Les ondes se déclenchent quand l’assistant « parle » (basé sur les logs), prêt à être branché sur un vrai **niveau RMS**.
- **Chat + Raccourcis** : `/web`, `/ddg`, `/fetch`, `!tool <name> {json}`.
- **Websearch MCP** (client stdio éphémère) : DuckDuckGo search/fetch, auto-web option.
- **Contrôles audio** : préférences TTS (Piper), langue, vitesse, speaker id, CUDA on/off.
- **Gestion des serveurs MCP** : démarrer/stopper, lister les tools, exécuter un tool.
- **Ollama** : test de connexion, pull/warmup, choix de modèle.
- **Persistance** : configuration stockée sous `~/.jarvis/ui_config.json`.

---

## 🧰 Prérequis

- Python **3.11** recommandé
- **Ollama** (local) pour les LLMs
- **Piper** (pour TTS) + voix `.onnx`
- (Optionnel) **modelcontextprotocol / mcp** pour le client MCP Python
- Node.js (si vous utilisez le **proxy MCP** `adamwattis_mcp-proxy-server`)

---

## 🚀 Installation

```bash
# 1) Créez un venv
python -m venv .venv
source .venv/bin/activate  # ou .venv\\Scripts\\activate sous Windows

# 2) Dépendances Python
pip install streamlit requests onnxruntime  # + mcp si besoin: pip install mcp

# 3) Lancer l'UI
streamlit run app_4.py
```

> **Astuce** : si vous utilisez **onnxruntime-gpu**, assurez-vous d’avoir les bibliothèques CUDA/cuDNN compatibles. Sinon, laissez l’option **Piper → CUDA** désactivée et utilisez `onnxruntime` CPU.

---

## 🗂️ Structure rapide

```
.
├── app_4.py                 # Cette application Streamlit
├── jarvis.py                # Backend audio local (appelé par l’UI)
├── README.md                # (ce fichier)
└── docs/
    └── interface.png        # Capture d’écran pour GitHub
```

---

## ⚙️ Configuration & Persistance

Un fichier est créé/lu en **JSON** :

- **Chemin** : `~/.jarvis/ui_config.json`
- **Extrait** (valeurs par défaut) :

```jsonc
{
  "whisper": {"model": "small", "lang": "fr", "vad": true, "device": "cpu", "compute": "int8"},
  "piper": {"base_dir": "~/.jarvis/voices", "voice": "", "speaker_id": 0, "speed": 0.9,
             "noise": 0.667, "noise_w": 0.8, "sentence_silence": 0.10, "use_cuda": false},
  "ollama": {"host": "http://127.0.0.1:11434", "model": "qwen2.5:latest", "temperature": 0.7,
             "num_ctx": 4096, "stream": false},
  "mcp": {"servers": [],
           "proxy": {"enabled": true, "command": "node",
                      "args": ["/ABSOLU/adamwattis_mcp-proxy-server/build/index.js"],
                      "env": {}, "tool_timeout_s": 60, "chat_shortcuts": true,
                      "auto_web": false, "auto_web_topk": 5}},
  "jarvis": {"path": "./jarvis.py", "audio_out": "analog", "tts_engine": "piper",
             "tts_lang": "fr", "wake_word": "jarvis", "wake_aliases": "",
             "require_wake": true, "wake_fuzzy": true, "wake_fuzzy_score": 80}
}
```

L’UI exporte les principaux paramètres en **variables d’environnement** (exemples) pour le backend :

- `PIPER_VOICE`, `PIPER_LENGTH`, `PIPER_NOISE`, `PIPER_NOISE_W`, `PIPER_SENT_SIL`, `PIPER_CUDA`, `PIPER_SPEAKER_ID`
- `FW_MODEL`, `FW_DEVICE`, `FW_COMPUTE`, `FW_LANGUAGE`, `FW_VAD_CMD`
- `WAKE_WORD`, `REQUIRE_WAKE`, `WAKE_FUZZY`, `WAKE_FUZZY_SCORE`, `AUDIO_OUT`
- `OLLAMA_HOST`, `LLM_ID`, `OLLAMA_TEMPERATURE`, `OLLAMA_NUM_CTX`, `OLLAMA_STREAM`
- `MCP_SERVERS_JSON`

---

## 🗣️ Radar vocal (HTML/JS)

Le composant est rendu via `st.components.v1.html` pour éviter l’échappement du SVG et autoriser l’animation CSS. Il expose :

- **Mode** : `vocal` / `chat` (étiquette sous le radar)
- **Speaking** : bascule automatique pendant ~6s après un message de l’assistant (branché sur les logs). Vous pouvez le connecter facilement à un **niveau RMS** (WebSocket ou audio worker) pour une réactivité en temps réel.

---

## 🔎 Websearch MCP (proxy)

- L’UI peut lancer un client MCP **stdio** éphémère pointant vers le **proxy** `adamwattis_mcp-proxy-server` (champ *command* = `node`, *args* = chemin absolu vers `build/index.js`).
- Renseignez l’`ENV` si nécessaire (ex. `MCP_CONFIG_PATH=/ABSOLU/.../config.json`).
- Raccourcis de chat :
  - `/web requête` ou `/ddg requête`
  - `/fetch URL`
  - `!tool <nom> {json}`

---

## 🧪 Démos rapides

- **Tester Ollama** : onglet *Ollama* → *Tester connexion* / *Pull modèle* / *Warm-up*.
- **Piper** : téléversez une voix `.onnx` et (optionnellement) le `.json` associé, choisissez-la dans la liste.
- **MCP** : ajoutez un serveur, démarrez-le, *Lister tools* ou *Appeler un tool*.

---

## 🛠️ Dépannage (FAQ courte)

### ONNXRuntime / CUDA introuvable (Piper)
- Symptôme : erreurs `libcublasLt.so.11` ou `CUDA_PATH is set but CUDA wasn’t able to be loaded`.
- Solutions :
  1) Désactiver **Piper → CUDA** (CPU), ou
  2) Installer la pile CUDA/cuDNN compatible avec `onnxruntime-gpu`, ou
  3) Utiliser `onnxruntime` (CPU) au lieu de la variante GPU.

### Node ESM / `require is not defined`
- Si votre wrapper JS est traité en **ESM**, utilisez `import`/`export` ou renommez le wrapper en `.cjs`. L’UI appelle directement `build/index.js` avec `node`.

### JSONRPC: "Failed to parse" (MCP)
- Vérifiez que le binaire proxy n’écrit **que du JSON** sur stdout (pas de lignes de log comme `Connecting to ...`).

---

## 🔒 Sécurité

L’UI peut démarrer des **processus locaux** (MCP, Jarvis backend) et exécuter des tools via MCP. Utilisez-la sur une machine de confiance.

---

## 📝 Licence

MIT — voir `LICENSE`.

---

## 🤝 Contribuer

Issues et PR bienvenues ! Merci d’indiquer votre OS, version de Python, version d’Ollama, et les logs pertinents.

