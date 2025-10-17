# Jarvis — UI Streamlit pour assistant vocal local

![Interface – Jarvis UI](docs/interface.png)

Jarvis est une interface Streamlit qui pilote un assistant vocal local. Elle orchestre la reconnaissance vocale (Whisper / Faster-Whisper), la synthèse vocale (Piper), la génération via Ollama et un gateway Model Context Protocol (MCP) pour appeler des tools. Tout est pensé pour un usage sur machine personnelle : aucune donnée ne quitte votre poste.

---

## ✨ Points clés

- **Topbar de statut** : visualisez en un coup d'œil l'état de Whisper, Piper, Ollama, MCP Gateway et ONNX Runtime.
- **Radar vocal animé** : composant HTML/JS isolé rendu via `st.components.v1.html` qui réagit aux réponses de l'assistant et au mode (chat ou vocal).
- **Chat enrichi** : conversation texte avec mémoire, raccourci `/tool <nom> {json}` pour appeler un tool MCP et déclenchement du radar lors des réponses.
- **Mode vocal** : basculez en un clic vers une interaction mains libres, synchronisée avec le backend `jarvis.py`.
- **Pilotage du backend local** : démarrez/arrêtez `jarvis.py`, synchronisez les paramètres (variables d'environnement) et consultez ses logs.
- **Ollama intégré** : test de connectivité, téléchargement (`ollama pull`) et warmup de modèles, plus chat direct via l'API `/api/chat`.
- **MCP Gateway (HTTP / SSE)** : compatibilité avec MCPJungle Gateway, listing et appel des tools via HTTP streamable ou SSE.
- **Persistance automatique** : configuration sauvegardée sous `~/.jarvis/ui_config.json` et rechargée au lancement.

---

## 🧰 Prérequis

- Python **3.11** (recommandé).
- [Ollama](https://ollama.com/) installé localement avec les modèles souhaités.
- [Piper TTS](https://github.com/rhasspy/piper) et au moins une voix `.onnx` (copiée dans `~/.jarvis/voices`).
- (Optionnel) [Model Context Protocol](https://github.com/modelcontextprotocol) pour utiliser le gateway MCP.
- Accès audio (micro + sortie) si vous exploitez le mode vocal via `jarvis.py`.

Toutes les dépendances Python nécessaires sont listées dans `requirements.txt` (Streamlit, Faster-Whisper, Piper-TTS, Ollama SDK, MCP SDK, etc.).

---

## 🚀 Installation rapide

```bash
# 1) Cloner le dépôt
git clone https://github.com/<votre_compte>/Streamlit_JARVIS.git
cd Streamlit_JARVIS

# 2) Créer un environnement virtuel (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate

# 3) Installer les dépendances Python
pip install --upgrade pip
pip install -r requirements.txt

# 4) Lancer l'interface Streamlit
streamlit run app.py
```

> 💡 **CUDA / GPU** : le projet s'appuie sur `onnxruntime` CPU par défaut. Si vous installez `onnxruntime-gpu`, assurez-vous d'avoir la pile CUDA/cuDNN adéquate et adaptez vos variables d'environnement en conséquence.

---

## 🗂️ Organisation du dépôt

```
.
├── app.py           # Interface Streamlit principale
├── jarvis.py        # Backend audio local (Whisper + Piper + logique vocale)
├── requirements.txt # Dépendances Python
├── docs/
│   └── interface.png
└── README.md
```

Le fichier `app.py` constitue l'unique point d'entrée Streamlit (l'ancien `app_4.py` n'existe plus).

---

## ⚙️ Configuration & persistance

La configuration utilisateur est stockée dans `~/.jarvis/ui_config.json`. Elle est fusionnée avec les valeurs par défaut et normalisée vers la nouvelle structure MCP (champ `gateway`). Exemple de contenu :

```json
{
  "whisper": {
    "model": "small",
    "lang": "fr",
    "vad": true,
    "device": "cpu",
    "compute": "int8",
    "prompt": "Transcris strictement en français."
  },
  "piper": {
    "base_dir": "~/.jarvis/voices",
    "voice": "fr_FR-0001",
    "speaker_id": 0,
    "speed": 0.9,
    "noise": 0.667,
    "noise_w": 0.8,
    "sentence_silence": 0.1,
    "use_cuda": false
  },
  "ollama": {
    "host": "http://127.0.0.1:11434",
    "model": "qwen2.5:latest",
    "temperature": 0.7,
    "num_ctx": 4096,
    "stream": false
  },
  "mcp": {
    "gateway": {
      "enabled": true,
      "base_url": "http://127.0.0.1:8080",
      "auth_header": ""
    }
  },
  "jarvis": {
    "path": "./jarvis.py",
    "audio_out": "analog",
    "tts_engine": "piper",
    "tts_lang": "fr",
    "wake_word": "jarvis",
    "wake_aliases": "",
    "require_wake": true,
    "wake_fuzzy": true,
    "wake_fuzzy_score": 80
  }
}
```

Chaque sauvegarde depuis l'UI synchronise également les variables d'environnement (`FW_MODEL`, `PIPER_VOICE`, `OLLAMA_HOST`, etc.) pour le backend `jarvis.py`.

---

## 🧑‍💻 Utilisation

1. **Lancer l'UI** via `streamlit run app.py`.
2. **Configurer les onglets "Settings"** :
   - Renseignez le chemin de `jarvis.py`, la sortie audio, les paramètres Piper et Ollama.
   - Activez ou désactivez le MCP Gateway (HTTP/SSE). Fournissez l'URL et un éventuel header d'authentification si nécessaire.
3. **Démarrer Jarvis** depuis l'onglet "Interface" (boutons démarrer/arrêter). Les logs du backend apparaissent en temps réel.
4. **Interagir** :
   - Mode **chat** : saisissez vos messages, utilisez `/tool <nom> {json}` pour appeler un tool MCP.
   - Mode **vocal** : activez le toggle, puis parlez (backend requis pour capturer l'audio et répondre via Piper).
5. **Superviser** via la topbar (statuts) et le radar vocal.

Le gateway MCP utilise prioritairement le client HTTP "streamable" du SDK MCP (`mcp.client.streamable_http`) et bascule automatiquement sur la variante SSE si nécessaire.

---

## 🔎 Intégration MCP Gateway

- Fournissez un endpoint MCPJungle Gateway (ex. `http://127.0.0.1:8080`).
- Activez le toggle "Activer le gateway MCPJungle" dans la colonne gauche.
- Le raccourci `/tool weather {"city": "Paris"}` appelle directement le tool `weather` avec les arguments JSON fournis.
- Les résultats sont formatés pour extraire `structuredContent` et `TextContent` du SDK MCP.

---

## 🛠️ Dépannage

- **Ollama KO** : vérifiez que `ollama serve` tourne (`app.py` peut lancer le daemon si besoin) et que l'URL correspond.
- **Voix Piper introuvable** : importez vos `.onnx` via l'onglet Piper ou copiez-les dans `~/.jarvis/voices`, puis sélectionnez-les.
- **Gateway MCP injoignable** : confirmez l'URL, le header d'authentification et l'installation du package `modelcontextprotocol`.
- **CUDA absente** : laissez `use_cuda` désactivé et restez sur `onnxruntime` CPU.

---

## 🤝 Contribuer

Les issues et PR sont les bienvenues. Pensez à préciser : OS, version de Python, version d'Ollama, logs pertinents et configuration MCP éventuelle.

---

## 📝 Licence

Projet distribué sous licence MIT (`LICENSE`).
