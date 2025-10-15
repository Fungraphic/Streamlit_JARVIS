# Analyse du dépôt Streamlite_JARVIS

## Problèmes identifiés

1. **Synchronisation totale UI ⇄ Jarvis**
   Toutes les préférences (Whisper, Piper, TTS, wake-word, Ollama, sortie audio) sont maintenant exportées en variables d'environnement avant le chargement de `jarvis.py`. Le backend consomme exactement les valeurs choisies dans l'interface, y compris la voix Piper et les paramètres de détection du mot-clé. 【F:app.py†L34-L126】【F:app.py†L206-L274】【F:jarvis.py†L36-L136】【F:jarvis.py†L512-L579】

2. **Dépendances alignées avec le code**
   Le fichier `requirements.txt` inclut désormais `numpy`, `ollama`, `rapidfuzz` et bascule sur `onnxruntime` CPU par défaut. Les installateurs disposent d'une note pour ajouter la version GPU si nécessaire. Cela évite les erreurs d'import et les installations inutiles sur des machines sans CUDA. 【F:requirements.txt†L1-L11】【F:jarvis.py†L17-L45】

3. **Paramètres Ollama unifiés**
   L'UI et Jarvis partagent le même modèle, la température, la taille de contexte et le mode streaming via `OLLAMA_*`. Les ajustements effectués depuis Streamlit s'appliquent immédiatement au backend, qui les loggue et utilise les options lors des appels `chat`. 【F:app.py†L71-L122】【F:jarvis.py†L94-L134】【F:jarvis.py†L420-L506】

## Points de vigilance restants

- **Démarrage Ollama externe** : l'interface peut lancer `ollama serve`, mais l'utilisateur doit toujours vérifier la disponibilité réseau et l'espace disque pour les modèles lourds. 【F:app.py†L135-L215】
- **Voix Piper à déployer manuellement** : l'UI permet l'upload, toutefois `jarvis.py` exige le binaire `piper` dans le PATH et les paires `.onnx`/`.onnx.json` valides côté serveur. 【F:app.py†L305-L351】【F:jarvis.py†L270-L374】

## Recommandations générales

- Harmoniser la gestion de configuration (idéalement un fichier partagé ou des variables d'environnement communes) afin que l'UI et le backend utilisent les mêmes valeurs par défaut.
- Ajouter un script de vérification des dépendances ou un guide d'installation précisant les modules optionnels versus obligatoires.
- Prévoir un mécanisme de validation dans l'UI pour alerter l'utilisateur lorsque Jarvis tourne avec une configuration incompatible (voix Piper absente, Ollama indisponible, etc.).
