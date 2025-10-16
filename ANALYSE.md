# Analyse du dépôt Streamlit_JARVIS

## Problèmes critiques à corriger

1. **Configuration par défaut mutée en mémoire**  
   `load_cfg()` part d'une copie superficielle de `DEFAULT_CFG`. Les sous-dictionnaires (`mcp`, `piper`, etc.) restent donc partagés avec l'original, puis `_normalize_mcp_servers()` ré-écrit ces structures sur place. Résultat : la première lecture de config modifie aussi `DEFAULT_CFG`, ce qui change les valeurs par défaut pour la session suivante ou pour tout utilisateur qui n'a pas encore de fichier `ui_config.json`. 【F:app.py†L69-L149】

2. **Proxy MCP inexploitable par défaut**  
   Le proxy activé par défaut pointe vers `node /ABSOLU/adamwattis_mcp-proxy-server/build/index.js`. Ce chemin n'existe pas dans le dépôt ni dans une installation standard, provoquant un échec systématique dès que l'utilisateur tente d'utiliser les raccourcis MCP. 【F:app.py†L40-L52】

3. **Dépendances Ollama/Piper impossibles à installer sur CPU**  
   Le fichier `requirements.txt` exige à la fois `onnxruntime` (CPU) et `onnxruntime-gpu`. Ces paquets fournissent les mêmes modules et entrent en conflit : l'installation échoue ou tente de charger CUDA (`libcuda.so`) sur des machines qui n'en disposent pas. Il faut choisir une variante selon la cible matérielle, pas les deux. 【F:requirements.txt†L1-L15】

## Risques de sécurité

- **Création de fichiers temporaires non sécurisée**  
  `write_wav_tmp()` et le repli Piper utilisent `tempfile.mktemp()`, une fonction déconseillée car sujette aux conditions de course (un processus tiers peut créer le fichier avant l'écriture). Il est préférable d'utiliser `NamedTemporaryFile(delete=False)` ou `mkstemp()` pour obtenir un chemin réservé de manière atomique. 【F:jarvis.py†L516-L522】【F:jarvis.py†L701-L744】

## Recommandations

- Cloner profondément (`copy.deepcopy`) `DEFAULT_CFG` avant toute mutation, ou reconstruire une nouvelle structure à partir des valeurs lues pour éviter de polluer les défauts.  
- Fournir une valeur proxy MCP neutre (désactivée ou pointant vers un script inclus) et proposer des champs à remplir dans l'UI.  
- Scinder les dépendances Piper en deux extras (`onnxruntime` **ou** `onnxruntime-gpu`) ou documenter clairement la marche à suivre selon le matériel.  
- Remplacer `tempfile.mktemp()` par `NamedTemporaryFile(delete=False)` / `mkstemp()` et gérer la fermeture/suppression en conséquence.
