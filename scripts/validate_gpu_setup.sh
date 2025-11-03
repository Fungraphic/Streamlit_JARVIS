#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "  Validation Architecture Hybride CPU/GPU"
echo "========================================="
echo

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Vérifier nvidia-smi
echo "=== 1. Vérification GPU NVIDIA ==="
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ nvidia-smi disponible${NC}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}❌ nvidia-smi non trouvé (pilotes NVIDIA manquants?)${NC}"
    echo "   Installer: sudo apt install nvidia-driver-XXX"
    exit 1
fi

# 2. Vérifier CUDA_VISIBLE_DEVICES dans Python
echo
echo "=== 2. Vérification CUDA_VISIBLE_DEVICES ==="
python3 << 'EOF'
import os
cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')
print(f'CUDA_VISIBLE_DEVICES: {cuda_env}')

if cuda_env == '':
    print('❌ ERREUR: GPU désactivé pour Python!')
    print('   → Vérifier jarvis.py:15-19 (doit être commenté)')
    exit(1)
elif cuda_env == 'NOT_SET':
    print('✅ OK: Tous les GPUs disponibles')
else:
    print(f'✅ OK: GPU {cuda_env} sélectionné')
EOF

# 3. Vérifier onnxruntime (doit être CPU pour Piper)
echo
echo "=== 3. Vérification onnxruntime (Piper TTS) ==="
python3 << 'EOF'
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f'Providers: {providers}')

    if 'CUDAExecutionProvider' in providers:
        print('⚠️  onnxruntime-gpu installé (peut causer compétition GPU avec Ollama)')
        print('   Recommandation: pip uninstall onnxruntime-gpu && pip install onnxruntime==1.18.1')
    elif 'CPUExecutionProvider' in providers:
        print('✅ onnxruntime CPU (optimal pour Piper)')
    else:
        print('❓ Providers inhabituels détectés')
except ImportError:
    print('❌ onnxruntime non installé')
    print('   → pip install onnxruntime==1.18.1')
    exit(1)
EOF

# 4. Vérifier Ollama
echo
echo "=== 4. Vérification Ollama ==="
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama installé${NC}"

    # Vérifier si Ollama tourne
    if pgrep -x ollama > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama en cours d'exécution${NC}"

        # Vérifier si sur GPU
        if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -qi ollama; then
            echo -e "${GREEN}✅ Ollama utilise le GPU${NC}"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep -i ollama || true
        else
            echo -e "${YELLOW}⚠️  Ollama ne semble pas utiliser le GPU${NC}"
            echo "   Vérifications:"
            echo "   1. Charger un modèle: ollama run qwen2.5:latest 'test'"
            echo "   2. Vérifier nvidia-smi pendant génération"
        fi
    else
        echo -e "${YELLOW}⚠️  Ollama non démarré${NC}"
        echo "   → Démarrer: ollama serve"
    fi
else
    echo -e "${RED}❌ Ollama non installé${NC}"
    echo "   → Installer depuis: https://ollama.ai/download"
    exit 1
fi

# 5. Vérifier Streamlit
echo
echo "=== 5. Vérification Streamlit ==="
python3 << 'EOF'
try:
    import streamlit as st
    print(f'✅ Streamlit {st.__version__} installé')
except ImportError:
    print('❌ Streamlit non installé')
    print('   → pip install "streamlit>=1.30.0,<2.0.0"')
    exit(1)
EOF

# 6. Résumé
echo
echo "========================================="
echo "  RÉSUMÉ - Architecture Cible"
echo "========================================="
echo
echo "Stack Vocal (CPU):"
echo "  ✓ Faster-Whisper: CPU + int8"
echo "  ✓ Piper TTS: CPU (onnxruntime)"
echo "  ✓ Audio I/O: CPU (sounddevice)"
echo
echo "Stack LLM (GPU):"
echo "  ✓ Ollama: CUDA (llama.cpp)"
echo "  ✓ Models: GPU VRAM"
echo
echo "Séparation CPU/GPU:"
echo "  → Audio processing = latence critique → CPU suffit"
echo "  → LLM generation = throughput massif → GPU requis"
echo "  → Pas de compétition GPU entre Piper et Ollama"
echo

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Validation terminée !${NC}"
echo -e "${GREEN}=========================================${NC}"
