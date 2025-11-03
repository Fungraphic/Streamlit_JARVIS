#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  Installation Dépendances Jarvis"
echo "=========================================="
echo

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Vérifier Python version
echo -e "${BLUE}=== 1. Vérification Python ===${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 non trouvé${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python ${PYTHON_VERSION}${NC}"

# Vérifier pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip non installé${NC}"
    echo "Installer: sudo apt install python3-pip"
    exit 1
fi

PIP_CMD=$(command -v pip3 || command -v pip)
echo -e "${GREEN}✅ pip disponible${NC}"
echo

# Mettre à jour pip
echo -e "${BLUE}=== 2. Mise à jour pip ===${NC}"
$PIP_CMD install --upgrade pip setuptools wheel
echo

# Installer les dépendances par groupe
echo -e "${BLUE}=== 3. Installation Dépendances ===${NC}"
echo

# Groupe 1: UI
echo -e "${YELLOW}[1/6] UI Streamlit...${NC}"
$PIP_CMD install "streamlit>=1.30.0,<2.0.0" requests==2.32.5
echo -e "${GREEN}✅ UI installée${NC}"
echo

# Groupe 2: Audio
echo -e "${YELLOW}[2/6] Audio I/O...${NC}"
$PIP_CMD install sounddevice==0.5.2 soundfile==0.13.1
echo -e "${GREEN}✅ Audio installé${NC}"
echo

# Groupe 3: STT
echo -e "${YELLOW}[3/6] Speech-to-Text (Faster-Whisper)...${NC}"
$PIP_CMD install faster-whisper==1.2.0 "ctranslate2>=4.6.0"
echo -e "${GREEN}✅ STT installé${NC}"
echo

# Groupe 4: TTS
echo -e "${YELLOW}[4/6] Text-to-Speech (Piper)...${NC}"
$PIP_CMD install piper-tts==1.3.0 onnxruntime==1.18.1
echo -e "${GREEN}✅ TTS installé${NC}"
echo

# Groupe 5: LLM
echo -e "${YELLOW}[5/6] LLM Client (Ollama)...${NC}"
$PIP_CMD install "ollama>=0.1.30" "numpy>=1.24" "rapidfuzz>=3.9"
echo -e "${GREEN}✅ LLM client installé${NC}"
echo

# Groupe 6: MCP
echo -e "${YELLOW}[6/6] MCP Protocol...${NC}"
$PIP_CMD install "mcp>=1.1.0" "anyio>=4.0.0"
echo -e "${GREEN}✅ MCP installé${NC}"
echo

# Vérification finale
echo -e "${BLUE}=== 4. Vérification Installation ===${NC}"
python3 << 'EOF'
import sys

packages = {
    'streamlit': 'Streamlit UI',
    'requests': 'HTTP client',
    'sounddevice': 'Audio I/O',
    'soundfile': 'Audio files',
    'faster_whisper': 'Speech-to-Text',
    'ctranslate2': 'STT backend',
    'onnxruntime': 'TTS runtime',
    'numpy': 'Numerical computing',
    'ollama': 'LLM client',
    'rapidfuzz': 'Fuzzy matching',
    'mcp': 'MCP Protocol',
    'anyio': 'Async I/O',
}

print()
errors = []
for pkg, desc in packages.items():
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'N/A')
        print(f'✅ {pkg:20s} {version:10s} ({desc})')
    except ImportError:
        print(f'❌ {pkg:20s} MANQUANT')
        errors.append(pkg)

print()
if errors:
    print(f'⚠️  {len(errors)} package(s) manquant(s): {", ".join(errors)}')
    sys.exit(1)
else:
    print('✅ Toutes les dépendances sont installées!')
EOF

echo
echo -e "${GREEN}=========================================="
echo "  Installation Terminée ✅"
echo "==========================================${NC}"
echo
echo "Prochaines étapes:"
echo "  1. Télécharger une voix Piper: voir DEMARRAGE_RAPIDE.md"
echo "  2. Installer Ollama: https://ollama.ai/download"
echo "  3. Valider le setup: ./scripts/validate_gpu_setup.sh"
echo "  4. Lancer l'app: streamlit run app.py"
echo
