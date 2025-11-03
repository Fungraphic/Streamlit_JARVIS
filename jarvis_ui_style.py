#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jarvis_ui_style.py - Style JARVIS HUD pour Streamlit

Contient:
- Le CSS JARVIS complet (HUD holographique)
- Les fonctions de rendu HTML pour le chat
- Les avatars et composants visuels
"""

import html
import re
import base64
from typing import List, Dict
from pathlib import Path

# ============================================================================
# CSS JARVIS - HUD Holographique complet
# ============================================================================

JARVIS_CSS = """
<style>
/* ===== Styliser la barre TOP Streamlit avec thème JARVIS ===== */
header[data-testid="stHeader"] {
  background: linear-gradient(180deg, rgba(10,22,28,0.85), rgba(10,22,28,0.95)) !important;
  border-bottom: 1px solid rgba(45,212,255,.25) !important;
  backdrop-filter: blur(10px) !important;
}

/* Styliser le bouton de la sidebar (hamburger) avec thème JARVIS */
button[data-testid="baseButton-header"],
button[data-testid="collapsedControl"] {
  color: var(--jarvis-cyan) !important;
  background: rgba(45,212,255,.15) !important;
  border: 1px solid rgba(45,212,255,.35) !important;
  border-radius: 8px !important;
  padding: 0.5rem !important;
}

button[data-testid="baseButton-header"]:hover,
button[data-testid="collapsedControl"]:hover {
  background: rgba(45,212,255,.25) !important;
  border-color: var(--jarvis-cyan) !important;
  box-shadow: 0 0 15px rgba(45,212,255,.35) !important;
}

/* SVG icon inside button (hamburger lines) */
button[data-testid="baseButton-header"] svg,
button[data-testid="collapsedControl"] svg {
  color: var(--jarvis-cyan) !important;
  fill: currentColor !important;
}

/* Styliser le toolbar (Deploy, Rerun, etc.) */
header[data-testid="stHeader"] [data-testid="stToolbar"] button {
  color: var(--jarvis-cyan) !important;
  background: rgba(45,212,255,.10) !important;
  border: 1px solid rgba(45,212,255,.25) !important;
  border-radius: 6px !important;
}

header[data-testid="stHeader"] [data-testid="stToolbar"] button:hover {
  background: rgba(45,212,255,.20) !important;
  box-shadow: 0 0 10px rgba(45,212,255,.25) !important;
}

/* Réduire l'espace en haut de la page de 75% (2rem → 0.5rem) */
.main .block-container {
  padding-top: 0.5rem !important;
  max-width: 100% !important;
}

/* Styliser la sidebar avec le thème JARVIS */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,22,28,0.95), rgba(10,22,28,0.98)) !important;
  border-right: 1px solid rgba(45,212,255,.25) !important;
}

section[data-testid="stSidebar"] > div {
  background: transparent !important;
}

/* Bouton de fermeture de la sidebar (X ou chevron) */
section[data-testid="stSidebar"] button[kind="header"] {
  color: var(--jarvis-cyan) !important;
  background: transparent !important;
  border: 1px solid rgba(45,212,255,.25) !important;
}

section[data-testid="stSidebar"] button[kind="header"]:hover {
  background: rgba(45,212,255,.15) !important;
  box-shadow: 0 0 10px rgba(45,212,255,.25) !important;
}

/* ===== Palette & tokens ===== */
:root {
  --jarvis-bg: #0a1116;
  --jarvis-panel: rgba(10, 22, 28, 0.85);
  --jarvis-cyan: #2dd4ff;
  --jarvis-ice: #9ae6ff;
  --jarvis-blue: #3b82f6;
  --jarvis-teal: #00e5ff;
  --jarvis-orange: #ffb020;
  --jarvis-grid: rgba(45,212,255,.06);
  --jarvis-text: #cfe8ff;
  --jarvis-muted: #89a8bd;
  --jarvis-input-bg: rgba(10,22,28,.85);
  --jarvis-input-text: #e6fbff;
  --jarvis-input-border: rgba(45,212,255,.25);
  --jarvis-input-focus: rgba(45,212,255,.35);
  --jarvis-placeholder: #9bb6c7;
}

/* ===== Fond grille holographique ===== */
.stApp {
  background:
    radial-gradient(1000px 600px at 70% -20%, rgba(45,212,255,.08), transparent 60%),
    radial-gradient(800px 500px at -10% 110%, rgba(59,130,246,.10), transparent 60%),
    var(--jarvis-bg);
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(transparent 95%, rgba(45,212,255,.08) 96%, transparent 97%),
    linear-gradient(90deg, transparent 95%, rgba(45,212,255,.06) 96%, transparent 97%);
  background-size: 100% 22px, 22px 100%;
  mix-blend-mode: screen;
  opacity: .25;
  z-index: 0;
}

/* ===== Messages chat JARVIS ===== */
.message {
  display: grid;
  align-items: start;
  grid-template-columns: 64px minmax(0, 1fr);
  gap: 12px;
  padding: 10px 0 22px 0;
  font-size: 18px;
  font-family: "Orbitron","Rajdhani","Segoe UI",Roboto,Arial,sans-serif;
  line-height: 1.5;
  position: relative;
  z-index: 1;
  margin-bottom: 16px;
}

/* Message USER : avatar à droite */
.message.message-user {
  grid-template-columns: minmax(0, 1fr) 64px;
  align-items: center;
}

.message.message-user .circle {
  grid-column: 2;
  grid-row: 1;
  justify-self: end;
}

.message.message-user .text {
  grid-column: 1;
  grid-row: 1;
  text-align: right;
  padding-left: 0;
  padding-right: 2px;
}

.message.message-user .username {
  padding-left: 0;
  padding-right: 2px;
  text-align: right;
}

.message.message-user .message-body {
  margin-left: auto;
}

/* Avatars circulaires avec halo */
.circle {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background:
    radial-gradient(60% 60% at 50% 40%, rgba(255,255,255,.15), rgba(255,255,255,.02) 60%, transparent 70%),
    #0d1a20;
  border: 1px solid rgba(45,212,255,.35);
  box-shadow:
    0 0 0 1px rgba(45,212,255,.15) inset,
    0 0 22px rgba(45,212,255,.25),
    0 8px 24px rgba(0,0,0,.6);
  position: relative;
  overflow: visible;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: var(--jarvis-cyan);
  isolation: isolate;
}

.circle::after {
  content: "";
  position: absolute;
  inset: -3px;
  border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0 70%, rgba(45,212,255,.6) 75% 85%, transparent 90% 100%);
  animation: ringPulse 2.8s linear infinite;
  pointer-events: none;
  z-index: 2;
}

/* Style pour les images d'avatar */
.circle img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 50%;
  position: absolute;
  top: 0;
  left: 0;
  z-index: 0;
}

/* Animation du ring autour de l'avatar */
@keyframes ringPulse {
  to { transform: rotate(360deg); }
}

/* Corps du message */
.text {
  padding-left: 2px;
  color: var(--jarvis-text);
}

.username {
  padding-left: 2px;
  font-size: 14px;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--jarvis-muted);
  margin-bottom: 6px;
}

.username:hover {
  color: var(--jarvis-ice);
  text-shadow: 0 0 10px rgba(45,212,255,.35);
}

.message-body {
  position: relative;
  border-radius: 12px;
  padding: 14px 16px;
  background:
    linear-gradient(180deg, rgba(59,130,246,.08), rgba(10,22,28,.4) 22%, rgba(10,22,28,.65)),
    var(--jarvis-panel);
  border: 1px solid rgba(45,212,255,.25);
  box-shadow:
    0 0 0 1px rgba(45,212,255,.10) inset,
    0 0 30px rgba(45,212,255,.10),
    0 10px 32px rgba(0,0,0,.55);
  overflow: hidden;
}

/* Coins "bracket" JARVIS */
.message-body::before,
.message-body::after {
  content: "";
  position: absolute;
  width: 18px;
  height: 18px;
  border: 2px solid var(--jarvis-cyan);
  opacity: .45;
  filter: drop-shadow(0 0 6px rgba(45,212,255,.35));
}

.message-body::before {
  top: 8px;
  left: 10px;
  border-right: none;
  border-bottom: none;
}

.message-body::after {
  bottom: 8px;
  right: 10px;
  border-left: none;
  border-top: none;
}

.message-body p {
  margin: 0 0 8px 0 !important;
  font-size: 16px !important;
  line-height: 1.6 !important;
  color: var(--jarvis-ice) !important;
}

.message-body p:last-child {
  margin-bottom: 0 !important;
}

.message-body em {
  color: #a8d8ee !important;
}

.message-body strong {
  color: #dff6ff !important;
  text-shadow: 0 0 10px rgba(45,212,255,.25);
}

/* Liens */
.message-body a {
  color: var(--jarvis-cyan);
  text-decoration: none;
  border-bottom: 1px dotted rgba(45,212,255,.35);
}

.message-body a:hover {
  color: #e6fbff;
  text-shadow: 0 0 10px rgba(45,212,255,.35);
}

/* Code */
.message-body pre, .message-body code {
  background: rgba(6,12,16,.65);
  border: 1px solid rgba(45,212,255,.22);
  border-radius: 10px;
  padding: 10px 12px;
  color: #d7f6ff;
  box-shadow: inset 0 0 18px rgba(45,212,255,.06);
  font-family: "JetBrains Mono","SFMono-Regular","Consolas",monospace;
  font-size: 13px;
}

.message-body pre {
  overflow: auto;
}

/* Container chat - Fenêtre scrollable avec hauteur fixe */
.chat-container {
  width: 1200px;
  height: 558px;
  margin: 0 auto;
  padding: 20px;
  position: relative;
  z-index: 1;

  /* Scrolling behavior */
  overflow-y: auto;
  overflow-x: hidden;
  scroll-behavior: smooth;
  padding-right: 15px;

  /* Cadre et fond légèrement plus clair */
  background: rgba(15, 30, 40, 0.4);
  border: 1px solid rgba(45, 212, 255, 0.25);
  border-radius: 12px;
  box-shadow: 0 0 20px rgba(45, 212, 255, 0.1);

  /* Force scroll container */
  display: flex;
  flex-direction: column;
}

/* Scrollbar custom JARVIS style */
.chat-container::-webkit-scrollbar {
  width: 10px;
}

.chat-container::-webkit-scrollbar-track {
  background: rgba(10,22,28,.6);
  border-radius: 5px;
  border: 1px solid rgba(45,212,255,.15);
}

.chat-container::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(45,212,255,.35), rgba(45,212,255,.20));
  border-radius: 5px;
  border: 1px solid rgba(45,212,255,.25);
}

.chat-container::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(45,212,255,.55), rgba(45,212,255,.35));
  box-shadow: 0 0 10px rgba(45,212,255,.25);
}

/* Input Streamlit adapté */
.stTextArea textarea,
.stTextInput input {
  background: var(--jarvis-input-bg) !important;
  color: var(--jarvis-input-text) !important;
  border: 1px solid var(--jarvis-input-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 0 0 1px rgba(45,212,255,.08) inset, 0 0 18px rgba(45,212,255,.10) !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
  border-color: var(--jarvis-input-focus) !important;
  box-shadow: 0 0 0 1px var(--jarvis-input-focus) inset, 0 0 22px rgba(45,212,255,.18) !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
  color: var(--jarvis-placeholder) !important;
}

/* Boutons */
.stButton button {
  background: linear-gradient(180deg, rgba(45,212,255,.15), rgba(45,212,255,.08)) !important;
  border: 1px solid rgba(45,212,255,.35) !important;
  color: var(--jarvis-cyan) !important;
  border-radius: 12px !important;
  box-shadow: 0 0 18px rgba(45,212,255,.15) !important;
}

.stButton button:hover {
  background: linear-gradient(180deg, rgba(45,212,255,.25), rgba(45,212,255,.15)) !important;
  border-color: var(--jarvis-cyan) !important;
  box-shadow: 0 0 24px rgba(45,212,255,.35) !important;
}

/* Toggle */
.stCheckbox label {
  color: var(--jarvis-text) !important;
}

/* Headers */
h1, h2, h3 {
  color: var(--jarvis-cyan) !important;
  text-shadow: 0 0 10px rgba(45,212,255,.25);
  font-family: "Orbitron", "Rajdhani", sans-serif;
  letter-spacing: .08em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(10,22,28,.6);
  border-radius: 12px;
  border: 1px solid rgba(45,212,255,.15);
  padding: 8px;
}

.stTabs [data-baseweb="tab"] {
  color: var(--jarvis-muted);
  border-radius: 8px;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, rgba(45,212,255,.15), rgba(45,212,255,.08));
  color: var(--jarvis-cyan) !important;
  box-shadow: 0 0 12px rgba(45,212,255,.25);
}

/* Mobile */
@media (max-width: 688px) {
  .message {
    grid-template-columns: 46px minmax(0,1fr);
    font-size: 15px;
  }

  .message.message-user {
    grid-template-columns: minmax(0, 1fr) 46px;
  }

  .circle {
    width: 46px;
    height: 46px;
    font-size: 18px;
  }

  .username {
    font-size: 12px;
  }

  .message-body p {
    font-size: 14px !important;
  }
}
</style>
"""

# ============================================================================
# Fonctions de rendu HTML
# ============================================================================

def get_image_base64(image_path: str) -> str:
    """
    Convertit une image en base64 pour l'utiliser dans du HTML.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Data URI base64 de l'image
    """
    try:
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            # Déterminer le type MIME
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml'
            }.get(ext, 'image/png')
            return f"data:{mime_type};base64,{img_data}"
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return ""

# Cache pour les images en base64
_IMAGE_CACHE = {}

def get_cached_image_base64(image_path: str) -> str:
    """
    Récupère une image en base64 avec cache pour éviter de recharger à chaque message.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Data URI base64 de l'image
    """
    if image_path not in _IMAGE_CACHE:
        _IMAGE_CACHE[image_path] = get_image_base64(image_path)
    return _IMAGE_CACHE[image_path]

def render_message(role: str, content: str, username: str = None) -> str:
    """
    Rend un message au format JARVIS HUD.

    Args:
        role: "user" ou "assistant"
        content: Contenu du message
        username: Nom d'utilisateur (optionnel)

    Returns:
        HTML du message
    """
    is_user = (role == "user")
    msg_class = "message message-user" if is_user else "message message-assistant"

    # Avatar
    if is_user:
        # Charger l'image en base64 pour qu'elle fonctionne dans le HTML
        img_data = get_cached_image_base64("assets/fun.png")
        if img_data:
            avatar = f'<img src="{img_data}" alt="User" />'
        else:
            avatar = "👤"  # Fallback emoji si l'image ne charge pas
        display_name = username or "VOUS"
    else:
        avatar = "🤖"  # Bot icon
        display_name = "JARVIS"

    # Échapper le HTML et traiter les liens
    safe_content = html.escape(str(content))

    # Détecter et créer des liens cliquables
    safe_content = re.sub(
        r'(https?://[^\s<>"\')]+)',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        safe_content
    )

    # Convertir les retours à la ligne
    safe_content = safe_content.replace("\n", "<br />")

    # Construire le HTML
    html_parts = [
        f'<div class="{msg_class}">',
        f'  <div class="circle">{avatar}</div>',
        f'  <div class="text">',
        f'    <div class="username">{display_name}</div>',
        f'    <div class="message-body">',
        f'      <p>{safe_content}</p>',
        f'    </div>',
        f'  </div>',
        f'</div>'
    ]

    return '\n'.join(html_parts)


def render_chat_messages(messages: List[Dict[str, str]], username: str = None) -> str:
    """
    Rend une liste de messages au format JARVIS HUD.

    Args:
        messages: Liste de dict avec "role" et "content"
        username: Nom d'utilisateur (optionnel)

    Returns:
        HTML complet du chat
    """
    if not messages:
        return '''
        <div class="chat-container">
          <div class="message message-assistant">
            <div class="circle">🤖</div>
            <div class="text">
              <div class="username">JARVIS</div>
              <div class="message-body">
                <p style="opacity: 0.6;">Aucun échange pour le moment. Comment puis-je vous aider?</p>
              </div>
            </div>
          </div>
        </div>
        '''

    # Filtrer seulement les messages user/assistant
    display_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

    # Rendre chaque message
    rendered = []
    for msg in display_msgs:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        rendered.append(render_message(role, content, username))

    # Envelopper dans un container avec auto-scroll
    # Ajouter un marqueur invisible à la fin pour forcer le scroll
    return f'''
    <div class="chat-container" id="jarvis-chat-container">
      <div style="min-height: 100%;">
        {''.join(rendered)}
        <div id="chat-bottom-marker" style="height: 1px;"></div>
      </div>
    </div>
    <script>
      // Auto-scroll vers le bas en utilisant scrollIntoView
      (function() {{
        function scrollToBottom() {{
          const marker = document.getElementById('chat-bottom-marker');
          if (marker) {{
            marker.scrollIntoView({{ behavior: 'auto', block: 'end' }});
          }}
        }}

        // Scroll immédiat
        scrollToBottom();

        // Retry après délais pour Streamlit
        setTimeout(scrollToBottom, 50);
        setTimeout(scrollToBottom, 150);
        setTimeout(scrollToBottom, 500);

        // Observer pour les futurs changements
        const container = document.getElementById('jarvis-chat-container');
        if (container) {{
          const observer = new MutationObserver(scrollToBottom);
          observer.observe(container, {{
            childList: true,
            subtree: true
          }});
        }}
      }})();
    </script>
    '''


def inject_jarvis_css() -> str:
    """
    Retourne le CSS JARVIS complet à injecter via st.markdown.

    Usage:
        st.markdown(inject_jarvis_css(), unsafe_allow_html=True)
    """
    return JARVIS_CSS


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    # Test rapide
    messages = [
        {"role": "user", "content": "Quelle est la capitale de la France?"},
        {"role": "assistant", "content": "La capitale de la France est **Paris**."},
        {"role": "user", "content": "Merci! Et le nombre d'habitants?"},
        {"role": "assistant", "content": "Paris intra-muros compte environ 2,2 millions d'habitants."},
    ]

    print(JARVIS_CSS)
    print(render_chat_messages(messages, username="Stéphane"))
