# Monica TTS Server - Chatterbox Multilingual

Serveur de synthese vocale pour le **Bourly Poker Tour**.
Utilise le modele [Chatterbox-Multilingual 500M](https://huggingface.co/ResembleAI/chatterbox-turbo) de ResembleAI pour cloner la voix de Monica en francais.

**Endpoint** : `POST /blabla` — recoit du texte, renvoie du WAV.

---

## Architecture

```
PokerTourIRL Backend                    Monica TTS Server
┌──────────────────┐     POST /blabla   ┌──────────────────────┐
│  party.controller│ ──────────────────► │  FastAPI (uvicorn)   │
│                  │    {"text":"..."}   │                      │
│                  │ ◄────────────────── │  ChatterboxMTL 500M  │
│  WebSocket push  │    audio/wav        │  GPU: CUDA / MPS     │
└──────────────────┘                    └──────────────────────┘
```

---

## Setup sur NVIDIA DGX Spark

### Specifications materiel

| Composant | Detail |
|-----------|--------|
| **SoC** | NVIDIA GB10 Grace Blackwell Superchip |
| **GPU** | Blackwell — 6144 CUDA cores, 192 Tensor Cores |
| **RAM** | 128 Go LPDDR5x unified (CPU+GPU partagee) |
| **Bande passante** | 273 Go/s |
| **CUDA Compute** | 12.0 (sm_120) |
| **OS** | DGX OS 7 (base Ubuntu 24.04 LTS) |
| **CUDA Toolkit** | Pre-installe (CUDA 13.0.1) |
| **Docker** | Pre-installe avec nvidia-container-toolkit |

### Pre-requis (deja presents sur DGX Spark)

Le DGX Spark est livre avec tout ce qu'il faut :

- **DGX OS 7** (Ubuntu 24.04) — installe d'usine
- **Docker Engine** + **nvidia-container-toolkit** — pre-configures
- **CUDA drivers** — integres au systeme

Verifier que tout est fonctionnel :

```bash
# Verifier le GPU
nvidia-smi

# Verifier Docker + GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Verifier la version CUDA
nvcc --version
```

### Etape 1 — Cloner le repo

```bash
cd ~
git clone https://github.com/Pablohassan/chatterbox.git
cd chatterbox
```

### Etape 2 — Ajouter le fichier audio de reference

Le fichier `26-monica--interview.wav` (74 Mo, ~387s) n'est pas dans le repo git (trop volumineux).
Il faut le copier manuellement dans le dossier :

```bash
# Depuis votre machine locale
scp 26-monica--interview.wav user@<dgx-spark-ip>:~/chatterbox/

# Ou via USB/transfert direct
cp /chemin/vers/26-monica--interview.wav ~/chatterbox/
```

### Etape 3 — Build de l'image Docker

```bash
cd ~/chatterbox
docker compose build
```

Le build effectue :
1. Installation de PyTorch 2.6.0 avec CUDA 12.4
2. Installation de `chatterbox-tts` et dependances
3. Pre-telechargement du modele (~2 Go) dans l'image
4. Copie du fichier audio de reference

> **Note** : Le premier build prend ~10-15 minutes (telechargement PyTorch + modele).
> Les builds suivants utilisent le cache Docker.

### Etape 4 — Lancement

```bash
docker compose up -d
```

Le serveur :
1. Detecte automatiquement le GPU CUDA
2. Charge le fichier audio de reference (crop a 120s)
3. Charge le modele Chatterbox-Multilingual sur GPU
4. Execute une inference de warm-up
5. Ecoute sur le port **8080**

Suivre les logs de demarrage :

```bash
docker compose logs -f
```

Sortie attendue :
```
CUDA detected: NVIDIA GB10 (128.0 GB VRAM)
Loading reference audio: 26-monica--interview.wav
Cropping reference from 387.6s to 120s
Loading ChatterboxMultilingualTTS on device=cuda ...
Model loaded in 4.2s
Warm-up inference ...
Server ready!
Uvicorn running on http://0.0.0.0:8080
```

### Etape 5 — Test

```bash
# Health check
curl http://localhost:8080/health

# Generer de la voix
curl -X POST http://localhost:8080/blabla \
  -H 'Content-Type: application/json' \
  -d '{"text": "Pablo vient de se faire eliminer par Rusmir!"}' \
  --output test.wav

# Ecouter (si PulseAudio disponible)
aplay test.wav
```

---

## Configuration

Toutes les variables sont configurables via `docker-compose.yml` > `environment` :

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Port d'ecoute du serveur |
| `HOST` | `0.0.0.0` | Adresse d'ecoute |
| `LOG_LEVEL` | `info` | Niveau de log (`debug`, `info`, `warning`, `error`) |
| `REFERENCE_AUDIO` | `26-monica--interview.wav` | Fichier audio de reference pour le clonage |
| `REFERENCE_DURATION_S` | `120` | Duree de reference utilisee (en secondes) |
| `MAX_TEXT_LENGTH` | `500` | Longueur max du texte en entree |
| `HF_HOME` | `/app/hf_cache` | Cache des modeles HuggingFace |

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true
}
```

### `POST /blabla`

**Request :**
```json
{
  "text": "Rusmir vient d'eliminer Pablo avec une paire d'as!"
}
```

**Response :** `audio/wav` (binaire)

**Headers de reponse :**
- `Content-Disposition: inline; filename="monica.wav"`
- `X-Generation-Time: 2.34s`

**Codes d'erreur :**
| Code | Raison |
|------|--------|
| 400 | Texte vide ou trop long (> 500 chars) |
| 422 | JSON invalide |
| 500 | Erreur de generation TTS |
| 503 | Modele pas encore charge |

---

## Integration avec PokerTourIRL

Dans la configuration du backend PokerTourIRL, definir l'endpoint TTS :

```env
# .env du backend PokerTourIRL
TTS_ENDPOINT=http://<dgx-spark-ip>:8080/blabla
```

Ou via le reverse proxy Nginx (recommande) :

```env
TTS_ENDPOINT=https://monica.bourlypokertour.fr/blabla
```

### Configuration Nginx (reverse proxy)

Sur le reverse proxy (192.168.1.60), ajouter un server block :

```nginx
server {
    listen 443 ssl;
    server_name monica.bourlypokertour.fr;

    ssl_certificate /etc/letsencrypt/live/monica.bourlypokertour.fr/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/monica.bourlypokertour.fr/privkey.pem;

    location / {
        proxy_pass http://<dgx-spark-ip>:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }
}
```

---

## Performances attendues sur DGX Spark

| Metrique | Valeur estimee |
|----------|---------------|
| Temps de chargement modele | ~4-5s |
| Inference (phrase courte) | ~2-4s |
| Inference (phrase longue) | ~5-10s |
| Memoire GPU utilisee | ~4-6 Go |
| Memoire RAM totale | ~8 Go (modele + runtime) |

Le DGX Spark dispose de 128 Go de RAM unifiee partagee entre CPU et GPU.
Le modele Chatterbox-Multilingual 500M est leger et laisse largement de la marge.

---

## Commandes utiles

```bash
# Demarrer
docker compose up -d

# Arreter
docker compose down

# Voir les logs
docker compose logs -f

# Rebuild apres modification
docker compose build --no-cache && docker compose up -d

# Verifier l'utilisation GPU
nvidia-smi

# Entrer dans le conteneur
docker exec -it monica-tts bash
```

---

## Compatibilite CUDA

Le Dockerfile utilise actuellement les wheels PyTorch pour **CUDA 12.4** (`cu124`).
Ces wheels sont compatibles avec les drivers CUDA 12.x et 13.x du DGX Spark grace a la retrocompatibilite NVIDIA.

Si une version native CUDA 13 de PyTorch devient disponible, mettre a jour le Dockerfile :

```dockerfile
# Remplacer
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Par (quand disponible)
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
```

---

## Structure du projet

```
chatterbox/
├── server.py              # Serveur FastAPI principal
├── Dockerfile             # Image Docker (Python 3.11 + CUDA)
├── docker-compose.yml     # Orchestration avec GPU passthrough
├── requirements.txt       # Dependances Python (FastAPI, uvicorn)
├── .dockerignore          # Exclusions Docker build
├── .gitignore             # Exclusions Git
├── 26-monica--interview.wav  # Audio de reference (non versionne)
└── README.md              # Ce fichier
```
