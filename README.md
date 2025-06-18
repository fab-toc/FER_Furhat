# TPE ISSD - Reconnaissance d'Expressions Faciales avec un Robot Furhat

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.6+-red.svg)

**Système de reconnaissance d'expressions faciales en temps réel intégré avec le robot humanoïde <a href="https://furhatrobotics.com/">Furhat</a>**

</div>

## 📋 Table des Matières

- [🎯 Description du Projet](#-description-du-projet)
- [📋 Architecture](#-architecture)
- [🚀 Installation](#-installation)
- [🎮 Utilisation](#-utilisation)
- [📊 Structure du Projet](#-structure-du-projet)
- [🧠 Modèles Supportés](#-modèles-supportés)
- [🤖 À propos de Furhat](#-à-propos-de-furhat)
- [📈 Performances](#-performances)

## 🎯 Description du Projet

Ce projet vise à implémenter un **système de reconnaissance des expressions faciales en temps réel** en utilisant des méthodes de deep learning. Il capture les visages à l'aide d'une caméra puis les analyse avec un modèle de classification d'images.

Le modèle est entraîné sur le jeu de données ImageNet (plusieurs modèles disponibles), fine-tuné une première fois sur le jeu de données d'expressions faciales FER2013 afin de classifier des expressions faciales, puis fine-tuné une nouvelle fois sur un jeu de données privé, élaboré spécialement pour le projet.
On fait ensuite réagir le robot humanoïde [Furhat](https://furhatrobotics.com/) en réaction à l'expression détectée, en utilisant notamment des expressions faciales disponibles sur le robot, des LED et de la synthèse vocale.

### 🎭 Expressions Reconnues

Le système peut détecter, à partir du dataset FER2013, **7 expressions faciales différentes**. On choisit ici de se limiter à **4 expressions** afin d'avoir de meilleurs résultats (on exclut la surprise, la peur et l'expression "neutre" qui sont plus difficiles à classifier):

- 😠 **Colère** (Angry) - LED rouge, expression de colère
- 😨 **Peur** (Fear) - LED violette, expression de peur
- 😊 **Joie** (Happy) - LED jaune, grand sourire
- 😢 **Tristesse** (Sad) - LED bleue, expression triste

## 📋 Architecture

**Pipeline de traitement :**

1. **Capture** : Capture des images en temps réel via une caméra connectée au PC via USB
2. **Détection** : Localisation des visages et redimensionnement avec OpenCV
3. **Traitement** : Augmentation des données (rotation, zoom, etc.) pour améliorer la robustesse de la prédiction
4. **Inférence** : Classification avec modèle pré-entraîné par batch d'images (permet une prédiction plus robuste par moyennage des prédictions obtenues sur une certaine période de temps)
5. **Réaction** : Synchronisation au robot Furhat et réaction en fonction de l'expression détectée (expression, voix, LEDs)

## 🚀 Installation

### Prérequis

- 🎥 **Caméra quelconque** / **Caméra RealSense** (_si vous souhaitez utiliser le projet directement_)
- 🤖 **Robot Furhat** avec connexion réseau (voir [documentation](https://docs.furhat.io) et configuration pour obtenir son adresse IP) et furhat-remote-api activée sur le robot

### 1. Installer uv (Gestionnaire de dépendances)

Pour gérer les dépendances, ce projet utilise [uv](https://docs.astral.sh/uv), un gestionnaire de paquets Python bien plus rapide que pip qui gère automatiquement les environnements virtuels.

#### 🐧 Linux / 🍎 macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 🪟 Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Cloner le Projet

```bash
git clone https://github.com/fab-toc/FER_Furhat.git
cd FER_Furhat
```

### 3. Installer les Dépendances

Le choix des dépendances PyTorch dépend de votre configuration matérielle :

#### 🔥 Avec GPU NVIDIA

Identifiez d'abord votre version CUDA :

```bash
nvidia-smi
```

Puis installez selon votre version :

```bash
# CUDA 11.8
uv sync --extra cu118

# CUDA 12.4
uv sync --extra cu124

# CUDA 12.6
uv sync --extra cu126

# CUDA 12.8
uv sync --extra cu128
```

#### 💻 Autres Configurations

```bash
# CPU uniquement
uv sync --extra cpu
```

### 4. Configuration Kaggle (Optionnel)

Pour télécharger automatiquement le dataset FER2013 (dans le cas où vous souhaitez fine-tuner le modèle et l'entraîner), vous pouvez configurer vos identifiants Kaggle :

```bash
# Copier le template
cp .env.example .env

# Éditer avec vos credentials Kaggle
nano .env
```

Ajoutez vos identifiants Kaggle :

```bash
export KAGGLE_USERNAME=votre_username
export KAGGLE_KEY=votre_api_key
```

## 🎮 Utilisation

### 🚀 Démarrage Rapide

1. **Connecter la caméra RealSense**
2. **S'assurer que Furhat est accessible sur le réseau et bien configuré**
3. **Lancer l'application principale** :

```bash
uv run src/main.py
```

### 📊 Entraînement d'un Nouveau Modèle

```bash
# Entraînement initial
uv run src/train/train.py

# Fine-tuning sur vos données
uv run src/train/fine-tuning.py
```

### 🧪 Test et Évaluation de Modèles

```bash
# Test sur dataset
uv run src/test.py
```

## 📊 Structure du Projet

```
FER_Furhat/
├── 📁 src/                    # Code source principal
│   ├── 🐍 main.py             # Application principale (pipeline complet de reconnaissance d'expressions faciales et réactions avec le robot)
│   ├── 🧪 test.py             # Test et évaluation de modèles
│   └── 📁 train/              # Scripts d'entraînement
│       ├── 🔧 utils.py        # Fonctions utilitaires, d'entraînement, de deep learning, etc.
│       ├── 🎯 train.py        # Entraînement d'un modèle (fine-tuning sur le dataset FER2013)
│       └── ⚡ fine-tuning.py   # Fine-tuning sur un dataset privé
├── 📁 trained/                # Répertoire des modèles entraînés sauvegardés (auto-généré lors de la sauvegarde d'un modèle après entraînement)
│   └── 📁 (model_name)        # Répertoire spécifique à une famille de modèles sauvegardés (auto-généré lors de la sauvegarde d'un modèle après entraînement)
├── 📁 dataset/                # Répertoire d'un potentiel dataset privé (à créer)
├── ⚙️ pyproject.toml          # Configuration du projet, des dépendances, des scripts, etc.
├── 🔒 .env.example            # Template variables d'environnement
└── 📖 README.md               # Cette documentation
```

## 🧠 Modèles Supportés

### 🏗️ Architectures

| Modèle       | Variantes                  | Paramètres  | Performance |
| ------------ | -------------------------- | ----------- | ----------- |
| **ConvNeXt** | Tiny, Small, Base, Large   | 28M - 197M  | ⭐⭐⭐⭐⭐  |
| **VGG**      | VGG11, VGG13, VGG16, VGG19 | 132M - 143M | ⭐⭐⭐⭐    |

### ⚙️ Configuration Recommandée

```python
MODEL_NAME = "convnext"
MODEL_VERSION = "large"        # Meilleure précision
UNFREEZE_LAYER = 3            # Fine-tuning optimal
BATCH_SIZE = 32               # Équilibre vitesse/mémoire
```

## 🤖 À propos de Furhat

[Furhat Robotics](https://furhatrobotics.com/) développe des robots sociaux avec des capacités d'interaction naturelle. Notre système utilise :

### 🎭 Capacités Furhat Utilisées

- **Expressions faciales** : 20+ expressions programmables
- **Synthèse vocale** : Voix multilingues (français supporté)
- **LEDs** : Éclairage RGB personnalisable
- **API REST** : Contrôle via HTTP/WebSocket

### 🔧 Configuration Réseau

Par défaut, le système cherche le robot Furhat sur `192.168.10.14:54321`. Modifiez dans `main.py` :

```python
furhat_controller = FurhatController(host="VOTRE_IP_FURHAT")
```

## 📈 Performances

### 🎯 Métriques Typiques

| Dataset                     | Modèle         | Précision |
| --------------------------- | -------------- | --------- |
| FER2013                     | ConvNeXt-Large | ~60%      |
| Fine-tuné sur notre dataset | ConvNeXt-Large | ~90%      |
