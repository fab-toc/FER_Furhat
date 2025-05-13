# TPE ISSD

# Description du projet

Ce projet vise à implémenter un algorithme de reconnaissance de certaines expressions faciales acquises avec la caméra du robot humanoïde Furhat (https://www.furhatrobotics.com/) et à à oraliser les réponses et/ou réagir avec le robot (feedback). Ces expressions sont à choisir entre les expressions que le robot peut exprimer (surprise, sourire par exemple).

# Installation

## Installer uv

### Pour macOS and Linux :

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pour Windows :

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installer les dépendances en fonction de sa carte graphique

### Si vous avez une carte graphique NVIDIA

#### Identifier la version de CUDA

Sous Linux, exécutez la commande suivante et notez la version de CUDA affichée :

```
nvidia-smi
```

#### Pour CUDA 11.8

```
uv sync --extra cu118
```

#### Pour CUDA 12.4

```
uv sync --extra cu124
```

#### Pour CUDA 12.6

```
uv sync --extra cu126
```

#### Pour CUDA 12.8

```
uv sync --extra cu128
```

### Si vous avez une carte graphique AMD

(uniquement compatible sous Linux, limitation de PyTorch)

```
uv sync --extra rocm
```

### Si vous avez une carte graphique INTEL

(uniquement compatible sous Linux et Windows, limitation de PyTorch)

```
uv sync --extra xpu
```

### Si vous n'avez pas de carte graphique

```
uv sync --extra cpu
```

## Créer ses variables d'environnement

Créer un fichier `.env` à la racine du projet et y ajouter les variables d'environnement mentionnées dans `.env.example`, en remplaçant les valeurs par celles de votre configuration.
