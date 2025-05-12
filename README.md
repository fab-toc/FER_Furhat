# TPE ISSD

## Description du projet

Ce projet vise à implémenter un algorithme de reconnaissance de certaines expressions faciales acquises avec la caméra du robot humanoïde Furhat (https://www.furhatrobotics.com/) et à à oraliser les réponses et/ou réagir avec le robot (feedback). Ces expressions sont à choisir entre les expressions que le robot peut exprimer (surprise, sourire par exemple).

## Installation

### Installer uv

#### Pour macOS and Linux :

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Pour Windows :

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installer les dépendances du projet

```
uv sync
```

<!-- #### Si vous avez une carte graphique NVIDIA

```
uv sync --extra cu128
```

#### Si vous avez une carte graphique Intel

```
uv sync --extra xpu
```

#### Si vous n'avez pas de carte graphique

```
uv sync --extra cpu
``` -->
