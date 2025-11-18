# 🎥 CamFX

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-success?logo=opencv)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-orange)
![VirtualCam](https://img.shields.io/badge/Camera-VirtualCam-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


**CamFX** est une application Python qui applique des effets visuels en temps réel à votre webcam, avec une interface Tkinter pour activer/désactiver les filtres et une sortie vers une **caméra virtuelle** (compatible OBS, Discord, Zoom, etc.).

---

## ✨ Fonctionnalités

- Interface graphique simple (Tkinter) pour activer/désactiver les effets.
- Plusieurs effets visuels disponibles :
  - Glitch RGB
  - Flou
  - Inversion des couleurs
  - Dessin (style esquisse)
  - Pixelisation
  - Tremblement
  - Effet vieux film (sepia + bruit)
  - Cartoon
  - Distorsion en vague
  - Inversion radiale sur le visage
  - Rotation dynamique (avec vitesse ajustable)
- Envoi du flux à une **caméra virtuelle** (via `pyvirtualcam`) pour l’utiliser dans Discord, OBS, etc.

---

## 🧰 Technologies utilisées

- [Python 3](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [PyVirtualCam](https://github.com/johnboiles/pyvirtualcam)

---

## 🚀 Installation

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/AlexerV/CamFX.git
cd CamFX
```

### 2️⃣ Installer les dépendances
Assure-toi d’avoir Python ≥ 3.8, puis installe les modules nécessaires :
```bash
pip install opencv-python numpy pyvirtualcam
```
> Tkinter est inclus avec Python sur la plupart des distributions.
> Sous Linux, tu peux l’installer via ton gestionnaire de paquets :
> ```bash
> sudo apt install python3-tk
> ```

---

## 🧩 Utilisation
Lance simplement le script principal :
```bash
python main.py
```

- Une fenêtre Tkinter s’ouvre pour gérer les effets.
- Tu peux choisir la caméra de ton choix si tu en as plusieurs.
- La caméra virtuelle apparaît dans tes applications (OBS, Discord, etc.).
- Pour arrêter, ferme la fenêtre ou appuie sur `Q`.

---

## 🖼️ Exemple de rendu
<img width="293" height="560" alt="image" src="https://github.com/user-attachments/assets/98c5183e-18cd-403a-a21c-05249641d212" />

Exemple de rendu du panneau de gestion des effets.

---

## 🧠 À venir
- Ajout de nouveaux effets

---

## 📜 Licence
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Projet open source sous licence MIT.

---

Projet développé et maintenu par [![GitHub](https://img.shields.io/badge/GitHub-AlexerV-181717?logo=github)](https://github.com/AlexerV)
