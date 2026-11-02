# ProjectStark

Controla tu computador con gestos de la mano usando la cámara. Hecho en Python con
[OpenCV](https://opencv.org/), [MediaPipe](https://ai.google.dev/edge/mediapipe) y
[PyAutoGUI](https://pyautogui.readthedocs.io/).

## Gestos

| Gesto | Acción |
|---|---|
| ☝️ Solo el índice | Mover el cursor |
| 🤏 Pellizco rápido (pulgar + índice) | Clic |
| 🤏 Pellizco mantenido | Arrastrar |
| 🖐️ Mano abierta quieta (1 s) | Play / Pausa |
| 🖐️ Mano abierta deslizando | Pestaña siguiente / anterior |
| ✌️ Paz | Siguiente canción |
| 🤟 Tres dedos (índice, medio, anular) | Scroll |
| 👍 Pulgar arriba | Subir volumen |
| 👎 Pulgar abajo | Bajar volumen |
| 🤘 Rock (índice + meñique) | Abrir una página web |
| 🤙 Pulgar + meñique | Captura de pantalla |

## Instalación (Windows)

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El modelo de detección de manos ya está incluido en `models/hand_landmarker.task`.

## Uso

Haz doble clic en `iniciar.bat` (o ejecuta `.venv\Scripts\python.exe gestos.py`).

- **ESPACIO**: activar / desactivar las acciones (empiezan desactivadas)
- **Q** o **ESC**: salir

Los ajustes (sensibilidad, tiempos, página que abre el gesto rock, etc.) están al
principio de `gestos.py`.
