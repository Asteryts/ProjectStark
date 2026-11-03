"""
ProjectStark - Control del computador con gestos de la mano.

Gestos (con las acciones ACTIVADAS):
  - Solo índice levantado              -> mueve el cursor del mouse
  - Juntar pulgar e índice (rápido)    -> clic izquierdo
  - Juntar pulgar e índice (mantener)  -> arrastrar (mantiene el clic hasta separar los dedos)
  - Mano abierta quieta (1 segundo)    -> Play / Pausa de música
  - Mano abierta deslizando a un lado  -> pestaña siguiente / anterior
  - Cuatro dedos (pulgar escondido)
    deslizando a un lado               -> cambiar de aplicación (Alt+Tab); baja la mano para elegirla
  - Signo de paz (índice + medio)      -> siguiente canción
  - Tres dedos (índice, medio, anular) -> scroll (sube o baja la mano)
  - Pulgar arriba                      -> subir volumen (se repite mientras lo mantienes)
  - Pulgar abajo                       -> bajar volumen (se repite mientras lo mantienes)
  - Rock (índice + meñique)            -> abrir una página web
  - Pulgar + meñique                   -> captura de pantalla (se guarda en la carpeta "capturas")

Teclas (con la ventana de la cámara seleccionada):
  - ESPACIO -> activar / desactivar acciones (empiezan desactivadas)
  - Q o ESC -> salir
"""

import math
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions, vision

CARPETA = Path(__file__).parent
MODELO = CARPETA / "models" / "hand_landmarker.task"
CARPETA_CAPTURAS = CARPETA / "capturas"

# Ajustes generales
TIEMPO_CONFIRMAR = 0.3       # segundos que debe mantenerse un gesto antes de actuar
ESPERA_ENTRE_ACCIONES = 1.0  # segundos mínimos entre dos acciones del mismo gesto
MARGEN = 0.15                # zona del borde de la cámara que no se usa para el cursor

# Cursor: el suavizado cambia según la velocidad (lento = preciso, rápido = ágil)
SUAVIZADO_MIN = 0.10         # para movimientos pequeños (más bajo = más preciso)
SUAVIZADO_MAX = 0.5          # para movimientos grandes (más alto = más rápido)
DISTANCIA_RAPIDA = 700       # píxeles a partir de los cuales el cursor va a máxima velocidad
ZONA_MUERTA_CURSOR = 1.5     # píxeles; movimientos menores se ignoran (quita el temblor)
RETROCESO_CLIC = 0.25        # el clic se hace donde estaba el cursor este tiempo ANTES del pellizco

# Pellizco (clic y arrastrar)
UMBRAL_PELLIZCO = 0.35       # distancia pulgar-índice para EMPEZAR el pellizco (relativa a la mano)
UMBRAL_SOLTAR = 0.55         # distancia para TERMINARLO (mayor = no se suelta solo al arrastrar)
TIEMPO_MIN_CLIC = 0.10       # pellizcos más cortos se ignoran (ruido)
TIEMPO_ARRASTRE = 0.5        # mantener el pellizco este tiempo empieza a arrastrar
TOLERANCIA_PELLIZCO = 0.2    # segundos sin pellizco antes de considerarlo terminado

# Mano abierta: deslizar vs. play/pausa
VENTANA_DESLIZAR = 0.5       # segundos de movimiento que se analizan para deslizar
DISTANCIA_DESLIZAR = 0.15    # fracción del ancho de la cámara que hay que recorrer
ESPERA_DESLIZAR = 0.7        # segundos mínimos entre dos deslizamientos
TIEMPO_QUIETO = 1.0          # mano abierta quieta este tiempo -> play/pausa
MOVIMIENTO_QUIETO = 0.04     # cuánto puede moverse la mano y seguir "quieta"
BLOQUEO_TRAS_DESLIZAR = 1.5  # después de deslizar, no hay play/pausa durante este tiempo
BLOQUEO_CONTRARIO = 1.0      # tras deslizar, se ignora el movimiento de vuelta en sentido contrario
SOLTAR_ALT = 0.8             # segundos sin mano abierta para elegir la aplicación (suelta Alt)
TECLAS_DESLIZAR_DERECHA = ("ctrl", "tab")
TECLAS_DESLIZAR_IZQUIERDA = ("ctrl", "shift", "tab")

# Otros
VELOCIDAD_SCROLL = 4000
ZONA_MUERTA_SCROLL = 0.003
REPETIR_VOLUMEN = 0.15       # segundos entre pasos de volumen al mantener el pulgar
PAGINA_ROCK = "https://www.google.com"

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True    # llevar el mouse a una esquina detiene pyautogui

DEDOS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (punta, articulación) índice..meñique
CONEXIONES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def distancia(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def dedos_levantados(lm):
    """Devuelve [pulgar, índice, medio, anular, meñique] como True/False.

    Un dedo está extendido si su punta está más lejos de la muñeca que su
    articulación; así funciona aunque la mano esté de lado o girada.
    """
    pulgar = distancia(lm[4], lm[17]) > distancia(lm[3], lm[17])
    otros = [distancia(lm[0], lm[p]) > distancia(lm[0], lm[a]) for p, a in DEDOS]
    return [pulgar] + otros


def distancia_3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def es_pellizco(lm, tam_mano, ya_pellizcando):
    """Pulgar tocando la PUNTA del índice (no la mitad del dedo, como en el puño).

    Para empezar se exige un pellizco claro; una vez empezado basta con que los
    dedos sigan cerca, así no se suelta solo mientras arrastras.
    """
    punta_con_punta = distancia_3d(lm[4], lm[8]) / tam_mano
    if ya_pellizcando:
        return punta_con_punta < UMBRAL_SOLTAR
    pulgar_a_mitad = distancia_3d(lm[4], lm[6]) / tam_mano
    indice_fuera_del_puno = distancia(lm[0], lm[8]) > distancia(lm[0], lm[5]) * 1.1
    return (punta_con_punta < UMBRAL_PELLIZCO
            and punta_con_punta < pulgar_a_mitad * 0.7
            and indice_fuera_del_puno)


def detectar_gesto(lm, ya_pellizcando=False):
    tam_mano = distancia(lm[0], lm[9]) or 1e-6
    pulgar, indice, medio, anular, menique = dedos_levantados(lm)

    if es_pellizco(lm, tam_mano, ya_pellizcando):
        return "pellizco"

    if not any([indice, medio, anular, menique]):
        altura_pulgar = (lm[2].y - lm[4].y) / tam_mano
        if altura_pulgar > 0.55:
            return "pulgar_arriba"
        if altura_pulgar < -0.55:
            return "pulgar_abajo"
        return "puno"
    if all([pulgar, indice, medio, anular, menique]):
        return "mano_abierta"
    if indice and medio and anular and menique:
        return "cuatro_dedos"
    if indice and medio and anular and not menique:
        return "tres_dedos"
    if indice and medio and not anular and not menique:
        return "paz"
    if indice and menique and not medio and not anular:
        return "rock"
    if pulgar and menique and not indice and not medio and not anular:
        return "pulgar_menique"
    if indice and not medio and not anular and not menique:
        return "indice"
    return None


def dibujar_mano(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONEXIONES:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, p in enumerate(pts):
        color = (0, 200, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
        cv2.circle(frame, p, 5, color, -1)


def captura_de_pantalla():
    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    ruta = CARPETA_CAPTURAS / f"captura_{datetime.now():%Y%m%d_%H%M%S}.png"
    pyautogui.screenshot(str(ruta))


# Acciones que se ejecutan una sola vez cada vez que haces el gesto
ACCIONES_UNICAS = {
    "paz": ("Siguiente cancion", lambda: pyautogui.press("nexttrack")),
    "rock": ("Abriendo pagina", lambda: webbrowser.open(PAGINA_ROCK)),
    "pulgar_menique": ("Captura guardada", captura_de_pantalla),
}


def main():
    if not MODELO.exists():
        raise SystemExit(f"No se encontró el modelo en {MODELO}")

    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    ancho_pantalla, alto_pantalla = pyautogui.size()
    cursor = list(pyautogui.position())
    historial_cursor = deque()      # (tiempo, x, y) del cursor, para clicar donde apuntabas

    def a_pantalla(px, py):
        """Convierte una posición de la cámara (0..1) en píxeles de la pantalla."""
        x = (px - MARGEN) / (1 - 2 * MARGEN)
        y = (py - MARGEN) / (1 - 2 * MARGEN)
        return (min(max(x, 0), 1) * (ancho_pantalla - 3) + 1,
                min(max(y, 0), 1) * (alto_pantalla - 3) + 1)

    def mover_cursor(x, y):
        # Los movimientos pequeños se frenan mucho (precisión) y los grandes casi nada (rapidez)
        dx, dy = x - cursor[0], y - cursor[1]
        dist = math.hypot(dx, dy)
        if dist < ZONA_MUERTA_CURSOR:
            return
        factor = min(SUAVIZADO_MAX, SUAVIZADO_MIN + dist / DISTANCIA_RAPIDA)
        cursor[0] += dx * factor
        cursor[1] += dy * factor
        pyautogui.moveTo(cursor[0], cursor[1])
        ahora = time.monotonic()
        historial_cursor.append((ahora, cursor[0], cursor[1]))
        while ahora - historial_cursor[0][0] > 1.0:
            historial_cursor.popleft()

    def posicion_antes_de(momento):
        """Dónde estaba el cursor en ese momento.

        Al pellizcar, el índice baja hacia el pulgar y arrastra el cursor un poco;
        por eso el clic se hace donde apuntabas justo antes de empezar el pellizco.
        """
        posicion = tuple(cursor)
        for t, x, y in reversed(historial_cursor):
            posicion = (x, y)
            if t <= momento:
                break
        return posicion

    activo = False
    gesto_actual, inicio_gesto, disparado = None, 0.0, False
    ultima_accion = {}
    ultima_repeticion = 0.0
    historial_mano = deque()        # (tiempo, x, y de la muñeca) con la mano abierta
    ultima_mano_abierta, ultimo_deslizar = -10.0, -10.0
    tipo_deslizar, ultima_direccion, alt_presionado = None, 0, False
    y_scroll_anterior = None
    inicio_pellizco, ultimo_pellizco, arrastrando = None, 0.0, False
    punto_clic, desfase_arrastre = None, (0.0, 0.0)
    texto_accion, hora_texto = "", 0.0

    def mostrar(texto):
        nonlocal texto_accion, hora_texto
        texto_accion, hora_texto = texto, time.monotonic()

    with vision.HandLandmarker.create_from_options(opciones) as detector:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # efecto espejo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = detector.detect_for_video(imagen, int(time.monotonic() * 1000))

            gesto, lm = None, None
            if resultado.hand_landmarks:
                lm = resultado.hand_landmarks[0]
                dibujar_mano(frame, lm)
                gesto = detectar_gesto(lm, ya_pellizcando=inicio_pellizco is not None)

            ahora = time.monotonic()
            if gesto != gesto_actual:
                gesto_actual, inicio_gesto, disparado = gesto, ahora, False
                y_scroll_anterior = None
            duracion = ahora - inicio_gesto

            # Al mover la mano rápido los dedos se ven borrosos y el gesto puede
            # cambiar un instante; por eso la mano abierta se sigue 0.3 s más.
            if gesto in ("mano_abierta", "cuatro_dedos"):
                ultima_mano_abierta = ahora
                tipo_deslizar = gesto
            siguiendo_mano = lm is not None and ahora - ultima_mano_abierta < 0.3
            if siguiendo_mano:
                historial_mano.append((ahora, lm[0].x, lm[0].y))
                while ahora - historial_mano[0][0] > max(VENTANA_DESLIZAR, TIEMPO_QUIETO):
                    historial_mano.popleft()
            else:
                historial_mano.clear()

            # --- Pellizco: clic corto o arrastrar ---
            if gesto == "pellizco":
                ultimo_pellizco = ahora
                if inicio_pellizco is None:
                    inicio_pellizco = ahora
                    punto_clic = posicion_antes_de(ahora - RETROCESO_CLIC)
                medio_x, medio_y = a_pantalla((lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2)
                if activo and not arrastrando and ahora - inicio_pellizco > TIEMPO_ARRASTRE:
                    cursor[:] = punto_clic
                    pyautogui.moveTo(*punto_clic)
                    pyautogui.mouseDown()
                    arrastrando = True
                    # El arrastre sigue a los dedos desde donde estaba el cursor, sin saltos
                    desfase_arrastre = (punto_clic[0] - medio_x, punto_clic[1] - medio_y)
                    mostrar("Arrastrando")
                elif arrastrando:
                    mover_cursor(medio_x + desfase_arrastre[0], medio_y + desfase_arrastre[1])
            elif inicio_pellizco is not None and ahora - ultimo_pellizco > TOLERANCIA_PELLIZCO:
                if arrastrando:
                    pyautogui.mouseUp()
                    arrastrando = False
                    mostrar("Soltar")
                elif activo and ultimo_pellizco - inicio_pellizco >= TIEMPO_MIN_CLIC:
                    cursor[:] = punto_clic
                    pyautogui.click(*punto_clic)
                    mostrar("Clic")
                inicio_pellizco = None

            if activo and inicio_pellizco is None:
                # --- Deslizar: mano abierta = pestañas, cuatro dedos = aplicaciones ---
                if siguiendo_mano and ahora - ultimo_deslizar > ESPERA_DESLIZAR:
                    xs = [x for t, x, _ in historial_mano if ahora - t <= VENTANA_DESLIZAR]
                    recorrido = xs[-1] - xs[0]
                    direccion = 1 if recorrido > 0 else -1
                    # Al volver la mano a su sitio después de deslizar no debe contar al revés
                    es_vuelta = (direccion != ultima_direccion
                                 and ahora - ultimo_deslizar < BLOQUEO_CONTRARIO)
                    if abs(recorrido) > DISTANCIA_DESLIZAR and es_vuelta:
                        historial_mano.clear()  # descartar ese movimiento para que no cuente luego
                    elif abs(recorrido) > DISTANCIA_DESLIZAR:
                        if alt_presionado or tipo_deslizar == "cuatro_dedos":
                            # Alt se mantiene apretado para ver el selector de ventanas;
                            # se suelta (y elige la app) al bajar la mano
                            if not alt_presionado:
                                pyautogui.keyDown("alt")
                                alt_presionado = True
                            if direccion > 0:
                                pyautogui.press("tab")
                                mostrar("Aplicacion siguiente >>")
                            else:
                                pyautogui.hotkey("shift", "tab")
                                mostrar("<< Aplicacion anterior")
                        elif direccion > 0:
                            pyautogui.hotkey(*TECLAS_DESLIZAR_DERECHA)
                            mostrar("Pestana siguiente >>")
                        else:
                            pyautogui.hotkey(*TECLAS_DESLIZAR_IZQUIERDA)
                            mostrar("<< Pestana anterior")
                        ultimo_deslizar, ultima_direccion = ahora, direccion
                        historial_mano.clear()
                        disparado = True

                # --- Índice: mover el cursor (sin esperar, para que sea fluido) ---
                if gesto == "indice" and not siguiendo_mano:
                    mover_cursor(*a_pantalla(lm[8].x, lm[8].y))

                # --- Mano abierta quieta: play/pausa ---
                elif gesto == "mano_abierta" and not alt_presionado:
                    quieta = [(x, y) for t, x, y in historial_mano if ahora - t <= TIEMPO_QUIETO]
                    if (quieta and not disparado and duracion > TIEMPO_QUIETO
                            and ahora - ultimo_deslizar > BLOQUEO_TRAS_DESLIZAR
                            and max(x for x, _ in quieta) - min(x for x, _ in quieta) < MOVIMIENTO_QUIETO
                            and max(y for _, y in quieta) - min(y for _, y in quieta) < MOVIMIENTO_QUIETO):
                        pyautogui.press("playpause")
                        mostrar("Play / Pausa")
                        disparado = True

                # --- Tres dedos: scroll ---
                elif gesto == "tres_dedos":
                    y = (lm[8].y + lm[12].y + lm[16].y) / 3
                    if y_scroll_anterior is not None and duracion > 0.15:
                        dy = y - y_scroll_anterior
                        if abs(dy) > ZONA_MUERTA_SCROLL:
                            pyautogui.scroll(int(-dy * VELOCIDAD_SCROLL))
                    y_scroll_anterior = y

                # --- Pulgar arriba / abajo: volumen (se repite) ---
                elif gesto in ("pulgar_arriba", "pulgar_abajo"):
                    if duracion > TIEMPO_CONFIRMAR and ahora - ultima_repeticion > REPETIR_VOLUMEN:
                        sube = gesto == "pulgar_arriba"
                        pyautogui.press("volumeup" if sube else "volumedown")
                        mostrar("Volumen +" if sube else "Volumen -")
                        ultima_repeticion = ahora

                # --- Gestos de una sola acción ---
                elif (gesto in ACCIONES_UNICAS and not disparado
                        and duracion > TIEMPO_CONFIRMAR
                        and ahora - ultima_accion.get(gesto, 0) > ESPERA_ENTRE_ACCIONES):
                    texto, accion = ACCIONES_UNICAS[gesto]
                    accion()
                    mostrar(texto)
                    ultima_accion[gesto] = ahora
                    disparado = True

            # Bajar la mano (o pausar) después de cambiar de app: soltar Alt para elegirla
            if alt_presionado and (not activo or ahora - ultima_mano_abierta > SOLTAR_ALT):
                pyautogui.keyUp("alt")
                alt_presionado = False
                mostrar("Aplicacion elegida")

            # --- Interfaz ---
            estado = "ACTIVO" if activo else "PAUSADO (ESPACIO para activar)"
            color = (0, 200, 0) if activo else (0, 0, 255)
            cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Gesto: {gesto or '-'}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if texto_accion and ahora - hora_texto < 1.5:
                cv2.putText(frame, texto_accion, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("ProjectStark - Gestos", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                activo = not activo
                if not activo and arrastrando:
                    pyautogui.mouseUp()
                    arrastrando = False

    if arrastrando:
        pyautogui.mouseUp()
    if alt_presionado:
        pyautogui.keyUp("alt")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
