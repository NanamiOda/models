import os
import time
import psutil 
import threading
import pyaudio
import numpy as np
import json
import serial
import time
import serial.tools.list_ports
import unicodedata
import wave
import subprocess


try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    print("Error: Vosk no está instalado")
    print("Instalar con: pip install vosk")
    VOSK_AVAILABLE = False

puerto = 'COM5'

arduino = serial.Serial(puerto, baudrate=9600, timeout=1)
time.sleep(2) 

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Vosk funciona bien con 16kHz
RECORD_SECONDS = 10 

peak_ram = 0.0
running = True
words_per_line = 12

def record_audio_streaming(seconds=RECORD_SECONDS, model=None, rec=None):
    """Grabar audio en streaming y procesar con Vosk en tiempo real"""
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"[Grabando {seconds}s desde micrófono con Vosk streaming...]")
    
    transcription_parts = []
    
    # Procesar audio en chunks
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        
        # Procesar cada chunk con Vosk
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                transcription_parts.append(result['text'])
                print(f"[Chunk {i+1}] Reconocido: {result['text']}")
        else:
            # Resultado parcial (opcional, para feedback en tiempo real)
            partial = json.loads(rec.PartialResult())
            if 'partial' in partial and partial['partial'].strip():
                print(f"[Parcial] {partial['partial']}", end='\r')

    # Obtener resultado final
    final_result = json.loads(rec.FinalResult())
    if 'text' in final_result and final_result['text'].strip():
        transcription_parts.append(final_result['text'])

    print(f"\n[Grabación finalizada]")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Unir todas las partes de la transcripción
    full_transcription = " ".join(transcription_parts).strip()
    return full_transcription

def track_ram():
    """Monitor de RAM en segundo plano"""
    global peak_ram
    while running:
        current_ram = process.memory_info().rss / (1024 * 1024)
        peak_ram = max(peak_ram, current_ram)
        time.sleep(0.1)  # chequea cada 100ms

def normalizar_texto(texto):
    """Normalizar texto para Arduino (sin tildes pero con ñ)"""
    texto = texto.lower()
    # Normaliza y elimina tildes (NFD separa base + tilde)
    texto = unicodedata.normalize('NFD', texto)
    texto_sin_tildes = []
    for c in texto:
        if c == 'ñ':  # dejamos la ñ intacta
            texto_sin_tildes.append(c)
        else:
            # ignorar caracteres que son marcas diacríticas
            if unicodedata.category(c) != 'Mn':
                texto_sin_tildes.append(c)
    return ''.join(texto_sin_tildes)

def enviar_texto_a_arduino(texto, arduino):
    """Enviar texto normalizado al Arduino letra por letra"""
    texto = normalizar_texto(texto)
    for letra in texto:
        if letra.isalpha() or letra == 'ñ':
            arduino.write(letra.encode('utf-8'))
            print(f"Enviando: {letra}")
            time.sleep(2)

def text_to_braille(text):
    """Convertir texto a braille usando el diccionario braille"""
    brailleDict = { 
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓',
        'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏',
        'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
        'y': '⠽', 'z': '⠵', 'á': '⠁', 'é': '⠑', 'í': '⠊', 'ó': '⠕', 'ú': '⠥', 'ñ': '⠝', 
        'ü': '⠥', ' ': ' ', '.': '⠲',                 
    }
    
    braille_text = ""
    for char in text:
        if char.lower() in brailleDict:
            braille_text += brailleDict[char.lower()]
        else:
            braille_text += char
    return braille_text

# Inicializar proceso para monitoreo
process = psutil.Process(os.getpid())

# Iniciar temporizador
start_time = time.time()

print("  Inicializando Vosk...")

# Verificar que Vosk esté disponible
if not VOSK_AVAILABLE:
    print(" Vosk no está disponible. Terminando programa.")
    exit(1)

# Cargar modelo Vosk (español)
try:
    # Intentar cargar modelo local si existe
    if os.path.exists("vosk-model-es"):
        print(" Cargando modelo Vosk local...")
        model = vosk.Model("vosk-model-es")
    else:
        print(" Descargando modelo Vosk automático...")
        model = vosk.Model(lang="es")
except Exception as e:
    print(f" Error cargando modelo Vosk: {e}")
    print(" Asegúrate de tener Vosk instalado: pip install vosk")
    exit(1)

# Crear reconocedor
rec = vosk.KaldiRecognizer(model, RATE)
rec.SetWords(True)  # Habilitar palabras individuales

print(" Modelo Vosk cargado correctamente")

# Comienza a monitorear RAM en segundo plano
monitor_thread = threading.Thread(target=track_ram)
monitor_thread.start()

# Grabar y transcribir en streaming
print("\n Iniciando grabación con reconocimiento en tiempo real...")
texto = record_audio_streaming(5, model, rec)

print(f"\n Texto reconocido: {texto}")

# Enviar texto a Arduino
if texto.strip():
    print("\n Enviando texto al Arduino...")
    enviar_texto_a_arduino(texto, arduino)
else:
    print("  No se reconoció texto para enviar")

arduino.close()

# Procesar el texto para formato (igual que whispermodel.py)
words = texto.split()
lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
formatted_text = "\n".join(lines)

# Finalizar monitoreo
end_time = time.time()
running = False
monitor_thread.join()

print(f"\n--- Estadísticas ---")
print(f"Tiempo de inferencia: {end_time - start_time:.2f} segundos")
print(f"Pico de RAM durante inferencia: {peak_ram:.2f} MB")

print(f"\n--- Resultados ---")
print(f"Texto original: {texto}")
print(f"Texto formateado:")
print(formatted_text)

# Convertir a braille
braille_text = text_to_braille(formatted_text)

print(f"\n--- Texto en Braille ---")
print(braille_text)

# Opcional: Guardar resultados
# with open("vosk_transcription.txt", "w", encoding="utf-8") as f:
#     f.write(formatted_text)
# 
# with open("vosk_braille.txt", "w", encoding="utf-8") as f:
#     f.write(braille_text)

print("\n🎉 Proceso completado!")
