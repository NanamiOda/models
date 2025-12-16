import whisper
import os
import time
import psutil 
import threading
import soundfile as sf
import pyaudio
import numpy as np
import tempfile
import serial
import time
import serial.tools.list_ports
import unicodedata


puerto = 'COM5'

arduino = serial.Serial(puerto, baudrate=9600, timeout=1)
time.sleep(2) 

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper trabaja con 16kHz
RECORD_SECONDS = 10 




peak_ram = 0.0
running = True
words_per_line = 12



def record_audio(seconds=RECORD_SECONDS):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"[Grabando {seconds}s desde micrófono...]")
    frames = []

    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("[Grabación finalizada]")
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_np = np.hstack(frames).astype(np.float32) / 32768.0  # normalizar a [-1, 1]
    return audio_np

def track_ram():
    global peak_ram
    while running:
        current_ram = process.memory_info().rss / (1024 * 1024)
        peak_ram = max(peak_ram, current_ram)
        time.sleep(0.1)  # chequea cada 100ms


def normalizar_texto(texto):
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
    texto = normalizar_texto(texto)
    for letra in texto:
        if letra.isalpha() or letra == 'ñ':
            arduino.write(letra.encode('utf-8'))
            print(f"Enviando: {letra}")
            time.sleep(2)




process = psutil.Process(os.getpid())

brailleDict = { 'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓',
                'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏',
                'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
                'y': '⠽', 'z': '⠵', 'á': '⠁', 'é': '⠑', 'í': '⠊', 'ó': '⠕', 'ú': '⠥', 'ñ': '⠝', 'ü': '⠥', ' ': ' ', '.': '⠲',                 
                }



# Iniciar temporizador
start_time = time.time()

model = whisper.load_model("base",download_root="E://Universidad//quinto//tesis otra//spech-to-text")

#audio = whisper.load_audio("E://Universidad//quinto//tesis otra//spech-to-text//audioCuentos//los3cerditos.mp3")
#audio = whisper.pad_or_trim(audio)

audio = record_audio(5)
audio = whisper.pad_or_trim(audio)


# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)



# Comienza a monitorear RAM en segundo plano
monitor_thread = threading.Thread(target=track_ram)
monitor_thread.start()

# decode the audio
options = whisper.DecodingOptions(language="es", without_timestamps=True)
result = model.transcribe(audio, language="es")

texto = result["text"]
# Enviar texto a Arduino
enviar_texto_a_arduino(texto,arduino)
arduino.close()



# Procesar el texto antes de escribirlo
words = result["text"].split()
lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
formatted_text = "\n".join(lines)

#  Finalizar monitoreo
end_time = time.time()
running = False
monitor_thread.join()

print(f"\n--- Estadísticas ---")
print(f"Tiempo de inferencia: {end_time - start_time:.2f} segundos")
print(f"Pico de RAM durante inferencia: {peak_ram:.2f} MB")

print(f"\n--- Resultados ---")
# print the recognized text
print(result["text"])



# save the result to a text file
#with open("E://Universidad//quinto//tesis otra//spech-to-text//textoSmallmodel//los3cerditos.txt", "w", encoding="utf-8") as f:
#    f.write(formatted_text)


# pass text file to braille

def text_to_braille(text):
    """
    Convert text to braille using the braille dictionary.
    """
    braille_text = ""
    for char in text:
        if char.lower() in brailleDict:
            braille_text += brailleDict[char.lower()]
        else:
            braille_text += char
    return braille_text




braille_text = text_to_braille(formatted_text)


#with open("E://Universidad//quinto//tesis otra//spech-to-text//result_los3cerditostasmallbrailecompleto.txt", "w", encoding="utf-8") as f:
#    f.write(braille_text)

print(f"\n--- Texto en Braille ---")
print(braille_text)

# print the braille text
#print(braille_text)
