import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from openwakeword.model import Model
import subprocess


VOICE_ID = "j57KDF72L6gxbLk4sOo5"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

openai_client = OpenAI(api_key=OPENAI_KEY)
tts_client = ElevenLabs(api_key=ELEVEN_KEY)

wake_model = Model(
    wakeword_models=["wakeword_models/jarvis.onnx"], 
    model_format="onnx"
)

def speak(text):
    print(f"Jarvis says: {text}")

    audio_stream = tts_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        text=text
    )

    with open("jarvis_output.mp3", "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    os.system("afplay jarvis_output.mp3")



def listen_for_command():
    print("Speak your command...")

    duration = 4
    fs = 44100

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()

    wav_path = "command.wav"
    write(wav_path, fs, audio)

    with open(wav_path, "rb") as f:
        transcription = openai_client.audio.transcriptions.create(
            file=f,
            model="gpt-4o-mini-transcribe"
        )

    text = transcription.text.lower().strip()
    print("Command recognized:", text)
    return text


def listen():
    print("\nListening for wake word 'jarvis'...")

    fs = 16000
    chunk_duration = 0.5
    chunk_samples = int(fs * chunk_duration)

    while True:
        audio = sd.rec(chunk_samples, samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()

        pred = wake_model.predict(audio.flatten())
        score = pred.get("jarvis", 0.0)

        if confidence > 0.6:
            print("Wake word detected!")
            speak("Yes sir?")
            return listen_for_command()


def ask_gpt(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Jarvis: calm, intelligent, precise, and professional."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content



def execute_mac_command(cmd):
    cmd = cmd.lower()

    if "open safari" in cmd:
        subprocess.run(["open", "-a", "Safari"])
        return "Opening Safari."

    if "open chrome" in cmd:
        subprocess.run(["open", "-a", "Google Chrome"])
        return "Opening Google Chrome."

    if "open vscode" in cmd or "open vs code" in cmd:
        subprocess.run(["open", "-a", "Visual Studio Code"])
        return "Opening Visual Studio Code."

    if "shutdown" in cmd:
        subprocess.run(["osascript", "-e", 'tell app "System Events" to shut down'])
        return "Shutting down."

    return None



def main():
    speak("Jarvis online and awaiting your call, sir.")

    while True:
        command = listen()  

        if not command or len(command.strip()) == 0:
            continue

        mac_action = execute_mac_command(command)
        if mac_action:
            speak(mac_action)
            continue

        response = ask_gpt(command)
        speak(response)



if __name__ == "__main__":
    main()
