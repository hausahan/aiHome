import requests
import wave
import pyaudio
import whisper
import datetime
import pyttsx3


model_audio = whisper.load_model("medium").to("cuda")

def printLogWithTime(s):
    current_time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{current_time} {s}")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def recordAndAnalyze(time):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = "./output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    printLogWithTime("* Recording")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    printLogWithTime("* Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio = whisper.load_audio(WAVE_OUTPUT_FILENAME)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_audio.device)
    options = whisper.DecodingOptions(language='English', fp16=True)
    result = whisper.decode(model_audio, mel, options)
    return result.text

if __name__ == '__main__':
    gotCall = False

    while True:
        re = recordAndAnalyze(2)
        printLogWithTime(re)
        if "hello" in re.lower():
            gotCall = True
            printLogWithTime("Got Call, sending request to HTTPServer")
            text_to_speech("What can I do for you?")
            user_input = recordAndAnalyze(10)
            printLogWithTime(f"Recognized input: {user_input}")
            printLogWithTime(f"Requesting LLama")
            response = requests.post(
                'http://192.168.1.3:50001/process',
                json={"text": user_input}
            )

            if response.status_code == 200:
                reply = response.json().get('response', 'Sorry, no response received.')
                printLogWithTime(reply)
                text_to_speech(reply)
            else:
                printLogWithTime("Error in response from server.")
                text_to_speech("Sorry, I couldn't process your request.")
        elif "shut down" in re.lower():
            printLogWithTime("Shutting down client.")
            break
