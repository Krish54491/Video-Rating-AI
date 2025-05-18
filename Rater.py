import cv2
import librosa
import moviepy as mp
import numpy as np
import os
import speech_recognition as sr
import textstat
import re
from openai import OpenAI
import sys
import os

# Add the folder 'Api_key' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Api_key'))
from key import OPENAI_API_KEY

transcript = ""
unclear_audio = 0
clear_audio= 0
transcribedCount = 0
print("This program will rate a video using AI to rate a transcript of the video. \nThen will rate the visuals on how blurry it is and the audio on how clear it is")

def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

def blurriness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def average_blurriness(video_path):
    frames = extract_frames(video_path)
    scores = [blurriness_score(f) for f in frames]
    return np.mean(scores)

def extract_audio(video_path, audio_path="temp_audio.wav"):
    global transcript
    global clear_audio
    global unclear_audio
    global transcribedCount
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    r = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        totalDuration = int(source.DURATION)
        chunkDuration = 30 # change this to change how long it listens before transcribing (less is recommended) the shorter it is the longer the rating will take
        print(f"Audio duration: {totalDuration:.2f} sec")
        for i in range(0, totalDuration, chunkDuration):
            print(f"Transcribing chunk {i}–{i+chunkDuration} seconds...")
            try:
                audio = r.record(source,duration=chunkDuration)
                text = r.recognize_google(audio)
                transcribedCount += 1
                if(len(text) > 10):
                    clear_audio+=1
                transcript += text + "\n "
                #print("Transcription: " + text)
                with open("transcription.txt", "w") as f:
                    f.write(transcript)
            except sr.UnknownValueError:
                print("Could not understand audio")
                unclear_audio+= 1
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return audio_path

#def audio_rms_score(audio_path):
#    y, sr = librosa.load(audio_path)
#    rms = librosa.feature.rms(y=y)[0]
#    return np.mean(rms)
def rate_script():
    if(not transcript):
       return 0.0
    #print(f"Rate how good this video is out of ten based on the transcript of it, before giving your rating saying \\\"My final rating for the video is\\\" Transcript:"+transcript)
    
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
    )



    completion = client.chat.completions.create(
    
        model="deepseek/deepseek-prover-v2:free",
        
        messages=
        [
          {
            "role": "user",
            "content": ("Rate how good this video is out of ten based on the transcript of it, before giving your rating saying \\\"My final rating for the video is\\\" Don't bold the rating only do the number out of ten! Transcript:" + transcript)
            }
        ]
    )
    
    try:
        print(completion.choices[0].message.content)
    except Exception as e:
        print("Error accessing response:", e)
        print("Full response:", completion)
    
    script_score = float((re.search(r'My final rating for the video is\s+(\d+(?:\.\d+)?)', completion.choices[0].message.content, re.IGNORECASE)).group(1))


    #readability_score = textstat.flesch_reading_ease(transcript) # based on readablity and not how good the script is


    print(f"Transcript Script Score: {script_score:.2f}")
    return script_score*.1

def rate_video(video_path):
    print("Analyzing video clarity...")
    visual_score = average_blurriness(video_path)
    norm_visual = min(visual_score / 1000, 1.0)
    print(f"Blurriness score: {(norm_visual*10):.2f}")
    print("Extracting and analyzing audio...")
    audio_path = extract_audio(video_path)

    #audio_score = audio_rms_score(audio_path)
    audio_score = (clear_audio-unclear_audio)/ transcribedCount
    print(f"Audio RMS score: {(audio_score*10):.5f}")
    
    # Normalize scores to 0–1 range for simplicity
    
    norm_audio = min(audio_score, 1.0)
    script_score = min(rate_script(), 1.0)
    # Weighted average
    final_score = 0.2 * norm_visual + 0.4 * norm_audio + .4 * script_score
    print(f"\nFinal Content Quality Score (0–10): {(final_score*10):.2f}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return final_score

if __name__ == "__main__":
    video_file = "Lab 10 Demonstration Video.mp4"  # Replace with your video name
    rate_video(video_file)