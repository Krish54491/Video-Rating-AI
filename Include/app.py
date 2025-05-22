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
from flask import Flask, request, redirect, url_for, render_template_string, send_file

app = Flask(__name__)
transcript = ""
aiComment = ""
unclear_audio = 0
clear_audio= 0
transcribedCount = 0
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# print("This program will rate a video using AI to rate a transcript of the video. \nThen will rate the visuals on how blurry it is and the audio on how clear it is")

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
        # print(f"Audio duration: {totalDuration:.2f} sec")
        for i in range(0, totalDuration, chunkDuration):
            # print(f"Transcribing chunk {i}–{i+chunkDuration} seconds...")
            transcribedCount += 1
            try:
                audio = r.record(source,duration=chunkDuration)
                text = r.recognize_google(audio)
                if(len(text) > 10):
                    clear_audio+=1
                transcript += text + "\n "
                #print("Transcription: " + text)s
                with open("transcription.txt", "w") as f:
                    f.write(transcript)
            except sr.UnknownValueError:
                # print("Could not understand audio")
                unclear_audio+= 1
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        video.audio.close()
        video.close()
        return audio_path


def rate_script():
    if(not transcript):
       return 0.0
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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", #feel free to use this key, it has a dollar limit on it doesn't matter if you use it
    )
    completion = client.chat.completions.create(
        model="deepseek/deepseek-prover-v2:free",
        messages=
        [
          {
            "role": "user",
            "content": ("Rate how good this video is out of ten based on the transcript of it, before giving your rating say exactly as follows \\\"My final rating for the video is\\\" You must say that before giving your number rating! Under no circumstances should you add asterisks! Transcript:" + transcript)
            }
        ]
    )

    try:
        global aiComment
        aiComment = completion.choices[0].message.content
        print(completion.choices[0].message.content)
        script_score = float((re.search(r'My final rating for the video is\s+(\d+(?:\.\d+)?)', completion.choices[0].message.content, re.IGNORECASE)).group(1))

    except Exception as e:
        # print("Error accessing response:", e)
        # print("Full response:", completion)
        script_score = -1.0

    
    # print(f"Transcript Script Score: {script_score:.2f}")
    return script_score*.1

@app.route('/download-transcript')
def download_transcript():
    transcript_path = "transcription.txt"
    if os.path.exists(transcript_path):
        return send_file(transcript_path, as_attachment=True)
    else:
        return "Transcript not found.", 404

@app.route("/rate")
def rate_video(video_path):
    # print("Analyzing video clarity...")
    visual_score = average_blurriness(video_path)
    norm_visual = min(visual_score / 1000, 1.0)
    # print(f"Blurriness score: {(norm_visual*10):.2f}")
    # print("Extracting and analyzing audio...")
    audio_path = extract_audio(video_path)

    audio_score = (clear_audio-unclear_audio)/ transcribedCount
    # print(f"Audio RMS score: {(audio_score*10):.5f}")

    norm_audio = min(audio_score, 1.0)
    script_score = min(rate_script(), 1.0)
    # Weighted average
    final_score = 0.2 * norm_visual + 0.4 * norm_audio + .4 * script_score
    # print(f"\nFinal Content Quality Score (0–10): {(final_score*10):.2f}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if(os.path.exists(video_path)):
        os.remove(video_path)
    return f'''
    <p>Visual Score: {(norm_visual*10):.2f}</p>
    <p>Audio Score: {(audio_score*10):.2f}</p>
    <p>Script Score: {(script_score*10):.2f}</p>
    <p>AI Comment: {aiComment}</p>
    <p>Final Score: {(final_score*10):.2f}</p>
    '''
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if 'video' not in request.files:
            return "No file part"
        file = request.files['video']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)
            # Call rate_video with the uploaded file path
            result = rate_video(video_path)
            return f'''
            <!doctype html>
            <title>AI Video Rater</title>
            <h1>AI Video Rater</h1>
            <p>If the script score is -1 it means the AI errored out and could not give a rating. Also only works for english videos and videos shorter than 30 minutes</p>
            <form method=post enctype=multipart/form-data>
              <input type=file name=video accept="video/*" required>
              <input type=submit value=Upload>
            </form>
            {result}
            <p><a href="/download-transcript" download>
               <button type="button">Download Transcript</button>
            </a></p>
            '''
        else:
            return "Invalid file type. Please upload a video file."
    return '''
    <!doctype html>
    <title>AI Video Rater</title>
    <h1>AI Video Rater</h1>
    <p>If the script score is -1 it means the AI errored out and could not give a rating. Also only works for english videos and videos shorter than 30 minutes</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=video accept="video/*" required>
      <input type=submit value=Upload>
    </form>
    '''

#if __name__ == "__main__":
#    video_file = "Lab 10 Demonstration Video.mp4"  # Replace with your video name
#    rate_video(video_file)