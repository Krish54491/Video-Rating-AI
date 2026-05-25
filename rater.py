import cv2
import moviepy as mp
import numpy as np
import os
import speech_recognition as sr
import re
import threading
from openai import OpenAI
from flask import Flask, request, redirect, url_for, render_template_string, send_file, jsonify
from flask_cors import CORS
import shutil

app = Flask(__name__)
CORS(app,
     resources={r"/*": {"origins": [
        "https://krish544.com",
        "http://localhost:5173",
        "http://localhost:8788",
     ]}},
     allow_headers=["Content-Type", "*"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)
openrouter_key= os.getenv("OPENROUTER_API_KEY")

# Use thread-local storage to prevent concurrent requests from interfering with each other
request_context = threading.local()

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

def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        totalDuration = int(source.DURATION)
        chunkDuration = 30 # change this to change how long it listens before transcribing (less is recommended) the shorter it is the longer the rating will take
        # print(f"Audio duration: {totalDuration:.2f} sec")
        for i in range(0, totalDuration, chunkDuration):
            # print(f"Transcribing chunk {i}–{i+chunkDuration} seconds...")
            request_context.transcribedCount += 1
            try:
                audio = r.record(source,duration=chunkDuration)
                text = r.recognize_google(audio)
                if(len(text) > 10):
                    request_context.clear_audio+=1
                request_context.transcript += text + "\n "
                #print("Transcription: " + text)s
                with open("transcription.txt", "w") as f:
                    f.write(request_context.transcript)
            except sr.UnknownValueError:
                # print("Could not understand audio")
                request_context.unclear_audio+= 1
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        video.audio.close()
        video.close()
        return audio_path


def rate_script():
    if(not request_context.transcript):
       return -1
    
    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not set!")
        return -1
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=
            [
              {
                "role": "user",
                "content": ("Rate how good this video is out of ten based on the transcript of it, before giving your rating say exactly as follows \\\"My final rating for the video is\\\" You must say that before giving your number rating! Under no circumstances should you add asterisks! Transcript:" + request_context.transcript)
                }
            ]
        )

        try:
            request_context.aiComment = completion.choices[0].message.content
            print(completion.choices[0].message.content)
            script_score = float((re.search(r'My final rating for the video is\s+(\d+(?:\.\d+)?)', completion.choices[0].message.content, re.IGNORECASE)).group(1))

        except Exception as e:
            print("Error parsing response:", e)
            script_score = -1.0

        
        # print(f"Transcript Script Score: {script_score:.2f}")
        return script_score*.1
    except Exception as e:
        print(f"Error calling OpenRouter API: {str(e)}")
        return -1

@app.route('/download-transcript')
def download_transcript():
    transcript_path = "transcription.txt"
    if os.path.exists(transcript_path):
        return send_file(transcript_path, as_attachment=True)
    else:
        return "Transcript not found.", 404

@app.route("/rate")
def rate_video(video_path):
    # Initialize thread-local context for this request
    request_context.transcript = ""
    request_context.aiComment = ""
    request_context.unclear_audio = 0
    request_context.clear_audio = 0
    request_context.transcribedCount = 0
    
    # Use unique audio filename based on video filename to support concurrent requests
    video_filename = os.path.basename(video_path)
    audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{video_filename}.wav")
    
    try:
        # print("Analyzing video clarity...")
        visual_score = average_blurriness(video_path)
        norm_visual = min(max(visual_score / 1000, 0), 1.0)
        # print(f"Blurriness score: {(norm_visual*10):.2f}")
        # print("Extracting and analyzing audio...")
        extract_audio(video_path, audio_path)

        # Safely calculate audio score
        if request_context.transcribedCount > 0:
            audio_score = (request_context.clear_audio - request_context.unclear_audio) / request_context.transcribedCount
        else:
            audio_score = 0
        # print(f"Audio RMS score: {(audio_score*10):.5f}")

        # Clamp all scores to [0, 1] to prevent negative scores
        norm_audio = min(max(audio_score, 0), 1.0)
        script_raw = rate_script()
        norm_script = min(max(script_raw, 0), 1.0)
        
        # Weighted average
        final_score = 0.2 * norm_visual + 0.4 * norm_audio + .4 * norm_script
        # print(f"\nFinal Content Quality Score (0–10): {(final_score*10):.2f}")

        temp_aiComment = request_context.aiComment
        return f'''
        <p>Visual Score: {(norm_visual*10):.2f}</p>
        <p>Audio Score: {(norm_audio*10):.2f}</p>
        <p>Script Score: {(norm_script*10):.2f}</p>
        <p>AI Comment: {temp_aiComment}</p>
        <p>Final Score: {(final_score*10):.2f}</p>
        '''
    
    except Exception as e:
        print(f"Error in rate_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return f'''
        <p>Error: {str(e)}</p>
        <p>Final Score: -1</p>
        '''
    
    finally:
        # Cleanup on success or error
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Failed to delete audio file {audio_path}: {e}")
        
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Failed to delete video file {video_path}: {e}")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        # Clear the uploads folder before saving the new file
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

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
            return '''
            <!doctype html>
            <html>
            <head>
            <title>AI Video Rater</title>
            <script>
              // Hide progress message after rating is done
              window.onload = function() {{
                document.getElementById('progress').style.display = 'none';
              }};
            </script>
            </head>
            <body>
            <h1>AI Video Rater</h1>
            <p>If the script score is -1 it means the AI errored out and could not give a rating. Also only works for english videos and videos shorter than 30 minutes</p>
            <form method=post enctype=multipart/form-data onsubmit="document.getElementById('progress').style.display='block';">
              <input type=file name=video accept="video/*" required>
              <input type=submit value=Upload>
            </form>
            <div id="progress" style="color:blue; font-weight:bold; display:none;">
              Video Rating in Progress...
            </div>''' + result + '''
            <p><a href="/download-transcript" download>
               <button type="button">Download Transcript</button>
            </a></p>
            </body>
            </html>
            '''
        else:
            return "Invalid file type. Please upload a video file."
    return '''
    <!doctype html>
    <html>
    <head>
    <title>AI Video Rater</title>
    <script>
      function showProgress() {
        document.getElementById('progress').style.display = 'block';
      }
    </script>
    </head>
    <body>
    <h1>AI Video Rater</h1>
    <p>If the script score is -1 it means the AI errored out and could not give a rating. Also only works for english videos and videos shorter than 30 minutes</p>
    <form method=post enctype=multipart/form-data onsubmit="showProgress()">
      <input type=file name=video accept="video/*" required>
      <input type=submit value=Upload>
    </form>
    <div id="progress" style="color:blue; font-weight:bold; display:none;">
      Video Rating in Progress...
    </div>
    </body>
    </html>
    '''

@app.route("/api/rate", methods=["POST", "OPTIONS"])
def api_rate():
    """
    POST /api/rate
    FormData:
      video: file
    Returns:
      {
        "success": bool,
        "result": html_string,
        "error": error_message (if success=false)
      }
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        if "video" not in request.files:
            return jsonify({"success": False, "error": "No video file uploaded"}), 400

        video = request.files["video"]
        
        if video.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(video.filename):
            return jsonify({"success": False, "error": "Invalid file type. Please upload a video file."}), 400

        filename = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(filename)
        
        print(f"Processing video: {filename}")
        result = rate_video(filename)
        
        return jsonify({"success": True, "result": result})
    
    except Exception as e:
        print(f"Error in /api/rate: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/api_transcript', methods=['GET', 'OPTIONS'])
def api_transcript():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    file_path = 'transcription.txt'
    try:
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route("/api/status")
def root():
    return jsonify({"status": "API running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)