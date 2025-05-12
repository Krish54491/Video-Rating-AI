# Video-Rating-AI
This program will rate a video using AI to rate a transcript of the video, it will also rate the visuals on how blurry it is and the audio on how clear it is.


# Setup
You will have to make an API key using OpenRouter for deepseek free or this link
https://openrouter.ai/deepseek/deepseek-prover-v2:free 
replace the variable OPENAI_API_KEY in
```
def rate_script():
    if(not transcript):
       return 0.0
    #print(f"Rate how good this video is out of ten based on the transcript of it, before giving your rating saying \\\"My final rating for the video is\\\" Transcript:"+transcript)
    
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
    )
```
Put in the file of the video you want to rate in the folder with the program.\
Finally replace the video_file value with the name of your video file (don't forget the file ending (Example: .mp4))
```
if __name__ == "__main__":
    video_file = "Lab 10 Demonstration Video.mp4"  # Replace with your video name
    rate_video(video_file)
```
Now run Rater.py and see how good(or bad) your video is!
