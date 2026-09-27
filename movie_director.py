import base64
from io import BytesIO
import asyncio
from PIL import Image
import numpy as np
import os
import io
from moviepy import (
    ImageClip,
    concatenate_videoclips,
    concatenate_audioclips,
    AudioFileClip,
)
import moviepy as mpe


async def make_movie(movie_script, manga, volume_number, narrator):
    print("Narrating movie script...")
    await add_narrations_to_script(movie_script, narrator)
    print("Editing movie together...")
    create_movie_from_script(movie_script, manga, volume_number)
    print("Movie created successfully!")
    return True


ESPEAK_DATA_DIRS = [
    "/usr/share/espeak-ng-data",
    "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
    "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/opt/homebrew/share/espeak-ng-data",
]


def use_system_espeak():
    # The espeak-ng copy bundled via espeakng-loader can ship without its phoneme
    # tables (phontab), which aborts the process, so prefer a system install.
    import ctypes.util
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    library = ctypes.util.find_library("espeak-ng")
    data_path = next(
        (d for d in ESPEAK_DATA_DIRS if os.path.isfile(os.path.join(d, "phontab"))),
        None,
    )
    if library and data_path:
        EspeakWrapper.set_library(library)
        EspeakWrapper.set_data_path(data_path)


class KokoroNarrator:
    """Free, local text-to-speech using the open-weight Kokoro-82M model."""

    SAMPLE_RATE = 24000

    def __init__(self):
        from kokoro import KPipeline

        use_system_espeak()

        self.voice = os.getenv("KOKORO_VOICE", "am_michael")
        self.speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        # Kokoro voice names start with their language code, e.g. "a" = American English
        lang_code = os.getenv("KOKORO_LANG_CODE", self.voice[0])
        self.pipeline = KPipeline(lang_code=lang_code)

    def narrate(self, text):
        import soundfile as sf

        chunks = [
            np.asarray(audio)
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed)
            if audio is not None
        ]
        audio_bytes_io = BytesIO()
        sf.write(audio_bytes_io, np.concatenate(chunks), self.SAMPLE_RATE, format="WAV")
        audio_bytes_io.seek(0)
        return audio_bytes_io


# Function to generate and update movie script with narrations
async def add_narrations_to_script(script, narrator):
    # Kokoro runs locally and its pipeline is not thread-safe, so segments are
    # narrated one at a time off the event loop
    for entry in script:
        entry["narration"] = await asyncio.to_thread(narrator.narrate, entry["text"])
        print("got bytes:", entry["narration"].getbuffer().nbytes)


def create_movie_from_script(script, manga, volume_number):
    video_clips = []
    audio_clips = []  # Collect audio clips here

    for segment_index, segment in enumerate(script):
        narration_audio = segment["narration"]  # This is a BytesIO object
        important_panels = segment["important_panels"]
        base64_images = segment[
            "images_unscaled"
        ]  # This should be a list of base64 strings
        print(f"Number of images in segment {segment_index}:", len(base64_images))
        scene_images = []

        if len(base64_images) == 0:
            print(f"No images found for segment {segment_index}. Skipping...")
            continue

        scene_images = important_panels
        if len(important_panels) == 0:
            scene_images = base64_images

        # Create a unique temporary audio file path for each segment
        temp_audio_path = f"{manga}/v{volume_number}/temp_audio_{segment_index}.wav"
        with open(temp_audio_path, "wb") as out_file:
            out_file.write(narration_audio.getvalue())

        audio_clip = AudioFileClip(temp_audio_path)
        audio_clips.append(audio_clip)  # Add to the list of audio clips

        audio_duration = audio_clip.duration
        print("Audio duration:", audio_duration)

        image_display_duration = audio_duration / len(scene_images)
        segment_clips = []
        for base64_image in scene_images:
            image_data = base64.b64decode(base64_image)
            scaled_image = scale_image_to_720p(
                image_data
            )  # Assume this function returns scaled image data

            image = Image.open(BytesIO(scaled_image))
            final_image = add_image_to_background(
                image
            )  # Assume this function adds image to background

            image_clip = mpe.ImageClip(np.array(final_image)).with_duration(
                image_display_duration
            )
            segment_clips.append(image_clip)

        # Concatenate all image clips for this segment and do not set audio here
        video_clip = concatenate_videoclips(segment_clips)
        video_clips.append(video_clip)

    # Concatenate all video segments together
    final_video_clip = concatenate_videoclips(video_clips)

    # Concatenate all audio clips together
    final_audio_clip = concatenate_audioclips(audio_clips)

    # Set the concatenated audio clip to the final video clip
    final_video_clip = final_video_clip.with_audio(final_audio_clip)

    # Final movie path
    final_movie_path = f"{manga}/v{volume_number}/recap.mp4"

    # Write the final video file with audio
    final_video_clip.write_videofile(
        final_movie_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=f"{manga}/v{volume_number}/temp-audio.m4a",
        remove_temp=True,
        fps=24,
    )

    # Cleanup temporary audio files
    for audio_clip in audio_clips:
        audio_clip.close()  # Close the clip to release the file
    for segment_index in range(len(script)):
        temp_audio_path = f"{manga}/v{volume_number}/temp_audio_{segment_index}.wav"
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return final_movie_path


def scale_image_to_720p(image_bytes, target_width=1280, target_height=720):
    # Convert bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Calculate the target size to maintain aspect ratio
    original_width, original_height = image.size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert the PIL Image back to bytes
    img_byte_arr = io.BytesIO()
    resized_image.save(
        img_byte_arr, format="JPEG"
    )  # Save as JPEG for better compatibility with video formats
    scaled_image_bytes = img_byte_arr.getvalue()

    return scaled_image_bytes


def add_image_to_background(image, background_size=(1280, 720)):
    background = Image.new("RGB", background_size, (0, 0, 0))
    # Calculate the position to paste the scaled image on the background
    bg_width, bg_height = background_size
    img_width, img_height = image.size
    x = (bg_width - img_width) // 2
    y = (bg_height - img_height) // 2
    background.paste(image, (x, y))
    return background
