from dotenv import load_dotenv
import asyncio
import json
import os
import argparse

from manga_extraction import (
    extract_all_pages_as_images,
    save_important_pages,
    split_volume_into_parts,
    save_all_pages,
    extract_panels,
    scale_base64_image,
)
from vision_analysis import (
    analyze_images_with_vision,
    create_client,
    detect_important_pages,
    vision_model,
)
from prompts import (
    DRAMATIC_PROMPT,
    BASIC_PROMPT,
    BASIC_PROMPT_WITH_CONTEXT,
    BASIC_INSTRUCTIONS,
    KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
)
from citation_processing import extract_text_and_citations, extract_script
from movie_director import make_movie, KokoroNarrator
from panel_selection import select_important_panels
from checkpoint import Checkpoint
from api_resilience import is_transient

load_dotenv()  # Load environment variables from .env file

def write_text_to_file(movie_script, manga, volume_number):
    output_dir = f"{manga}/v{volume_number}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/extracted_text.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in movie_script:
            f.write(f"Segment Text:\n{segment['text']}\n\n")
            if 'citations' in segment:
                f.write(f"Citations: {', '.join(map(str, segment['citations']))}\n\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"Extracted text has been written to {output_file}")

async def main(volume_number, manga, text_only=False, fresh=False):
    # Initialize Ollama client (Ollama Cloud if OLLAMA_API_KEY is set, else local)
    client = create_client()
    print(f"Using Ollama at {client.base_url} with model {vision_model()}")
    # Only load the Kokoro TTS model if we're not in text-only mode
    narrator = None
    if not text_only:
        narrator = KokoroNarrator()

    print("Extracting all pages from the volume...")
    volume_pdf = f"{manga}/v{volume_number}/v{volume_number}.pdf"
    volume_scaled_and_unscaled = extract_all_pages_as_images(volume_pdf)
    volume = volume_scaled_and_unscaled["scaled"]
    volume_unscaled = volume_scaled_and_unscaled["full"]
    print("Total pages in volume:", len(volume))

    if len(volume) == 0:
        print("Error: No images extracted from the PDF. Please check the PDF file.")
        return

    profile_reference = extract_all_pages_as_images(f"{manga}/profile-reference.pdf")[
        "scaled"
    ]
    chapter_reference = extract_all_pages_as_images(f"{manga}/chapter-reference.pdf")[
        "scaled"
    ]

    profile_pages = []
    chapter_pages = []

    important_page_tokens = 0

    batch_size = 20
    NUMBER_OF_JOBS = 7

    # Progress is saved after every API request so a crash can resume here
    checkpoint = Checkpoint(
        f"{manga}/v{volume_number}/checkpoint.json",
        fingerprint={
            "pdf_bytes": os.path.getsize(volume_pdf),
            "pages": len(volume),
            "page_batch_size": batch_size,
            "number_of_jobs": NUMBER_OF_JOBS,
        },
        fresh=fresh,
    )
    detected_pages = checkpoint.get("important_pages", {})

    print("Identifying important pages in the volume...")

    # Sequential on purpose: the free Ollama Cloud tier rate-limits bursts
    for start_idx in range(0, len(volume), batch_size):
        end_index = min(start_idx + batch_size, len(volume)) - 1
        if str(start_idx) in detected_pages:
            print(f"Pages {start_idx} to {end_index} already processed (checkpoint)")
        else:
            print(f"Processing pages {start_idx} to {end_index}")
            response = detect_important_pages(
                profile_reference,
                chapter_reference,
                volume[start_idx : start_idx + batch_size],
                client,
                KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
                KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
            )
            detected_pages[str(start_idx)] = response
            checkpoint.set("important_pages", detected_pages)

        response = detected_pages[str(start_idx)]
        ip = response["parsed_response"]
        print(json.dumps(ip, indent=2))
        for page in ip:
            if page["type"] == "profile":
                profile_pages.append(page["image_index"] + start_idx)
            elif page["type"] == "chapter":
                chapter_pages.append(page["image_index"] + start_idx)

        important_page_tokens += response["total_tokens"]

    profile_pages.sort()
    chapter_pages.sort()

    print("Total tokens to extract profiles and chapters:", important_page_tokens)
    print("\n__________\n")
    print("Profile pages:", profile_pages)
    print("Chapter pages:", chapter_pages)

    #chapter_pages = [1]
    #profile_pages = [0]
    print(f"{len(volume)}")
    print("\n__________\n")
    print("Saving important pages to disk for QA...")
    save_important_pages(volume, profile_pages, chapter_pages, manga, volume_number)

    character_profiles = [volume[i] for i in profile_pages]
    jobs = split_volume_into_parts(
        volume, volume_unscaled, chapter_pages, NUMBER_OF_JOBS
    )
    parts = jobs["parts"]
    jobs_unscaled = jobs["unscaled_images"]
    jobs = jobs["scaled_images"]

    # Summarize each job in order, adding context from previous ones. Finished
    # jobs are checkpointed, so a resumed run continues from the first missing one.
    summaries = checkpoint.get("summaries", [])
    recap = ""
    tokens = 0
    movie_script = []
    for i, job in enumerate(jobs):
        if i < len(summaries):
            print(f"Summary for part {i} already generated (checkpoint)")
        else:
            prompt = (
                BASIC_PROMPT
                if i == 0
                else recap + "\n-----\n" + BASIC_PROMPT_WITH_CONTEXT
            )
            response = analyze_images_with_vision(
                character_profiles, job, client, prompt, BASIC_INSTRUCTIONS
            )
            summaries.append(
                {
                    "content": response.choices[0].message.content,
                    "total_tokens": response.usage.total_tokens,
                }
            )
            checkpoint.set("summaries", summaries)

        content = summaries[i]["content"]
        recap = content if i == 0 else recap + "\n\n" + content
        tokens += summaries[i]["total_tokens"]
        print("\n\n\n_____________\n\n\n")
        print(content)
        movie_script = movie_script + extract_text_and_citations(
            content, job, jobs_unscaled[i]
        )

    print("\n\n\n_____________\n\n\n")
    print("\n\n\n_____________\n\n\n")
    print("\n\n\n_____________\n\n\n")

    narration_script = extract_script(movie_script)
    print(narration_script)
    print("\n___________\n")

    extract_panels(movie_script)
    print("Extracting panels from movie script...")
    for i, segment in enumerate(movie_script):
        print(f"Processing segment {i}")
        print(f"Number of images in segment: {len(segment['images'])}")
        print(f"Number of unscaled images in segment: {len(segment['images_unscaled'])}")

        if len(segment['images']) == 0:
            print(f"Warning: No images found in segment {i}. Skipping panel extraction.")
            continue

        #extract_panels(segment)

        all_panels_base64 = [
            panel for sublist in segment["panels"].values() for panel in sublist
        ]
        print(f"Number of panels extracted: {len(all_panels_base64)}")


    print("number of segments:", len(movie_script))

    for i, segment in enumerate(movie_script):
        print("segment", i, ": ", segment["text"])
        all_panels_base64 = [
            panel for sublist in segment["panels"].values() for panel in sublist
        ]
        print(len(all_panels_base64))
        print("number of panels:", len(all_panels_base64))
        print("number of images:", len(segment["images"]))

    # One request per batch of segments instead of one per segment. Text-only
    # output never shows panels, so it skips these requests entirely.
    panel_tokens = 0
    if not text_only:
        panel_tokens = select_important_panels(
            movie_script, profile_reference, client, checkpoint
        )

    # Ollama (local or free cloud tier) and Kokoro are free, so only usage is reported
    print("Tokens for extracting profiles and chapters:", important_page_tokens)
    print("Tokens for summarization:", tokens)
    print("Tokens for extracting important panels:", panel_tokens)
    print("Total LLM tokens:", important_page_tokens + tokens + panel_tokens)
    if not text_only:
        print("Total narration characters:", len(extract_script(movie_script)))

    if text_only:
        write_text_to_file(movie_script, manga, volume_number)
        print("Text-only mode: Skipping narration and video creation.")
    else:
        await make_movie(movie_script, manga, volume_number, narrator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process manga volumes.")
    parser.add_argument(
        "--manga",
        type=str,
        default="naruto",
        help="Manga name to process (default: naruto)",
    )
    parser.add_argument(
        "--volume-number",
        type=int,
        default=10,
        help="Volume number to process (default: 10)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Output extracted text to a file instead of creating a video",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved checkpoint and redo every API request",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.volume_number, args.manga, args.text_only, args.fresh))
    except Exception as e:
        if is_transient(e):
            print(
                f"\nOllama Cloud is still unavailable after all retries ({e}).\n"
                f"Progress is saved in {args.manga}/v{args.volume_number}/checkpoint.json; "
                "re-run the same command to resume."
            )
        raise
