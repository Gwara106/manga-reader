import hashlib
import os

from manga_extraction import scale_base64_image
from prompts import KEY_PANEL_BATCH_IDENTIFICATION_INSTRUCTIONS
from vision_analysis import get_important_panels_batch


def segment_candidates(segment):
    # A segment's candidates are the extracted panels of each cited page, or the
    # page itself when no panels were extracted from it
    panels = []
    for j, page in enumerate(segment["images"]):
        if j in segment.get("panels", {}):
            panels.extend(segment["panels"][j])
        else:
            panels.append(page)
    return panels


def _digest(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def plan_batches(movie_script, pending, max_images, max_segments):
    """Group pending segments into requests of at most max_images unique panels.

    Neighbouring segments usually cite the same pages, so panels are
    de-duplicated and each shared panel is only sent once per request.
    """
    batches = []
    current = None
    for i in pending:
        candidates = segment_candidates(movie_script[i])
        if not candidates:
            continue
        keys = [_digest(p) for p in candidates]
        if current is not None:
            new_images = len(set(keys) - current["keys"])
            if (
                len(current["segments"]) >= max_segments
                or len(current["keys"]) + new_images > max_images
            ):
                batches.append(current)
                current = None
        if current is None:
            current = {"segments": [], "keys": set()}
        current["segments"].append((i, candidates, keys))
        current["keys"].update(keys)
    if current is not None:
        batches.append(current)
    return batches


def select_important_panels(movie_script, profile_reference, client, checkpoint=None):
    """Set segment["important_panels"] for every segment, batching requests.

    Completed segments are saved to the checkpoint after every request and are
    never re-sent. Returns the total tokens used (including resumed runs).
    """
    max_images = int(os.getenv("PANEL_BATCH_MAX_IMAGES", "40"))
    max_segments = int(os.getenv("PANEL_BATCH_MAX_SEGMENTS", "8"))
    done = checkpoint.get("panel_selection", {}) if checkpoint else {}
    total_tokens = checkpoint.get("panel_tokens", 0) if checkpoint else 0

    pending = []
    for i, segment in enumerate(movie_script):
        segment["important_panels"] = []
        saved = done.get(str(i))
        if saved and saved["text"] == _digest(segment["text"]):
            candidates = segment_candidates(segment)
            segment["important_panels"] = [
                candidates[n] for n in saved["picks"] if n < len(candidates)
            ]
        else:
            pending.append(i)

    batches = plan_batches(movie_script, pending, max_images, max_segments)
    print(
        f"Panel selection: {len(movie_script) - len(pending)} segments resumed, "
        f"{len(pending)} pending in {len(batches)} requests."
    )

    failed_batches = 0
    for b, batch in enumerate(batches):
        # Assign each unique panel in this batch a number the model can refer to
        batch_index = {}
        unique_panels = []
        for _, candidates, keys in batch["segments"]:
            for panel, key in zip(candidates, keys):
                if key not in batch_index:
                    batch_index[key] = len(unique_panels)
                    unique_panels.append(scale_base64_image(panel))
        segments = [
            {
                "segment": i,
                "text": movie_script[i]["text"],
                "candidates": sorted({batch_index[k] for k in keys}),
            }
            for i, _, keys in batch["segments"]
        ]
        print(
            f"Panel batch {b + 1}/{len(batches)}: segments "
            f"{[s['segment'] for s in segments]}, {len(unique_panels)} panels"
        )

        try:
            response = get_important_panels_batch(
                profile_reference,
                unique_panels,
                segments,
                client,
                KEY_PANEL_BATCH_IDENTIFICATION_INSTRUCTIONS,
            )
        except Exception as e:
            # Panel picks only improve the video; full pages are a valid fallback.
            # Not checkpointed, so a re-run retries just these segments.
            failed_batches += 1
            print(
                f"Panel batch {b + 1} failed ({type(e).__name__}: {e}). "
                "Its segments will use full pages; re-run to retry them."
            )
            continue

        total_tokens += response["total_tokens"]
        selections = response["parsed_response"]
        for i, candidates, keys in batch["segments"]:
            # Map batch panel numbers back to this segment's own candidate list
            local_index = {}
            for n, key in enumerate(keys):
                local_index.setdefault(batch_index[key], n)
            picks = [local_index[p] for p in selections.get(i, [])]
            movie_script[i]["important_panels"] = [candidates[n] for n in picks]
            done[str(i)] = {"text": _digest(movie_script[i]["text"]), "picks": picks}
            status = "exist." if picks else "missing; using full pages."
            print("Important panels for segment", i, status)

        if checkpoint:
            checkpoint.data["panel_tokens"] = total_tokens
            checkpoint.set("panel_selection", done)

    if failed_batches:
        print(
            f"{failed_batches} panel batch(es) failed and fell back to full pages. "
            "Re-run the same command to retry only those segments."
        )
    return total_tokens
