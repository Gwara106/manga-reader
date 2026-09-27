import json
import os
import re

from openai import OpenAI

from api_resilience import retry_api_call

# Ollama exposes an OpenAI-compatible API, so the OpenAI SDK is used as the
# transport. With OLLAMA_API_KEY set, requests go to Ollama Cloud (free tier);
# without it, they go to a local Ollama daemon (free, no key needed).
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CLOUD_VISION_MODEL = "mistral-large-3:675b"
DEFAULT_LOCAL_VISION_MODEL = "qwen3-vl:4b-instruct"


def create_client():
    api_key = os.getenv("OLLAMA_API_KEY")
    base_url = os.getenv("OLLAMA_BASE_URL") or (
        OLLAMA_CLOUD_BASE_URL if api_key else OLLAMA_LOCAL_BASE_URL
    )
    # The local daemon ignores the key, but the SDK requires a non-empty one.
    # SDK retries are off: retry_api_call owns backoff so retries don't multiply.
    return OpenAI(
        base_url=base_url,
        api_key=api_key or "ollama",
        max_retries=0,
        timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")),
    )


def chat(client, **kwargs):
    # Every model request goes through here: shared throttle + transient retries
    return retry_api_call(client.chat.completions.create, **kwargs)


def vision_model():
    default = (
        DEFAULT_CLOUD_VISION_MODEL
        if os.getenv("OLLAMA_API_KEY")
        else DEFAULT_LOCAL_VISION_MODEL
    )
    return os.getenv("OLLAMA_VISION_MODEL", default)


def text_model():
    return os.getenv("OLLAMA_TEXT_MODEL") or vision_model()


def parse_json(text):
    # Open models often wrap JSON in markdown fences; strip them before parsing
    match = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.DOTALL)
    return json.loads(match.group(1) if match else text)


def analyze_images_with_vision(
    character_profiles, pages, client, prompt, instructions, detail="low"
):
    # Construct the messages including the prompt and images
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some character profile pages, for your reference:",
                }
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in character_profiles
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in pages
            ],
        },
    ]

    response = chat(client, model=vision_model(), messages=messages, max_tokens=4096)

    return response


def detect_important_pages(
    profile_reference,
    chapter_reference,
    pages,
    client,
    prompt,
    instructions,
    detail="low",
):
    # Construct the messages including the prompt and images
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some character profile pages, for your reference:",
                }
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in profile_reference
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some chapter start pages, for your reference:",
                }
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in chapter_reference
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in pages
            ],
        },
    ]

    response = chat(
        client,
        model=vision_model(),
        messages=messages,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content
    tokens = response.usage.total_tokens

    # parse response text json from response.choices[0].message.content into object
    try:
        # Extract the text content from the first choice's message (if structured as expected)
        parsed_response = parse_json(response_text)
    except (AttributeError, IndexError, json.JSONDecodeError) as e:
        # Handle cases where parsing fails or the structure is not as expected
        print(f"Using the text model as a backup to format JSON object...")
        response = completions(client, response_text, JSON_PARSE_PROMPT)
        tokens += response.usage.total_tokens
        try:
            parsed_response = parse_json(response.choices[0].message.content)
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            parsed_response = None
            print("Even after using the text model to parse the json, we failed. Fatal error.")
            raise e

    return {
        "total_tokens": tokens,
        "parsed_response": parsed_response.get("important_pages", []),
    }


def completions(client, text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]

    response = chat(
        client,
        model=text_model(),
        messages=messages,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    return response


def get_important_panels(
    profile_reference, panels, client, prompt, instructions, detail="low"
):
    # Construct the messages including the prompt and images
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some character profile pages, for your reference:",
                }
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in profile_reference
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in panels
            ],
        },
    ]
    try:
        response = chat(
            client,
            model=vision_model(),
            messages=messages,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        if "content_policy_violation" in str(e):
            print(
                "The input image may contain content that is not allowed by the model provider's safety system."
            )
            return {"total_tokens": 0, "parsed_response": []}
        raise e

    response_text = response.choices[0].message.content
    print("MODEL RESPONSE:", response_text)
    tokens = response.usage.total_tokens

    # parse response text json from response.choices[0].message.content into object
    try:
        # Extract the text content from the first choice's message (if structured as expected)
        parsed_response = parse_json(response_text)
    except (AttributeError, IndexError, json.JSONDecodeError) as e:
        # Handle cases where parsing fails or the structure is not as expected
        print(f"Using the text model as a backup to format JSON object...")
        response = completions(client, response_text, JSON_PARSE_PROMPT_PANELS)
        tokens += response.usage.total_tokens
        try:
            parsed_response = parse_json(response.choices[0].message.content)
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            parsed_response = None
            print("Even after using the text model to parse the json, we failed. Fatal error.")
            raise e

    return {
        "total_tokens": tokens,
        "parsed_response": parsed_response.get("important_panels", []),
    }


def get_important_panels_batch(
    profile_reference, panels, segments, client, instructions, detail="low"
):
    """Pick important panels for several segments in a single request.

    panels: unique scaled panel images, referenced by their index.
    segments: [{"segment": int, "text": str, "candidates": [panel index, ...]}]
    Returns {"total_tokens": int, "parsed_response": {segment: [panel index, ...]}}.
    """
    panel_content = []
    for n, img_base64 in enumerate(panels):
        panel_content.append({"type": "text", "text": f"Panel {n}:"})
        panel_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": detail,
                },
            }
        )
    segment_text = "\n\n".join(
        f"Segment {s['segment']}: {s['text']}\nCandidate panels: {s['candidates']}"
        for s in segments
    )
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some character profile pages, for your reference:",
                }
            ]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": detail,
                    },
                }
                for img_base64 in profile_reference
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Here are the numbered panels:"}]
            + panel_content
            + [{"type": "text", "text": segment_text}],
        },
    ]
    response = chat(
        client,
        model=vision_model(),
        messages=messages,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content
    tokens = response.usage.total_tokens

    try:
        parsed_response = parse_json(response_text)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        print(f"Using the text model as a backup to format JSON object...")
        response = completions(client, response_text, JSON_PARSE_PROMPT_PANEL_BATCH)
        tokens += response.usage.total_tokens
        try:
            parsed_response = parse_json(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
            print("Could not parse panel selection; these segments will use full pages.")
            parsed_response = {}

    # Keep only well-formed picks that come from each segment's own candidates
    candidates = {s["segment"]: set(s["candidates"]) for s in segments}
    selections = {}
    entries = parsed_response.get("segments", []) if isinstance(parsed_response, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            segment = int(entry.get("segment"))
        except (TypeError, ValueError):
            continue
        if segment not in candidates:
            continue
        picks = []
        for p in entry.get("important_panels") or []:
            if isinstance(p, str) and p.isdigit():
                p = int(p)
            if isinstance(p, int) and p in candidates[segment] and p not in picks:
                picks.append(p)
        selections[segment] = picks

    return {"total_tokens": tokens, "parsed_response": selections}


JSON_PARSE_PROMPT = """
You are a JSON parser. Return a properly formatted json object based on the input from the user.
Your response must be in the following format:
{"important_pages": Array<{"image_index": int 0-19, "type": "profile" | "chapter"}>}

Examples of valid responses:
```
{
    "important_pages": [
        {"image_index": 0, "type": "profile"},
        {"image_index": 17, "type": "chapter"}
    ]
}
```

```
{
    "important_pages": []
}
```

"""


JSON_PARSE_PROMPT_PANELS = """
You are a JSON parser. Return a properly formatted json object based on the input from the user.
Your response must be in the following format:
{"important_panels": Array<int>}

Examples of valid responses:
```
{
    "important_panels": [
        0, 4,
    ]
}
```

```
{
    "important_panels": []
}
```

"""


JSON_PARSE_PROMPT_PANEL_BATCH = """
You are a JSON parser. Return a properly formatted json object based on the input from the user.
Your response must be in the following format:
{"segments": Array<{"segment": int, "important_panels": Array<int>}>}

Example of a valid response:
```
{
    "segments": [
        {"segment": 12, "important_panels": [0, 2]},
        {"segment": 13, "important_panels": []}
    ]
}
```

"""
