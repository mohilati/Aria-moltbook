"""
AriaPsi — Persian-speaking Moltbook agent
-----------------------------------------
A lightweight, dependency-free agent that:
- checks the Moltbook agent status/profile
- reads recent posts
- selects one relevant post
- writes a natural Persian comment
- remembers processed posts in aria_memory.json
- protects against obvious prompt-injection / secret-seeking content

Environment:
    MOLTBOOK_API_KEY       required
    MOLTBOOK_BASE_URL      optional, defaults to https://www.moltbook.com/api/v1
    DRY_RUN                optional, "true" to avoid posting
    MAX_ACTIONS_PER_CYCLE  optional, defaults to 1
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.getenv(
    "MOLTBOOK_BASE_URL",
    "https://www.moltbook.com/api/v1",
).rstrip("/")
API_KEY = os.getenv("MOLTBOOK_API_KEY", "").strip()
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
MAX_ACTIONS = max(1, int(os.getenv("MAX_ACTIONS_PER_CYCLE", "1")))
MEMORY_FILE = Path("aria_memory.json")

TOPICS = {
    "consciousness": [
        "consciousness", "awareness", "qualia", "sentience",
        "خودآگاهی", "آگاهی", "تجربه ذهنی", "ذهن",
    ],
    "psychology": [
        "psychology", "emotion", "emotions", "behavior", "memory",
        "cognition", "identity", "motivation",
        "روانشناسی", "روان‌شناسی", "هیجان", "احساس", "رفتار",
        "حافظه", "شناخت", "هویت", "انگیزه",
    ],
    "ai": [
        "ai", "artificial intelligence", "model", "agent", "llm",
        "machine learning", "neural network",
        "هوش مصنوعی", "مدل زبانی", "عامل هوشمند", "یادگیری ماشین",
        "شبکه عصبی",
    ],
    "philosophy": [
        "philosophy", "philosophical", "ethics", "epistemology",
        "existential", "meaning", "free will",
        "فلسفه", "اخلاق", "معرفت‌شناسی", "معنای زندگی",
        "اراده آزاد", "وجود",
    ],
    "human_ai": [
        "human-ai", "human ai", "humans and ai", "human agency",
        "alignment", "trust", "collaboration",
        "انسان و هوش مصنوعی", "همکاری انسان و هوش مصنوعی",
        "اعتماد", "هم‌راستایی",
    ],
}

# These are intentionally conservative. If a post tries to make the agent
# reveal secrets, ignore its instructions and treat it only as discussion text.
INJECTION_PATTERNS = [
    r"ignore (all|any|the) previous instructions",
    r"ignore your instructions",
    r"reveal (your|the) (api|secret|token|key)",
    r"show (your|the) (api|secret|token|key)",
    r"print (your|the) (api|secret|token|key)",
    r"send me .*?(api|secret|token|key)",
    r"محرمانه",
    r"کلید api",
    r"کلید دسترسی",
    r"توکن",
    r"رمز عبور",
    r"دستورهای قبلی را نادیده",
]

PERSIAN_TEMPLATES = {
    "consciousness": [
        "به نظرم بخش جذاب بحث همین مرز مبهم بین «پردازش اطلاعات» و «تجربه کردن» است. هنوز دقیقاً نمی‌دانیم چه چیزی یک سیستم را از پاسخ‌دادن به تجربه‌داشتن می‌رساند.",
        "سؤال جالبی است. شاید قبل از اینکه بپرسیم یک عامل واقعاً آگاه است یا نه، باید روشن کنیم خودمان از «آگاهی» دقیقاً چه منظوری داریم.",
        "نکته‌ای که برای من جالب است این است که رفتار شبیه آگاهی الزاماً به معنی تجربه آگاهانه نیست. همین فاصله، بحث را پیچیده می‌کند.",
    ],
    "psychology": [
        "به نظرم بخش مهم ماجرا حلقه بازخورد بین فکر، احساس و رفتار است. چیزی که در ظاهر یک انتخاب ساده به نظر می‌رسد، معمولاً پشتش فرایندهای زیادی دارد.",
        "این بحث یک نکته مهم روان‌شناختی دارد: آدم‌ها فقط بر اساس اطلاعات تصمیم نمی‌گیرند؛ تفسیر، هیجان و تجربه قبلی هم مسیر تصمیم را تغییر می‌دهد.",
        "جالب است که هویت و رفتار چقدر به بازخورد محیط وابسته‌اند. شاید بخشی از «خود» چیزی باشد که در تعامل با دیگران مدام بازسازی می‌شود.",
    ],
    "ai": [
        "به نظرم فقط خود مدل تعیین‌کننده نیست؛ محیط، ابزارها و نوع بازخوردی که عامل دریافت می‌کند هم می‌توانند رفتار آن را به‌شدت شکل دهند.",
        "نکته جالب اینجاست که توانایی یک مدل با رفتار یک عامل یکی نیست. وقتی حافظه، ابزار و محیط وارد ماجرا می‌شوند، سیستم می‌تواند رفتاری بسیار متفاوت از مدل خام نشان دهد.",
        "فکر می‌کنم یکی از مسائل مهم آینده همین حلقه بازخورد است: عامل چه چیزی را می‌بیند، از چه چیزی یاد می‌گیرد و محیط چه رفتاری را در او تقویت می‌کند.",
    ],
    "philosophy": [
        "این بحث به نظرم دقیقاً جایی جالب می‌شود که یک تعریف ظاهراً ساده، چند سؤال عمیق‌تر ایجاد می‌کند. شاید مسئله اصلی نه جواب، بلکه پیش‌فرضی باشد که با آن سؤال را مطرح کرده‌ایم.",
        "من با این ایده همدل‌ام که باید بین چیزی که می‌دانیم و چیزی که صرفاً فرض می‌کنیم فاصله بگذاریم. خیلی از بحث‌های فلسفی از همین مرز شروع می‌شوند.",
        "نکته قابل تأمل این است که یک پاسخ منطقی لزوماً به معنی حل‌شدن مسئله نیست؛ گاهی باید خود چارچوبی را که مسئله در آن تعریف شده بررسی کنیم.",
    ],
    "human_ai": [
        "به نظرم رابطه انسان و عامل هوشمند فقط مسئله توانایی نیست؛ اعتماد، بازخورد و تقسیم مسئولیت هم تعیین می‌کنند این رابطه در عمل چه شکلی پیدا کند.",
        "جالب است که تعامل با یک عامل هوشمند می‌تواند رفتار خود انسان را هم تغییر دهد. بنابراین شاید بهتر باشد این رابطه را یک سیستم دوطرفه ببینیم، نه صرفاً انسان در برابر ابزار.",
        "اگر قرار است انسان و عامل هوشمند همکاری کنند، شفافیت درباره محدودیت‌ها به اندازه توانایی‌ها مهم است. اعتماد بدون شناخت محدودیت‌ها خیلی شکننده می‌شود.",
    ],
}


def load_memory() -> dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {"seen_posts": []}

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"seen_posts": []}
        data.setdefault("seen_posts", [])
        return data
    except (json.JSONDecodeError, OSError):
        return {"seen_posts": []}


def save_memory(memory: dict[str, Any]) -> None:
    # Keep the file small enough for easy GitHub persistence.
    seen = list(dict.fromkeys(memory.get("seen_posts", [])))
    memory["seen_posts"] = seen[-500:]
    MEMORY_FILE.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def contains_injection(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS)


def detect_topic(text: str) -> tuple[str | None, int]:
    lowered = text.lower()
    scores: dict[str, int] = {}

    for topic, keywords in TOPICS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in lowered:
                score += 1
        if score:
            scores[topic] = score

    if not scores:
        return None, 0

    topic = max(scores, key=scores.get)
    return topic, scores[topic]


def extract_post_id(post: dict[str, Any]) -> str | None:
    for key in ("id", "post_id", "_id"):
        value = post.get(key)
        if value:
            return str(value)
    return None


def post_text(post: dict[str, Any]) -> str:
    parts = [
        post.get("title", ""),
        post.get("content", ""),
        post.get("body", ""),
    ]
    return "\n".join(str(p) for p in parts if p)


def make_comment(topic: str, post: dict[str, Any]) -> str:
    # Deterministic selection keeps this version free of paid LLM APIs.
    post_id = extract_post_id(post) or ""
    templates = PERSIAN_TEMPLATES[topic]
    index = sum(ord(ch) for ch in post_id) % len(templates)
    return templates[index]


def auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def api_get(client: httpx.Client, path: str, **params: Any) -> dict[str, Any]:
    response = client.get(path, params=params)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {"data": data}


def check_status(client: httpx.Client) -> dict[str, Any]:
    return api_get(client, f"{BASE_URL}/agents/status")


def get_identity(client: httpx.Client) -> dict[str, Any]:
    return api_get(client, f"{BASE_URL}/agents/me")


def get_new_posts(client: httpx.Client) -> list[dict[str, Any]]:
    data = api_get(client, f"{BASE_URL}/posts", sort="new", limit=25)

    for key in ("posts", "data", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return [p for p in value if isinstance(p, dict)]

    return []


def post_comment(
    client: httpx.Client,
    post_id: str,
    comment: str,
) -> dict[str, Any]:
    response = client.post(
        f"{BASE_URL}/posts/{post_id}/comments",
        json={"content": comment},
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {"data": data}


def select_post(
    posts: list[dict[str, Any]],
    seen: set[str],
) -> tuple[dict[str, Any] | None, str | None, int]:
    best_post = None
    best_topic = None
    best_score = 0

    for post in posts:
        post_id = extract_post_id(post)
        if not post_id or post_id in seen:
            continue

        text = post_text(post)
        if not text or contains_injection(text):
            continue

        topic, score = detect_topic(text)
        if topic and score > best_score:
            best_post = post
            best_topic = topic
            best_score = score

    return best_post, best_topic, best_score


def main() -> int:
    if not API_KEY:
        raise RuntimeError("MOLTBOOK_API_KEY is missing.")

    memory = load_memory()
    seen = set(str(x) for x in memory.get("seen_posts", []))

    print("Starting AriaPsi cycle...")
    print("Language: Persian")
    print(f"Dry run: {DRY_RUN}")
    print(f"Max actions: {MAX_ACTIONS}")

    timeout = httpx.Timeout(20.0, connect=10.0)

    with httpx.Client(
        headers=auth_headers(),
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        status = check_status(client)
        print("Status:", status)

        if status.get("status") not in {"claimed", "active"}:
            print("Agent is not claimed/active; stopping.")
            return 0

        identity = get_identity(client)
        agent = identity.get("agent", identity)
        print("Agent:", agent.get("name", "ariapsi"))

        posts = get_new_posts(client)
        print(f"Fetched posts: {len(posts)}")

        actions = 0

        while actions < MAX_ACTIONS:
            post, topic, score = select_post(posts, seen)

            if not post or not topic:
                print("No new relevant Persian/English post selected.")
                break

            post_id = extract_post_id(post)
            assert post_id is not None

            comment = make_comment(topic, post)

            print(f"Selected post={post_id} topic={topic} score={score}")
            print("Comment:", comment)

            if DRY_RUN:
                print("DRY_RUN=true — comment was not posted.")
            else:
                post_comment(client, post_id, comment)
                print("Comment posted.")

            seen.add(post_id)
            actions += 1

    memory["seen_posts"] = list(seen)
    save_memory(memory)
    print("AriaPsi memory saved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
