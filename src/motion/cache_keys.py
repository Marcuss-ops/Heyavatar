import hashlib

def generate_segment_cache_key(
    avatar_id: str,
    text: str,
    voice_id: str,
    language: str,
    gesture_timeline_str: str,
    render_profile: str
) -> str:
    """Compute sha256 cache key for final output segment."""
    h = hashlib.sha256()
    h.update(avatar_id.encode())
    h.update(text.encode())
    h.update(voice_id.encode())
    h.update(language.encode())
    h.update(gesture_timeline_str.encode())
    h.update(render_profile.encode())
    return h.hexdigest()
