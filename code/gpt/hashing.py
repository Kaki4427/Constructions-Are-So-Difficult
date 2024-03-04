import hashlib


def short_hash(input_str):
    return hashlib.sha256(input_str.encode()).hexdigest()[:8]
