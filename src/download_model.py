from os import getenv
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    model = getenv("MODEL", "Philogicae/whisper-large-v3-french-ct2")
    print(f"Downloading Whisper model: {model}")
    snapshot_download(model)
