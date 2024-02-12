from os import getenv
from faster_whisper import WhisperModel


class Transcriber:
    model_path = getenv("MODEL", "Philogicae/whisper-large-v3-french-ct2")
    device = "cuda"
    compute_type = "int8_float16"

    def __init__(self):
        self.model = WhisperModel(
            self.model_path, self.device, compute_type=self.compute_type
        )

    def process(self, file, custom_params: dict = {}):
        params = (
            dict(
                language="fr",
                word_timestamps=True,
                vad_filter=True,
                beam_size=8,
                best_of=12,
                temperature=[0.65, 0.70, 0.75, 0.80],
            )
            | custom_params
        )
        segments, _ = self.model.transcribe(file, **params)
        text, stats = "", []
        for segment in segments:
            for word in segment.words:
                text += word.word
                stats.append(
                    [
                        word.word.strip(),
                        f"{word.start}:{word.end}",
                        f"{word.probability:.4f}",
                    ]
                )
        text = text.strip()
        stats = list(sorted(stats, key=lambda x: x[2]))
        return dict(text=text, stats=stats, params=params)
