from faster_whisper import WhisperModel


class Transcriber:
    model_path = "Philogicae/whisper-large-v3-french-ct2"
    device = "cuda"
    compute_type = "int8_float16"

    def __init__(self):
        self.model = WhisperModel(
            self.model_path, self.device, compute_type=self.compute_type
        )

    def process(self, file):
        segments, _ = self.model.transcribe(
            file,
            language="fr",
            vad_filter=True,
            beam_size=8,
            best_of=12,
            temperature=[0.65, 0.70, 0.75, 0.80],
            word_timestamps=True,
        )
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
        return text, stats
