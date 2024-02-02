PARAMS = dict(
    beam_size="Beam size to use for decoding. Type: int. Default: 8",
    best_of="Number of candidates when sampling with non-zero temperature. Type: int. Default: 12",
    temperature="Temperature for sampling. It can be a tuple of temperatures, which will be successively used upon failures according to either `compression_ratio_threshold` or `log_prob_threshold`. Type: float | float[]. Default:[0.65, 0.70, 0.75, 0.80]",
    patience="Beam search patience factor. Type: int. Default: 1",
    length_penalty="Exponential length penalty constant. Type: int. Default: 1",
    compression_ratio_threshold="If the gzip compression ratio is above this value, treat as failed. Type: float. Default: 2.4",
    log_prob_threshold="If the average log probability over sampled tokens is below this value, treat as failed. Type: float. Default: -1.0",
    no_speech_threshold="If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `log_prob_threshold`, consider the segment as silent. Type: float. Default: 0.6",
    condition_on_previous_text="If True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync. Type: bool. Default: true",
    initial_prompt="Optional text string or iterable of token ids to provide as a prompt for the first window. Type: [str, int[], ...]. Default: none",
    prefix="Optional text to provide as a prefix for the first window. Type: str. Default: none",
    suppress_blank="Suppress blank outputs at the beginning of the sampling. Type: bool. Default: true",
    suppress_tokens="List of token IDs to suppress. -1 will suppress a default set of symbols as defined in the model config. Type: int[]. Default: true",
    max_initial_timestamp="The initial timestamp cannot be later than this. Type: float. Default: 1.0",
    vad_filter="Enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad. Type: bool. Default: true",
    vad_parameters="Dictionary of Silero VAD parameters. Type: dict. Default: none",
)


def is_valid_params(params: dict = {}):
    return set(params).issubset(set(PARAMS))


def get_schema_server():
    return {
        "/ping": "check if status",
        "/schema": "return api schema dict",
        "/transcribe": dict(
            expected=dict(file_raw="base64_file.wav", file_url="str", params=PARAMS),
            returned=dict(
                text="str",
                stats="list[[word, from:to, probability], ...]",
                params="dict",
                error="str",
                time="float",
            ),
        ),
    }


def get_schema_serverless():
    return dict(
        expected=dict(
            for_schema=dict(input=dict(schema=True)),
            for_transcribe=dict(
                input=dict(
                    file_raw="base64_file.wav",
                    urlEncoded="bool",
                    file_url="str",
                    params=PARAMS,
                )
            ),
        ),
        returned=dict(
            text="str",
            stats="list[[word, from:to, probability], ...]",
            params="dict",
            error="str",
            time="float",
            schema="dict",
        ),
    )
