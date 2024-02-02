import base64
import tempfile
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
import runpod
from time import time as now
from transcriber import Transcriber

MODEL = Transcriber()


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


@rp_debugger.FunctionTimer
def faster_whisper_fr(job):
    started = now()
    job_input = job["input"]
    if job_input.get("file_raw", False):
        audio_input = base64_to_tempfile(job_input["file_raw"])
    elif job_input.get("file_url", False):
        with rp_debugger.LineTimer("download_step"):
            audio_input = download_files_from_urls(job["id"], [job_input["file_url"]])[
                0
            ]
    if not audio_input:
        return {"error": "Must provide either file_raw or file_url"}
    with rp_debugger.LineTimer("prediction_step"):
        text, stats = MODEL.process(audio_input)
    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])
    stop = now() - started
    return {"time": stop, "text": text, "stats": stats}


runpod.serverless.start({"handler": faster_whisper_fr})
