import base64
import tempfile
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
import runpod
from time import time as now
from transcriber import Transcriber
from schema import get_schema_serverless, is_valid_params
from urllib import parse

MODEL = Transcriber()


def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


def download_file(jobId, file_url):
    return download_files_from_urls(jobId, file_url)[0]


@rp_debugger.FunctionTimer
def faster_whisper_fr(job):
    started = now()
    job_input = job["input"]
    result = {}
    if "schema" in job_input:
        result = dict(schema=get_schema_serverless())
    else:
        try:
            params = job_input.get("params", {})
            if is_valid_params(params):
                if "file_raw" in job_input:
                    urlEncoded = (
                        job_input["urlEncoded"] if "urlEncoded" in job_input else False
                    )
                    audio_input = base64_to_tempfile(
                        parse.unquote(job_input["file_raw"])
                        if urlEncoded
                        else job_input["file_raw"]
                    )
                elif "file_url" in job_input:
                    with rp_debugger.LineTimer("download_step"):
                        audio_input = download_file(job["id"], [job_input["file_url"]])
                else:
                    result = dict(error="Must provide either file_raw or file_url")
                if not result:
                    with rp_debugger.LineTimer("prediction_step"):
                        result = MODEL.process(audio_input, params)
            else:
                result = dict(error="Invalid params, check schema")
        except Exception as e:
            result = dict(error=str(e))
    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])
    return result | {"time": now() - started}


runpod.serverless.start({"handler": faster_whisper_fr})
