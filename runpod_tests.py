import runpod
import base64
from rich import print
from dotenv import load_dotenv
from os import getenv
from json import dumps

load_dotenv()

runpod.api_key = getenv("RUNPOD_API_KEY")
endpoint_id = getenv("RUNPOD_ENDPOINT_ID")
endpoint = runpod.Endpoint(endpoint_id)
TIMEOUT = 120


def sync_call(data):
    print("Start sync call...")
    return endpoint.run_sync({"input": data})


def async_call(data):
    print("Start async call...")
    job = endpoint.run({"input": data})
    print(job.status())
    return lambda: job.output(TIMEOUT)


def load_wav(filename):
    with open("input/" + filename + ".wav", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_output(filename, result):
    if "stats" in result:
        with open("output/" + filename + "-data.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([" -> ".join(item) for item in result["stats"]]))
    del result["stats"]
    with open("output/" + filename + ".txt", "w", encoding="utf-8") as f:
        f.write(dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    params = dict()
    # data = dict(file_url= getenv("AUDIO_URL"), params= params)
    data = dict(file_raw=load_wav(input("Filename .wav: ")), params=params)
    # data = "schema"
    # result = sync_call(data)
    result = async_call(data)()
    print(result)
    save_output("test", result)
