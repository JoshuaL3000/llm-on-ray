import requests
import time
import argparse

parser = argparse.ArgumentParser("Model Inference Script", add_help=False)
parser.add_argument("--model_endpoint", default="http://127.0.0.1:8000", type=str, help="deployed model endpoint")
parser.add_argument("--streaming_response", default=False, action="store_true", help="whether to enable streaming response")
args = parser.parse_args()
prompt = "Once upon a time,"
sample_input = {"text": prompt, "stream": args.streaming_response}
total_time = 0.0
num_iter = 10
num_warmup = 3
for i in range(num_iter):
    print("iter: ", i)
    tic = time.time()
    proxies = { "http": None, "https": None}
    outputs = requests.post(args.model_endpoint, proxies=proxies, json=[sample_input], stream=args.streaming_response)
    if args.streaming_response:
        outputs.raise_for_status()
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            print(output, flush=True)
    else:
        print(outputs.text, flush=True)
    toc = time.time()
    if i >= num_warmup:
        total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))
