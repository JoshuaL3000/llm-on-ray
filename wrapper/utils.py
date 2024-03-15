import json
import os
import re
import subprocess

def ray_status_parser( head_address, head_port = "6379" ):

    command = "ray status --address " + head_address + ":" + head_port
    # Create a new process
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Capture stdout and stderr
    stdout, stderr = process.communicate()
    output =stdout.decode("utf-8")

    print (output)

    parsed_data = {"active": [], "pending": [], "demands": {}}

    # Extract active and pending nodes
    active_match = re.search(r"Active:\s*(.*)", output)
    if active_match:
        parsed_data["active"] = [node.strip() for node in active_match.group(1).split(",")]

    pending_match = re.search(r"Pending:\s*(.*)", output)
    if pending_match:
        parsed_data["pending"] = [node.strip() for node in pending_match.group(1).split(",")]

    # Extract resource usage
    usage_match = re.search(r"Usage:\s*(.*)", output)
    if usage_match:
        usage_data = usage_match.group(1).split("\n")
        for line in usage_data:
            try:
                resource, value = line.split(":")
                parsed_data["demands"][resource.strip()] = value.strip()
            except: 
                pass

    return parsed_data

def is_simple_api(request_url, model_name):
    if model_name is None or len(model_name) == 0:
        return True
    return model_name in request_url

def history_to_messages(history, image=None):
    messages = []
    for human_text, bot_text in history:
        if image is not None:
            import base64
            from io import BytesIO

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            human_text = [
                {"type": "text", "text": human_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        messages.append(
            {
                "role": "user",
                "content": human_text,
            }
        )
        if bot_text is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": bot_text,
                }
            )
    return messages

def add_knowledge(prompt, enhance_knowledge):
    description = "Known knowledge: {knowledge}. Then please answer the question based on follow conversation: {conversation}."
    if not isinstance(prompt[-1]["content"], list):
        prompt[-1]["content"] = description.format(
            knowledge=enhance_knowledge, conversation=prompt[-1]["content"]
        )
    return prompt