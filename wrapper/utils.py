import json
import os
import re
import subprocess

def ray_status_parser():

    command = "ray status"
    # Create a new process
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Capture stdout and stderr
    stdout, stderr = process.communicate()
    output =stdout.decode("utf-8")

    result = {}

    # Parse node status
    node_status_pattern = re.compile(r'Active:\s*(.*?)Pending:', re.DOTALL)
    node_status_match = node_status_pattern.search(output)
    if node_status_match:
        active_nodes = re.findall(r'\d+\s+node_\w+', node_status_match.group(1))
        result['active_nodes'] = active_nodes

    # Parse resources
    resources_pattern = re.compile(r'Usage:\s*(.*?)Demands:', re.DOTALL)
    resources_match = resources_pattern.search(output)
    if resources_match:
        usage = re.findall(r'(\d+(\.\d+)?[KMGTPEZY]?B)/(\d+(\.\d+)?[KMGTPEZY]?B) (\w+)', resources_match.group(1))
        result['usage'] = {item[5]: {'used': item[0], 'total': item[3]} for item in usage}

    # Parse demands
    demands_pattern = re.compile(r'Demands:\s*(.*)$', re.DOTALL)
    demands_match = demands_pattern.search(output)
    if demands_match:
        demands = re.findall(r'\{(.+?)\}:\s*(\d+)\+ pending tasks/actors', demands_match.group(1))
        result['demands'] = {item[0]: int(item[1]) for item in demands}

    return result

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