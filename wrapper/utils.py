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