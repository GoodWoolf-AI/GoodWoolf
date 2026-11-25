import torch


def render_mc(question, letters, choices):
    """Render a multiple-choice question with one labeled choice per line.

    The letter follows the choice with no whitespace (``- choice=A``) so the
    assistant target ``A`` is the exact same token as in the prompt.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


def get_bos_token_id(tokenizer):
    getter = getattr(tokenizer, "get_bos_token_id", None)
    if callable(getter):
        token_id = getter()
        if token_id is not None:
            return token_id
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return token_id
    return 0


def get_pad_token_id(tokenizer):
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return token_id
    return 0


def _message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def _chat_messages(messages):
    return [{"role": message["role"], "content": _message_text(message)} for message in messages]


def _encode_chat_text(tokenizer, text):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def render_conversation(tokenizer, conversation, max_tokens=2048):
    renderer = getattr(tokenizer, "render_conversation", None)
    if callable(renderer):
        return renderer(conversation, max_tokens=max_tokens)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Tokenizer must provide render_conversation() or apply_chat_template()")
    messages = _chat_messages(conversation["messages"])
    full_text = apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
    prompt_text = apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=bool(prompt_messages))
    ids = _encode_chat_text(tokenizer, full_text)
    prompt_ids = _encode_chat_text(tokenizer, prompt_text)
    bos_token_id = get_bos_token_id(tokenizer)
    if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids
        prompt_ids = [bos_token_id] + prompt_ids
    prompt_len = 0
    for prompt_id, full_id in zip(prompt_ids, ids):
        if prompt_id != full_id:
            break
        prompt_len += 1
    mask = [0] * prompt_len + [1] * max(0, len(ids) - prompt_len)
    return ids[:max_tokens], mask[:max_tokens]


def render_prompt(tokenizer, messages, max_tokens=2048):
    renderer = getattr(tokenizer, "render_prompt", None)
    if callable(renderer):
        return renderer(messages, max_tokens=max_tokens)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Tokenizer must provide render_prompt() or apply_chat_template()")
    text = apply_chat_template(_chat_messages(messages), tokenize=False, add_generation_prompt=True)
    ids = _encode_chat_text(tokenizer, text)
    bos_token_id = get_bos_token_id(tokenizer)
    if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids
    return ids[:max_tokens]


def record_text(record):
    """Return the plain text of a conversation record (used for statistics)."""
    return "\n".join(_message_text(message) for message in record["messages"])


def _model_context_length(model):
    config = getattr(model, "config", None)
    context = getattr(config, "max_position_embeddings", None) or getattr(config, "sequence_len", None)
    return int(context or 4096)


@torch.inference_mode()
def respond(
    model,
    tokenizer,
    messages,
    max_tokens=256,
    temperature=0.6,
    top_k=50,
    do_sample=True,
    max_position_embeddings=None,
    seed=42,
):
    context = _model_context_length(model) if max_position_embeddings is None else int(max_position_embeddings)
    prompt_limit = max(1, context - max_tokens)
    prompt_ids = render_prompt(tokenizer, messages, max_tokens=prompt_limit)
    if not prompt_ids:
        raise ValueError("Chat template rendered an empty prompt")
    sequences, _ = model.generate(
        tokenizer,
        prompt_ids,
        num_samples=1,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        seed=seed,
    )
    response_ids = sequences[0][len(prompt_ids):]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def chat(
    model_name,
    max_tokens=256,
    temperature=0.6,
    top_k=50,
    device="",
    seed=42,
    title=None,
    input_fn=input,
    output_fn=print,
):
    from goodwoolf.utils.models import load_config, load_model
    from goodwoolf.utils.system import model_path

    load_path = model_path(model_name)
    configs = load_config(model_name)
    tokenizer = configs["tokenizer"]
    model = load_model(model_name, load_path)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    if "cuda" in str(target_device):
        model.to(dtype=torch.bfloat16)
    model.eval()

    messages = []
    if title is None:
        title = f"\nChat with {model_name}\n"
    if title:
        output_fn(title)
    output_fn("-" * 50)
    output_fn("Type 'quit' or 'exit' to end the conversation")
    output_fn("Type 'clear' to start a new conversation")
    output_fn("-" * 50)
    while True:
        try:
            user_input = input_fn("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit"):
            output_fn("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            output_fn("Conversation cleared.")
            continue
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        response = respond(
            model,
            tokenizer,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            seed=seed,
        )
        output_fn(f"\nAssistant: {response}")
        messages.append({"role": "assistant", "content": response})
    return messages
