import logging
import os
from typing import Dict, List, Mapping

from typing_extensions import Literal

try:
    import openai
    import tiktoken
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ModuleNotFoundError as e:
    raise ImportError(
        "The `llm` feature requires the `openai`, `tenacity`, and `tiktoken` packages. Run `pip install functime[llm]` to install."
    ) from e


openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable must be set to use the `llm` feature."
    )

MODEL_T = Literal["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k", "gpt-4-32k"]

_MAX_RETRIES = 6
_DEFAULT_SYSTEM_CONTEXT = (
    "You are an expert trends analyst and data communicator."
    " You always write clearly and concisely."
    " You use simple percentage and difference calculations to explain time series and trends."
    " You MUST follow the instructions for your given task exactly."
    " When returning a JSON object, you MUST return a valid and well-formatted JSON object following exactly the schema provided."
    " Your returned JSON object MUST NOT have a trailing comma after the last entry. This would make it invalid."
    " Do your best!"
)

_supported_models_token_limit: Mapping[MODEL_T, int] = {
    "gpt-3.5-turbo": 4000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-4": 8000,
    "gpt-4-32k": 32000,
}
# Allow models of the same kind to choose the next largest model
_dag: Mapping[MODEL_T, MODEL_T] = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    "gpt-4": "gpt-4-32k",
}


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(_MAX_RETRIES),
    retry_error_callback=lambda e: logging.error(f"Retry raised an OpenAI error: {e}"),
)
def openai_call(
    prompt: str,
    model: MODEL_T = "gpt-3.5-turbo",
    temperature: float = 0.5,
    system_context: str = _DEFAULT_SYSTEM_CONTEXT,
    auto_adjust_model: bool = True,
    **kwargs,
) -> str:
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": prompt},
    ]
    n_tokens = _openai_count_tokens(messages)
    while n_tokens > _supported_models_token_limit[model]:
        next_model = _dag.get(model) if auto_adjust_model else None
        if next_model is None:
            raise ValueError(
                f"Prompt exceeds token limit for model {model}."
                f" Either no larger model is available or try set auto_adjust_model=True."
            )
        logging.warning(
            f"Prompt exceeds token limit for model {model}. Checking with larger model..."
        )
        model = next_model
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return response["choices"][0]["message"]["content"].strip()


def _openai_count_tokens(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
):
    """Return the number of tokens used by a list of messages.

    Code obtained from OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logging.warning(
            "gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return _openai_count_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logging.warning(
            "gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return _openai_count_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""_openai_count_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
