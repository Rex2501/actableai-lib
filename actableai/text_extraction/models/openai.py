from functools import lru_cache
from typing import Iterable, List, Dict, Tuple

from actableai.parameters.numeric import FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.parameters.string import StringParameter
from actableai.text_extraction.models.base import BaseTextExtractionModel, Model
from actableai.utils.typing import JSONType


class OpenAI(BaseTextExtractionModel):
    @classmethod
    @lru_cache(maxsize=0)
    def get_parameters(cls) -> Parameters:
        parameters = [
            *cls.get_base_parameters(),
            OptionsParameter[str](
                name="model",
                display_name="Model",
                description="Model to use for text extraction",
                default="gpt-3.5-turbo",
                is_multi=False,
                options={
                    "gpt-3.5-turbo": {
                        "display_name": "GPT 3.5 Turbo",
                        "value": "gpt-3.5-turbo",
                    },
                    "gpt-3.5-turbo-16k": {
                        "display_name": "GPT 3.5 Turbo 16K",
                        "value": "gpt-3.5-turbo-16k",
                    },
                    "gpt-4": {
                        "display_name": "GPT 4",
                        "value": "gpt-4",
                    },
                    "gpt-4-32k": {
                        "display_name": "GPT 4 32K",
                        "value": "gpt-4-32k",
                    },
                },
            ),
            StringParameter(
                name="output_schema",
                display_name="Output JSON Schema",
                description="Schema used for the output",
                default="""[
    {
        "field_name": "string",
        "field_value": "string" or "array of string"
    }
]""",
            ),
            FloatParameter(
                name="rate_limit_per_minute",
                display_name="Rate Limit per Minute",
                description="Rate limit to use when calling openai API",
                default=3500,
                min=0.000001,
            ),
            StringParameter(
                name="openai_api_key",
                display_name="OpenAI API Key",
                description="API key to use when calling openai API",
                default="",
            ),
        ]

        return Parameters(
            name=Model.openai,
            display_name="OpenAI",
            parameters=parameters,
        )

    def _prepare_message(
        self,
        model: str,
        document: str,
        prompt: str,
    ) -> Tuple[List[Dict[str, str]], int]:
        import tiktoken

        messages = [
            {"role": "system", "content": document},
            {"role": "user", "content": prompt},
        ]

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if (
            model == "gpt-3.5-turbo-0301" or model == "gpt-3.5-turbo"
        ):  # note: future models may deviate from this
            max_tokens = 4096
            document_max_token = 3000

            num_tokens = 0
            for message_index, message in enumerate(messages):
                num_tokens += (
                    4
                    # every message follows <im_start>{role/name}\n{content}<im_end>\n
                )
                for key, value in message.items():
                    message_tokens = encoding.encode(value)
                    message_num_tokens = len(message_tokens)

                    if message_num_tokens > document_max_token:
                        new_message = " ".join(
                            str(e) for e in message_tokens[:document_max_token]
                        )
                        messages[message_index][key] = new_message
                        message_num_tokens = document_max_token

                    num_tokens += message_num_tokens

                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant

            return messages, max_tokens - num_tokens - 1

        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    def _open_ai_completion(
        self,
        fields_to_extract: List[str],
        output_schema: str,
        document: str,
        model: str,
    ) -> str:
        import openai

        prompt = f"""Extract the following fields from the provided document: {fields_to_extract}

Using this JSON format as a result:
{output_schema}

The JSON Object:
"""
        messages, max_tokens = self._prepare_message(
            model=model, document=document, prompt=prompt
        )

        chat_completion_result = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )

        # TODO check if max tokens is reached

        return chat_completion_result["choices"][0]["message"]["content"]

    def _predict(self, data: Iterable[str]) -> Iterable[JSONType]:
        import time
        import openai
        import backoff

        _open_ai_completion_func = backoff.on_exception(
            backoff.expo, openai.error.RateLimitError
        )(self._open_ai_completion)

        rate_limit_per_minute = self.parameters["rate_limit_per_minute"]
        delay = 60.0 / rate_limit_per_minute

        fields_to_extract = self.parameters["fields_to_extract"]
        model = self.parameters["model"]
        output_schema = self.parameters["output_schema"]

        extracted_data = []
        for document in data:
            time.sleep(delay)
            content = _open_ai_completion_func(
                fields_to_extract=fields_to_extract,
                output_schema=output_schema,
                document=document,
                model=model,
            )
            extracted_data.append(content)

        return extracted_data
