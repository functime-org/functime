from typing import Mapping, Optional, Union

import polars as pl

from functime.llm.common import MODEL_T, openai_call
from functime.llm.formatting import FORMAT_T, format_dataframes, format_instructions

_LLM_NAMESPACE = "llm"


@pl.api.register_dataframe_namespace(_LLM_NAMESPACE)
class LLMActions:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def analyze(
        self,
        context: Optional[str] = None,
        model: MODEL_T = "gpt-3.5-turbo",
        format: FORMAT_T = "markdown_bullet_list",
        **kwargs,
    ) -> str:
        """Analyze a polars DataFrame."""

        task, formatting = format_instructions(format)
        prompt_context = f" The context is: {context}."
        constraints = (
            " Be specific and respond with non-obvious statistical analyses in the tone of a McKinsey consultant."
            " Describe trend, seasonality, and anomalies. Do not provide recommendations. Do not describe the table."
            " Do not introduce yourself or your role."
        )
        prompt = (
            f"{task}"
            f"{prompt_context}"
            f"{constraints}"
            f"{format_dataframes(self._df)}"
            f"{formatting}"
        )
        response = openai_call(prompt, model, **kwargs)
        return response

    def compare(
        self,
        others: Union[pl.DataFrame, Mapping[str, pl.DataFrame]],
        *,
        as_label: str = "This",
        model: MODEL_T = "gpt-3.5-turbo",
        target_feature: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Compare 2 or more polars DataFrames."""

        task = "Compare and contrast the following time series data."

        if target_feature is not None:
            prompt_context = f" Compare the dataframe entities against the target feature `{target_feature}`."
        else:
            prompt_context = " Compare the dataframe entities against each other."
        constraints = (
            " Be specific and respond with non-obvious statistical analyses in the tone of a McKinsey consultant."
            " Compare trend, seasonality, and anomalies. Do not provide recommendations. Do not describe the tables."
            " Do not introduce yourself or your role."
        )
        output_format = "{{ Insert your comparative analysis here }}"
        dfs = {as_label: self._df}
        if isinstance(others, pl.DataFrame):
            dfs.update({"other": others})
        else:
            dfs.update(others)
        prompt = (
            f"{task}"
            f"{prompt_context}"
            f"{constraints}"
            f"{format_dataframes(dfs)}"
            f"{output_format}"
        )
        response = openai_call(prompt, model, **kwargs)
        return response
