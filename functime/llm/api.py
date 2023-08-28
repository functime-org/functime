from typing import List, Optional

import polars as pl

from functime.llm.common import MODEL_T, openai_call
from functime.llm.formatting import FORMAT_T, format_dataframes, format_instructions

_LLM_NAMESPACE = "llm"


@pl.api.register_dataframe_namespace(_LLM_NAMESPACE)
class LLMActions:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _filter_panel_entities(self, panel_df: pl.DataFrame, basket: List[str]):
        entity_col = panel_df.columns[0]
        df = panel_df.filter(pl.col(entity_col).is_in(basket))
        if df.is_empty():
            raise ValueError(
                f"No matching entities found in panel given basket: {basket}"
            )
        return df

    def _panel_to_wide(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        entity_col, time_col, target_col = panel_df.columns[:3]
        wide_df = panel_df.pivot(
            index=time_col,
            values=target_col,
            columns=entity_col,
            aggregate_function=None,
            sort_columns=True,
        ).sort(time_col)
        return wide_df

    def analyze(
        self,
        basket: List[str],
        context: Optional[str] = None,
        model: MODEL_T = "gpt-3.5-turbo",
        format: FORMAT_T = "markdown_bullet_list",
        **kwargs,
    ) -> str:
        """Analyze a `basket` (list of entity / series IDs) of forecasts."""

        task, formatting = format_instructions(format)
        prompt_context = f" The context is: {context}."
        constraints = (
            " Be specific and respond with non-obvious statistical analyses in the tone of a McKinsey consultant."
            " Describe trend, seasonality, and anomalies. Do not provide recommendations. Do not describe the table."
            " Do not introduce yourself or your role."
        )
        forecasts = self._filter_panel_entities(self.df, basket=basket)
        wide_forecasts = self._panel_to_wide(forecasts)
        prompt = (
            f"{task}"
            f"{prompt_context}"
            f"{constraints}"
            f"{format_dataframes(wide_forecasts)}"
            f"{formatting}"
        )
        response = openai_call(prompt, model, **kwargs)
        return response

    def compare(
        self,
        basket: List[str],
        other_basket: List[str],
        model: MODEL_T = "gpt-3.5-turbo",
        target_feature: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Compare two basket of forecasts."""

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
        forecasts = {
            "This": self._filter_panel_entities(self._df, basket=basket),
            "Other": self._filter_panel_entities(self._df, basket=other_basket),
        }
        wide_forecasts = {k: self._panel_to_wide(df) for k, df in forecasts.items()}
        prompt = (
            f"{task}"
            f"{prompt_context}"
            f"{constraints}"
            f"{format_dataframes(wide_forecasts)}"
            f"{output_format}"
        )
        response = openai_call(prompt, model, **kwargs)
        return response
