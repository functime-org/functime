# Functime

Time-series transformation and forecasting domain covering single and multi-entity sequential observations, leak-free temporal state estimation, and out-of-sample forecast evaluation.

## Language

### Temporal Data

**Temporal Dataset**:
An ordered or unordered collection of sequential observations representing either a single **Time Series** or a **Panel** indexed along a **Time Axis**.
_Avoid_: Panel-only dataset, sequential table, raw record set

**Time Series**:
A sequence of observations indexed along a **Time Axis** for a single unit of study without **Entity** partitioning.
_Avoid_: Univariate dataset, flat series, single-subject sequence

**Panel**:
A **Temporal Dataset** containing multiple **Time Series**, each partitioned by a distinct **Entity**.
_Avoid_: Longitudinal table, cross-sectional stack, multi-series dataset

**Entity**:
An independent unit of observation—identified by one or more identifying attributes—whose **Time Series** evolves within a **Panel**.
_Avoid_: Group ID, series ID, partition key, panel key

**Time Axis**:
The temporal coordinate along which observations in a **Time Series** or **Panel** are ordered and spaced.
_Avoid_: Timestamp, date index, step number, period key

**Target**:
The primary quantity observed along a **Time Axis** that a **Temporal Transform** modifies or a **Forecaster** predicts across a **Forecast Horizon**.
_Avoid_: Dependent variable, response, outcome series

**Exogenous Feature**:
An explanatory variable observed alongside a **Target** whose known or scheduled values inform a **Forecaster** or **Temporal Transform** without itself being forecast.
_Avoid_: Covariate, side feature, external regressor, predictor

**Chronological Order**:
The monotonic ascending arrangement of observations along the **Time Axis** within each **Entity** (or across a single **Time Series**).
_Avoid_: Row order, arrival order, storage order

### Transformations & State

**Temporal Transform**:
A deterministic mapping from an input **Temporal Dataset** to an output **Temporal Dataset** that respects **Entity** boundaries, **Chronological Order**, and **Point-in-Time Validity**.
_Avoid_: Panel transformer, preprocessor, feature operator

**Stateless Transform**:
A **Temporal Transform** whose output depends solely on the input observations and fixed configuration, with no **Fitted State**.
_Avoid_: Memoryless transform, uncalibrated transform

**Stateful Transform**:
A **Temporal Transform** that estimates **Fitted State** from historical observations before applying that state to new observations.
_Avoid_: Learned transform, calibrated preprocessor

**Fitted State**:
The parameters or summary statistics learned per **Entity** (or globally for a **Time Series**) by a **Stateful Transform** or **Forecaster** from historical observations up to a **Cutoff**.
_Avoid_: Training artifacts, cached statistics, learned weights

**Unseen Entity Policy**:
The rule governing how a **Stateful Transform** resolves missing **Fitted State** for an **Entity** absent during calibration.
_Avoid_: Unknown-key fallback, cold-start heuristic

**Invertible Transform**:
A **Temporal Transform** paired with a reverse mapping that reconstructs original-scale **Target** values from transformed values using stored configuration or **Fitted State**.
_Avoid_: Reversible transform, back-transformation

**Point-in-Time Validity**:
The invariant that every value produced by a **Temporal Transform** or **Fitted State** at a given point on the **Time Axis** depends exclusively on observations at or before that point.
_Avoid_: Causal consistency, no-lookahead constraint, anti-leakage

### Forecasting & Evaluation

**Forecast Horizon**:
The fixed number of future steps beyond a **Cutoff** along the **Time Axis** over which a **Forecaster** generates predictions.
_Avoid_: Lead time, prediction window, horizon length

**Cutoff**:
The final observed coordinate on the **Time Axis** separating historical observations available for calibration from the future **Forecast Horizon**.
_Avoid_: Split date, anchor point, forecast origin

**Forecaster**:
A predictive model calibrated on historical **Temporal Dataset** observations up to a **Cutoff** to estimate future **Target** values across a **Forecast Horizon**.
_Avoid_: Time-series regressor, prediction estimator

**Recursive Forecasting**:
A multi-step forecasting strategy where a **Forecaster** predicts one step beyond the **Cutoff** and feeds each prediction back as input to predict subsequent steps in the **Forecast Horizon**.
_Avoid_: Autoregressive rollout, iterative forecasting

**Direct Forecasting**:
A multi-step forecasting strategy where a **Forecaster** predicts each step in the **Forecast Horizon** independently from information available at or before the **Cutoff**.
_Avoid_: Horizon-specific forecasting, non-iterative forecasting

**Backtest**:
An out-of-sample evaluation procedure that evaluates a **Forecaster** across one or more **Cutoff** points along the **Time Axis** while enforcing **Point-in-Time Validity**.
_Avoid_: Historical simulation, rolling cross-validation

**Prediction Interval**:
An interval bounded by lower and upper quantiles expressing calibrated predictive uncertainty around a point estimate for a **Target** across a **Forecast Horizon**.
_Avoid_: Confidence interval, error band, uncertainty range
