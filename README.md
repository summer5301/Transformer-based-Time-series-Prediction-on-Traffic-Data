# Transformer-Based Time Series Prediction on Traffic Data
This project explores whether the advantages of the "self-attention mechanism" translate effectively to time series forecasting based on two transformer-based models, TimeSeriesTransformer and Autoformer, on a timer-series transportation dataset. 

# Data Source
We fine-tune and validate two target models with a time-series traffic dataset. Our dataset is sourced from police-reported motor vehicle collisions in NYC since 1998 to 2025. [Motor-Vehicle-Collisions-Crashes data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)

One can download original data using Socrata API:

```
import pandas as pd
from sodapy import Socrata

# download dataset vis Socrate API for data only from 2021 to 2025
results = Socrata("data.cityofnewyork.us", None).get("h9gi-nx95",
                                                     select="collision_id, crash_date, borough",
                                                     where="crash_date between '2020-12-31T12:00:00' and '2024-12-31T14:00:00'", # 48 months
                                                     limit=500000)
```

We intended to focus on the most recent four years' motor vehicle collisions in NYC. The data was cleaned and filtered from January 1, 2021, to December 31, 2024, encompassing 48 months of crash records. These records were aggregated by borough and by month, resulting in monthly collision counts for each borough. A 12-month prediction window (one year) was defined for long-horizon forecasting. The training set spans the first 36 months (January 2021 – December 2023) while the test set covers the full 48-month period, enabling forecasts for the final 12 months (January 2024 – December 2024). The data transformation code is included in ./src/code.py script.

![download (5)](https://github.com/user-attachments/assets/91063f9f-3940-4d93-a03d-f1b2b95ece67)

#  Model Selection

This work chose two recent emerged transformer-based models: "TimeSeriesTransformer" and "Autoformer". The former is a vanilla encoder-decoder Transformer for time series forecasting contributed by [Kashif, etc](https://arxiv.org/abs/2310.08278). 

This decoder-only transformer architecture that uses lags as co-variates with is pretrained on a large corpus of diverse time series data from several domains, and demonstrates strong zero-shot generalization capabilities.  "Autoformer" was introduced by [Wu, ect](https://arxiv.org/abs/2106.13008) as a novel architecture integrates a decomposition framework with an Auto-Correlation mechanism.

Together, TimeSeriesTransformer and Autoformer represent state-of-the-art approaches to our best knowledge that blend traditional forecasting insights with modern deep learning innovations, offering promising new directions for complex, real-world time-series prediction tasks. By comparing these models, we investigate whether their attention-based structures can effectively learn and predict monthly collision trends in NYC over a 12-month horizon, and provide explanations for the results we obtained.

# Packages Required

```
! pip install -q transformers datasets evaluate accelerate "gluonts[torch]" ujson tqdm
! pip install pandas
! pip install sodapy
```

# Results

![download](https://github.com/user-attachments/assets/e99bdb13-eca6-448d-9425-975e27a43134)
![download (2) (1)](https://github.com/user-attachments/assets/787d404c-3829-4ea7-9c1c-84bb5a70cb64)

Our result in loss curve shows that both models exhibit a general downward trend in the training loss as the number of epochs increases, indicating successful learning; nevertheless, some distinct trait are observed.

TimeSeriesTransformer shows a steady decrease in loss through the initial epochs, occasionally experiencing spikes in training loss but generally trending downward. Near the later epochs (beyond approximately 150), the model experiences a few larger fluctuations, suggesting sensitivity to learning rate or hyperparameter settings. On the contrary, Autoformer also displays a gradual reduction in loss, with some oscillations but fewer large spikes compared to TimeSeriesTransformer. Ends training at a loss value comparable to or slightly higher than the lowest points of the TimeSeriesTransformer, though overall differences are modest.

![download (1)](https://github.com/user-attachments/assets/baf2fb93-2af6-4eac-a9ee-149855dcfde4){:height="36px" width="36px"}
![download (3)](https://github.com/user-attachments/assets/6f8e86c5-fe3a-4ef2-88df-143b63138890)


The forecasting plots for Bronx, Brooklyn, Manhattan, Queens, and Staten Island are shown below. We observed that both models track the general upward/downward trends reasonably well. Additionally, the displayed ±1 standard deviation intervals suggest that both models maintain relatively tight uncertainty bounds, though some widening occurs toward the end of the forecast window. TimeSeriesTransformer sometimes appears to yield narrower intervals, suggesting a more confident prediction, while Autoformer intervals can be slightly broader in certain months. We also point that both models tend to underestimate the result.

In borough-by-borough comparison, Manhattan and Brooklyn tend to show higher collision counts, with both models reflecting similar seasonal fluctuations. Bronx and Queens exhibit moderate collision counts, and predictions across these boroughs are generally aligned between the two models. As the timeline moves from 2023 to 2024, the forecasts of both model indicate a gradual increase in monthly collision counts for some boroughs which coincides with ground truth. We hold the reason to believe that both model are able to capture the  long-term direction of the time series. Visually, TimeSeriesTransformer showing less alignment to ground truth than Autoformer in most cases. 

# Conclusion

This project demonstrates that transformer-based models—originally designed for NLP—can be successfully adapted for long-term time-series forecasting of monthly motor vehicle collisions in NYC. Both the TimeSeriesTransformer and Autoformer successfully captured the temporal dependencies, seasonal patterns, and underlying trends in the monthly collision data, providing reliable forecasts over a 12-month horizon. The study also underscores the broader implications of applying transformer-based architectures to time-series analysis. The ability to capture intricate temporal dynamics not only improves forecasting accuracy but also provides a deeper understanding of the underlying processes driving motor vehicle collisions. This insight is crucial for developing proactive traffic safety strategies and informing policy decisions in dynamic urban environments.

# References

This work referred to Hugging Face official tutorial of "Probabilistic Time Series Forecasting with 🤗 Transformers" by [Kashif, etc](https://huggingface.co/blog/time-series-transformers) to define time feature and data transformer align with GluonTS style. 
