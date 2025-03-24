# Transformer-based-Time-series-Prediction-on-Traffic-Data
This project explores whether the advantages of the ``self-attention mechanism" translate effectively to time series forecasting based on two transformer-based models, TimeSeriesTransformer and Autoformer, on a timer-series transportation dataset. 

# Data Source
We fine-tune and validate two target models with a time-series traffic dataset. Our dataset is sourced from police-reported motor vehicle collisions in NYC since 1998 to 2025: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data

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


# Packages Required

```
! pip install -q transformers datasets evaluate accelerate "gluonts[torch]" ujson tqdm
! pip install pandas
! pip install sodapy
```

