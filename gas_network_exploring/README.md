# prepare environment:
* ```pip install -r requirements.txt``` <br>
  ```jupyter nbextension enable --py widgetsnbextension```

# about gas_network_dataset.csv
    gas_network_generation.ipynb
* this dataset is stored in folder ./data/gas_network_dataset.csv

* if some problems about ```df.progress_apply()``` or ```series.progress_apply()``` come out: <br>
please change ```df.progress_apply()``` or ```series.progress_apply()``` with normal ```df.apply()``` or ```series.apply()```

* step to generate dataset:
    1. create entsog_dataset
    2. train lasso regression x=diameter y=capacity dataset=Internet raw
    3. fill missing value with help entsog_dataset
    4. fill missing diameter data with diameter level EMAP
    5. fill still missing diameter with IGG diameter_mm
    5. fill missing capacity with lasso regression
    7. dealing with negative value,reset negative value as missing value
    6. capacity spread(dont need)
