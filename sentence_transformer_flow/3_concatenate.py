"""
Add date to hashtags. 
"""
import numpy as np
import pandas as pd

hashtags = np.load("data/st_hashtags_pca.npy")
dates = np.load("data/st_dates.npy")

df = pd.DataFrame(dates, columns=["date"])

df_datetime = pd.to_datetime(df.date, format="%Y-%m-%dT%H:%M:%S.%fZ")
df_datetime = pd.DatetimeIndex(df_datetime)

months = df_datetime.month
months = months.to_numpy()

days = df_datetime.day
days = days.to_numpy()

hours = df_datetime.hour
hours = hours.to_numpy()

minutes = df_datetime.minute
minutes = minutes.to_numpy()

data = np.concatenate((hashtags, months[:, None], days[:, None], hours[:, None], minutes[:, None]), axis=1)
print(data.shape)

np.save("data/st_concatenated.npy", data)