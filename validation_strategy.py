import joblib
import numpy as np
import pandas as pd

np.random.seed(0)

df_tracks = pd.read_hdf('df_data/df_tracks.hdf')
df_playlists = pd.read_hdf('df_data/df_playlists.hdf')
df_playlists_info = pd.read_hdf('df_data/df_playlists_info.hdf')
df_playlists_test = pd.read_hdf('df_data/df_playlists_test.hdf')
df_playlists_test_info = pd.read_hdf('df_data/df_playlists_test_info.hdf')

num_tracks = df_playlists_info.groupby('num_tracks').pid.apply(np.array)

validation_playlists = {}
for i, j in df_playlists_test_info.num_tracks.value_counts().reset_index().values:
    validation_playlists[i] = np.random.choice(num_tracks.loc[i], 2 * j, replace=False)

val1_playlist = {}
val2_playlist = {}
for i in [0, 1, 5, 10, 25, 100]:

    val1_playlist[i] = []
    val2_playlist[i] = []

    value_counts = df_playlists_test_info.query('num_samples==@i').num_tracks.value_counts()
    for j, k in value_counts.reset_index().values:

        val1_playlist[i] += list(validation_playlists[j][:k])
        validation_playlists[j] = validation_playlists[j][k:]

        val2_playlist[i] += list(validation_playlists[j][:k])
        validation_playlists[j] = validation_playlists[j][k:]

val1_index = df_playlists.pid.isin(val1_playlist[0])
val2_index = df_playlists.pid.isin(val2_playlist[0])

for i in [1, 5, 10, 25, 100]:
    val1_index = val1_index | (df_playlists.pid.isin(val1_playlist[i]) & (df_playlists.pos >= i))
    val2_index = val2_index | (df_playlists.pid.isin(val2_playlist[i]) & (df_playlists.pos >= i))

train = df_playlists[~(val1_index | val2_index)]

val1 = df_playlists[val1_index]
val2 = df_playlists[val2_index]

val1_pids = np.hstack([val1_playlist[i] for i in val1_playlist])
val2_pids = np.hstack([val2_playlist[i] for i in val2_playlist])

train = pd.concat([train, df_playlists_test])

train.to_hdf('df_data/train.hdf', key='abc')
val1.to_hdf('df_data/val1.hdf', key='abc')
val2.to_hdf('df_data/val2.hdf', key='abc')
joblib.dump(val1_pids, 'df_data/val1_pids.pkl')
joblib.dump(val2_pids, 'df_data/val2_pids.pkl')


