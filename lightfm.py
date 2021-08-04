import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import os
# import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
# from sklearn.preprocessing import LabelEncoder
from lightfm import LightFM

os.makedirs('models', exist_ok=True)

df_tracks = pd.read_hdf('df_data/df_tracks.hdf')
df_playlists = pd.read_hdf('df_data/df_playlists.hdf')
df_playlists_info = pd.read_hdf('df_data/df_playlists_info.hdf')
df_playlists_test = pd.read_hdf('df_data/df_playlists_test.hfd')
df_playlists_test_info = pd.read_hdf('df_data/df_playlists_test_info.hdf')

train = pd.read_hdf('df_data/train.hdf')
val = pd.read_hdf('df_data/val1.hdf')
val1_pids = joblib.load('df_data/val1_pids.pkl')

user_seen = train.groupby('pid').tid.apply(set).to_dict()
val_tracks = val.groupby('pid').tid.apply(set).to_dict()

config = {
    'num_playlists': df_playlists_test_info.pid.max() +1,
    'num_tracks': df_tracks.tid.max() +1
}

X_train = sp.coo_matrix((np.ones(len(train)), (train.pid, train.tid)),
                        shape=(config['num_playlists'], config['num_tracks']))
config['model_path'] = 'models/lightfm_model.pkl'

model = LightFM(no_components=200, loss='warp', learning_rate=0.02, max_sampled=400,
                random_state=1, user_alpha=1e-05)

best_score = 0
for i in range(60):
    model.fit_partial(X_train, epochs=5, num_threads=50)

    model.batch_setup(
        item_chunks={0: np.arange(config['num_tracks'])},
        n_process=50
    )
    res = model.batch_predict(chun_id=0, user_ids=val1_pids, top_k=600)
    model.batch_cleanup()

    score=[]
    for pid in val1_pids:
        tracks_t = val_tracks[pid]
        tracks = [i for i in res[pid][0] if i not in user_seen.get(pid, set())][:len(tracks_t)]
        guess = np.sum([i in tracks_t for i in tracks])
        score.append(guess / len(tracks_t))

    score = np.mean(score)
    print(score)
    if score > best_score:
        joblib.dump(model, open(config['model_path'], 'wb'))
        best_score = score
