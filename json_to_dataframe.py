import os
import json
import pandas as pd

os.makedirs('df_data', exist_ok=True)


def create_df_data():
    path = 'data/data'
    playlist_col = ['collaborative', 'duration_ms', 'modified_at',
                    'name', 'num_albums', 'num_artists', 'num_edits',
                    'num_followers', 'num_tracks', 'pid']
    tracks_col = ['album_name', 'album_uri', 'artist_name', 'artist_uri',
                  'duration_ms', 'track_name', 'track_uri']
    playlist_test_col = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']

    filenames = os.listdir(path)

    data_playlists = []
    data_tracks = []
    playlists = []

    tracks = set()

    for filename in filenames:  # delete mpd.slice.833000-833999.json
        fullpath = os.sep.join((path, filename))
        f = open(fullpath)
        js = f.read()
        f.close()

        mpd_slice = json.loads(js)

        for playlist in mpd_slice['playlists']:
            data_playlists.append([playlist[col] for col in playlist_col])
            for track in playlist['tracks']:
                playlists.append([playlist['pid'], track['track_uri'], track['pos']])
                if track['track_uri'] not in tracks:
                    data_tracks.append([track[col] for col in tracks_col])
                    tracks.add(track['track_uri'])

    df_playlists_info = pd.DataFrame(data_playlists, columns=playlist_col)
    df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false': False, 'true': True})

    df_tracks = pd.DataFrame(data_tracks, columns=tracks_col)
    df_tracks['tid'] = df_tracks.index

    track_uri2tid = df_tracks.set_index('track_uri').tid

    df_playlists = pd.DataFrame(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists.tid = df_playlists.tid.map(track_uri2tid)

    df_tracks.to_hdf('df_data/df_tracks.hdf', key='abc')
    df_playlists.to_hdf('df_data/df_playlists.hdf', key='abc')
    df_playlists_info.to_hdf('df_data/df_playlists_info.hdf', key='abc')

    f = open('data/challenge_set.json')
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    data_playlists_test = []
    playlists_test = []

    for playlist in mpd_slice['playlists']:
        data_playlists_test.append([playlist.get(col, '') for col in playlist_test_col])
        for track in playlist['tracks']:
            playlists_test.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_col])
                tracks.add(track['track_uri'])

    df_playlists_test_info = pd.DataFrame(data_playlists_test, columns=playlist_test_col)

    df_playlists_test = pd.DataFrame(playlists_test, columns=['pid', 'tid', 'pos'])
    df_playlists_test.tid = df_playlists_test.tid.map(track_uri2tid)

    df_playlists_test.to_hdf('df_data/df_playlists_test.hdf', key='abc')
    df_playlists_test_info.to_hdf('df_data/df_playlists_test_info.hdf', key='abc')


create_df_data()
