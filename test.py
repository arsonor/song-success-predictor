import requests

song_id = '62yJjFtgkhUrXktIoSjgP2'
song = {
    'release_year': 2013,
    'key': 'A',
    'acousticness': 0.114,
    'danceability': 0.457,
    'energy': 0.793,
    'instrumentalness': 0.000101,
    'liveness': 0.671,
    'loudness': -3.699,
    'speechiness': 0.0634,
    'valence': 0.195,
    'tempo': 136.382,
    'popularity': 91.0,
    'artist_type': 'band',
    'genre': 'country',
    'duration_log': 12.137869,
    'followers_log': 16.877702
}

url = 'http://localhost:9696/predict'

# host = 'hit-serving-env.eba-haikumet.eu-west-3.elasticbeanstalk.com'
# url = f'http://{host}/predict'

response = requests.post(url, json=song).json()
print(response)

if response['hit'] == True:
    print('this song %s is a hit!' % song_id)
else:
    print('this song %s is not a hit' % song_id)