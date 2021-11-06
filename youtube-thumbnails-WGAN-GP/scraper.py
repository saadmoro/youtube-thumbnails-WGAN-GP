import requests
from googleapiclient.discovery import build

api_key = "API KEY"

youtube = build('youtube', 'v3', developerKey=api_key)

playlist_id = 'COPY PLAYLIST ID FROM YOUTUBE PLAYLIST URL'

vids = []

nextPageToken = None
while True:
    pl_req = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50,
        pageToken=nextPageToken
    )

    pl_res = pl_req.execute()
    for item in pl_res['items']:
        vids.append(item['contentDetails']['videoId'])

    nextPageToken = pl_res.get('nextPageToken')
    if not nextPageToken:
        break


for vid in vids:
    url = f'http://img.youtube.com/vi/{vid}/0.jpg'
    res = requests.get(url, allow_redirects=True)
    open(f'thumbs/{vid}.jpg', 'wb').write(res.content)
