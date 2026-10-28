import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

df = pd.read_csv('Spotify_tracks_final.csv', index_col=0)

print(df.columns.tolist())

# Aantal dubbele tracknamen berekenen
duplicates = df['track_name'].duplicated().sum()
print(f"Duplicates: {duplicates}")

# Dubbele tracks verwijderen, alle kolommen behouden
df = df.drop_duplicates(subset=['track_name'])

# Unieke genres weergeven
print(f"Unique genres:\n{df['genre'].unique()}\n")

# Voor de eerste analyse kijken we naar de numerieke features.

numeric_features = df.select_dtypes(include=np.number).columns.tolist()


# In deze dataset zijn er enorm veel genres. Veel van deze genres lijken ook sterk op elkaar.
# Een machine learning model zal dus redelijk wat problemen hebben bij het onderscheiden van bijvoorbeeld hard-rock en alt-rock.

ROCK_GENRES = [
    "alt-rock",
    "alternative",
    "hard-rock",
    "rock",
    "punk",
    "punk-rock",
    "rock-n-roll",
    "rockabilly",
    "groove",
    "grunge",
    "goth",
    "hardcore",
    "psych-rock",
    "power-pop",
    "indie-rock",
    "emo",
    "j-rock",
    "guitar"]

df_rock = df[df['genre'].isin(ROCK_GENRES)]
genre_means = df_rock.groupby('genre')[numeric_features].mean()

genre_scaled = StandardScaler().fit_transform(genre_means)

distance_matrix = euclidean_distances(genre_scaled)
distance_df = pd.DataFrame(distance_matrix,
                           index=genre_means.index,
                           columns=genre_means.index)

plt.figure(figsize=(12, 10))
sn.heatmap(distance_df, cmap='YlGnBu',
            xticklabels=True, yticklabels=True)
plt.title('Genre similarity')
plt.tight_layout()
plt.savefig('./eda_visualisations/genre_similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Buiten het donkerblauwe kruis op "guitar" zijn meeste genres binnen de rockfamilie redelijk gelijkaardig. Tijdens het testn van vorige modellen struikelde het model ook het sterkst over genres die sterk bij elkaar liggen. Voor deze oefening zullen we vervolgens meerdere genres samen groeperen

# Kolommen die verwijdert kunnen worden
# De eerste en meest duidelijke kolom die verwijdert kan worden zijn de verschillende tijdkolommen. Deze geven exact dezelfde waarden maar zijn gewoon een verschillende precisie.
# Bijgevolg nemen we enkel duration_ms mee. 
heatmap = sn.heatmap(df[['duration_seconds','duration_minutes','duration_ms']].corr(), annot=True, cmap="Blues", fmt='.1g', cbar=False)
plt.savefig(f"./eda_visualisations/heatmap_duration.png", dpi=150, bbox_inches="tight")
plt.close()
# 
# Andere kolommne die weg kunnen gelaten worden zijn kolommen die randinformatie geven over een track. Dit gaat dan bijvoorbeeld over de track name, de artist, uit welk album het kwam.
# Deze gegevens zijn moeilijk om te zetten naar numerieke waarden. Hoewel een artist of een album duiding kan geven over uit welk genre het kwam, zal het model wellicht geen voorspellingen
# kunnen doen op basis van deze gegevens als het bv. een nieuw nog niet gekend album or artist tegenkomt.
#
# Daarnaast kan de kolom databaseshard_id verder ook weggelaten worden. Deze kolom is een database artifact en geeft geen gevolg op het genre. 
# Gelijkaardig met preview_url. Dit is enkel een weburl en geeft geen impact op het genre.

# Van numerieke features kunnen meeste behouden worden doordat deze een grote impact hebben op genre.
#
CURRENT_FEATURES = ['popularity', 'duration_ms', 'explicit', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
SAMPLE_GENRES = ['Classical', 'hard-rock', 'blues', 'country', 'latin','funk','disco','electronic','pop']
genre_means_z = (genre_means - genre_means.mean()) / genre_means.std()
df_sample = df[df['genre'].isin(SAMPLE_GENRES)]

genre_means = df_sample.groupby('genre')[CURRENT_FEATURES].mean()
genre_means_z = (genre_means - genre_means.mean()) / genre_means.std()

plt.figure(figsize=(12, 8))
sn.heatmap(genre_means_z, cmap='coolwarm', center=0, annot=True, fmt='.1f')
plt.title('Feature means per genre (z-scored across genres)')
plt.tight_layout()
plt.savefig('./eda_visualisations/genre_feature_means.png', dpi=150)
plt.close()

# In de gegeven grafiek lijkt er een goeie spread over de verschillende genres over het gemiddelde van elke feature.
# Enkele features die weinig verschil tonen is time_signature.
# Samen met key en mode zijn dit geen ordinale waarde. 
# Key geeft de muzieknoot aan encoded in een nummer. Mode is dan of deze noot in major of in minor is.
# Deze kunnen weliswaar invloed hebben op het genre maar moeten geencode worden omdat dit geen ordinale waarden zijn.

CATEGORICAL_FEATURES = ['mode','key','time_signature']
df_sample[CATEGORICAL_FEATURES].apply(lambda s: s.value_counts().sort_index()).plot(kind='bar', subplots=True, layout=(1,3), figsize=(12,4), legend=False)
plt.savefig('./eda_visualisations/categorical_bars.png', dpi=150)
plt.close()

# Zoals in deze grafiek kan gezien worden zijn mode en key over het algemeen redelijk wijd verspreidt. time_signature aan de andere kant heeft de overgrote meerderheid op 4 met weinig variatie.
# Doordat meeste tracks op time signature 4 bevinden zal het model wellicht geen geoeie voorspellingen kunnen maken op deze kolom. De waarden van mode en key kunnen wel behouden worden maar moeten
# geencode worden. Doordat er geen ordinaal verband is tussen verschillende muzieknoten

# Explictiness
# Eer zijn maar een heel klein aantal nummers die explicit zijn over de volledige dataset geconcentreerd onder enkele genres.
# Enkel kijken naar de samples dataframe geeft niks. 
# Doordat we al genres samen gaan groeperen zal deze feature zijn nut snel verliezen. 
df.groupby('genre')['explicit'].value_counts().unstack().plot(kind='bar', stacked=True, figsize=(12,5))
plt.savefig('./eda_visualisations/explicit.png', dpi=150)
plt.close()
explicit_by_genre = df.groupby('genre')['explicit'].mean().sort_values(ascending=False)
explicit_by_genre.head(15).plot(kind='bar', figsize=(12,5))
plt.ylabel('proportion explicit')
plt.tight_layout()
plt.savefig('./eda_visualisations/explicit_by_genre.png', dpi=150)
plt.close()

CURRENT_FEATURES = ['album_release_date', 'album_release_date_precision' ,'popularity', 'duration_ms', 'genre', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']

# album_release_date
# De datum waarop een album is uitgebracht kan een impact hebben op het genre. Genres zoals jazz en funk zullen wellicht vaak meer oudere release dates hebben dan andere genres zoals rocl.
# Wellicht hebben we hier niet de volledige precisie voor nodg en volstaat het gewoon om het jaar op te geven.
# Over hte algemeen lijkt er wel een verband te zijn met wanneer de track uitgebracht werd en het genre. 

df_sample["release_year"] = pd.to_datetime(df_sample["album_release_date"], format="mixed", errors="coerce").dt.year

order = df_sample.groupby('genre')['release_year'].median().sort_values().index

plt.figure(figsize=(10, 8))
sn.boxplot(data=df_sample, y='genre', x='release_year', order=order)
plt.tight_layout()
plt.savefig('./eda_visualisations/year_by_genre_box.png', dpi=150)
plt.close()


# Omdat de dataset enorm groot is en er per genre mogelijks maar enkele entries zijn is het best om deze te groeperen per genrefamilie
# De verdeling van genres is over het algemeen vrij subjectief en is er vaak overlap tussen verschillende groepen.
# Over het algemeen worden genres gegroepeerd per 'familie' of in welk breder genre ze kunnen geplaatst worde.
# De verschillende rock genres worden hier bv. onder Rock geplaatst
# Genres die geen overlappend genre hebben zoals Ambient werden hieruit weggelaten. Deze zouden te klein zijn in de dataset om er redelijke predicities mee te maken.
# De verschillende genres zoals "French" en "German" werden hier ook uit weggelaten. Deze vertegenwoordigen eerder het land waar ze uit kwamen dan het effectieve genre.
GENRE_MAPPING = {
    "alt-rock": "Rock",
    "alternative": "Rock",
    "hard-rock": "Rock",
    "rock": "Rock",
    "punk": "Rock",
    "punk-rock": "Rock",
    "rock-n-roll": "Rock",
    "rockabilly": "Rock",
    "groove": "Rock",
    "grunge": "Rock",
    "goth": "Rock",
    "hardcore": "Rock",
    "psych-rock": "Rock",
    "power-pop": "Rock",
    "indie-rock": "Rock",
    "emo": "Rock",
    "j-rock": "Rock",
    
    "metal": "Metal",
    "metalcore": "Metal",
    "death-metal": "Metal",
    "black-metal": "Metal",
    "grindcore": "Metal",

    "pop": "Pop",
    "indie-pop": "Pop",
    "synth-pop": "Pop",
    "k-pop": "Pop",
    "j-pop": "Pop",
    "j-idol": "Pop",
    "cantopop": "Pop",
    "mandopop": "Pop",
    "pop-film": "Pop",
    
    "dance": "Electronic",
    "edm": "Electronic",
    "electronic": "Electronic",
    "house": "Electronic",
    "deep-house": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "dubstep": "Electronic",
    "drum-and-bass": "Electronic",
    "breakbeat": "Electronic",
    "chicago-house": "Electronic",
    "detroit-techno": "Electronic",
    "minimal-techno": "Electronic",
    "progressive-house": "Electronic",
    "club": "Electronic",
    "dub": "Electronic",
    "idm": "Electronic",
    "hardstyle": "Electronic",
    
    "hip-hop": "Hip-Hop/R&B",
    "r-n-b": "Hip-Hop/R&B",
    "reggae": "Hip-Hop/R&B",
    "reggaeton": "Hip-Hop/R&B",
    "dancehall": "Hip-Hop/R&B",
    
    "jazz": "Jazz/Blues",
    "blues": "Jazz/Blues",
    
    "classical": "Classical/Opera",
    "opera": "Classical/Opera",
    "piano": "Classical/Opera",
    "romance": "Classical/Opera",
    "show-tunes": "Classical/Opera",
    
    "country": "Country/Folk",
    "folk": "Country/Folk",
    "bluegrass": "Country/Folk",
    "honky-tonk": "Country/Folk",
    "singer-songwriter": "Country/Folk",
    "songwriter": "Country/Folk",
    
    "latin": "Latin",
    "latino": "Latin",
    "salsa": "Latin",
    "samba": "Latin",
    "brazil": "Latin",
    "forro": "Latin",
    "mpb": "Latin",
    "pagode": "Latin",
    "sertanejo": "Latin",

    "funk": "Funk/Disco",
    "disco": "Funk/Disco",
}
