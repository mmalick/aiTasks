import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


pd.set_option('display.max_columns', None)
df = pd.read_csv("G:/PSI zaliczenie/Zad1/games.csv", encoding='latin-1')


def punkt1():

    print(df.columns)
    print(df[['Windows','Mac','Linux']])

    platform_counts = {
        "Windows": df['Windows'].sum(),
        "Mac": df['Mac'].sum(),
        "Linux": df['Linux'].sum()
    }
    print(platform_counts)

    platform_df = pd.DataFrame(list(platform_counts.items()), columns=['Platform', 'Game Count'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Platform', y='Game Count', data=platform_df, palette='viridis')
    plt.title('Liczba gier na platformy')
    plt.xlabel('Platforma')
    plt.ylabel('Liczba gier')
    plt.show()


def punkt2(df):
    print(df[['Positive', 'Negative']])
    
    filtered_df = df[(df['Negative'] != 0) | (df['Positive'] != 0)]

    filtered_df['Positive_rating_percentage'] = (filtered_df['Positive'] / (filtered_df['Positive'] + filtered_df['Negative'])) * 100

    if filtered_df.empty:
        print("Brak ocen")
    else:
        positive_percentage_frame = pd.DataFrame({
            "Percentage": filtered_df['Positive_rating_percentage'],
            "Positive": filtered_df['Positive'],
            "Negative": filtered_df['Negative'],
        })

        print(positive_percentage_frame)
    return filtered_df


def punkt3(filtered_df):
    filtered_df = filtered_df.dropna(subset=['Price'])
    data = filtered_df[['Price', 'Positive_rating_percentage']]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    wcss = [] 
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    # plt.title('Metoda łokciowa')
    # plt.xlabel('Liczba klastrów')
    # plt.ylabel('WCSS')
    # plt.show()

    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(data_scaled)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for cluster in range(optimal_clusters):
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
        plt.scatter(cluster_data['Price'], cluster_data['Positive_rating_percentage'], 
                    label=f'Grupa {cluster}', color=colors[cluster])

    plt.title('Podział na grupy za pomocą K-means')
    plt.xlabel('Cena gry (Price)')
    plt.ylabel('Procent pozytywnych ocen')
    plt.legend()
    plt.show()


def punkt4():
    filtered_df = df.dropna(subset=['Genres', 'Price'])
    filtered_df = filtered_df.sample(n=2000, random_state=22)

    #One-Hot Encoding
    genres_encoded = filtered_df['Genres'].str.split(',').apply(pd.Series).stack().str.strip().reset_index(drop=True)
    genres_encoded = pd.get_dummies(genres_encoded, drop_first=True).groupby(level=0).sum()

    data2 = pd.concat([filtered_df[['Price']].reset_index(drop=True), genres_encoded], axis=1)

    if data2.isnull().sum().any():
        print("Warning: Missing values detected in data2, handling with imputation.")
        data2 = data2.fillna(data2.mean())  # Impute missing values with the mean for simplicity

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data2)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(data_scaled)

    def find_similar_games(game_name):
        if game_name not in filtered_df['Name'].values:
            print(f"Game '{game_name}' not found in the database.")
            print(f"Available games: {filtered_df['Name'].head()}")  # Provide some context
            return

        game_row = filtered_df[filtered_df['Name'] == game_name]
        game_genres = game_row['Genres'].iloc[0].split(', ')
        game_price = game_row['Price'].iloc[0]

        game_genres_encoded = [
            1 if genre in game_genres else 0 for genre in genres_encoded.columns.tolist()
        ]

        game_input = pd.DataFrame([game_genres_encoded + [game_price]], columns=genres_encoded.columns.tolist() + ['Price'])
        game_input = game_input[data2.columns]
        game_input_scaled = scaler.transform(game_input)

        distances, indices = knn.kneighbors(game_input_scaled)
        recommended_games = filtered_df.iloc[indices[0]]

        print(f"\nRecommendations for the game '{game_name}':")
        for i, (_, row) in enumerate(recommended_games.iterrows()):
            print(f"{i + 1}. Name: {row['Name']}, Price: {row['Price']}, Genres: {row['Genres']}, Distance: {distances[0][i]:.2f}")

    find_similar_games("Memento Infernum")




    # user_input_genres = ['Indie','Casual','Action','Exploration']
    # user_input_price = 15.00

    # user_input_genres_encoded = [
    #     2 if genre in user_input_genres else 1 for genre in genres_encoded.columns.tolist()
    # ]

    # user_input = pd.DataFrame([user_input_genres_encoded + [user_input_price]], columns=genres_encoded.columns.tolist() + ['Price'])

    # user_input = user_input[data2.columns.tolist()]

    # user_input_scaled = scaler.transform(user_input)

    # distances, indices = knn.kneighbors(user_input_scaled)
    # recommended_games = filtered_df.iloc[indices[0]]

    # recommended_games['Genres'] = recommended_games['Genres'].apply(lambda x: ', '.join(x.split(', ')))

    # print("\nRekomendacje dla użytkownika:")
    # for i, (_, row) in enumerate(recommended_games.iterrows()):
    #     print(f"{i + 1}. Name: {row['Name']}, Price: {row['Price']}, Genres: {row['Genres']}, Distance: {distances[0][i]:.2f}")



#punkt1()
#punkt2(df)
#filtered_df = punkt2(df)
#punkt3(filtered_df)
punkt4()