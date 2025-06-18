import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'title': ['The Matrix', 'RRR', 'Baahubali 2', 'Dangal', 'KGF: Chapter 2',
              'Inception', 'Drishyam', 'Pulp Fiction', 'Premam', 'Lucifer',
              'Interstellar', '3 Idiots', 'The Godfather', 'Vikram', 'Bangalore Days',
              'The Dark Knight', 'Kabali', 'Pather Panchali', 'Ustad Hotel', 'Thani Oruvan'],
    'genre': ['action sci-fi', 'action drama', 'epic action', 'sports drama', 'action crime',
              'sci-fi thriller', 'crime thriller', 'crime drama', 'romantic drama', 'political thriller',
              'sci-fi adventure', 'comedy drama', 'crime drama', 'action thriller', 'romantic comedy',
              'action crime drama', 'gangster drama', 'art drama', 'family drama', 'action thriller'],
    'director': ['Lana Wachowski', 'S.S. Rajamouli', 'S.S. Rajamouli', 'Nitesh Tiwari', 'Prashanth Neel',
                 'Christopher Nolan', 'Jeethu Joseph', 'Quentin Tarantino', 'Alphonse Puthren', 'Prithviraj Sukumaran',
                 'Christopher Nolan', 'Rajkumar Hirani', 'Francis Ford Coppola', 'Lokesh Kanagaraj', 'Anjali Menon',
                 'Christopher Nolan', 'Pa. Ranjith', 'Satyajit Ray', 'Anwar Rasheed', 'Mohan Raja'],
    'year': [1999, 2022, 2017, 2016, 2022,
             2010, 2013, 1994, 2015, 2019,
             2014, 2009, 1972, 2022, 2014,
             2008, 2016, 1955, 2012, 2016],
    'rating': [8.7, 8.0, 8.2, 8.4, 8.4,
               8.8, 8.3, 8.9, 8.1, 7.6,
               8.6, 8.4, 9.2, 8.4, 7.9,
               9.0, 7.1, 8.5, 8.3, 8.2],
    'language': ['English', 'Telugu', 'Telugu', 'Hindi', 'Kannada',
                 'English', 'Malayalam', 'English', 'Malayalam', 'Malayalam',
                 'English', 'Hindi', 'English', 'Tamil', 'Malayalam',
                 'English', 'Tamil', 'Bengali', 'Malayalam', 'Tamil'],
    'country': ['USA', 'India', 'India', 'India', 'India',
                'USA', 'India', 'USA', 'India', 'India',
                'USA', 'India', 'USA', 'India', 'India',
                'USA', 'India', 'India', 'India', 'India']
})

def create_soup(x):
    return f"{x['genre']} {x['director']} {x['year']} {x['language']} {x['country']}"
movies['soup'] = movies.apply(create_soup, axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['soup'])
cosine_sim = cosine_similarity(tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, n=5, rating_threshold=None,
                        preferred_language=None, preferred_country=None):
    try:
        idx = movies[movies['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return None
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies.iloc[movie_indices][['title', 'genre', 'director', 'year', 'rating', 'language', 'country']]
    if rating_threshold is not None:
        recommendations = recommendations[recommendations['rating'] >= rating_threshold]
    if preferred_language:
        recommendations = recommendations[recommendations['language'].str.lower() == preferred_language.lower()]
    if preferred_country:
        recommendations = recommendations[recommendations['country'].str.lower() == preferred_country.lower()]
    return recommendations.head(n).reset_index(drop=True)

def recommend_movies():
    print("Available Movies:")
    print(movies['title'].to_string(index=False))
    while True:
        print("\nEnter a movie title( or'q' to quit):")
        user_input =input().strip()
        if user_input.lower()=='q':
            break
        if not any(movies['title'].str.lower()==user_input.lower()):
            continue
        print("how many reccomendations would you like?(1-10):")
        try:
            n=int(input().strip())
            if n < 1 or n > 10:
                print("using default value 5")
                n=5
        except:
            n= 5
        print("Minimum rating threshold(0-10)? leave blank for no filter")
        try:
            rating_input = input().strip()
            rating_threshold = float(rating_input) if rating_input else None
            
        except:
            rating_threshold = None
            
        print("Prefered Language(optional): ")
        prefered_language = input().strip() or None
        
        print("Prefered Country(optional): ")
        prefered_country = input().strip() or None
        
        results = get_recommendations(
            title=user_input,
            n=n,
            rating_threshold=rating_threshold,
            preferred_language=prefered_language,
            preferred_country=prefered_country
        )
        
        if results is None or results.empty:
            print("no matching reccomendations found.")
        else:
            print(f"\n Recommendations based on '{user_input}':\n")
            print(results.to_string(index=False))
            print("\n would you like another reccomendation?(y/n):")
            if input().strip().lower()!='y':
                break
if __name__ == "__main__":
    print(" MOVIE RECOMMENDATION  SYSTEM with Language & Country Preferences")
    print("-------------------------------------------------------------------")
    recommend_movies()
