import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 1rem;
    font-weight: 600;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}
.recommendation-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
    border-left: 4px solid #2ecc71;
}
.movie-card {
    background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid #e0e0e0;
}
.stats-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    background-color: #2ecc71;
    color: white;
    font-weight: bold;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING WITH CACHING
# ============================================================================

@st.cache_data
def load_data(base_path):
    """Load processed data with error handling"""
    try:
        movies = pd.read_csv(os.path.join(base_path, 'movies_processed.csv'))
        ratings = pd.read_csv(os.path.join(base_path, 'ratings_processed.csv'))
        user_stats = pd.read_csv(os.path.join(base_path, 'user_statistics.csv'))
        genre_stats = pd.read_csv(os.path.join(base_path, 'genre_statistics.csv'))
        
        return movies, ratings, user_stats, genre_stats
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("💡 Please ensure the data files are in the correct directory.")
        return None, None, None, None

@st.cache_resource
def load_models(models_path):
    """Load trained models"""
    try:
        baseline = joblib.load(os.path.join(models_path, 'baseline_popularity.pkl'))
        best_model = joblib.load(os.path.join(models_path, 'lasso_model.pkl'))
        
        return baseline, best_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Adjust these paths for your environment
BASE_PATH = '/content/drive/MyDrive/MovieLens_Processed'
MODELS_PATH = os.path.join(BASE_PATH, 'models')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
REPORTS_PATH = os.path.join(BASE_PATH, 'reports')

# Load data and models
movies, ratings, user_stats, genre_stats = load_data(BASE_PATH)
baseline_model, best_model = load_models(MODELS_PATH)

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

def get_user_recommendations(user_id, n_recommendations=20, genre_filter=None):
    """Generate personalized recommendations for a user"""
    if baseline_model is None or ratings is None:
        return None, None
    
    # Get user's watched movies
    user_ratings = ratings[ratings['userId'] == user_id]
    watched_movies = set(user_ratings['movieId'])
    
    # Get recommendations (movies not watched)
    recommendations = baseline_model['movie_scores'][
        ~baseline_model['movie_scores']['movieId'].isin(watched_movies)
    ].copy()
    
    # Apply genre filter if specified
    if genre_filter and genre_filter != 'All Genres':
        recommendations = recommendations[
            recommendations['genres'].str.contains(genre_filter, na=False)
        ]
    
    return recommendations.head(n_recommendations), user_ratings

def search_similar_movies(movie_title, n_similar=10):
    """Find similar movies based on genre similarity"""
    if movies is None or baseline_model is None:
        return None
    
    # Find the movie
    movie_matches = movies[
        movies['title'].str.contains(movie_title, case=False, na=False)
    ]
    
    if len(movie_matches) == 0:
        return None
    
    # Get the first match
    target_movie = movie_matches.iloc[0]
    target_genres = set(target_movie['genres'].split('|'))
    
    # Calculate genre similarity
    def genre_similarity(genres_str):
        if pd.isna(genres_str):
            return 0
        movie_genres = set(genres_str.split('|'))
        return len(target_genres & movie_genres) / len(target_genres | movie_genres)
    
    similar_movies = movies.copy()
    similar_movies['similarity'] = similar_movies['genres'].apply(genre_similarity)
    
    # Merge with scores and filter
    similar_movies = similar_movies.merge(
        baseline_model['movie_scores'][['movieId', 'weighted_score', 'avg_rating', 'vote_count']],
        on='movieId',
        how='left'
    )
    
    # Remove the target movie and sort
    similar_movies = similar_movies[
        similar_movies['movieId'] != target_movie['movieId']
    ].sort_values(['similarity', 'weighted_score'], ascending=[False, False])
    
    return similar_movies.head(n_similar)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🎬 MovieLens Recommender")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "📍 Navigate",
    ["🏠 Home", "👤 User Recommendations", "🔍 Movie Search & Similar",
     "📊 Business Insights", "🤖 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info(
    """
    This recommendation system uses advanced machine learning 
    to provide personalized movie recommendations based on 
    the MovieLens dataset.
    
    **Features:**
    - Personalized recommendations
    - Similar movie discovery
    - Business analytics
    - Model comparisons
    """
)

# Display statistics in sidebar
if movies is not None and ratings is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Statistics")
    st.sidebar.metric("🎬 Total Movies", f"{len(movies):,}")
    st.sidebar.metric("⭐ Total Ratings", f"{len(ratings):,}")
    st.sidebar.metric("👥 Total Users", f"{ratings['userId'].nunique():,}")
    st.sidebar.metric("🎯 Avg Rating", f"{ratings['rating'].mean():.2f}")

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">🎬 MovieLens Recommendation System</p>',
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the MovieLens Recommender! 🎉
    
    This enterprise-grade recommendation system provides:
    - 🎯 **Personalized Recommendations**: Get movie suggestions tailored to your taste
    - 🔍 **Similar Movie Discovery**: Find movies similar to your favorites
    - 📊 **Business Insights**: Explore genre trends and user behavior analytics
    - 🤖 **ML Model Performance**: Compare different recommendation algorithms
    
    **Getting Started:**
    1. Choose a feature from the sidebar
    2. Explore personalized recommendations
    3. Discover hidden gems in the movie database
    
    ---
    """)
    
    # Display key metrics
    if movies is not None and ratings is not None:
        st.markdown("## 📊 Platform Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "🎬 Movies",
                f"{len(movies):,}",
                help="Total number of movies in the database"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "⭐ Ratings",
                f"{len(ratings):,}",
                help="Total number of user ratings"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "👥 Users",
                f"{ratings['userId'].nunique():,}",
                help="Total number of active users"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            avg_rating = ratings['rating'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "🎯 Avg Rating",
                f"{avg_rating:.2f} ⭐",
                help="Average rating across all movies"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top rated movies preview
        if baseline_model is not None:
            st.markdown("## 🏆 Top Rated Movies")
            
            top_movies = baseline_model['movie_scores'].head(10)
            
            for idx, row in top_movies.iterrows():
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"🎭 Genres: {row['genres']}")
                
                with col2:
                    st.metric("⭐ Rating", f"{row['avg_rating']:.2f}")
                
                with col3:
                    st.metric("👥 Votes", f"{row['vote_count']:,}")
                
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: USER RECOMMENDATIONS
# ============================================================================

elif page == "👤 User Recommendations":
    st.markdown('<p class="main-header">👤 Personalized Movie Recommendations</p>',
                unsafe_allow_html=True)
    
    if movies is None or ratings is None or baseline_model is None:
        st.error("⚠️ Data or models not loaded. Please check your paths.")
    else:
        # User selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sample first 1000 users for demo
            available_users = sorted(ratings['userId'].unique()[:1000])
            user_id = st.selectbox(
                "🔍 Select User ID",
                options=available_users,
                help="Choose a user to see their personalized recommendations"
            )
        
        with col2:
            n_recommendations = st.slider(
                "📊 Number of Recommendations",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
        
        # Genre filter
        genres_list = ['All Genres'] + sorted(
            set(g for sublist in movies['genres'].str.split('|').dropna()
            for g in sublist if g != '(no genres listed)')
        )
        genre_filter = st.selectbox("🎭 Filter by Genre (optional)", genres_list)
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("🔄 Generating personalized recommendations..."):
                recommendations, user_ratings = get_user_recommendations(
                    user_id, n_recommendations, genre_filter
                )
                
                if recommendations is not None and user_ratings is not None:
                    # Display user profile
                    st.markdown("---")
                    st.markdown("### 📊 User Profile")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎬 Movies Rated", len(user_ratings))
                    
                    with col2:
                        st.metric("⭐ Avg Rating", f"{user_ratings['rating'].mean():.2f}")
                    
                    with col3:
                        st.metric("📊 Std Dev", f"{user_ratings['rating'].std():.2f}")
                    
                    with col4:
                        avg_user_rating = user_ratings['rating'].mean()
                        rating_bias = ("🎉 Generous" if avg_user_rating > 3.5 else 
                                     "😐 Moderate" if avg_user_rating > 2.5 else "🤔 Harsh")
                        st.metric("🎭 Rating Style", rating_bias)
                    
                    # Display recommendations
                    st.markdown("---")
                    st.markdown(f"### 🎬 Top {len(recommendations)} Recommendations for User {user_id}")
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{idx}. {row['title']}**")
                            st.caption(f"🎭 Genres: {row['genres']}")
                        
                        with col2:
                            st.metric("⭐ Score", f"{row['weighted_score']:.2f}")
                            st.caption(f"👥 {row['vote_count']:,} votes")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: MOVIE SEARCH & SIMILAR
# ============================================================================

elif page == "🔍 Movie Search & Similar":
    st.markdown('<p class="main-header">🔍 Movie Search & Similar Movies</p>',
                unsafe_allow_html=True)
    
    if movies is None:
        st.error("⚠️ Movies data not loaded.")
    else:
        # Search functionality
        search_query = st.text_input(
            "🔎 Search for a movie",
            placeholder="Enter movie title (e.g., 'Inception', 'Matrix')...",
            help="Search by movie title"
        )
        
        if search_query:
            # Search movies
            search_results = movies[
                movies['title'].str.contains(search_query, case=False, na=False)
            ].head(20)
            
            if len(search_results) > 0:
                st.markdown(f"### 🎬 Found {len(search_results)} movies matching '{search_query}'")
                
                for _, movie in search_results.iterrows():
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"🎭 Genres: {movie['genres']}")
                        if pd.notna(movie['release_year']):
                            st.caption(f"📅 Year: {int(movie['release_year'])}")
                    
                    with col2:
                        if baseline_model is not None:
                            movie_score = baseline_model['movie_scores'][
                                baseline_model['movie_scores']['movieId'] == movie['movieId']
                            ]
                            if len(movie_score) > 0:
                                st.metric("⭐ Score", f"{movie_score.iloc[0]['weighted_score']:.2f}")
                        
                        # Similar movies button
                        if st.button(f"🔍 Find Similar", key=f"similar_{movie['movieId']}"):
                            similar = search_similar_movies(movie['title'], n_similar=10)
                            
                            if similar is not None:
                                st.markdown(f"#### 🎯 Movies Similar to '{movie['title']}'")
                                
                                for idx, sim_movie in similar.iterrows():
                                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                                    
                                    col_a, col_b = st.columns([3, 1])
                                    
                                    with col_a:
                                        st.markdown(f"**{sim_movie['title']}**")
                                        st.caption(f"🎭 {sim_movie['genres']}")
                                    
                                    with col_b:
                                        st.metric("🎯 Similarity", f"{sim_movie['similarity']:.0%}")
                                        if pd.notna(sim_movie['weighted_score']):
                                            st.caption(f"⭐ {sim_movie['weighted_score']:.2f}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning(f"😕 No movies found matching '{search_query}'. Try a different search term.")
        
        # Browse by genre
        st.markdown("---")
        st.markdown("### 🎭 Browse by Genre")
        
        genres_list = sorted(
            set(g for sublist in movies['genres'].str.split('|').dropna()
            for g in sublist if g != '(no genres listed)')
        )
        
        selected_genre = st.selectbox("Select a genre", genres_list)
        
        if selected_genre:
            genre_movies = movies[
                movies['genres'].str.contains(selected_genre, na=False)
            ]
            
            if baseline_model is not None:
                genre_movies = genre_movies.merge(
                    baseline_model['movie_scores'][['movieId', 'weighted_score', 'avg_rating', 'vote_count']],
                    on='movieId',
                    how='left'
                ).sort_values('weighted_score', ascending=False).head(20)
            
            st.markdown(f"#### 🏆 Top 20 {selected_genre} Movies")
            
            for _, movie in genre_movies.iterrows():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{movie['title']}**")
                
                with col2:
                    if 'weighted_score' in movie and pd.notna(movie['weighted_score']):
                        st.metric("⭐ Score", f"{movie['weighted_score']:.2f}")
                        st.caption(f"👥 {int(movie['vote_count']):,} votes")
                
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: BUSINESS INSIGHTS
# ============================================================================

elif page == "📊 Business Insights":
    st.markdown('<p class="main-header">📊 Business Analytics Dashboard</p>',
                unsafe_allow_html=True)
    
    if genre_stats is not None and ratings is not None:
        # Genre performance visualization
        st.markdown("### 🎬 Genre Performance Analysis")
        
        tab1, tab2 = st.tabs(["📊 By Rating", "👥 By Popularity"])
        
        with tab1:
            fig = px.bar(
                genre_stats.sort_values('avg_rating', ascending=False).head(15),
                x='avg_rating',
                y='genre',
                orientation='h',
                title='Top 15 Genres by Average Rating',
                labels={'avg_rating': 'Average Rating', 'genre': 'Genre'},
                color='avg_rating',
                color_continuous_scale='RdYlGn',
                text='avg_rating'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig2 = px.bar(
                genre_stats.sort_values('n_ratings', ascending=False).head(15),
                x='n_ratings',
                y='genre',
                orientation='h',
                title='Top 15 Most Popular Genres (by Number of Ratings)',
                labels={'n_ratings': 'Number of Ratings', 'genre': 'Genre'},
                color='n_ratings',
                color_continuous_scale='Blues',
                text='n_ratings'
            )
            fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig2.update_layout(height=600)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Rating distribution
        st.markdown("---")
        st.markdown("### 📈 Rating Distribution")
        
        fig3 = px.histogram(
            ratings,
            x='rating',
            title='Distribution of All Ratings',
            labels={'rating': 'Rating', 'count': 'Frequency'},
            color_discrete_sequence=['#3498db']
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_genre = genre_stats.loc[genre_stats['avg_rating'].idxmax()]
            st.markdown(f"**🏆 Highest Rated Genre:**")
            st.markdown(f"### {best_genre['genre']}")
            st.markdown(f"⭐ Average Rating: **{best_genre['avg_rating']:.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            most_popular = genre_stats.loc[genre_stats['n_ratings'].idxmax()]
            st.markdown(f"**👥 Most Popular Genre:**")
            st.markdown(f"### {most_popular['genre']}")
            st.markdown(f"📊 Total Ratings: **{most_popular['n_ratings']:,}**")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">🤖 ML Model Performance</p>',
                unsafe_allow_html=True)
    
    # Load model comparison data
    comparison_path = os.path.join(REPORTS_PATH, 'model_comparison.csv')
    
    if os.path.exists(comparison_path):
        model_comparison = pd.read_csv(comparison_path)
        
        st.markdown("### 📊 Model Comparison Summary")
        
        # Display comparison table with highlighting
        st.dataframe(
            model_comparison.style.highlight_min(
                subset=['Test RMSE', 'Test MAE'],
                color='lightgreen'
            ).highlight_max(
                subset=['Test R²'],
                color='lightgreen'
            ).format({
                'Test RMSE': '{:.4f}',
                'Test MAE': '{:.4f}',
                'Test R²': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Best model highlight
        best_model_row = model_comparison.loc[model_comparison['Test RMSE'].idxmin()]
        
        st.markdown("---")
        st.markdown("### 🏆 Best Performing Model")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🤖 Model", best_model_row['Model'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📉 RMSE", f"{best_model_row['Test RMSE']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 MAE", f"{best_model_row['Test MAE']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 R²", f"{best_model_row['Test R²']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Performance Visualizations")
        
        tab1, tab2 = st.tabs(["RMSE Comparison", "R² Comparison"])
        
        with tab1:
            fig = px.bar(
                model_comparison.sort_values('Test RMSE'),
                x='Test RMSE',
                y='Model',
                orientation='h',
                title='Model Comparison by RMSE (Lower is Better)',
                labels={'Test RMSE': 'RMSE', 'Model': 'Model'},
                color='Test RMSE',
                color_continuous_scale='RdYlGn_r',
                text='Test RMSE'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig2 = px.bar(
                model_comparison.sort_values('Test R²', ascending=False),
                x='Test R²',
                y='Model',
                orientation='h',
                title='Model Comparison by R² (Higher is Better)',
                labels={'Test R²': 'R²', 'Model': 'Model'},
                color='Test R²',
                color_continuous_scale='RdYlGn',
                text='Test R²'
            )
            fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ Model comparison data not found. Please run the training pipeline first.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🎬 MovieLens Recommendation System</strong></p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
    <p>Data Science Project • December 2024</p>
    <p>⭐ Trained on 2M+ ratings from 247K+ users</p>
</div>
""", unsafe_allow_html=True)
