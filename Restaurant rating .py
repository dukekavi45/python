import pandas as pd
import matplotlib.pyplot as plt
import os

def recommend_restaurants(df, city=None, cuisine=None, min_rating=4.0, top_n=10):
    """
    Recommends restaurants based on specified criteria.

    Args:
        df (pd.DataFrame): The DataFrame containing restaurant data.
        city (str, optional): Filter by city. Defaults to None.
        cuisine (str, optional): Filter by cuisine. Defaults to None.
        min_rating (float, optional): Minimum aggregate rating. Defaults to 4.0.
        top_n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame of recommended restaurants.
    """
    print("\n--- Generating Restaurant Recommendations ---")
    filtered_df = df[df['Aggregate rating'] >= min_rating].copy()

    if city:
        filtered_df = filtered_df[filtered_df['City'].str.contains(city, case=False, na=False)]
        print(f"Filtering by City: {city}")

    if cuisine:
        filtered_df = filtered_df[filtered_df['Cuisines'].str.contains(cuisine, case=False, na=False)]
        print(f"Filtering by Cuisine: {cuisine}")

    if filtered_df.empty:
        print("No restaurants found matching your criteria. Try adjusting the filters.")
        return pd.DataFrame()

    
    recommended_restaurants = filtered_df.sort_values(
        by=['Aggregate rating', 'Votes'],
        ascending=[False, False]
    ).head(top_n)

    print(f"\nTop {len(recommended_restaurants)} Recommended Restaurants (Rating >= {min_rating}):")
    if not recommended_restaurants.empty:
        
        display_cols = [
            'Restaurant Name',
            'City',
            'Cuisines',
            'Aggregate rating',
            'Rating text',
            'Votes',
            'Average Cost for two',
            'Currency'
        ]
        
        display_cols = [col for col in display_cols if col in recommended_restaurants.columns]
        print(recommended_restaurants[display_cols].to_string(index=False))
    else:
        print("No recommendations to display.")

    return recommended_restaurants

def plot_rating_distribution(df):
    """
    Generates and displays a pie chart of restaurant rating colors.

    Args:
        df (pd.DataFrame): The DataFrame containing restaurant data.
    """
    print("\n--- Generating Rating Distribution Pie Chart ---")
    rating_counts = df['Rating color'].value_counts()

    if rating_counts.empty:
        print("No rating data to plot.")
        return

    plt.figure(figsize=(8, 8))
    plt.pie(
        rating_counts,
        labels=rating_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#9E9E9E'] # Example colors
    )
    plt.title('Distribution of Restaurant Rating Colors')
    plt.axis('equal') 
    plt.show()
    print("Pie chart displayed in a new window.")

def main():
    """Main function to run the restaurant recommendation project."""
    file_path = 'Dataset .csv' 

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'Dataset .csv' is in the same directory as the script.")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_path}' with {len(df)} entries.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    
    recommend_restaurants(df, top_n=5)

 
    recommend_restaurants(df, city='Istanbul', cuisine='Turkish', top_n=3)

    recommend_restaurants(df, city='New Delhi', top_n=5)
   
    recommend_restaurants(df,city='chennai',top_n=5)
    
    recommend_restaurants(df,city='coimbatore',top_n=5)
    
    recommend_restaurants(df,city='goa',top_n=5)
    
    
   
    
    
    

    plot_rating_distribution(df)

if __name__ == "__main__":
    main()
