import pandas as pd
from geopy.distance import geodesic
import folium

# Load the CSV data
data = pd.read_csv('cleaned_parking_violations.csv')

# Function to find violations within a 500-meter radius
def find_violations_within_radius(lat, lon, radius=500):
    selected_point = (lat, lon)
    data['distance'] = data.apply(lambda row: geodesic(selected_point, (row['LATITUDE'], row['LONGITUDE'])).meters, axis=1)
    within_radius = data[data['distance'] <= radius]
    return within_radius

# Example usage
selected_lat = 38.89
selected_lon = -77.004
violations_within_500m = find_violations_within_radius(selected_lat, selected_lon)

# Create a map centered at the selected location
map_center = [selected_lat, selected_lon]
m = folium.Map(location=map_center, zoom_start=15)

# Add a circle for the radius
folium.Circle(
    radius=500,
    location=map_center,
    popup='500m radius',
    color='crimson',
    fill=True,
).add_to(m)

# Add markers for each violation within the radius
for idx, row in violations_within_500m.iterrows():
    folium.Marker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        popup=(
            f"<b>Violation:</b> {row['VIOLATION_PROC_DESC']}<br>"
            f"<b>Issue Date:</b> {row['ISSUE_DATE']}<br>"
            f"<b>Fine Amount:</b> ${row['FINE_AMOUNT']}"
        ),
    ).add_to(m)

# Save the map as an HTML file
m.save('violations_map.html')
