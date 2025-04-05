# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:08:29 2025

@author: cavus
"""

import pandas as pd
import folium
from folium.plugins import MiniMap, HeatMap
import random

# Load datasets
west_midlands_data = pd.ExcelFile('C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\Survey_West_Midlands.xlsx')
newcastle_data = pd.ExcelFile('C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\Survey_Newcastle.xlsx')

# Parse sheets
west_midlands_df = west_midlands_data.parse('Form Responses 1')
newcastle_df = newcastle_data.parse('Form Responses 1')

# Strip column names
west_midlands_df.columns = west_midlands_df.columns.str.strip()
newcastle_df.columns = newcastle_df.columns.str.strip()

# Filter relevant data
west_midlands_filtered = west_midlands_df[['First three digit postcode', 'EV?']].rename(
    columns={'First three digit postcode': 'Postcode', 'EV?': 'Owns_EV'}
)
newcastle_filtered = newcastle_df[['First three digit postcode', 'EV?']].rename(
    columns={'First three digit postcode': 'Postcode', 'EV?': 'Owns_EV'}
)

# Define counts
newcastle_ev_count = 147
newcastle_non_ev_count = 11
west_midlands_ev_count = 154
west_midlands_non_ev_count = 10

# Filter EV and non-EV users for Newcastle
newcastle_ev = newcastle_filtered[newcastle_filtered['Owns_EV'] == 'Yes'].head(newcastle_ev_count)
newcastle_non_ev = newcastle_filtered[newcastle_filtered['Owns_EV'] == 'No'].head(newcastle_non_ev_count)

# Filter EV and non-EV users for West Midlands
west_midlands_ev = west_midlands_filtered[west_midlands_filtered['Owns_EV'] == 'Yes'].head(west_midlands_ev_count)
west_midlands_non_ev = west_midlands_filtered[west_midlands_filtered['Owns_EV'] == 'No'].head(west_midlands_non_ev_count)

# Combine data for mapping
newcastle_combined = pd.concat([newcastle_ev, newcastle_non_ev], ignore_index=True)
west_midlands_combined = pd.concat([west_midlands_ev, west_midlands_non_ev], ignore_index=True)

# Function to generate approximate coordinates for a postcode
def generate_coordinates(base_coords, count):
    lat, lon = base_coords
    return [
        (lat + random.uniform(-0.005, 0.005), lon + random.uniform(-0.005, 0.005))
        for _ in range(count)
    ]

# Define base coordinates for regions
postcode_base_coordinates = {
    'West Midlands': (52.4862, -1.8904),  # Approximate central coordinates for West Midlands
    'Newcastle': (54.9784, -1.6174)       # Approximate central coordinates for Newcastle
}

# Create map
m = folium.Map(location=[53.0, -1.5], zoom_start=6)
minimap = MiniMap()
m.add_child(minimap)

# Heatmap data
heatmap_data = []

# Add EV and non-EV markers with symbols
for region, coords in postcode_base_coordinates.items():
    data = west_midlands_combined if region == 'West Midlands' else newcastle_combined
    generated_coords = generate_coordinates(coords, len(data))

    for i, (_, row) in enumerate(data.iterrows()):
        if row['Owns_EV'] == 'Yes':
            icon = folium.Icon(color='green', icon='car', prefix='fa')
        else:
            icon = folium.Icon(color='red', icon='ban', prefix='fa')

        folium.Marker(
            location=generated_coords[i],
            icon=icon,
            popup=f"Postcode: {row['Postcode']}<br>Owns EV: {row['Owns_EV']}"
        ).add_to(m)

        # Add to heatmap data (all users included)
        heatmap_data.append(generated_coords[i])

# Add heatmap to the map
HeatMap(heatmap_data, radius=10, blur=15).add_to(m)

# Add draggable legend
legend_html = '''
<div id="legend" style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 100px; 
            background-color: white; z-index:9999; font-size:14px; 
            border:2px solid grey; padding: 10px; cursor: move;">
    <b>Legend</b><br>
    <i class="fa fa-car" style="font-size:14px; color:green;"></i> EV User<br>
    <i class="fa fa-ban" style="font-size:14px; color:red;"></i> Non-EV User<br>
</div>

<script>
    var legend = document.getElementById('legend');
    legend.onmousedown = function(event) {
        let shiftX = event.clientX - legend.getBoundingClientRect().left;
        let shiftY = event.clientY - legend.getBoundingClientRect().top;

        function moveAt(pageX, pageY) {
            legend.style.left = pageX - shiftX + 'px';
            legend.style.top = pageY - shiftY + 'px';
        }

        function onMouseMove(event) {
            moveAt(event.pageX, event.pageY);
        }

        document.addEventListener('mousemove', onMouseMove);

        legend.onmouseup = function() {
            document.removeEventListener('mousemove', onMouseMove);
            legend.onmouseup = null;
        };
    };

    legend.ondragstart = function() {
        return false;
    };
</script>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
map_path = 'C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\EV_Dots_Heatmap_Improved.html'
m.save(map_path)
print(f"Map saved at: {map_path}")
