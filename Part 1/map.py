# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:08:29 2025

@author: cavus
"""

import pandas as pd
import folium
from folium.plugins import MiniMap, HeatMap
import random

# Load the provided datasets
west_midlands_data = pd.ExcelFile('C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\Survey_West_Midlands.xlsx')
newcastle_data = pd.ExcelFile('C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\Survey_Newcastle.xlsx')

# Parse sheets
west_midlands_df = west_midlands_data.parse('Form Responses 1')
newcastle_df = newcastle_data.parse('Form Responses 1')

# Strip column names
west_midlands_df.columns = west_midlands_df.columns.str.strip()
newcastle_df.columns = newcastle_df.columns.str.strip()

# Filter relevant data
west_midlands_filtered = west_midlands_df[[
    'First three digit postcode', 'Age', 'EV?', 'Gender', 'Ethnic group'
]].rename(
    columns={
        'First three digit postcode': 'Postcode', 
        'Age': 'Age_Group',
        'EV?': 'Owns_EV',
        'Gender': 'Gender',
        'Ethnic group': 'Ethnic_Group'
    }
)
newcastle_filtered = newcastle_df[[
    'First three digit postcode', 'Age', 'EV?', 'Gender', 'Ethnic group'
]].rename(
    columns={
        'First three digit postcode': 'Postcode', 
        'Age': 'Age_Group',
        'EV?': 'Owns_EV',
        'Gender': 'Gender',
        'Ethnic group': 'Ethnic_Group'
    }
)

# Combine data
combined_data = pd.concat([west_midlands_filtered, newcastle_filtered], ignore_index=True)

# Function to generate approximate coordinates for a postcode
def generate_coordinates(base_coords, count):
    lat, lon = base_coords
    return [
        (lat + random.uniform(-0.005, 0.005), lon + random.uniform(-0.005, 0.005))
        for _ in range(count)
    ]

# Define base coordinates for regions (representative values)
postcode_base_coordinates = {
    'West Midlands': (52.4862, -1.8904),  # Approximate central coordinates for West Midlands
    'Newcastle': (54.9784, -1.6174)       # Approximate central coordinates for Newcastle
}

# Ensure Ethnic_Group is consistent and stripped of extra spaces
combined_data['Ethnic_Group'] = combined_data['Ethnic_Group'].str.strip()

# Updated ethnic_icons dictionary using FontAwesome icons
ethnic_icons = {
    'White': 'circle',
    'Black / African / Caribbean / Black British': 'bolt',  # Example icon for representation
    'Asian / Asian British': 'adjust',  # Example icon for representation
    'Mixed': 'star',  # Already compatible
    'Other': 'question-circle',  # Example icon for undefined groups
}

# Handle undefined ethnic groups
def get_ethnic_icon(ethnic_group):
    return ethnic_icons.get(ethnic_group, 'circle')  # Default to 'circle' if undefined

# Create map
m = folium.Map(location=[53.0, -1.5], zoom_start=6)
minimap = MiniMap()
m.add_child(minimap)

# Add markers for each region
def add_markers_region(data, base_coords, total_users):
    region_data = data.head(total_users)
    coords = generate_coordinates(base_coords, total_users)
    for i, (_, row) in enumerate(region_data.iterrows()):
        age_group = row['Age_Group']
        ethnic_group = row['Ethnic_Group']

        # Determine marker color based on age
        if age_group == 'Under 18':
            color = 'yellow'
        elif age_group == '18-24':
            color = 'blue'
        elif age_group == '25-34':
            color = 'green'
        elif age_group == '35-44':
            color = 'purple'
        elif age_group == '45-54':
            color = 'orange'
        elif age_group == '55-64':
            color = 'red'
        elif age_group == '65+':
            color = 'brown'
        else:
            color = 'gray'

        # Determine marker icon based on ethnic group
        icon = get_ethnic_icon(ethnic_group)

        folium.Marker(
            location=coords[i],
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            popup=f"Postcode: {row['Postcode']}<br>Age: {row['Age_Group']}<br>Owns EV: {row['Owns_EV']}<br>Gender: {row['Gender']}<br>Ethnic Group: {row['Ethnic_Group']}"
        ).add_to(m)

# Add markers for each region
add_markers_region(west_midlands_filtered, postcode_base_coordinates['West Midlands'], len(west_midlands_filtered))
add_markers_region(newcastle_filtered, postcode_base_coordinates['Newcastle'], len(newcastle_filtered))

# Add heatmap for EV density
ev_users_coords = generate_coordinates(postcode_base_coordinates['West Midlands'], len(west_midlands_filtered[west_midlands_filtered['Owns_EV'] == 'Yes'])) + generate_coordinates(postcode_base_coordinates['Newcastle'], len(newcastle_filtered[newcastle_filtered['Owns_EV'] == 'Yes']))
HeatMap(ev_users_coords).add_to(m)

# Add draggable legend with "Age" as the title
legend_html = '''
<div id="legend" style="position: fixed; 
            bottom: 50px; left: 50px; width: 400px; height: 175px; 
            background-color: white; z-index:9999; font-size:14px; 
            border:2px solid grey; padding: 10px; cursor: move;">
    <div style="float: left; width: 50%; border-right: 1px solid grey; padding-right: 10px;">
        <b>Age</b><br>
        <i style="background:yellow; color:yellow; font-size:14px;">&emsp;</i> Under 18<br>
        <i style="background:blue; color:blue; font-size:14px;">&emsp;</i> 18-24<br>
        <i style="background:green; color:green; font-size:14px;">&emsp;</i> 25-34<br>
        <i style="background:purple; color:purple; font-size:14px;">&emsp;</i> 35-44<br>
        <i style="background:orange; color:orange; font-size:14px;">&emsp;</i> 45-54<br>
        <i style="background:red; color:red; font-size:14px;">&emsp;</i> 55-64<br>
        <i style="background:brown; color:brown; font-size:14px;">&emsp;</i> 65+<br>
    </div>
    <div style="float: left; width: 50%; padding-left: 10px;">
        <b>Ethnic Group</b><br>
        <i class="fa fa-circle" style="font-size:14px; color:black;">&emsp;</i> White<br>
        <i class="fa fa-bolt" style="font-size:14px; color:black;">&emsp;</i> African / Black British<br>
        <i class="fa fa-adjust" style="font-size:14px; color:black;">&emsp;</i> Asian / Asian British<br>
        <i class="fa fa-star" style="font-size:14px; color:black;">&emsp;</i> Mixed<br>
        <i class="fa fa-question-circle" style="font-size:14px; color:black;">&emsp;</i> Other<br>
    </div>
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
