"""
Aadhaar Sentinel - Maps Module
===============================
Folium-based geospatial visualization for intervention mapping.
"""
import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import STATE_COORDINATES, MAP_CONFIG, COLORS, ACTIONS


def get_state_coordinates(state: str) -> tuple:
    """
    Get coordinates for a state.
    
    Args:
        state: State name
        
    Returns:
        Tuple of (latitude, longitude)
    """
    # Try exact match
    if state in STATE_COORDINATES:
        return STATE_COORDINATES[state]
    
    # Try case-insensitive match
    state_lower = state.lower()
    for key, coords in STATE_COORDINATES.items():
        if key.lower() == state_lower:
            return coords
    
    # Default to India center with random offset
    return (
        MAP_CONFIG['center'][0] + np.random.uniform(-5, 5),
        MAP_CONFIG['center'][1] + np.random.uniform(-5, 5)
    )


def get_district_coordinates(state: str, district: str, index: int = 0) -> tuple:
    """
    Get approximate coordinates for a district.
    Uses state center with offset based on district index.
    
    Args:
        state: State name
        district: District name
        index: Index for offset calculation
        
    Returns:
        Tuple of (latitude, longitude)
    """
    state_coords = get_state_coordinates(state)
    
    # Add small offset for districts within a state
    # Use a spiral pattern for distribution
    angle = index * 0.5
    radius = 0.3 + (index * 0.1)
    
    lat_offset = radius * np.cos(angle)
    lon_offset = radius * np.sin(angle)
    
    return (
        state_coords[0] + lat_offset,
        state_coords[1] + lon_offset
    )


def get_marker_color(action: str) -> str:
    """
    Get marker color based on strategic action.
    
    Args:
        action: Strategic action string
        
    Returns:
        Color name for Folium marker
    """
    color_map = {
        ACTIONS['update_center']: 'red',
        ACTIONS['mobile_camp']: 'orange',
        ACTIONS['school_camp']: 'beige',
        ACTIONS['migration_hub']: 'blue',
        ACTIONS['stable']: 'green'
    }
    return color_map.get(action, 'gray')


def get_marker_icon(action: str) -> str:
    """
    Get Font Awesome icon based on strategic action.
    
    Args:
        action: Strategic action string
        
    Returns:
        Icon name
    """
    icon_map = {
        ACTIONS['update_center']: 'refresh',
        ACTIONS['mobile_camp']: 'truck',
        ACTIONS['school_camp']: 'graduation-cap',
        ACTIONS['migration_hub']: 'plane',
        ACTIONS['stable']: 'check'
    }
    return icon_map.get(action, 'info')


def create_popup_html(row: pd.Series) -> str:
    """
    Create HTML content for marker popup.
    
    Args:
        row: DataFrame row with district data
        
    Returns:
        HTML string for popup
    """
    action_color = {
        ACTIONS['update_center']: '#E74C3C',
        ACTIONS['mobile_camp']: '#F39C12',
        ACTIONS['school_camp']: '#F1C40F',
        ACTIONS['migration_hub']: '#3498DB',
        ACTIONS['stable']: '#27AE60'
    }
    
    color = action_color.get(row.get('Strategic_Action', ''), '#7F8C8D')
    
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 250px; padding: 10px;">
        <h4 style="margin: 0 0 10px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;">
            {row.get('district', 'Unknown')}
        </h4>
        <p style="margin: 5px 0; color: #666;">
            <strong>State:</strong> {row.get('state', 'Unknown')}
        </p>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
        <table style="width: 100%; font-size: 13px;">
            <tr>
                <td style="color: #666;">Total Enrollments</td>
                <td style="text-align: right; font-weight: bold; color: #2E86AB;">
                    {row.get('Total_Enrollment', 0):,.0f}
                </td>
            </tr>
            <tr>
                <td style="color: #666;">Total Updates</td>
                <td style="text-align: right; font-weight: bold; color: #8E44AD;">
                    {row.get('Total_Updates', 0):,.0f}
                </td>
            </tr>
            <tr>
                <td style="color: #666;">Migration Index</td>
                <td style="text-align: right; font-weight: bold; color: #E67E22;">
                    {row.get('Migration_Index', 0):.2f}
                </td>
            </tr>
            <tr>
                <td style="color: #666;">Youth Ratio</td>
                <td style="text-align: right; font-weight: bold; color: #27AE60;">
                    {row.get('Youth_Ratio', 0)*100:.1f}%
                </td>
            </tr>
        </table>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">
        <div style="background: {color}; color: white; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
            {row.get('Strategic_Action', 'Unknown')}
        </div>
    </div>
    """
    return html


def create_intervention_map(
    district_df: pd.DataFrame,
    show_only_action_needed: bool = False
) -> folium.Map:
    """
    Create an intervention map with district markers.
    
    Args:
        district_df: DataFrame with district analysis
        show_only_action_needed: If True, only show districts needing action
        
    Returns:
        Folium Map object
    """
    # Create base map
    m = folium.Map(
        location=MAP_CONFIG['center'],
        zoom_start=MAP_CONFIG['zoom'],
        tiles=MAP_CONFIG['tiles']
    )
    
    # Filter if needed
    if show_only_action_needed:
        district_df = district_df[district_df['Action_Needed'] == True]
    
    # Create marker cluster for better performance
    marker_cluster = plugins.MarkerCluster(
        name='Districts',
        overlay=True,
        control=True
    )
    
    # Add markers for each district
    for idx, row in district_df.iterrows():
        # Get coordinates
        lat, lon = get_district_coordinates(
            row['state'],
            row['district'],
            idx
        )
        
        # Get marker properties
        color = get_marker_color(row.get('Strategic_Action', ''))
        icon = get_marker_icon(row.get('Strategic_Action', ''))
        
        # Create marker
        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                create_popup_html(row),
                max_width=300
            ),
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix='fa'
            ),
            tooltip=f"{row['district']}, {row['state']}"
        )
        
        marker.add_to(marker_cluster)
    
    # Add cluster to map
    marker_cluster.add_to(m)
    
    # Add legend
    add_legend(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def create_state_markers(
    state_df: pd.DataFrame
) -> folium.Map:
    """
    Create a map with state-level markers.
    
    Args:
        state_df: DataFrame with state summaries
        
    Returns:
        Folium Map object
    """
    m = folium.Map(
        location=MAP_CONFIG['center'],
        zoom_start=MAP_CONFIG['zoom'],
        tiles=MAP_CONFIG['tiles']
    )
    
    # Calculate size based on enrollment
    max_enrollment = state_df['Total_Enrollment'].max()
    
    for _, row in state_df.iterrows():
        state = row['State']
        if state not in STATE_COORDINATES:
            continue
        
        lat, lon = STATE_COORDINATES[state]
        
        # Size based on enrollment (10-50 range)
        size = 10 + (row['Total_Enrollment'] / max_enrollment) * 40
        
        # Color based on migration index
        if row['Migration_Index'] > 2.0:
            color = '#E74C3C'
        elif row['Migration_Index'] > 1.0:
            color = '#F39C12'
        else:
            color = '#27AE60'
        
        # Create circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=size,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=folium.Popup(
                f"""
                <div style="font-family: Arial; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0;">{state}</h4>
                    <p><strong>Enrollments:</strong> {row['Total_Enrollment']:,.0f}</p>
                    <p><strong>Updates:</strong> {row['Total_Updates']:,.0f}</p>
                    <p><strong>Districts:</strong> {row['Districts']}</p>
                    <p><strong>Migration Index:</strong> {row['Migration_Index']:.2f}</p>
                </div>
                """,
                max_width=250
            ),
            tooltip=state
        ).add_to(m)
    
    return m


def add_legend(m: folium.Map) -> None:
    """
    Add a legend to the map.
    
    Args:
        m: Folium Map object
    """
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: rgba(14, 17, 23, 0.95); padding: 15px; border-radius: 8px;
                border: 1px solid #2E86AB; font-family: 'Segoe UI', Arial, sans-serif;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 12px 0; color: #ECF0F1; font-size: 14px; 
                   border-bottom: 1px solid #2E86AB; padding-bottom: 8px;">
            🗺️ Intervention Legend
        </h4>
        <div style="font-size: 12px; color: #BDC3C7;">
            <p style="margin: 6px 0;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                             background: #E74C3C; border-radius: 50%; margin-right: 8px;"></span>
                Update Center Required
            </p>
            <p style="margin: 6px 0;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                             background: #F39C12; border-radius: 50%; margin-right: 8px;"></span>
                Mobile Camp Needed
            </p>
            <p style="margin: 6px 0;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                             background: #F1C40F; border-radius: 50%; margin-right: 8px;"></span>
                School Camp Needed
            </p>
            <p style="margin: 6px 0;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                             background: #3498DB; border-radius: 50%; margin-right: 8px;"></span>
                Migration Hub
            </p>
            <p style="margin: 6px 0;">
                <span style="display: inline-block; width: 12px; height: 12px; 
                             background: #27AE60; border-radius: 50%; margin-right: 8px;"></span>
                Operations Normal
            </p>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def create_heatmap(
    df: pd.DataFrame,
    value_column: str = 'Total_Enrollment'
) -> folium.Map:
    """
    Create a heatmap visualization.
    
    Args:
        df: DataFrame with state/district data
        value_column: Column to use for heat intensity
        
    Returns:
        Folium Map with heatmap layer
    """
    m = folium.Map(
        location=MAP_CONFIG['center'],
        zoom_start=MAP_CONFIG['zoom'],
        tiles=MAP_CONFIG['tiles']
    )
    
    # Prepare heatmap data
    heat_data = []
    
    for idx, row in df.iterrows():
        lat, lon = get_district_coordinates(
            row['state'],
            row['district'],
            idx
        )
        intensity = row[value_column]
        heat_data.append([lat, lon, intensity])
    
    # Normalize intensities
    max_val = max(d[2] for d in heat_data) if heat_data else 1
    heat_data = [[d[0], d[1], d[2]/max_val] for d in heat_data]
    
    # Add heatmap
    plugins.HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=25,
        blur=15,
        gradient={
            0.4: 'blue',
            0.6: 'lime',
            0.8: 'orange',
            1.0: 'red'
        }
    ).add_to(m)
    
    return m
