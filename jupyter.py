import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import folium
from ipyleaflet import Map, Marker, CircleMarker

class PortfolioManager:
    def __init__(self):
        self.step = 1
        # Mock data - replace with your actual data
        self.aus = pd.DataFrame({
            'AU_ID': ['AU1', 'AU2', 'AU3'],
            'lat': [40.7128, 41.8781, 34.0522],
            'lon': [-74.0060, -87.6298, -118.2437]
        })
        
        # Create widgets
        self.au_dropdown = widgets.Dropdown(
            options=self.aus['AU_ID'].tolist(),
            description='Select AU:'
        )
        
        self.range_slider = widgets.FloatSlider(
            min=1, max=20, step=1,
            description='Range (km):'
        )
        
        self.customer_type = widgets.RadioButtons(
            options=['All', 'Assigned', 'Unassigned'],
            description='Customer Type:'
        )
        
        self.create_button = widgets.Button(description='Create Portfolio')
        self.next_button = widgets.Button(description='Next')
        self.prev_button = widgets.Button(description='Previous')
        
        # Set up widget callbacks
        self.next_button.on_click(self.next_step)
        self.prev_button.on_click(self.prev_step)
        self.create_button.on_click(self.create_portfolio)
        
        # Display initial step
        self.display_current_step()
    
    def next_step(self, b):
        if self.step < 5:
            self.step += 1
            self.display_current_step()
    
    def prev_step(self, b):
        if self.step > 1:
            self.step -= 1
            self.display_current_step()
    
    def create_portfolio(self, b):
        # Implement portfolio creation logic
        self.display_map()
    
    def display_map(self):
        # Get selected AU coordinates
        au_data = self.aus[self.aus['AU_ID'] == self.au_dropdown.value].iloc[0]
        
        # Create map centered on selected AU
        m = Map(center=(au_data['lat'], au_data['lon']), zoom=10)
        
        # Add AU marker
        marker = Marker(location=(au_data['lat'], au_data['lon']))
        m.add_layer(marker)
        
        # Add circle for range
        circle = CircleMarker(
            location=(au_data['lat'], au_data['lon']),
            radius=self.range_slider.value * 1000,  # Convert km to meters
            color='red',
            fill_color='red'
        )
        m.add_layer(circle)
        
        display(m)
    
    def display_current_step(self):
        clear_output(wait=True)
        
        if self.step == 1:
            display(widgets.VBox([
                widgets.HTML('<h3>Step 1: Select Administrative Unit</h3>'),
                self.au_dropdown,
                self.next_button
            ]))
        
        elif self.step == 2:
            display(widgets.VBox([
                widgets.HTML('<h3>Step 2: Select Range</h3>'),
                self.range_slider,
                widgets.HBox([self.prev_button, self.next_button])
            ]))
        
        elif self.step == 3:
            display(widgets.VBox([
                widgets.HTML('<h3>Step 3: Select Customer Type</h3>'),
                self.customer_type,
                widgets.HBox([self.prev_button, self.next_button])
            ]))
        
        elif self.step == 4:
            display(widgets.VBox([
                widgets.HTML('<h3>Step 4: Portfolio Allocation</h3>'),
                widgets.HTML('Mock portfolio data...'),
                widgets.HBox([self.prev_button, self.next_button])
            ]))
        
        elif self.step == 5:
            display(widgets.VBox([
                widgets.HTML('<h3>Step 5: Create Portfolio</h3>'),
                self.create_button,
                self.prev_button
            ]))

# Usage
portfolio_manager = PortfolioManager()
