import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyleaflet import Map, Marker, CircleMarker
import numpy as np
from geopy.distance import geodesic

class PortfolioManager:
    def __init__(self, au_data, banker_data, customer_data):
        self.step = 1
        self.au_data = pd.DataFrame(au_data, columns=['AU_Number', 'Latitude', 'Longitude'])
        self.banker_data = pd.DataFrame(banker_data, columns=['Banker_ID', 'Banker_AU', 'Portfolio_Code'])
        self.customer_data = pd.DataFrame(customer_data, columns=[
            'Customer_ID', 'Latitude', 'Longitude', 'Portfolio_Code', 
            'Bank_Revenue', 'Deposit_Balance', 'Gross_Sales'
        ])
        
        self.filtered_customers = None
        
        # Widgets
        self.au_dropdown = widgets.Dropdown(
            options=self.au_data['AU_Number'].tolist(),
            description='AU:'
        )
        
        self.range_slider = widgets.FloatSlider(
            min=1, max=20, step=0.5,
            description='Range (km):'
        )
        
        self.customer_type = widgets.RadioButtons(
            options=['All', 'Assigned', 'Unassigned'],
            description='Type:'
        )
        
        self.revenue_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Bank_Revenue'].max(),
            description='Min Revenue:'
        )
        
        self.deposit_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Deposit_Balance'].max(),
            description='Min Deposit:'
        )
        
        self.sales_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Gross_Sales'].max(),
            description='Min Sales:'
        )
        
        self.portfolio_inputs = {}
        self.customer_stats = widgets.HTML()
        self.portfolio_table = widgets.HTML()
        
        # Navigation
        self.next_button = widgets.Button(description='Next')
        self.prev_button = widgets.Button(description='Previous')
        self.finish_button = widgets.Button(description='Finish')
        
        self.next_button.on_click(self.next_step)
        self.prev_button.on_click(self.prev_step)
        self.finish_button.on_click(self.save_portfolio)
        
        self.display_current_step()

    def calculate_distance(self, au_lat, au_lon, customer_lat, customer_lon):
        return geodesic((au_lat, au_lon), (customer_lat, customer_lon)).kilometers

    def filter_customers(self):
        selected_au = self.au_dropdown.value
        au_loc = self.au_data[self.au_data['AU_Number'] == selected_au].iloc[0]
        
        self.customer_data['Distance'] = self.customer_data.apply(
            lambda row: self.calculate_distance(
                au_loc['Latitude'], au_loc['Longitude'],
                row['Latitude'], row['Longitude']
            ), axis=1
        )
        
        # Filter by range
        in_range = self.customer_data[self.customer_data['Distance'] <= self.range_slider.value]
        
        # Filter by customer type
        if self.customer_type.value == 'Assigned':
            in_range = in_range[in_range['Portfolio_Code'].notna()]
        elif self.customer_type.value == 'Unassigned':
            in_range = in_range[in_range['Portfolio_Code'].isna()]
        
        # Filter by metrics
        self.filtered_customers = in_range[
            (in_range['Bank_Revenue'] >= self.revenue_slider.value) &
            (in_range['Deposit_Balance'] >= self.deposit_slider.value) &
            (in_range['Gross_Sales'] >= self.sales_slider.value)
        ]

    def update_customer_stats(self):
        self.filter_customers()
        assigned = len(self.filtered_customers[self.filtered_customers['Portfolio_Code'].notna()])
        unassigned = len(self.filtered_customers[self.filtered_customers['Portfolio_Code'].isna()])
        
        stats_html = f"""
        <h4>Customers in Range:</h4>
        <p>Assigned: {assigned}</p>
        <p>Unassigned: {unassigned}</p>
        """
        self.customer_stats.value = stats_html

    def update_portfolio_table(self):
        self.filter_customers()
        portfolio_stats = self.filtered_customers.groupby('Portfolio_Code').agg({
            'Customer_ID': 'count',
            'Bank_Revenue': 'sum',
            'Deposit_Balance': 'sum',
            'Gross_Sales': 'sum'
        }).reset_index()
        
        portfolio_stats = portfolio_stats.merge(
            self.banker_data[['Portfolio_Code', 'Banker_ID']],
            on='Portfolio_Code',
            how='left'
        )
        
        table_html = "<table><tr><th>Portfolio</th><th>Banker</th><th>Customers</th><th>Select</th></tr>"
        for _, row in portfolio_stats.iterrows():
            portfolio = row['Portfolio_Code']
            if pd.isna(portfolio):
                continue
                
            input_id = f"input_{portfolio}"
            self.portfolio_inputs[portfolio] = widgets.IntSlider(
                min=0, max=int(row['Customer_ID']),
                description=f'Portfolio {portfolio}:'
            )
            
            table_html += f"""
            <tr>
                <td>{portfolio}</td>
                <td>{row['Banker_ID']}</td>
                <td>{int(row['Customer_ID'])}</td>
                <td>{self.portfolio_inputs[portfolio]}</td>
            </tr>
            """
        table_html += "</table>"
        
        self.portfolio_table.value = table_html

    def save_portfolio(self, b):
        selected_customers = pd.DataFrame()
        
        for portfolio, input_widget in self.portfolio_inputs.items():
            portfolio_customers = self.filtered_customers[
                self.filtered_customers['Portfolio_Code'] == portfolio
            ].head(input_widget.value)
            selected_customers = pd.concat([selected_customers, portfolio_customers])
        
        if self.customer_type.value in ['All', 'Unassigned']:
            unassigned = self.filtered_customers[self.filtered_customers['Portfolio_Code'].isna()]
            selected_customers = pd.concat([selected_customers, unassigned])
        
        selected_customers.to_csv('new_portfolio.csv', index=False)
        print("Portfolio saved to 'new_portfolio.csv'")

    def next_step(self, b):
        if self.step < 6:
            self.step += 1
            self.display_current_step()

    def prev_step(self, b):
        if self.step > 1:
            self.step -= 1
            self.display_current_step()

    def display_current_step(self):
        clear_output(wait=True)
        
        steps = {
            1: widgets.VBox([
                widgets.HTML('<h3>Select AU</h3>'),
                self.au_dropdown,
                self.next_button
            ]),
            2: widgets.VBox([
                widgets.HTML('<h3>Set Range</h3>'),
                self.range_slider,
                widgets.HBox([self.prev_button, self.next_button])
            ]),
            3: widgets.VBox([
                widgets.HTML('<h3>Customer Type & Stats</h3>'),
                self.customer_type,
                self.customer_stats,
                widgets.HBox([self.prev_button, self.next_button])
            ]),
            4: widgets.VBox([
                widgets.HTML('<h3>Set Minimum Values</h3>'),
                self.revenue_slider,
                self.deposit_slider,
                self.sales_slider,
                widgets.HBox([self.prev_button, self.next_button])
            ]),
            5: widgets.VBox([
                widgets.HTML('<h3>Portfolio Selection</h3>'),
                self.portfolio_table,
                widgets.HBox([self.prev_button, self.next_button])
            ]),
            6: widgets.VBox([
                widgets.HTML('<h3>Save Portfolio</h3>'),
                self.finish_button,
                self.prev_button
            ])
        }
        
        if self.step == 3:
            self.update_customer_stats()
        elif self.step == 5:
            self.update_portfolio_table()
            
        display(steps[self.step])

# Usage example:
au_data = [
    [1, 40.7128, -74.0060],
    [2, 41.8781, -87.6298]
]

banker_data = [
    ['B1', 1, 'P1'],
    ['B2', 1, 'P2'],
    ['B3', 2, 'P3']
]

customer_data = [
    ['C1', 40.7128, -74.0060, 'P1', 1000, 5000, 10000],
    ['C2', 40.7128, -74.0060, None, 2000, 6000, 12000]
]

portfolio_manager = PortfolioManager(au_data, banker_data, customer_data)
