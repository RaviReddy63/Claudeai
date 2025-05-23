import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import numpy as np

class PortfolioManager:
    def __init__(self, au_data, banker_data, customer_data):
        # Updated column definitions for each DataFrame
        self.au_data = pd.DataFrame(au_data, columns=[
            'AU_Number', 'Banker_Type', 'Latitude', 'Longitude'
        ])
        
        self.banker_data = pd.DataFrame(banker_data, columns=[
            'Banker_Firstname', 'Banker_Lastname', 'Portfolio_Code'
        ])
        
        self.customer_data = pd.DataFrame(customer_data, columns=[
            'Customer_ID', 'Latitude', 'Longitude', 'Portfolio_Code',
            'Bank_Revenue', 'Deposit_Balance', 'Total_Sales'
        ])
        
        self.output = widgets.Output()
        self.current_step = 1
        self.setup_widgets()
        self.initialize_portfolio_inputs()
        self.setup_observers()
        self.display_interface()

    def setup_widgets(self):
        self.au_dropdown = widgets.Dropdown(
            options=self.au_data['AU_Number'].tolist(),
            description='AU:',
            layout=widgets.Layout(width='300px')
        )
        
        self.range_slider = widgets.FloatSlider(
            min=1, max=20, step=0.5,
            value=1,
            description='Range (km):',
            layout=widgets.Layout(width='300px'),
            style={'description_width': 'initial'}
        )
        
        self.customer_type = widgets.RadioButtons(
            options=['All', 'Assigned', 'Unassigned'],
            value='All',
            description='Type:',
            layout=widgets.Layout(width='300px')
        )
        
        self.revenue_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Bank_Revenue'].max(),
            value=0,
            description='Min Revenue:',
            layout=widgets.Layout(width='300px')
        )
        
        self.deposit_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Deposit_Balance'].max(),
            value=0,
            description='Min Deposit:',
            layout=widgets.Layout(width='300px')
        )
        
        self.sales_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Total_Sales'].max(),
            value=0,
            description='Min Sales:',
            layout=widgets.Layout(width='300px')
        )

        self.next_button = widgets.Button(
            description='Next',
            layout=widgets.Layout(width='100px')
        )
        self.next_button.on_click(self.on_next_click)

        self.customer_stats = widgets.HTML()
        self.portfolio_table = widgets.HTML()

    def on_next_click(self, b):
        """Handle the next button click event"""
        if self.current_step < 5:
            self.current_step += 1
            self.refresh_display()

    def on_au_change(self, change):
        """Handle AU dropdown change"""
        if self.current_step > 1:
            self.current_step = 1
            self.reset_all_filters()
            self.refresh_display()

    def on_distance_change(self, change):
        """Handle distance slider change"""
        if self.current_step > 2:
            self.current_step = 2
            self.reset_filters_from_customer_type()
            self.refresh_display()

    def on_customer_type_change(self, change):
        """Handle customer type selection change"""
        if self.current_step > 3:
            self.current_step = 3
            self.reset_filters_from_min_values()
            self.refresh_display()

    def on_min_values_change(self, change):
        """Handle minimum value sliders change"""
        if self.current_step > 4:
            self.current_step = 4
            self.reset_portfolio()
            self.refresh_display()

    def reset_all_filters(self):
        """Reset all filters to initial values"""
        self.range_slider.value = self.range_slider.min
        self.reset_filters_from_customer_type()

    def reset_filters_from_customer_type(self):
        """Reset filters from customer type onwards"""
        self.customer_type.value = 'All'
        self.reset_filters_from_min_values()

    def reset_filters_from_min_values(self):
        """Reset minimum value filters"""
        self.revenue_slider.value = self.revenue_slider.min
        self.deposit_slider.value = self.deposit_slider.min
        self.sales_slider.value = self.sales_slider.min
        self.reset_portfolio()

    def reset_portfolio(self):
        """Reset portfolio selection"""
        for slider in self.portfolio_inputs.values():
            slider.value = 0

    def filter_customers(self):
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            return distance

        # Get selected AU coordinates
        selected_au = self.au_data[self.au_data['AU_Number'] == self.au_dropdown.value].iloc[0]
        au_lat, au_lon = selected_au['Latitude'], selected_au['Longitude']

        # Calculate distances for all customers
        distances = self.customer_data.apply(
            lambda row: haversine_distance(au_lat, au_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

        # Apply all filters
        filtered = self.customer_data[
            (distances <= self.range_slider.value) &
            (self.customer_data['Bank_Revenue'] >= self.revenue_slider.value) &
            (self.customer_data['Deposit_Balance'] >= self.deposit_slider.value) &
            (self.customer_data['Total_Sales'] >= self.sales_slider.value)
        ]

        # Apply customer type filter
        if self.customer_type.value == 'Assigned':
            filtered = filtered[filtered['Portfolio_Code'].notna()]
        elif self.customer_type.value == 'Unassigned':
            filtered = filtered[filtered['Portfolio_Code'].isna()]

        return filtered

    def update_customer_stats(self):
        """Update the customer statistics display"""
        filtered_customers = self.filter_customers()
        assigned = len(filtered_customers[filtered_customers['Portfolio_Code'].notna()])
        unassigned = len(filtered_customers[filtered_customers['Portfolio_Code'].isna()])
        
        self.customer_stats.value = f"""
        <div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>
            <h4 style='margin-top: 0;'>Customers in Range:</h4>
            <p>Assigned: {assigned}</p>
            <p>Unassigned: {unassigned}</p>
            <p>Total: {assigned + unassigned}</p>
        </div>
        """

    def setup_observers(self):
        """Set up widget observers"""
        self.au_dropdown.observe(self.on_au_change, names='value')
        self.range_slider.observe(self.on_distance_change, names='value')
        self.customer_type.observe(self.on_customer_type_change, names='value')
        self.revenue_slider.observe(self.on_min_values_change, names='value')
        self.deposit_slider.observe(self.on_min_values_change, names='value')
        self.sales_slider.observe(self.on_min_values_change, names='value')

    def initialize_portfolio_inputs(self):
        """Initialize portfolio selection widgets"""
        self.portfolio_inputs = {}
        portfolio_codes = self.banker_data['Portfolio_Code'].unique()
        for portfolio in portfolio_codes:
            if pd.notna(portfolio):
                self.portfolio_inputs[portfolio] = widgets.IntSlider(
                    min=0, max=100,
                    value=0,
                    description=f'Select:',
                    layout=widgets.Layout(width='200px')
                )

    def refresh_display(self):
        """Refresh the interface display based on current step"""
        widgets_to_show = [
            widgets.HTML('<h3>Select Administrative Unit</h3>'),
            self.au_dropdown
        ]

        if self.current_step >= 2:
            widgets_to_show.extend([
                widgets.HTML('<h3>Set Distance Range</h3>'),
                self.range_slider
            ])

        if self.current_step >= 3:
            self.update_customer_stats()
            widgets_to_show.extend([
                widgets.HTML('<h3>Customer Type</h3>'),
                self.customer_type,
                self.customer_stats
            ])

        if self.current_step >= 4:
            widgets_to_show.extend([
                widgets.HTML('<h3>Set Minimum Values</h3>'),
                self.revenue_slider,
                self.deposit_slider,
                self.sales_slider
            ])

        if self.current_step >= 5:
            portfolio_widgets = self.update_portfolio_table()
            widgets_to_show.extend([
                widgets.HTML('<h3>Portfolio Selection</h3>'),
                self.portfolio_table,
            ])
            for i, widget in enumerate(portfolio_widgets, 1):
                widget.description = f'Select {i}:'
            widgets_to_show.extend(portfolio_widgets)
            save_button = widgets.Button(
                description='Create Portfolio',
                layout=widgets.Layout(width='150px')
            )
            save_button.on_click(self.save_portfolio)
            widgets_to_show.append(save_button)
        else:
            widgets_to_show.append(self.next_button)

        with self.output:
            clear_output(wait=True)
            display(widgets.VBox(widgets_to_show))

    def display_interface(self):
        """Display the main interface"""
        display(self.output)
        self.refresh_display()

    def update_portfolio_table(self):
        """Update the portfolio table display"""
        filtered_customers = self.filter_customers()
        portfolio_stats = filtered_customers.groupby('Portfolio_Code').agg({
            'Customer_ID': 'count',
            'Bank_Revenue': 'sum',
            'Deposit_Balance': 'sum',
            'Total_Sales': 'sum'
        }).reset_index()
        
        portfolio_stats = portfolio_stats.merge(
            self.banker_data,
            on='Portfolio_Code',
            how='left'
        )
        
        table_html, portfolio_widgets = self.create_portfolio_table(portfolio_stats)
        self.portfolio_table.value = table_html
        return portfolio_widgets

    def create_portfolio_table(self, portfolio_stats):
        """Create the HTML table for portfolio display"""
        table_html = """
        <table style='width: 100%; border-collapse: collapse;'>
            <tr style='background-color: #e0e0e0;'>
                <th style='padding: 8px; text-align: left;'>Portfolio</th>
                <th style='padding: 8px; text-align: left;'>Banker</th>
                <th style='padding: 8px; text-align: right;'>Customers</th>
                <th style='padding: 8px; text-align: right;'>Revenue</th>
                <th style='padding: 8px; text-align: right;'>Deposits</th>
                <th style='padding: 8px; text-align: right;'>Sales</th>
                <th style='padding: 8px; text-align: center;'>Select</th>
            </tr>
        """
        
        portfolio_widgets = []
        for _, row in portfolio_stats.iterrows():
            portfolio = row['Portfolio_Code']
            if pd.isna(portfolio):
                continue
                
            banker_name = f"{row['Banker_Firstname']} {row['Banker_Lastname']}"
            self.portfolio_inputs[portfolio].max = int(row['Customer_ID'])
            portfolio_widgets.append(self.portfolio_inputs[portfolio])
                
            table_html += f"""
            <tr style='border-bottom: 1px solid #ddd;'>
                <td style='padding: 8px;'>{portfolio}</td>
                <td style='padding: 8px;'>{banker_name}</td>
                <td style='padding: 8px; text-align: right;'>{int(row['Customer_ID'])}</td>
                <td style='padding: 8px; text-align: right;'>${row['Bank_Revenue']:,.2f}</td>
                <td style='padding: 8px; text-align: right;'>${row['Deposit_Balance']:,.2f}</td>
                <td style='padding: 8px; text-align: right;'>${row['Total_Sales']:,.2f}</td>
                <td style='padding: 8px; text-align: center;'>{len(portfolio_widgets)}</td>
            </tr>
            """
        
        table_html += "</table>"
        return table_html, portfolio_widgets

    def save_portfolio(self, b):
        """Save the selected portfolio to a CSV file"""
        filtered_customers = self.filter_customers()
        selected_customers = pd.DataFrame()
        
        for portfolio, input_widget in self.portfolio_inputs.items():
            portfolio_customers = filtered_customers[
                filtered_customers['Portfolio_Code'] == portfolio
            ].head(input_widget.value)
            selected_customers = pd.concat([selected_customers, portfolio_customers])
        
        selected_customers.to_csv('new_portfolio.csv', index=False)
        with self.output:
            print("\nPortfolio Summary:")
            print(f"Total Customers: {len(selected_customers)}")
            print(f"Total Revenue: ${selected_customers['Bank_Revenue'].sum():,.2f}")
            print(f"Total Deposits: ${selected_customers['Deposit_Balance'].sum():,.2f}")
            print(f"Total Sales: ${selected_customers['Total_Sales'].sum():,.2f}")

au_data = [
    [1, 'Private', 40.7128, -74.0060],  # New York
    [2, 'Business', 41.8781, -87.6298],  # Chicago
    [3, 'Private', 34.0522, -118.2437]   # Los Angeles
]

banker_data = [
    ['John', 'Doe', 'P1'],
    ['Jane', 'Smith', 'P2'],
    ['Bob', 'Johnson', 'P3']
]

customer_data = [
    # CustomerID, Lat, Long, Portfolio, Revenue, Deposits, Sales
    ['C1', 40.7128, -74.0060, 'P1', 100000, 500000, 1000000],
    ['C2', 40.7300, -74.0200, None, 200000, 600000, 1200000],
    ['C3', 41.8781, -87.6298, 'P3', 150000, 550000, 1100000],
    ['C4', 40.7500, -74.0100, 'P2', 180000, 700000, 1300000],
    ['C5', 41.8900, -87.6200, None, 220000, 800000, 1400000],
    ['C6', 34.0522, -118.2437, 'P1', 250000, 900000, 1500000]
]

# Create an instance of PortfolioManager
portfolio_manager = PortfolioManager(au_data, banker_data, customer_data)
