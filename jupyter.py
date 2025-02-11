import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

class PortfolioManager:
    def __init__(self, au_data, banker_data, customer_data):
        self.au_data = pd.DataFrame(au_data, columns=['AU_Number', 'Latitude', 'Longitude'])
        self.banker_data = pd.DataFrame(banker_data, columns=['Banker_ID', 'Banker_AU', 'Portfolio_Code'])
        self.customer_data = pd.DataFrame(customer_data, columns=[
            'Customer_ID', 'AU_Number', 'Latitude', 'Longitude', 'Portfolio_Code', 
            'Bank_Revenue', 'Deposit_Balance', 'Gross_Sales'
        ])
        
        self.output = widgets.Output()
        self.setup_widgets()
        self.initialize_portfolio_inputs()
        self.display_interface()

    def setup_widgets(self):
        self.au_dropdown = widgets.Dropdown(
            options=self.au_data['AU_Number'].tolist(),
            description='AU:',
            layout=widgets.Layout(width='300px')
        )
        
        self.range_slider = widgets.FloatSlider(
            min=1, max=20, step=0.5,
            description='Range (km):',
            layout=widgets.Layout(width='300px'),
            style={'description_width': 'initial'}
        )
        
        self.customer_type = widgets.RadioButtons(
            options=['All', 'Assigned', 'Unassigned'],
            description='Type:',
            layout=widgets.Layout(width='300px')
        )
        
        self.revenue_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Bank_Revenue'].max(),
            description='Min Revenue:',
            layout=widgets.Layout(width='300px')
        )
        
        self.deposit_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Deposit_Balance'].max(),
            description='Min Deposit:',
            layout=widgets.Layout(width='300px')
        )
        
        self.sales_slider = widgets.FloatSlider(
            min=0, max=self.customer_data['Gross_Sales'].max(),
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

    def initialize_portfolio_inputs(self):
        self.portfolio_inputs = {}
        portfolio_codes = self.banker_data['Portfolio_Code'].unique()
        for portfolio in portfolio_codes:
            if pd.notna(portfolio):
                self.portfolio_inputs[portfolio] = widgets.IntSlider(
                    min=0, max=100,
                    description=f'Count:',
                    layout=widgets.Layout(width='200px')
                )

    def display_interface(self):
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML('<h3>Select Administrative Unit</h3>'),
                self.au_dropdown,
                self.next_button
            ]))
        display(self.output)

    def on_next_click(self, b):
        if not hasattr(self, 'current_step'):
            self.current_step = 1
        
        if self.current_step == 1 and self.au_dropdown.value:
            self.current_step = 2
            self.show_distance_section()
        elif self.current_step == 2:
            self.current_step = 3
            self.show_customer_section()
        elif self.current_step == 3:
            self.current_step = 4
            self.show_min_values_section()
        elif self.current_step == 4:
            self.current_step = 5
            self.show_portfolio_section()

    def show_distance_section(self):
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML('<h3>Select Administrative Unit</h3>'),
                self.au_dropdown,
                widgets.HTML('<h3>Set Distance Range</h3>'),
                self.range_slider,
                self.next_button
            ]))

    def show_customer_section(self):
        self.update_customer_stats()
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML('<h3>Select Administrative Unit</h3>'),
                self.au_dropdown,
                widgets.HTML('<h3>Set Distance Range</h3>'),
                self.range_slider,
                widgets.HTML('<h3>Customer Type</h3>'),
                self.customer_type,
                self.customer_stats,
                self.next_button
            ]))

    def show_min_values_section(self):
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML('<h3>Select Administrative Unit</h3>'),
                self.au_dropdown,
                widgets.HTML('<h3>Set Distance Range</h3>'),
                self.range_slider,
                widgets.HTML('<h3>Customer Type</h3>'),
                self.customer_type,
                self.customer_stats,
                widgets.HTML('<h3>Set Minimum Values</h3>'),
                self.revenue_slider,
                self.deposit_slider,
                self.sales_slider,
                self.next_button
            ]))

    def show_portfolio_section(self):
        self.update_portfolio_table()
        finish_button = widgets.Button(description='Create Portfolio')
        finish_button.on_click(self.save_portfolio)
        
        with self.output:
            clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML('<h3>Select Administrative Unit</h3>'),
                self.au_dropdown,
                widgets.HTML('<h3>Set Distance Range</h3>'),
                self.range_slider,
                widgets.HTML('<h3>Customer Type</h3>'),
                self.customer_type,
                self.customer_stats,
                widgets.HTML('<h3>Set Minimum Values</h3>'),
                self.revenue_slider,
                self.deposit_slider,
                self.sales_slider,
                widgets.HTML('<h3>Portfolio Selection</h3>'),
                self.portfolio_table,
                finish_button
            ]))

    def update_customer_stats(self):
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

    def filter_customers(self):
        return self.customer_data[
            (self.customer_data['AU_Number'] == self.au_dropdown.value) &
            (self.customer_data['Bank_Revenue'] >= self.revenue_slider.value) &
            (self.customer_data['Deposit_Balance'] >= self.deposit_slider.value) &
            (self.customer_data['Gross_Sales'] >= self.sales_slider.value)
        ]

    def update_portfolio_table(self):
        filtered_customers = self.filter_customers()
        portfolio_stats = filtered_customers.groupby('Portfolio_Code').agg({
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
        
        self.portfolio_table.value = self.create_portfolio_table(portfolio_stats)

    def create_portfolio_table(self, portfolio_stats):
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
        
        for _, row in portfolio_stats.iterrows():
            portfolio = row['Portfolio_Code']
            if pd.isna(portfolio):
                continue
                
            table_html += f"""
            <tr style='border-bottom: 1px solid #ddd;'>
                <td style='padding: 8px;'>{portfolio}</td>
                <td style='padding: 8px;'>{row['Banker_ID']}</td>
                <td style='padding: 8px; text-align: right;'>{int(row['Customer_ID'])}</td>
                <td style='padding: 8px; text-align: right;'>${row['Bank_Revenue']:,.2f}</td>
                <td style='padding: 8px; text-align: right;'>${row['Deposit_Balance']:,.2f}</td>
                <td style='padding: 8px; text-align: right;'>${row['Gross_Sales']:,.2f}</td>
                <td style='padding: 8px; text-align: center;'>{self.portfolio_inputs[portfolio]}</td>
            </tr>
            """
        
        table_html += "</table>"
        return table_html

    def save_portfolio(self, b):
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
            print(f"Total Sales: ${selected_customers['Gross_Sales'].sum():,.2f}")

# Example usage
au_data = [
    [1, 40.7128, -74.0060],
    [2, 41.8781, -87.6298],
    [3, 34.0522, -118.2437]
]

banker_data = [
    ['B1', 1, 'P1'],
    ['B2', 1, 'P2'],
    ['B3', 2, 'P3']
]

customer_data = [
    ['C1', 1, 40.7128, -74.0060, 'P1', 100000, 500000, 1000000],
    ['C2', 1, 40.7300, -74.0200, None, 200000, 600000, 1200000],
    ['C3', 2, 41.8781, -87.6298, 'P3', 150000, 550000, 1100000]
]

portfolio_manager = PortfolioManager(au_data, banker_data, customer_data)
