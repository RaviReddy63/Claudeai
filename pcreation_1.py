from io import BytesIO
import pandas as pd
import math
from math import sin, cos, atan2, radians, sqrt
import streamlit as st
import pydeck as pdk
import numpy as np



def haversine_distance(clat, clon, blat, blon):
	if math.isnan(clat) or math.isnan(clon) or math.isnan(blat) or math.isnan(blon):
		return 0
		
	delta_lat = radians(clat - blat)
	delta_lon = radians(clon - blon)
	
	a = sin(delta_lat/2)**2 + cos(radians(clat))*cos(radians(blat))*sin(delta_lon/2)**2
	c = 2*atan2(sqrt(a), sqrt(1-a))
	distance = 6371*c
	return distance
	

def merge_dfs(customer_data, banker_data, branch_data):
	# customer_data = pd.concat([customer_data, ua_cust_data], axis = 0)
	customer_data = customer_data.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
	# banker_branch_merge = banker_data[["AU", "BRANCH_NAME", "EID", "EMPLOYEE_NAME", "PORT_CODE", "ROLE_TYPE"]].merge(branch_data[["AU", "BRANCH_LAT_NUM", "BRANCH_LON_NUM"]], on = "AU", how = "left")
	final_table = customer_data.merge(banker_data, on = "PORT_CODE", how = "left")
	final_table.fillna(0, inplace = True)
	return final_table
	

def to_excel(form_results):
	output = BytesIO()
	with pd.ExcelWriter(output , engine='openpyxl') as writer:
		for form_id, df in form_results.items():
			df.to_excel(writer , sheet_name = f"Form_{form_id}", index = False)
	output.seek(0)
	return output
			

def data_filteration(customer_data, branch_data, banker_data, form_id):
	st.subheader(f"Form {form_id}")
	col1, col2, col3 = st.columns(3)
	
	with col1:
		state = st.selectbox(f"State (Form {form_id})", branch_data['STATECODE'].dropna().unique(), key=f"State_{form_id}")
		
	filter_data = branch_data[branch_data['STATECODE'] == state]
	
	with col2:
		city = st.selectbox(f"City (Form {form_id})", filter_data['CITY'].dropna().unique(), key=f"City_{form_id}")
		
	au_options = filter_data[filter_data['CITY'] == city]['AU'].dropna().unique()
	with col3:
		selected_au = st.selectbox(f"AU (Form {form_id})", au_options, key = f"AU_{form_id}")
		
		
	use_role = st.checkbox("Filter by Role", key = f"role_check_{form_id}")
	role = st.selectbox(f"Role (Form {form_id})", customer_data['TYPE'].dropna().unique(), key=f"Role_{form_id}") if use_role else None
	
	max_dist = st.slider(f"Max Distance (Form {form_id})", 0, 20000, 20, key=f"Distance_{form_id}")
	
	
	
	AU_row = branch_data[branch_data['AU'] == int(selected_au)].iloc[0]
	AU_lat = AU_row['BRANCH_LAT_NUM']
	AU_lon = AU_row['BRANCH_LON_NUM']
	
	box_lat = max_dist/111 ###################------cause 1 degree lat = ~111km
	box_lon = max_dist/ (111 * np.cos(np.radians(AU_lat)))
	
	customer_data_boxed = customer_data[(customer_data['LAT_NUM'] >= AU_lat - box_lat) &
										(customer_data['LAT_NUM'] <= AU_lat + box_lat) &
										(customer_data['LON_NUM'] <= AU_lon + box_lon) &
										(customer_data['LON_NUM'] >= AU_lon - box_lon)]
										
	customer_data_boxed['Distance'] = 0
	
	
	customer_data_boxed['Distance'] = customer_data_boxed.apply(lambda row : haversine_distance(row['LAT_NUM'], row['LON_NUM'], AU_lat, AU_lon), axis = 1)
	
	customer_data_boxed = customer_data_boxed.rename(columns={'CG_PORTFOLIO_CD': 'PORT_CODE'})
	filtered_data = customer_data_boxed.merge(banker_data , on = "PORT_CODE", how = 'left')
	
	
####################other filter condition will come here---------------------

	if (role is not None) and (role.lower() == 'IN-MARKET'.lower()):
		filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
		filtered_data = filtered_data[filtered_data['TYPE'].str.lower() == role.lower()]
	elif (role is not None) and (role.lower() == 'CENTRALIZED'.lower()):
		filtered_data = filtered_data[filtered_data['TYPE'].str.lower() == role.lower()]
	elif (role is not None) and (role.lower() == 'Unassigned'.lower()):
		filtered_data = filtered_data[filtered_data['TYPE'].str.lower() == role.lower()]
	elif (role is not None) and (role.lower() == 'Unmanaged'.lower()):
		filtered_data = filtered_data[filtered_data['TYPE'].str.lower() == role.lower()]
	else:
		filtered_data = filtered_data[filtered_data['Distance'] <= int(max_dist)]
		
		
	return [filtered_data, role, city, state, max_dist, selected_au]
	
	
	
def distance_filteration(role_type , filtered_df, max_dist):
	if role_type is not None:
		if role_type == 'CENTRALIZED':
			
			if filtered_df.empty:
				st.info("no data matched")
				return filtered_df
				
			else:
				filtered_df['Distance'] = filtered_df.apply( lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], row['BRANCH_LAT_NUM'], row['BRANCH_LON_NUM']), axis = 1)
				filtered_df = filtered_df[filtered_df['Distance'] <= max_dist]
				return filtered_df
				
		else:
			if filtered_df.empty:
				st.info("no data matched")
				return filtered_df
				
			else:
				filtered_df['Distance'] = filtered_df.apply( lambda row: haversine_distance(row['LAT_NUM'], row['LON_NUM'], row['BRANCH_LAT_NUM'], row['BRANCH_LON_NUM']), axis = 1)
				filtered_df = filtered_df[filtered_df['Distance'] <= max_dist]
				return filtered_df
				
	else:
		return filtered_df
		
		
		
def recommend_reassignment(form_res: dict) -> pd.DataFrame:
	combine_df = pd.concat([df.assign(original_form = form_id) for form_id , df in form_res.items()], ignore_index = True)
	
	au_map = { form_id: ( df["BRANCH_LAT_NUM"].iloc[0], df["BRANCH_LON_NUM"].iloc[0])
				for form_id , df in form_res.items()
				if not df.empty}
				
	records = []
	for _, row in combine_df.iterrows():
		best_form = None
		min_dist = float("inf")
		for form_id , (au_lat , au_lon) in au_map.items():
			dist = haversine_distance(row['LAT_NUM'], row['LON_NUM'] , au_lat , au_lon)
			if dist < min_dist:
				best_form = form_id
				min_dist = dist
				
		row_data = row.to_dict()
		row_data['recommended_form'] = best_form
		row_data['recommended_dist'] = min_dist
		records.append(row_data)
		
	return pd.DataFrame(records)
	
	
def optional_filters(data , form_id):
	out = data.copy()
	
	with st.expander("Apply optional filters on the customer data"):
		use_rev = st.checkbox("Filter by minimum revenue", key = f"revenue_check_{form_id}")
		min_rev = st.number_input("Enter minimum revenue" , min_value = 0.0 , step = 1000.0 , key = f"revenue_{form_id}") if use_rev else None
		out = out[out['BANK_REVENUE'] >= min_rev] if min_rev is not None else out
		
		use_deposit = st.checkbox("Filter by minimum deposit" , key = f"deposit_check_{form_id}")
		min_deposit = st.number_input("Enter minimum deposit" , min_value = 0.0 , step = 1000.0 , key = f"deposit_{form_id}") if use_deposit else None
		out = out[out['DEPOSIT_BAL'] >= min_deposit] if min_deposit is not None else out
		
		use_state = st.checkbox("Filter by customers' state" , key = f"cust_state_check_{form_id}")
		cust_state = st.selectbox(f"State (Form {form_id})", out['BILLINGSTATE'].dropna().unique(), key= f"state_{form_id}") if use_state else None
		out = out[out['BILLINGSTATE'] == cust_state] if cust_state is not None else out
		
		use_portcd = st.checkbox("Filter by Portfolio Code", key = f"port_cd_check_{form_id}")
		cust_portcd = st.multiselect(f"PortCD (Form {form_id})", out['PORT_CODE'].dropna().unique(), key= f"port_cd_{form_id}") if use_portcd else None
		out = out[out['PORT_CODE'].isin(cust_portcd)] if cust_portcd is not None else out
		
	return out
	

#------------------------Streamlit App---------------------------------------------------------------
st.set_page_config("Portfolio Creation tool", layout = "wide")
st.title("Portfolio creation tool")
page = st.selectbox("Select Page", ["Portfolio Assignment", "Portfolio Mapping"])

#------------------------File Upload----------------------------------------------------------------------------------
st.sidebar.header("Upload CSV files")
cust_file = st.sidebar.file_uploader("Customer Data" , type = ["csv"])
banker_file = st.sidebar.file_uploader("Banker Data" , type = ["csv"])
branch_file = st.sidebar.file_uploader("Branch Data", type = ["csv"])
# ua_cust_file = st.sidebar.file_uploader("Unassigned Customer Data", type = ["csv"])

if 'form_results' not in st.session_state:
	st.session_state.form_results = {}
	
if 'form_controls' not in st.session_state:
	st.session_state.form_controls = {}
	
if 'recommend_reassignment' not in st.session_state:
	st.session_state.recommend_reassignment = {}
	
if page == "Portfolio Assignment":

	if cust_file and banker_file and branch_file:
		customer_data = pd.read_csv(cust_file)
		banker_data = pd.read_csv(banker_file)
		branch_data = pd.read_csv(branch_file)
		data = merge_dfs(customer_data, banker_data, branch_data)
		
		st.sidebar.header("Form Configuration")
		num_forms = st.sidebar.number_input("Number of Portfolios", min_value = 1 , max_value = 10, value = 1)
		
		tab_titles = [f"Form {i}" for i in range(1, num_forms+1)]
		tabs = st.tabs(tab_titles)
		
		assigned_customers = set()
		form_results = {}
		
		for form_id, tab in enumerate(tabs , start = 1):
			with tab:
				filtered_data , role, city , state , max_dist , selected_au = data_filteration(customer_data, branch_data, banker_data, form_id)
				
				
				
#------------------------------ Calculating distance to filter-----------------------------

				filtered_data = optional_filters(filtered_data , form_id)
				
				
#------------------------------ Calculating distance ends here---------------------

				filtered_data['FormID'] = form_id
				st.session_state.form_controls.setdefault(form_id, {})
				
				
#------------------------------ Change clean up stale portfolioIDs-------------------
				valid_pids = set(filtered_data['PORT_CODE'].unique())
				
				st.session_state.form_controls[form_id] = {
					pid: val for pid, val in st.session_state.form_controls[form_id].items()
					if pid in valid_pids}
					
					
					
##---------Change Live conflict detection right after filtering-------------------
				assigned = {cid for fid , df in st.session_state.form_results.items() if fid != form_id for cid in df['CG_ECN']}
				conflicts = filtered_data[filtered_data['CG_ECN'].isin(assigned)]
				if not conflicts.empty:
					st.warning(f" {len(conflicts)} customer already assigned and removed")
					filtered_data = filtered_data[~filtered_data['CG_ECN'].isin(assigned)]
					
					
				grouped = filtered_data.groupby("PORT_CODE")
				
				for pid , group in grouped:
					kk = len(group)
					total_customer = len(data[data["PORT_CODE"] == pid])
					
					
					with st.expander(f"portfolio {pid} - {total_customer} customers (s)"):
						st.session_state.form_controls[form_id][pid] = {"n": len(group), "exclude":[]}
						ctrl = st.session_state.form_controls[form_id][pid]
						ctrl["n"] = st.number_input(f"Top N customer to select from Portfolio {pid}" ,
													min_value = 0,
													max_value = len(group),
													value = min(ctrl["n"] , len(group)),
													key = f"slider_{form_id}_{pid}")
													
						ctrl["exclude"] = []            ##########This part needs to be fixed, not unselecting the customers, check if it can get the to_un_select 
						
#--------------------Checking for conflicts----------------


				if st.button(f"Save Form {form_id}", key = f"save_{form_id}"):
					result = []
					au_row = branch_data[branch_data['AU'] == selected_au].iloc[0]
					b_au , b_lat , b_lon = au_row['AU'] , au_row['BRANCH_LAT_NUM'] , au_row['BRANCH_LON_NUM']
					for pid , group in grouped:
						ctrl = st.session_state.form_controls[form_id][pid]
						selected_customers = group[~group["CG_ECN"].isin(ctrl["exclude"])]
						top_n = selected_customers.sort_values(by='Distance').head(ctrl["n"])
						top_n['AU'] = b_au
						top_n['BRANCH_LAT_NUM'] = b_lat
						top_n['BRANCH_LON_NUM'] = b_lon
						result.append(top_n)
						
					form_df = pd.concat(result) if result else pd.DataFrame()
					
					
					conflicted_ids = []
					reassigned_rows = []
					for cid in form_df["CG_ECN"]:
						for other_fid , other_df in st.session_state.form_results.items():
							if other_fid == form_id:
								continue
							if cid in other_df["CG_ECN"].values:
								old_row = other_df[other_df["CG_ECN"] == cid].iloc[0]
								new_row = form_df[form_df["CG_ECN"] == cid].iloc[0]
								
								if new_row['Distance'] < old_row['Distance']:
									#replace in new form as per distance
									st.session_state.form_results[other_fid] = other_df[other_df["CG_ECN"] != cid]
									reassigned_rows.append(new_row)
									conflicted_ids.append((cid , other_fid , old_row['Distance'] , form_id , new_row['Distance']))
									
								else:
									form_df = form_df[form_df["CG_ECN"] != cid]
									
									
					if reassigned_rows:
						form_df = pd.concat([form_df , pd.DataFrame(reassigned_rows)])
						
					st.session_state.form_results[form_id] = form_df
					st.success(f"Form {form_id} saved with {len(form_df)} customers")
					
					
#----------------------------show conflicted IDS


					if conflicted_ids:
						with st.expander("conflict resolutions (Auto Handeled)"):
							conflict_df = pd.DataFrame(conflicted_ids , columns = ["CG_ECN", "Previous Form", "Previous Distance", "Assigned Form", "New Distance"])
							st.warning("Some customers were reassigned based on distance:")
							st.dataframe(conflict_df, use_container_width = True)
							
							
#------------------------------- exporting the downloading files---------------------

#----------------Live tracking--------------------

		from collections import defaultdict
		tracker = defaultdict(int)
		data_for_pivot = []
		pid_track = {}
		already_assigned = {cid for df in st.session_state.form_results.values() for cid in df['CG_ECN']}
		for fid , controls in st.session_state.form_controls.items():
			pid_track[fid] = 0
			counts_for_form = 0
			for pid, ctrl in controls.items():
				df_pid = data[data['PORT_CODE'] == pid]
				valid = df_pid[~df_pid['CG_ECN'].isin(ctrl['exclude']) &
								~df_pid['CG_ECN'].isin(already_assigned)]
								
				sel_count = min(len(valid) , ctrl['n'])
				if sel_count > 0:
					tracker[pid] += sel_count
					data_for_pivot.append([pid , "Form"+str(fid) , sel_count])
					
				pid_track[fid] += sel_count
				
		tracker_df = pd.DataFrame(data_for_pivot, columns = ["PortfolioID", "FormID", "Customers"])
		tracker_df = pd.pivot_table(tracker_df, values = "Customers", index = "PortfolioID", columns = "FormID", aggfunc = 'sum')
		
		st.markdown("-----")
		st.sidebar.subheader("Live Portfolio Tracker")
		if tracker:
			tdf = pd.DataFrame([{"PortfolioID":pid, "Customers selected":n} for pid , n in tracker.items()])
			st.sidebar.dataframe(tdf)
			
		st.sidebar.markdown("**Customer count per Form**")
		st.sidebar.dataframe(tracker_df)
		for fid, counts in pid_track.items():
			st.sidebar.write(f"Form {fid} -> {counts} Customer (s)")
			
			
#--------------------------Save all forms


		if st.button("Recommended form"):
			rec_df = recommend_reassignment(st.session_state.form_results)
			st.session_state.recommend_reassignment = rec_df
			st.subheader("Form reassignment")
			st.dataframe(st.session_state.recommend_reassignment)
			st.session_state.recommend_reassignment.to_csv(r'file_rec.csv')
			
		st.markdown("----")
		if st.button("Save all form"):
			combined_result = pd.concat(st.session_state.form_results.values())
			st.session_state.final_result = combined_result
			st.success("All Forms are saved successfully")
			st.write(st.session_state.final_result)
			st.session_state.final_result.to_csv(r'file.csv')
			
	else:
		st.info("Please upload all 3 files to begin")
