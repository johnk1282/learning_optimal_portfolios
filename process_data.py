import os
import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import wrds

data_input_path = 'input/'
data_output_path = ''
data_output_file = 'characteristics.pkl'
n_stocks = 3000

#----------------------------------------------------------
# function to check whether stock characteristics have been 
# calculated, process, and return completed dataframe  
# see below for functions that calculate from input data  
#----------------------------------------------------------

def multi_groupby(df,groupby_cols,func):
	#helper function to do a pandas groupby().apply() using multi-processing
	pool = mp.Pool(processes=mp.cpu_count())
	results = pool.map(func,[df_sub for _,df_sub in df.groupby(groupby_cols)])
	pool.close()
	pool.join()
	return pd.concat(results)

def normalize_chars(df_sub):
	#helper function that normalizes characteristics in cross-section and within industry; will be fed into the multi-groupby function
	char_cols = [x for x in df_sub.columns if x not in ['permno','date','industry']]
	df_sub[[x+'_ia' for x in char_cols]] = df_sub.groupby('industry')[char_cols].transform(lambda x: (x-x.mean())/x.std())
	df_sub[char_cols] = (df_sub[char_cols]-df_sub[char_cols].mean())/df_sub[char_cols].std()
	return df_sub

def get_characteristics(overwrite=False):
	#calls functions below to calculate characteristics from input data and append to dataframe
	#by default does not overwrite steps that have already been done and logged
	#after characteristics are calculated, they are subset/normalized
	if overwrite:
		log = []
	elif os.path.isfile(data_output_path+'characteristics_output_log.txt'):
		with open(data_output_path+'characteristics_output_log.txt','r') as f:
			log = f.read().split('\n')
	else:
		log = []
	if 'technical' not in log:
		calc_technical()
	if 'fundamental' not in log:
		calc_fundamental()
	if 'earnmom' not in log:
		calc_earnmom()
	if 'estimates' not in log:
		calc_estimates()
	if 'processed' not in log:
		chars = pd.read_pickle(data_output_path+data_output_file)
		char_cols = list(chars.columns)
		chars.reset_index(inplace=True)
		#subset by date
		chars = chars.loc[chars.date>='1960-01-01']
		#subset to largest n_stocks in each month
		chars['mve_rank'] = chars.groupby('date')['mve'].rank(ascending=False)
		chars = chars.loc[chars.mve_rank<=n_stocks]
		#merge on SIC codes and get industry categorizations from map
		crsp = get_crsp_m()
		crsp['date'] = pd.to_datetime(crsp['date'])-MonthBegin(1)
		chars = chars.merge(crsp[['permno','date','hsiccd']],how='left',on=['permno','date'])
		industry_map = get_industry_sic_map()
		chars['industry'] = chars.hsiccd.replace(industry_map)
		chars['industry'] = chars['industry'].where(chars['industry'].isin(list(range(48))),49)
		chars = chars.drop(columns=['mve_rank','hsiccd'])
		#normalize characteristics and create new set of normalized characteristics within industry groups
		chars = multi_groupby(chars,['date'],normalize_chars)
		out = chars.set_index(['permno','date'])[char_cols+[x+'_ia' for x in char_cols]]
		out.to_pickle(data_output_path+data_output_file)
		with open(data_output_path+'characteristics_output_log.txt','a') as f:
			f.write('processed\n')
	else:
		out = pd.read_pickle(data_output_path+data_output_file)
	return out

#----------------------------------------------------------
# functions to calculate stock characteristics from input 
# data, lag appropriately, and append to output file
# see below for functions that import input files from WRDS
#----------------------------------------------------------

def output_characteristics(df,cols):
	#append columns (cols; list of column names) from dataframe (df) to output file 
	#note that output file is indexed by ['permno', 'date'], which must exist in df
	try:
		df = df.set_index(['permno','date'])
	except:
		return ValueError('df must contain [permno,date] as columns for indexing')
	if os.path.isfile(data_output_path+data_output_file):
		#overwrite output dataframe after appending new characteristics
		characteristics = pd.read_pickle(data_output_path+data_output_file)
		characteristics[cols] = df[cols]
		characteristics.to_pickle(data_output_file)
	else:
		#initialize index with CRSP universe
		characteristics = get_crsp_m()[['permno','date']]
		characteristics['date'] -= MonthBegin(1)
		characteristics = characteristics.set_index(['permno','date'])
		characteristics[cols] = df[cols]
		characteristics.to_pickle(data_output_path+data_output_file)

def calc_technical_sub(df_sub):
	#calculate mean returns over horizons
	for i in [5,10,22,198,252,756]:
		df_sub['mean_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).mean()
	#calculate volatility, correlation, and beta over horizons
	for i in [22,252,756]:
		df_sub['std_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).std()
		df_sub['corr_sp_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).corr(df_sub['sprtrn'])
		df_sub['beta_sp_'+str(i)] = df_sub['corr_sp_'+str(i)]*df_sub['std_'+str(i)]/df_sub['sp_std_'+str(i)]
		df_sub['beta_sp_'+str(i)+'_sq'] = df_sub['beta_sp_'+str(i)]**2
	#calculate idiosyncratic volatility 
	for i in [22,252]:	
		df_sub['e_'+str(i)] = df_sub['ret']-df_sub['beta_sp_'+str(i)]*df_sub['sprtrn']
		df_sub['e_std_'+str(i)] = df_sub['e_'+str(i)].rolling(i,min_periods=i).std()
	#calculate liquidity metrics
	df_sub['volume_5'] = df_sub['vol'].rolling(5,min_periods=5).sum()
	df_sub['stdvolume_22'] = df_sub['vol'].rolling(22,min_periods=22).std()
	df_sub['bidask'] = (df_sub['ask'].abs()-df_sub['bid'].abs())/df_sub['prc'].abs()
	df_sub['bidask_5'] = df_sub['bidask'].rolling(5,min_periods=5).sum()
	df_sub['volume_5_shrout'] = df_sub['volume_5']/df_sub['shrout']
	#change in shares outstanding
	df_sub['shrout_pct'] = df_sub['shrout']/df_sub.groupby(['permno'])['shrout'].shift(22)-1
	#final characteristics
	cols = ['mean_5','mean_10','mean_22','mean_198',
			'mean_252','mean_756','std_22','std_252','volume_5',
			'volume_5_shrout','stdvolume_22','bidask_5','mve',
			'corr_sp_22','corr_sp_252','corr_sp_756','beta_sp_22','beta_sp_252',
			'beta_sp_756','beta_sp_22_sq','beta_sp_252_sq',
			'beta_sp_756_sq','shrout_pct','e_std_22','e_std_252']
	#lag by 2 business days to ensure availability + 1 month to align with t+1 return
	df_sub['date'] = df_sub['date']+BDay(2)+MonthBegin(1)
	return df_sub.groupby(['permno','date'])[cols].last()

def calc_technical():
	#calculate technical characteristics with crsp daily data
	crsp_d = get_crsp_d()
	crsp_d['date'] = pd.to_datetime(crsp_d['date'])
	crsp_d = crsp_d.loc[crsp_d.date>pd.to_datetime('1950-01-01')]
	crsp_d['mve'] = crsp_d['prc'].abs()*crsp_d['shrout']
	crsp_d['mve_rank'] = crsp_d.groupby('date')['mve'].rank(ascending=False)
	min_rank = crsp_d.groupby('permno')['mve_rank'].min()
	crsp_d = crsp_d.loc[crsp_d.permno.isin(min_rank.index[min_rank<=3000])]
	
	crsp_d.sort_values(by=['permno','date'],inplace=True)
	ind = get_indices_d()
	ind['date'] = pd.to_datetime(ind['date'])
	ind.sort_values(by=['date'],inplace=True)
	for i in [22,252,756]:
		ind['sp_std_'+str(i)] = ind['sprtrn'].rolling(i,min_periods=i).std()
	crsp_d = crsp_d.merge(ind[['date','sprtrn']+['sp_std_'+str(i) for i in [22,252,756]]],how='left',on=['date'])
	out = multi_groupby(crsp_d,['permno'],calc_technical_sub)
	cols = out.columns
	out.reset_index(inplace=True)
	#append to output file
	output_characteristics(out,cols)
	with open(data_output_path+'characteristics_output_log.txt','a') as f:
		f.write('technical\n')

def calc_fundamental():
	#calculate fundamental characteristics with compustat quarterly data
	comp = get_compustat_q()
	#merge filing dates onto compustat data
	aod = get_comp_dates()
	aod.sort_values(by=['rdq'],ascending=False,inplace=True)
	aod.drop_duplicates(subset=['gvkey','datadate'],inplace=True)
	comp = comp.merge(aod,how='left',on=['gvkey','datadate'])
	comp = comp.sort_values(by=['datadate','gvkey'])
	#take differences of some fiscal YTD variables
	y_vars = ['capxy','recchy','invchy','apalchy','aqcy']
	comp[[var+'_lag' for var in y_vars]] = comp.groupby('gvkey')[y_vars].shift(1)
	#comp[[var[:-1]+'q' for var in y_vars]] = np.where(comp.fqtr==1,comp[y_vars],comp[y_vars]-comp[[var+'_lag' for var in y_vars]].values)
	comp[[var[:-1]+'q' for var in y_vars]] = comp[y_vars].where(comp.fqtr==1,comp[y_vars]-comp[[var+'_lag' for var in y_vars]].values)
	#calculate book equity
	comp['ps'] = np.where(comp['pstkq'].isnull(),0,comp['pstkq'])
	comp['txditc'] = comp['txditcq'].fillna(0)
	comp['be'] = (comp['seqq']+comp['txditc']-comp['ps']).combine_first(comp['atq']-comp['ltq'])
	comp['be'] = comp['be'].where(comp['be']>0,np.nan)
	#growth in book equity
	comp['be_growth'] = comp.groupby('gvkey')['be'].pct_change(1)
	cols_out = ['be_growth']
	#make assets non-negative
	comp['atq'] = comp['atq'].where(comp['atq']>0,np.nan)
	#asset growth and investment
	comp['asset_growth'] = comp.groupby('gvkey')['atq'].pct_change(4)
	comp['inv'] = comp['aqcq']/comp['atq']
	comp['inv2'] = comp['aqcq']/comp['be']
	cols_out += ['asset_growth','inv','inv2']
	#total debt
	comp['td'] = comp['dlttq']+comp['dlcq'].fillna(0)
	#calculate accruals, sales, profitability, cash flow
	comp['wcap'] = comp['actq']-comp['lctq']-comp['cheq']+comp['dlcq']+comp['txtq']
	comp['wc_d'] = comp.groupby('gvkey')['wcap'].diff(1)
	comp['acc'] = comp['dpq'] - comp['wc_d']
	comp['acc2'] = comp['recchq'] - comp['invchq'] - comp['apalchq']
	comp['sales'] = comp['saleq'].combine_first(comp['revtq'])
	comp['prof'] = comp['sales'] - comp['cogsq'] - comp['xsgaq'].fillna(0) + comp['xrdq'].fillna(0)
	comp['prof2'] = comp['prof'] - comp['recchq'].fillna(0) - comp['invchq'].fillna(0) - comp['apalchq'].fillna(0)
	comp['cf'] = (comp['ibq'] + comp['dpq'].fillna(0) - comp['wc_d'].fillna(0) - comp['capxq'].fillna(0))
	comp['cf2'] = comp['prof'] - comp['capxq'].fillna(0)
	comp['eps'] = comp['epspxq']
	#all cash flow variables
	cf_vars = ['sales','eps','prof','prof2','cf','cf2']
	#Novy-Marx fundamental momentum applied to flow variables
	comp[[var+'_d' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].diff(4)
	comp[[var+'_d_std' for var in cf_vars]] = comp.groupby('gvkey')[[var+'_d' for var in cf_vars]].rolling(8,min_periods=6).std().reset_index(0,drop=True)
	comp[[var+'_momentum' for var in cf_vars]] = comp[[var+'_d' for var in cf_vars]]/comp[[var+'_d_std' for var in cf_vars]].values
	cols_out += [var+'_momentum' for var in cf_vars]
	#growth and volatility applied to flow variables
	comp[[var+'_l' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].shift(4)
	comp[[var+'_growth' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].pct_change(4)
	comp[[var+'_growth' for var in cf_vars]] = comp[[var+'_growth' for var in cf_vars]].where(comp[[var+'_l' for var in cf_vars]].values>0,np.nan)
	comp[[var+'_std' for var in cf_vars]] = comp.groupby('gvkey')[[var+'_growth' for var in cf_vars]].rolling(10,min_periods=6).std().reset_index(0,drop=True)
	cols_out += [var+'_growth' for var in cf_vars]+[var+'_std' for var in cf_vars]
	#profitability, cash flow metrics to assets/book equity
	ratio_vars = cf_vars+['acc','acc2','td']
	comp[[var+'_asset' for var in ratio_vars]] = comp[ratio_vars].divide(comp['atq'],0)
	comp[[var+'_be' for var in ratio_vars]] = comp[ratio_vars].divide(comp['be'],0)
	cols_out += [var+'_asset' for var in ratio_vars]+[var+'_be' for var in ratio_vars]
	#change in leverage
	comp['td_l'] = comp.groupby('gvkey')['td'].shift(1)
	comp['td_d'] = comp.groupby('gvkey')['td'].diff(1)
	comp['td_d_std'] = comp.groupby('gvkey')['td_d'].rolling(8,min_periods=6).std().reset_index(0,drop=True)
	comp['chdebt_asset'] = comp['td_d']/comp['atq']
	comp['chdebt_z'] = comp['td_d']/comp['td_d_std']
	comp['chdebt_growth'] = comp['td_d']/comp['td_l']
	comp['chdebt_growth'] = comp['chdebt_growth'].where(comp['td_l']>0,np.nan)
	cols_out += ['chdebt_asset','chdebt_z','chdebt_growth']
	comp[cols_out] = comp[cols_out].replace({np.inf:np.nan,-np.inf:np.nan})
	#import crsp
	crsp = get_crsp_m()
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp['date'] = crsp['date']-MonthBegin(1)
	crsp = crsp.sort_values(by=['date','permno'])
	#calculate lagged market equity
	crsp['me'] = crsp['shrout']*crsp['prc'].abs()/1000
	crsp['me_lag'] = crsp.groupby('permno')['me'].shift(1)
	crsp['me_lag'] = np.where(crsp['me_lag'].isnull(),crsp['me']/(1+crsp['retx']),crsp['me_lag'])
	crsp['date_lag'] = crsp.groupby(['permno'])['date'].shift(1)
	crsp['date_lag'] = np.where(crsp['date_lag'].isnull(),crsp['date']-MonthBegin(1),crsp['date_lag'])
	crsp['date_comp'] = crsp['date']-MonthBegin(1)
	crsp['me_lag'] = np.where(crsp['date_comp']==crsp['date_lag'],crsp['me_lag'],np.nan)
	#get linking table to join crsp and compustat
	link = get_crspcomp_link()
	crsp = crsp.merge(link[['permno','gvkey','linkdt','linkenddt']],how='left',on='permno')
	crsp['gvkey'] = crsp.gvkey.where(~((crsp.date<crsp.linkdt)|(crsp.date>crsp.linkenddt)),np.nan)
	crsp = crsp.dropna(subset=['gvkey'])
	crsp = crsp.drop_duplicates(subset=['permno','date'])
	#join crsp and compustat after lagging compustat by 2 business days after filing date + 1 month to align with t+1 return
	#until now compustat was implicitly indexed by the period of filing, now switch to filing date 
	crsp['jdate'] = pd.to_datetime(crsp.date)
	comp['jdate'] = pd.to_datetime(comp.rdq)+BDay(2)+MonthBegin(1)
	comp = comp.sort_values(by=['gvkey','jdate','datadate'],ascending=[True,True,False]).dropna(subset=['jdate']).drop_duplicates(subset=['gvkey','jdate'])
	#merge on exact date
	to_merge = ['be','td','td_d','sales','prof','prof2','cf','cf2']
	crsp = crsp.merge(comp[['gvkey','jdate']+cols_out+to_merge],how='left',on=['gvkey','jdate'])
	#create a copy of signals forward filled for at most 12 months
	cols_asof = [x+'_asof' for x in cols_out]
	crsp[cols_asof+to_merge] = crsp.groupby('permno')[cols_out+to_merge].fillna(method='ffill',limit=12)
	cols = cols_out + cols_asof
	#calculate enterprise value, value signals, leverage on me/ev
	crsp['ev'] = crsp['me_lag']+crsp['td'].fillna(0)
	ratio_vars2 = ['td_d','be','sales','prof','prof2','cf','cf2']
	crsp[[var+'_me' for var in ratio_vars2]] = crsp[ratio_vars2].divide(crsp['me_lag'],0)
	crsp[[var+'_ev' for var in ratio_vars2]] = crsp[ratio_vars2].divide(crsp['ev'],0)
	cols += [var+'_me' for var in ratio_vars2]+[var+'_ev' for var in ratio_vars2]
	output_characteristics(crsp,cols)
	with open(data_output_path+'characteristics_output_log.txt','a') as f:
		f.write('fundamental\n')

def calc_earnmom():
	#calculate earnings momentum using returns around quarterly filing dates (Novy-Marx)
	#get crsp daily file, merge with compustat linking table and subset to companies in compustat
	crsp = get_crsp_d()
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp = crsp.sort_values(by=['date','permno'])
	link = get_crspcomp_link()
	crsp = crsp.merge(link[['permno','gvkey','linkdt','linkenddt']],how='left',on='permno')
	crsp['gvkey'] = crsp.gvkey.where(~((crsp.date<crsp.linkdt)|(crsp.date>crsp.linkenddt)),np.nan)
	crsp = crsp.dropna(subset=['gvkey'])
	#merge on S&P 500 index returns
	ind = get_indices_d()
	ind['date'] = pd.to_datetime(ind['date'])
	crsp = crsp.merge(ind[['date','sprtrn']],how='left',on='date')
	#estimate idiosyncratic return as r-r_m
	crsp['e'] = crsp['ret']-crsp['sprtrn']
	#calculate 3 and 5 day centered mean returns (this seems quicker than .rolling())
	for i in range(-2,3):
		crsp['ret_'+str(i)] = crsp.groupby('permno')['ret'].shift(-i)
		crsp['e_'+str(i)] = crsp.groupby('permno')['e'].shift(-i)
	crsp['momentum_earn_3'] = crsp[['ret_'+str(i) for i in range(-1,2)]].mean(axis=1)
	crsp['momentum_earn_5'] = crsp[['ret_'+str(i) for i in range(-2,3)]].mean(axis=1)
	crsp['momentum_idio_3'] = crsp[['e_'+str(i) for i in range(-1,2)]].mean(axis=1)
	crsp['momentum_idio_5'] = crsp[['e_'+str(i) for i in range(-2,3)]].mean(axis=1)
	#characteristics to output
	cols = ['momentum_earn_3','momentum_earn_5','momentum_idio_3','momentum_idio_5']
	#get earnings dates as quarterly filing dates and merge 3/5 day returns/idiosyncratic returns on
	aod = get_comp_dates()
	aod['jdate'] = pd.to_datetime(aod['rdq'])
	aod = aod.dropna(subset=['jdate','gvkey'])
	aod = aod.sort_values(by=['jdate','gvkey'])
	crsp['jdate'] = pd.to_datetime(crsp.date)
	out = pd.merge_asof(aod,crsp[['gvkey','jdate','permno','date']+cols],on='jdate',by=['gvkey'])
	out = out.groupby(['permno','rdq'])[['date']+cols].last().reset_index()
	#lag characteristics by 5 business days + 1 month to align with t+1 return
	out['date'] = out['date']+BDay(5)+MonthBegin(1)
	out = out[['date','permno']+cols].dropna().drop_duplicates(subset=['permno','date'])
	output_characteristics(out,cols)
	with open(data_output_path+'characteristics_output_log.txt','a') as f:
		f.write('earnmom\n')

def calc_estimates():
	#calculate signals from earnings estimate date (IBES), including earnings surprises, fwd P/E, change in analyst coverage, fwd EPS growth
	ibes = get_ibes_summary()
	#keep estimates for all fwd financial periods which will be used later to calculate fwd P/E ratios
	all_fpi = ibes.loc[ibes.fpi.isin(['0','1','2','3','4','6','7','8','9'])]
	#lag by a month to align with t+1 return
	all_fpi['date'] = pd.to_datetime(all_fpi['statpers']) + DateOffset(months=1)
	#reshape to create columns for each fwd financial period, indexed by company/date
	all_fpi = all_fpi[['ticker','date','fpi','meanest']].pivot_table(index=['ticker','date'],columns=['fpi'],values='meanest',aggfunc=np.sum)
	all_fpi = all_fpi.sort_values(by=['date','ticker'])
	#rename columns 
	replace_dict = dict(zip([str(i) for i in range(5)]+[str(i) for i in range(6,10)],['ltg']+['ann'+str(i) for i in range(1,5)]+['qtr'+str(i) for i in range(1,5)]))
	all_fpi.columns = [replace_dict[x] for x in all_fpi.columns]
	#subset ibes data to 1 qtr fwd EPS estimate and select the most recent estimate before the earnings date
	ibes = ibes.loc[(ibes.measure=='EPS')&(ibes.fpi=='6')]
	ibes['fpedats'] = pd.to_datetime(ibes.fpedats)
	ibes = ibes.sort_values(by=['statpers','ticker'])
	ibes = ibes.groupby(['fpedats','ticker'])[['statpers','meanest','medest','stdev','numest']].last().reset_index()
	#merge on the actual earnings release figures
	actuals = get_ibes_actual()
	actuals = actuals.rename(columns={'value':'actual'})
	actuals = actuals.loc[(actuals.measure=='EPS')&(actuals.pdicity=='QTR')]
	actuals['fpedats'] = pd.to_datetime(actuals.pends) 
	ibes = ibes.merge(actuals[['fpedats','ticker','actual','anndats']],how='left',on=['fpedats','ticker'])
	#set the date index to the earnings releast date and lag by 2 business days + 1 month to align with t+1 return
	ibes['anndats'] = pd.to_datetime(ibes['anndats'])
	ibes['date'] = ibes['anndats'] + BDay(2) + MonthBegin(1)
	ibes = ibes.loc[pd.notnull(ibes.anndats)]
	ibes = ibes.sort_values(by=['date','ticker']).drop_duplicates(subset=['ticker','date'])
	#calculate standardized unexpected earnings (SUE) and earnings surprise z-score
	ibes['sue'] = (ibes['actual']-ibes['meanest'])/ibes['stdev']
	ibes['sue'] = ibes['sue'].where(ibes.stdev!=0,np.nan)
	ibes['surprise'] = ibes['actual']-ibes['meanest']
	ibes['surprise_z'] = ibes.groupby('ticker')['surprise'].rolling(8,min_periods=6).apply(lambda x: (x[-1]-x.mean())/x.std(),raw=True).reset_index(0,drop=True)
	#calculate change in analyst coverage
	ibes['numest_lag'] = ibes.groupby('ticker')['numest'].shift(1)
	ibes['chanalyst'] = ibes['numest']-ibes['numest_lag'] 
	ibes['pchanalyst'] = ibes['numest']/ibes['numest_lag']-1 
	#characteristics to output
	cols = ['sue','surprise_z','chanalyst','pchanalyst']
	#get crsp file,  merge with IBES linking table, and merge characteristics
	crsp = get_crsp_m()
	crsp = crsp.sort_values(by=['date','permno'])
	link = get_crspibes_link()
	crsp = crsp.merge(link[['permno','ticker','sdate','edate']],how='left',on='permno')
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp['ticker'] = crsp.ticker.where(~((crsp.date<crsp.sdate)|(crsp.date>crsp.edate)),np.nan)
	crsp['date'] = pd.to_datetime(crsp.date)-MonthBegin(1)
	crsp = crsp.dropna(subset=['ticker']).drop_duplicates(subset=['permno','date'])
	crsp = crsp.merge(ibes[['ticker','date']+cols],how='left',on=['ticker','date'])
	#merge all fwd earning period estimates onto crsp data
	crsp = pd.merge_asof(crsp,all_fpi,on='date',by=['ticker'])
	#fwd earnings yield at various horizons
	crsp['prc_lag'] = crsp.groupby('permno')['prc'].shift(1)
	crsp['prc_lag'] = crsp.prc_lag.abs()
	crsp['pe0'] = crsp['ann1']/crsp['prc_lag']
	crsp['pe1'] = crsp['ann2']/crsp['prc_lag']
	crsp['pe2'] = crsp['ann3']/crsp['prc_lag']
	crsp['pe3'] = crsp['ann4']/crsp['prc_lag']
	crsp['pe4'] = crsp['qtr1']/crsp['prc_lag']
	crsp['pe5'] = crsp[['qtr1','qtr2','qtr3','qtr4']].sum(axis=1)/crsp['prc_lag']
	#add to list of characteristics to output
	cols += ['pe'+str(x) for x in range(6)]+['ltg']
	output_characteristics(crsp,cols)
	with open(data_output_path+'characteristics_output_log.txt','a') as f:
		f.write('estimates\n')

#----------------------------------------------------------
# functions to retrieve input data from WRDS (login needed) 
#----------------------------------------------------------

def get_crsp_m():
	#download and process monthly CRSP file from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'crsp/crsp_m.pkl'):
		crsp_m = pd.read_pickle(data_input_path+'crsp/crsp_m.pkl')
	else:
		db = wrds.Connection()
		#CRSP file with share/exchange data joined on
		crsp_m = db.raw_sql('''
		                  select a.permno, a.permco, a.cusip, a.date, 
	                      	b.shrcd, b.exchcd, a.ret, a.retx, a.shrout, 
	                      	a.prc, a.bid, a.ask, a.vol, a.hsiccd, a.cfacpr, a.cfacshr
	                      from crspq.msf as a
	                      left join crspq.msenames as b
	                      on a.permno=b.permno
	                      	and b.namedt<=a.date
	                      	and a.date<=b.nameendt
	                      ''') 
		#de-listing returns to be joined to crsp file
		dlret = db.raw_sql('''
		                   select permno, dlret, dlstdt 
		                   from crspq.msedelist
		                   ''')
		#merge de-listing returns and calcualte adjusted return 
		crsp_m['jdate'] = crsp_m['date']+MonthEnd(0)
		crsp_m = crsp_m.set_index(['permno','jdate'])
		dlret['jdate'] = dlret['dlstdt']+MonthEnd(0)
		dlret = dlret.set_index(['permno','jdate'])
		crsp_m['dlret'] = dlret.dlret
		crsp_m.reset_index(inplace=True)
		crsp_m = crsp_m.loc[pd.notnull(crsp_m.ret)|pd.notnull(crsp_m.dlret)]
		crsp_m['dlret'] = crsp_m['dlret'].fillna(0)
		crsp_m['ret'] = crsp_m['ret'].fillna(0)
		crsp_m['retadj'] = (1+crsp_m['ret'])*(1+crsp_m['dlret'])-1
		#calculate market equity
		crsp_m['me'] = crsp_m['prc'].abs()*crsp_m['shrout']
		crsp_m.to_pickle(data_input_path+'crsp/crsp_m.pkl')
		db.close()
	return crsp_m

def get_crsp_d():
	#download daily CRSP file from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'crsp/crsp_d.pkl'):
		crsp_d = pd.read_pickle(data_input_path+'crsp/crsp_d.pkl')
	else:
		db = wrds.Connection()
		crsp_d = db.raw_sql('''
		                    select a.permno, a.permco, a.cusip, a.date, 
		                    	b.shrcd, b.exchcd, a.ret, a.retx, a.shrout, 
		                      	a.prc, a.bid, a.ask, a.vol
		                    from crspq.dsf as a
		                    left join crspq.msenames as b
		                    on a.permno=b.permno
		                     	and b.namedt<=a.date
		                      	and a.date<=b.nameendt
		                    ''') 
		crsp_d.to_pickle(data_input_path+'crsp/crsp_d.pkl')
		db.close()
	return crsp_d

def get_indices_m():
	#download monthly index data from CRSP with WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'crsp/indices_m.pkl'):
		indices_m = pd.read_pickle(data_input_path+'crsp/indices_m.pkl')
	else:
		db = wrds.Connection()
		indices_m = db.raw_sql('''
		                       select *
		                       from crsp.msi
		                       ''') 
		indices_m.to_pickle(data_input_path+'crsp/indices_m.pkl')
		db.close()
	return indices_m

def get_indices_d():
	#download daily index data from CRSP with WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'crsp/indices_d.pkl'):
		indices_d = pd.read_pickle(data_input_path+'crsp/indices_d.pkl')
	else:
		db = wrds.Connection()
		indices_d = db.raw_sql('''
		                      select *
		                      from crsp.dsi
		                   	   ''') 
		indices_d.to_pickle(data_input_path+'crsp/indices_d.pkl')
		db.close()
	return indices_d

def get_crsp_divs():
	#download CRSP dividend data from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'crsp/crsp_divs.pkl'):
		crsp_divs = pd.read_pickle(data_input_path+'crsp/crsp_divs.pkl')
	else:
		db = wrds.Connection()
		crsp_divs = db.raw_sql('''
		                      select a.permno, a.divamt, a.dclrdt, a.exdt, a.paydt
		                      from crsp.msedist as a
			                   ''') 
		crsp_divs.to_pickle(data_input_path+'crsp/crsp_divs.pkl')
		db.close()
	return crsp_divs

def get_crspcomp_link():
	#download CRSP/Compustat linking table from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'linking/crspcomp.pkl'):
		link = pd.read_pickle(data_input_path+'linking/crspcomp.pkl')
	else:
		db = wrds.Connection()
		link = db.raw_sql('''
		                      select gvkey,lpermno,linkdt,linkenddt
		                      from crspq.ccmxpf_lnkhist 
		                      ''') 
		link.to_pickle(data_input_path+'linking/crspcomp.pkl')
		db.close()
	#make sure there's a 1-1 mapping
	link = link.rename(columns={'lpermno':'permno'}).drop_duplicates()
	link['linkdt'] = pd.to_datetime(link.linkdt)
	link['linkenddt'] = pd.to_datetime(link.linkenddt)
	link['linkenddt'] = link.linkenddt.where(pd.notnull(link.linkenddt),pd.to_datetime('today'))
	link = link.groupby(['permno','gvkey']).agg({'linkdt':'min','linkenddt':'max'}).reset_index()
	return link

def get_compustat_q():
	#download quarterly Compustat file from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'compustat/compustat_q.pkl'):
		compustat_q = pd.read_pickle(data_input_path+'compustat/compustat_q.pkl')
	else:
		db = wrds.Connection()
		compustat_q = db.raw_sql('''
			                      select gvkey,cusip,datadate,fqtr,atq,actq,ltq,cheq,teqq,seqq,lseq,lctq,dlttq,dlcq,
			                      	chq,gdwlq,req,revtq,saleq,cogsq,xsgaq,txtq,capxy,dpq,niq,rectq,invtq,
			                      	apq,apalchy,recchy,xaccq,xrdq,epspxq,aqcy,pstkq,txditcq,invchy,ibq,
			                      	dvy,cshoq
			                      from comp.fundq
		                         ''') 
		compustat_q.to_pickle(data_input_path+'compustat/compustat_q.pkl')
		db.close()
	return compustat_q

def get_comp_dates():
	#download and process mapping table for Compustat filing dates from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'compustat/as_of_dates.pkl'):
		comp_dates = pd.read_pickle(data_input_path+'compustat/as_of_dates.pkl')
	else:
		db = wrds.Connection()
		comp_dates = db.raw_sql('''
			                      select distinct gvkey,datadate,rdq
			                      from comp.fundq
		                    	''') 
		comp_dates.to_pickle(data_input_path+'compustat/as_of_dates.pkl')
		db.close()
	return comp_dates

def get_ibes_summary():
	#download IBES summary file from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'ibes/ibes_summary.pkl'):
		ibes_summary = pd.read_pickle(data_input_path+'ibes/ibes_summary.pkl')
	else:
		db = wrds.Connection()
		#EPS measure estimates
		ibes_summary = db.raw_sql('''
				                      select ticker,cusip,statpers,fpedats,fiscalp,fpi,measure,
				                      	numest,numup,numdown,medest,meanest,stdev,highest,lowest
				                      from ibes.statsumu_epsus
		                    	  ''') 
		#non-EPS measure estimates
		ibes_summary2 = db.raw_sql('''
				                      select ticker,cusip,statpers,fpedats,fiscalp,fpi,measure,
				                      	numest,numup,numdown,medest,meanest,stdev,highest,lowest
				                      from ibes.statsumu_xepsus
		                    	   ''') 
		
		ibes_summary = pd.concat([ibes_summary,ibes_summary2],axis=0)
		ibes_summary.to_pickle(data_input_path+'ibes/ibes_summary.pkl')
		db.close()
	return ibes_summary

def get_ibes_actual():
	#download IBES actuals file from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'ibes/ibes_actual.pkl'):
		ibes_actual = pd.read_pickle(data_input_path+'ibes/ibes_actual.pkl')
	else:
		db = wrds.Connection()
		#EPS measure actuals
		ibes_actual = db.raw_sql('''
			                      select ticker,cusip,anndats,pends,measure,value,pdicity
			                      from ibes.actu_epsus
	  		                     ''') 

		#non-EPS measure actuals
		ibes_actual2 = db.raw_sql('''
			                      select ticker,cusip,anndats,pends,measure,value,pdicity
			                      from ibes.actu_xepsus
			              	      ''') 

		ibes_actual = pd.concat([ibes_actual,ibes_actual2],axis=0)
		ibes_actual.to_pickle(data_input_path+'ibes/ibes_actual.pkl')
		db.close()
	return ibes_actual

def get_crspibes_link():
	#get crsp/ibes linking table from https://wrds-www.wharton.upenn.edu/pages/grid-items/linking-suite-wrds/
	link = pd.read_csv(data_input_path+'linking/crspibes.csv')
	link = link.rename(columns = {'TICKER':'ticker','PERMNO':'permno'})
	link = link.drop_duplicates()
	link['sdate'] = pd.to_datetime(link.sdate)
	link['edate'] = pd.to_datetime(link.edate)
	link = link.groupby(['permno','ticker']).agg({'sdate':'min','edate':'max'}).reset_index()
	return link

def get_inst_hold():
	#download aggregated TR institutional ownership data from WRDS api and save as pandas df
	if os.path.isfile(data_input_path+'tr/13f_agg.pkl'):
		inst_hold = pd.read_pickle(data_input_path+'tr/13f_agg.pkl')
	else:
		inst_hold = db.raw_sql('''
		                      select rdate, cusip, avg(shrout1) shrout1, avg(shrout2) shrout2, sum(shares) shares
		                      from tr_13f.s34
		                      group by cusip, rdate
		                     ''') 
		inst_hold.to_pickle(data_input_path+'tr/13f_agg.pkl')
		db.close()
	return inst_hold

def get_industry_sic_map():
	#map from SIC codes to Fama-French industries (from Kenneth French's data website)
	sic_ranges = [list(range(100,200))+list(range(200,300))+list(range(700,800))+list(range(910,919))+list(range(2048,2049)),
					   list(range(2000,2047))+list(range(2050,2064))+list(range(2070,2080))+list(range(2090,2093))+list(range(2095,2096))+list(range(2098,2100)),
					   list(range(2064,2069))+list(range(2086,2088))+list(range(2096,2098)),
					   list(range(2080,2081))+list(range(2082,2086)),
					   list(range(2100,2200)),
					   list(range(920,1000))+list(range(3650,3653))+list(range(3732,3733))+list(range(3930,3932))+list(range(3940,3950)),
					   list(range(7800,7834))+list(range(7840,7842))+list(range(7900,7901))+list(range(7910,7912))+list(range(7920,7934))+list(range(7940,7950))+list(range(7980,7981))+list(range(7990,8000)),
					   list(range(2700,2750))+list(range(2770,2772))+list(range(2780,2800)),
					   list(range(2047,2048))+list(range(2391,2393))+list(range(2510,2520))+list(range(2590,2600))+list(range(2840,2845))+list(range(3160,3162))+list(range(3170,3173))+list(range(3190,3200))+list(range(3229,3230))+list(range(3260,3261))+list(range(3262,3264))+list(range(3269,3270))+list(range(3230,3232))+list(range(3630,3640))+list(range(3750,3752))+list(range(3800,3801))+list(range(3860,3862))+list(range(3870,3874))+list(range(3910,3912))+list(range(3914,3916))+list(range(3960,3963))+list(range(3991,3992))+list(range(3995,3996)),
					   list(range(2300,2391))+list(range(3020,3022))+list(range(3100,3112))+list(range(3130,3132))+list(range(3140,3152))+list(range(3963,3966)),
					   list(range(8000,8100)),
					   list(range(3693,3694))+list(range(3840,3852)),
					   list(range(2830,2832))+list(range(2833,2837)),
					   list(range(2800,2830))+list(range(2850,2880))+list(range(2890,2900)),
					   list(range(3031,3032))+list(range(3041,3042))+list(range(3050,3054))+list(range(3060,3100)),
					   list(range(2200,2285))+list(range(2290,2296))+list(range(2297,2300))+list(range(2393,2396))+list(range(2397,2399)),
					   list(range(800,900))+list(range(2400,2440))+list(range(2450,2460))+list(range(2490,2500))+list(range(2660,2662))+list(range(2950,2953))+list(range(2950,2953))+list(range(3200,3201))+list(range(3210,3212))+list(range(3240,3242))+list(range(3250,3260))+list(range(3261,3262))+list(range(3264,3265))+list(range(3270,3276))+list(range(3280,3282))+list(range(3290,3294))+list(range(3295,3300))+list(range(3420,3443))+list(range(3446,3447))+list(range(3448,3453))+list(range(3490,3500))+list(range(3996,3997)),
					   list(range(1500,1512))+list(range(1520,1550))+list(range(1600,1800)),
					   list(range(3300,3301))+list(range(3310,3318))+list(range(3320,3326))+list(range(3330,3342))+list(range(3350,3358))+list(range(3360,3380))+list(range(3390,3400)),
					   list(range(3400,3401))+list(range(3443,3445))+list(range(3460,3480)),
					   list(range(3510,3537))+list(range(3538,3539))+list(range(3540,3570))+list(range(3580,3583))+list(range(3585,3587))+list(range(3589,3600)),
					   list(range(3600,3601))+list(range(3610,3614))+list(range(3620,3622))+list(range(3623,3630))+list(range(3640,3647))+list(range(3648,3650))+list(range(3660,3661))+list(range(3690,3693))+list(range(3699,3700)),
					   list(range(2296,2297))+list(range(2396,2397))+list(range(3010,3012))+list(range(3537,3538))+list(range(3647,3648))+list(range(3694,3695))+list(range(3700,3701))+list(range(3710,3712))+list(range(3713,3717))+list(range(3790,3793))+list(range(3799,3800)),
					   list(range(3720,3722))+list(range(3723,3726))+list(range(3728,3730)),
					   list(range(3730,3732))+list(range(3740,3744)),
					   list(range(3760,3770))+list(range(3795,3796))+list(range(3480,3490)),
					   list(range(1040,1050)),
					   list(range(1000,1040))+list(range(1050,1120))+list(range(1400,1500)),
					   list(range(1200,1300)),
					   list(range(1300,1301))+list(range(1310,1340))+list(range(1370,1383))+list(range(1389,1390))+list(range(2900,2913))+list(range(2990,3000)),
					   list(range(4900,4901))+list(range(4910,4912))+list(range(4920,4926))+list(range(4930,4933))+list(range(4939,4943)),
					   list(range(4800,4801))+list(range(4810,4814))+list(range(4820,4823))+list(range(4830,4842))+list(range(4880,4893))+list(range(4899,4900)),
					   list(range(7020,7022))+list(range(7030,7034))+list(range(7200,7201))+list(range(7210,7213))+list(range(7214,7215))+list(range(7215,7218))+list(range(7219,7222))+list(range(7230,7232))+list(range(7240,7242))+list(range(7250,7252))+list(range(7260,7300))+list(range(7395,7396))+list(range(7500,7501))+list(range(7520,7550))+list(range(7600,7601))+list(range(7620,7621))+list(range(7620,7621))+list(range(7622,7624))+list(range(7629,7632))+list(range(7640,7641))+list(range(7690,7700))+list(range(8100,8500))+list(range(8600,8700))+list(range(8800,8900))+list(range(7510,7516)),
					   list(range(2750,2760))+list(range(3993,3994))+list(range(7218,7219))+list(range(7300,7301))+list(range(7310,7343))+list(range(7349,7354))+list(range(7359,7373))+list(range(7374,7386))+list(range(7389,7395))+list(range(7396,7398))+list(range(7399,7400))+list(range(7519,7420))+list(range(8700,8701))+list(range(8710,8714))+list(range(8720,8722))+list(range(8730,8735))+list(range(8740,8749))+list(range(8900,8912))+list(range(8920,9000))+list(range(4220,4230)),
					   list(range(3570,3580))+list(range(3680,3690))+list(range(3695,3696))+list(range(7373,7374)),
					   list(range(3622,3623))+list(range(3661,3667))+list(range(3669,3680))+list(range(3810,3811))+list(range(3812,3813)),
					   list(range(3811,3812))+list(range(3820,3828))+list(range(3829,3840)),
					   list(range(2520,2550))+list(range(2600,2640))+list(range(2670,2700))+list(range(2760,2762))+list(range(3950,3956)),
					   list(range(2440,2450))+list(range(2640,2660))+list(range(3220,3222))+list(range(3410,3413)),
					   list(range(4000,4014))+list(range(4040,4050))+list(range(4100,4101))+list(range(4110,4122))+list(range(4130,4132))+list(range(4140,4143))+list(range(4150,4152))+list(range(4170,4174))+list(range(4190,4201))+list(range(4210,4220))+list(range(4230,4232))+list(range(4240,4250))+list(range(4400,4701))+list(range(4710,4713))+list(range(4720,4750))+list(range(4780,4781))+list(range(4782,4786))+list(range(4789,4790)),
					   list(range(5000,5001))+list(range(5010,5016))+list(range(5020,5024))+list(range(5030,5061))+list(range(5063,5066))+list(range(5070,5089))+list(range(5080,5089))+list(range(5090,5095))+list(range(5099,5101))+list(range(5110,5114))+list(range(5120,5123))+list(range(5130,5173))+list(range(5180,5183))+list(range(5190,5200)),
					   list(range(5200,5201))+list(range(5210,5232))+list(range(5250,5252))+list(range(5260,5262))+list(range(5270,5272))+list(range(5300,5301))+list(range(5310,5312))+list(range(5320,5321))+list(range(5330,5332))+list(range(5334,5335))+list(range(5340,5350))+list(range(5390,5401))+list(range(5410,5413))+list(range(5420,5470))+list(range(5490,5501))+list(range(5510,5580))+list(range(5590,5701))+list(range(5710,5723))+list(range(5730,5737))+list(range(5750,5800))+list(range(5900,5901))+list(range(5910,5913))+list(range(5920,5933))+list(range(5940,5991))+list(range(5992,5996))+list(range(5999,6000)),
					   list(range(5800,5830))+list(range(5890,5900))+list(range(7000,7001))+list(range(7010,7020))+list(range(7040,7050))+list(range(7213,7214)),
					   list(range(6000,6001))+list(range(6010,6037))+list(range(6040,6063))+list(range(6080,6083))+list(range(6090,6101))+list(range(6110,6114))+list(range(6120,6180))+list(range(6190,6199)),
					   list(range(6300,6301))+list(range(6310,6332))+list(range(6350,6352))+list(range(6360,6362))+list(range(6370,6380))+list(range(6390,6412)),
					   list(range(6500,6501))+list(range(6510,6511))+list(range(6512,6516))+list(range(6517,6533))+list(range(6540,6542))+list(range(6550,6554))+list(range(6590,6600))+list(range(6610,6612)),
					   list(range(6200,6300))+list(range(6700,6701))+list(range(6710,6727))+list(range(6730,6734))+list(range(6740,6780))+list(range(6790,6796))+list(range(6798,6800)),
					   list(range(4950,4962))+list(range(4970,4972))+list(range(4990,4992))]
	industry_map = {}
	for i,sic_range in enumerate(sic_ranges):
		industry_map.update({x:i for x in sic_range})
	return industry_map

if __name__=='__main__':
	df = get_characteristics()