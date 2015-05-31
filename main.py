import pandas as pd;
import sys;
import numpy as np;
import warnings;

def read(train_file, test_file, bids_file):
	train = pd.read_csv(train_file);
	test  = pd.read_csv(test_file);
	bids  = pd.read_csv(bids_file);

	train_bids = pd.merge(train, bids, on='bidder_id', how='inner');
	train_bids.drop(["outcome", "payment_account", "address"],inplace=True,axis=1);

	test_bids  = pd.merge(test, bids, on='bidder_id', how='inner');
	test_bids.drop(["payment_account", "address"],inplace=True,axis=1);

	return train, test, train_bids, test_bids;

def getStats(grouped):
	return len(grouped.keys()), \
		   np.min(grouped.values), \
		   np.max(grouped.values), \
		   np.mean(grouped.values), \
		   np.median(grouped.values), \
		   np.std(grouped.values), \
		   len(grouped.values[grouped.values==1]), \
		   len(grouped.values[grouped.values==1]) / float(len(grouped.values));



def computeFeatures(data_bids):
	#Convert into a matrix
	data_bids = data_bids.groupby('bidder_id')

	bidder_features  = dict();

	for name,group in data_bids:
		auctions  			    = group['auction'].value_counts(ascending=False,sort=True,dropna=True);
		devices  			    = group['device'].value_counts(ascending=False,sort=True,dropna=True);
		merchandise  			= group['merchandise'].value_counts(ascending=False,sort=True,dropna=True);
		countries  			    = group['country'].value_counts(ascending=False,sort=True,dropna=True);
		ips		  			    = group['ip'].value_counts(ascending=False,sort=True,dropna=True);
		urls  					= group['url'].value_counts(ascending=False,sort=True,dropna=True);

		auction_devices         = group[['auction','device']].groupby('auction');
		auction_countries       = group[['auction','country']].groupby('auction');
		auction_ips             = group[['auction','ip']].groupby('auction');

		#Auction Features
		no_of_distinct_auctions, min_bids_in_an_auction, \
		max_bids_in_an_auction, avg_bids_in_an_auction, \
		median_bids_in_an_auction, std_bids_in_an_auction, \       
		no_auctions_with_single_bid, per_auctions_with_single_bid \
		= getStats(auctions);


		#Device Features
		no_of_distinct_devices, min_bids_from_device, \
		max_bids_from_device, avg_bids_from_device, \
		median_bids_from_device, std_bids_from_device, \       
		no_devices_with_single_bid, per_devices_with_single_bid \
		= getStats(devices);


		#Merchandize Features
		no_of_distinct_merchandize, min_bids_for_merchandize, \
		max_bids_for_merchandize, avg_bids_for_merchandize, \
		med_bids_for_merchandize, std_bids_for_merchandize, \       
		no_merch_with_single_bid, per_merch_with_single_bid \
		= getStats(merchandise);



		#Countries Features
		no_of_distinct_countries, min_bids_from_countries, \
		max_bids_from_countries, avg_bids_from_countries, \
		med_bids_from_countries, std_bids_from_countries, \       
		no_countries_with_single_bid, per_countries_with_single_bid \
		= getStats(countries);

		#IPS Features
		no_of_distinct_ips, min_bids_from_ip, \
		max_bids_from_ip, avg_bids_from_ip, \
		med_bids_from_ip, std_bids_from_ip, \       
		no_ip_with_single_bid, per_ip_with_single_bid \
		= getStats(ips);


		#URLS Features
		no_of_distinct_ips, min_bids_from_ip, \
		max_bids_from_ip, avg_bids_from_ip, \
		med_bids_from_ip, std_bids_from_ip, \       
		no_ip_with_single_bid, per_ip_with_single_bid \
		= getStats(ips);


		per_auction_devices_list = [len(d['device'].unique()) for a,d in auction_devices]
		per_auction_country_list = [len(c['country'].unique()) for a,c in auction_countries]
		per_auction_ip_list 	 = [len(i['ip'].unique()) for a,i in auction_ips]

		#Features
		min_diff_devices_used_in_auction = np.min(per_auction_devices_list);
		max_diff_devices_used_in_auction = np.max(per_auction_devices_list);
		med_diff_devices_used_in_auction = np.median(per_auction_devices_list);

		#Features
		min_diff_countries_used_in_auction = np.min(per_auction_country_list);
		max_diff_countries_used_in_auction = np.max(per_auction_country_list);
		med_diff_countries_used_in_auction = np.median(per_auction_country_list);

		#Features
		min_diff_ip_used_in_auction = np.min(per_auction_ip_list);
		max_diff_ip_used_in_auction = np.max(per_auction_ip_list);
		med_diff_ip_used_in_auction = np.median(per_auction_ip_list);






if __name__ == '__main__':
	warnings.filterwarnings("ignore");
	train, test, train_bids, test_bids = read("./data/train.csv", "./data/test.csv", "./data/bids.csv");

	computeFeatures(train_bids);