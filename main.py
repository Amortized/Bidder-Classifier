import pandas as pd;
import sys;
import numpy as np;
import warnings;
import math;
import model;



def read(train_file, test_file, bids_file):
	train = pd.read_csv(train_file);
	test  = pd.read_csv(test_file);
	bids  = pd.read_csv(bids_file);

	train_bids = pd.merge(train, bids, on='bidder_id', how='inner');
	train_bids.drop(["outcome", "payment_account", "address"],inplace=True,axis=1);

	test_bids  = pd.merge(test, bids, on='bidder_id', how='inner');
	test_bids.drop(["payment_account", "address"],inplace=True,axis=1);

	test_bidders_without_bids      = pd.merge(test, bids, on='bidder_id', how='left');
	test_bidders_ids_without_bids  = test_bidders_without_bids[test_bidders_without_bids.bid_id.isnull()]['bidder_id'].values 

	#Prepare a dict
	label_train = dict(zip(list(train.bidder_id), list(train.outcome) ) );

	return label_train, train_bids, test_bids, test_bidders_ids_without_bids;

def getStats(grouped):
	try :

		return len(grouped.keys()), \
			   np.min(grouped.values), \
			   np.max(grouped.values), \
			   np.mean(grouped.values), \
			   np.median(grouped.values), \
			   np.std(grouped.values), \
			   len(grouped.values[grouped.values==1]), \
			   len(grouped.values[grouped.values==1]) / float(len(grouped.values));

	except :

		return  "NaN",\
				"NaN",\
				"NaN",\
				"NaN",\
				"NaN",\
				"NaN",\
				"NaN",\
				"NaN";


def computeFeatures(data_bids):
	#Compute auction stats
	auction_group = data_bids.groupby('auction');
	auction_stats = dict();
	for name,group in auction_group:
		auction_stats[name] = group['bidder_id'].value_counts(ascending=False,sort=True,dropna=True).to_dict();


	#Convert into a matrix
	data_bids = data_bids.groupby('bidder_id');

	feature_names = ["no_of_distinct_auctions", "min_bids_in_an_auction", \
					 "max_bids_in_an_auction", "avg_bids_in_an_auction", \
				  	 "median_bids_in_an_auction", "std_bids_in_an_auction", \
				 	 "no_auctions_with_single_bid", "per_auctions_with_single_bid", \
				 	 "no_of_distinct_devices", "min_bids_from_device", \
					 "max_bids_from_device", "avg_bids_from_device", \
					 "median_bids_from_device", "std_bids_from_device", \
					 "no_devices_with_single_bid", "per_devices_with_single_bid", \
					 "no_of_distinct_merchandize", "min_bids_for_merchandize", \
					 "max_bids_for_merchandize", "avg_bids_for_merchandize", \
					 "med_bids_for_merchandize", "std_bids_for_merchandize", \
					 "no_merch_with_single_bid", "per_merch_with_single_bid", \
				 	 "no_of_distinct_countries", "min_bids_from_countries", \
					 "max_bids_from_countries", "avg_bids_from_countries", \
					 "med_bids_from_countries", "std_bids_from_countries", \
					 "no_countries_with_single_bid", "per_countries_with_single_bid", \
					 "no_of_distinct_ips", "min_bids_from_ip", \
					 "max_bids_from_ip", "avg_bids_from_ip", \
					 "med_bids_from_ip", "std_bids_from_ip", \
					 "no_ip_with_single_bid", "per_ip_with_single_bid", \
					 "no_of_distinct_urls", "min_bids_from_url", \
					 "max_bids_from_url", "avg_bids_from_url", \
					 "med_bids_from_url", "std_bids_from_url", \
					 "no_url_with_single_bid", "per_url_with_single_bid", \
					 "min_diff_devices_used_in_auction", "max_diff_devices_used_in_auction", \
					 "med_diff_devices_used_in_auction", \
					 "min_diff_countries_used_in_auction", "max_diff_countries_used_in_auction", \
					 "med_diff_countries_used_in_auction",\
					 "min_diff_ip_used_in_auction","max_diff_ip_used_in_auction",\
					 "med_diff_ip_used_in_auction",\
					 "per_bids_at_distinct_unit_of_time", "bids_at_distinct_unit_of_time",\
					 "max_bids_at_same_unit_of_time","med_bids_at_same_unit_of_time",\
					 "avg_diff_in_time_between_bids","min_diff_in_time_between_bids",\
<<<<<<< HEAD
					 "max_diff_in_time_between_bids","med_diff_in_time_between_bids",
					 "country_with_max_bids_is_risky"];
=======
					 "max_diff_in_time_between_bids","med_diff_in_time_between_bids",\
					 "avg_no_bidders_in_all_my_auctions","avg_my_per_bids_in_all_my_auctions"];
>>>>>>> a686d370514b2d38b099ee0b045944b811b15714

	feature_black_list = ["no_of_distinct_urls", "max_bids_in_an_auction", \
						  "max_diff_ip_used_in_auction", "no_of_distinct_auctions", \
						  "med_diff_ip_used_in_auction", "bids_at_distinct_unit_of_time", \
						  "no_countries_with_single_bid", "max_bids_for_merchandize", \
						  "no_of_distinct_countries", "no_of_distinct_devices", \
						  "median_bids_from_device", "no_of_distinct_ips", \
						  "med_diff_devices_used_in_auction", "median_bids_in_an_auction",\
						  "min_bids_from_url"];


	bidder_features  = dict();


	for name,group in data_bids:
		auctions  			    = group['auction'].value_counts(ascending=False,sort=True,dropna=True);
		devices  			    = group['device'].value_counts(ascending=False,sort=True,dropna=True);
		merchandise  			= group['merchandise'].value_counts(ascending=False,sort=True,dropna=True);
		countries  			    = group['country'].value_counts(ascending=False,sort=True,dropna=True);
		ips		  			    = group['ip'].value_counts(ascending=False,sort=True,dropna=True);
		urls  					= group['url'].value_counts(ascending=False,sort=True,dropna=True);
		times  					= group['time'].value_counts(ascending=False,sort=True,dropna=True);

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
		no_of_distinct_urls, min_bids_from_url, \
		max_bids_from_url, avg_bids_from_url, \
		med_bids_from_url, std_bids_from_url, \
		no_url_with_single_bid, per_url_with_single_bid \
		= getStats(urls);


		per_auction_devices_list = [len(d['device'].unique()) for a,d in auction_devices]
		per_auction_country_list = [len(c['country'].unique()) for a,c in auction_countries]
		per_auction_ip_list 	 = [len(i['ip'].unique()) for a,i in auction_ips]

		try :
			#Features
			min_diff_devices_used_in_auction = np.min(per_auction_devices_list);
			max_diff_devices_used_in_auction = np.max(per_auction_devices_list);
			med_diff_devices_used_in_auction = np.median(per_auction_devices_list);
		except :
			min_diff_devices_used_in_auction = "NaN";
			max_diff_devices_used_in_auction = "NaN";
			med_diff_devices_used_in_auction = "NaN";


		try:	
			#Features
			min_diff_countries_used_in_auction = np.min(per_auction_country_list);
			max_diff_countries_used_in_auction = np.max(per_auction_country_list);
			med_diff_countries_used_in_auction = np.median(per_auction_country_list);
		except :
			#Features
			min_diff_countries_used_in_auction = "NaN";
			max_diff_countries_used_in_auction = "NaN";
			med_diff_countries_used_in_auction = "NaN";

		try:
			#Features
			min_diff_ip_used_in_auction = np.min(per_auction_ip_list);
			max_diff_ip_used_in_auction = np.max(per_auction_ip_list);
			med_diff_ip_used_in_auction = np.median(per_auction_ip_list);
		except:
			#Features
			min_diff_ip_used_in_auction = "NaN";
			max_diff_ip_used_in_auction = "NaN";
			med_diff_ip_used_in_auction = "NaN";


		try:	
			#Time based features
			per_bids_at_distinct_unit_of_time  = len(times.values[times.values==1]) / float(len(times.values))
			bids_at_distinct_unit_of_time      = len(times.values[times.values==1]) 
			max_bids_at_same_unit_of_time      = np.max(times.values)
			med_bids_at_same_unit_of_time      = np.median(times.values);
		except :
			#Time based features
			per_bids_at_distinct_unit_of_time  = "NaN";
			bids_at_distinct_unit_of_time      = "NaN";
			max_bids_at_same_unit_of_time      = "NaN";
			med_bids_at_same_unit_of_time      = "NaN";



		if len(times.keys()) > 1:
			avg_diff_in_time_between_bids      = math.log(0.00001 + np.mean(np.ediff1d(np.sort(times.keys()))));
			min_diff_in_time_between_bids      = math.log(0.00001 + np.min(np.ediff1d(np.sort(times.keys()))));
			max_diff_in_time_between_bids      = math.log(0.00001 + np.max(np.ediff1d(np.sort(times.keys()))));
			med_diff_in_time_between_bids      = math.log(0.00001 + np.median(np.ediff1d(np.sort(times.keys()))));
		else:
			avg_diff_in_time_between_bids      = "NaN";
			min_diff_in_time_between_bids      = "NaN";
			max_diff_in_time_between_bids      = "NaN";
			med_diff_in_time_between_bids      = "NaN";


<<<<<<< HEAD
		try:
			country_with_max_bids_is_risky			       = countries.argmax();
			if country_with_max_bids_is_risky in ["mo", "jp", "kr", "fo", "je", "mp", "tm", "mn", "bs", "dm", "do", "ag", "pf", "de", "ps", "at", "sr", "ca", "au", "tw"]:
				country_with_max_bids_is_risky = 1;
			else:
				country_with_max_bids_is_risky = 0;
		except :
			country_with_max_bids_is_risky	  = 0;

			
=======
		other_bidders_in_auctions = [];	
		#Check stats in various auctions
		for my_auction in auctions.keys():
			if my_auction in auction_stats and name in auction_stats[my_auction].keys():
				no_other_bidders       = len(auction_stats[my_auction].keys()) - 1;
				if no_other_bidders < 0:
					no_other_bidders = 0;
				my_per_bids_in_auction = auction_stats[my_auction][name] / float(sum(auction_stats[my_auction].values())); 
				other_bidders_in_auctions.append( (no_other_bidders,  my_per_bids_in_auction) )

		try :
			avg_no_bidders_in_all_my_auctions  = np.mean([x[0] for x in other_bidders_in_auctions]);
			avg_my_per_bids_in_all_my_auctions = np.mean([x[1] for x in other_bidders_in_auctions]);
		except :
			avg_no_bidders_in_all_my_auctions  = "NaN";
			avg_my_per_bids_in_all_my_auctions = "NaN";
			
		

>>>>>>> a686d370514b2d38b099ee0b045944b811b15714
		#Add the features
		bidder_features[name] = [no_of_distinct_auctions, min_bids_in_an_auction, \
								 max_bids_in_an_auction, avg_bids_in_an_auction, \
							  	 median_bids_in_an_auction, std_bids_in_an_auction, \
							 	 no_auctions_with_single_bid, per_auctions_with_single_bid, \
							 	 no_of_distinct_devices, min_bids_from_device, \
								 max_bids_from_device, avg_bids_from_device, \
								 median_bids_from_device, std_bids_from_device, \
								 no_devices_with_single_bid, per_devices_with_single_bid, \
								 no_of_distinct_merchandize, min_bids_for_merchandize, \
								 max_bids_for_merchandize, avg_bids_for_merchandize, \
								 med_bids_for_merchandize, std_bids_for_merchandize, \
								 no_merch_with_single_bid, per_merch_with_single_bid, \
							 	 no_of_distinct_countries, min_bids_from_countries, \
								 max_bids_from_countries, avg_bids_from_countries, \
								 med_bids_from_countries, std_bids_from_countries, \
								 no_countries_with_single_bid, per_countries_with_single_bid, \
								 no_of_distinct_ips, min_bids_from_ip, \
								 max_bids_from_ip, avg_bids_from_ip, \
								 med_bids_from_ip, std_bids_from_ip, \
								 no_ip_with_single_bid, per_ip_with_single_bid, \
								 no_of_distinct_urls, min_bids_from_url, \
								 max_bids_from_url, avg_bids_from_url, \
								 med_bids_from_url, std_bids_from_url, \
								 no_url_with_single_bid, per_url_with_single_bid, \
								 min_diff_devices_used_in_auction, max_diff_devices_used_in_auction, \
								 med_diff_devices_used_in_auction, \
								 min_diff_countries_used_in_auction, max_diff_countries_used_in_auction, \
								 med_diff_countries_used_in_auction,\
								 min_diff_ip_used_in_auction,max_diff_ip_used_in_auction,\
								 med_diff_ip_used_in_auction,\
								 per_bids_at_distinct_unit_of_time, bids_at_distinct_unit_of_time,\
								 max_bids_at_same_unit_of_time,med_bids_at_same_unit_of_time,\
								 avg_diff_in_time_between_bids,min_diff_in_time_between_bids,\
<<<<<<< HEAD
								 max_diff_in_time_between_bids,med_diff_in_time_between_bids,
								 country_with_max_bids_is_risky];
=======
								 max_diff_in_time_between_bids,med_diff_in_time_between_bids,\
								 avg_no_bidders_in_all_my_auctions,avg_my_per_bids_in_all_my_auctions];
>>>>>>> a686d370514b2d38b099ee0b045944b811b15714

		#Use this to remove blacklisted features						 
		#bidder_features[name] = [bidder_features[name][i] for i in range(0, len(feature_names)) if feature_names[i] not in feature_black_list];
								 
	return bidder_features, feature_names;


if __name__ == '__main__':
	warnings.filterwarnings("ignore");
	label_train, train_bids, test_bids, test_bidders_ids_without_bids = read("./data/train.csv", "./data/test.csv", "./data/bids.csv");

	print("Training Set Features");
	train_bidder_features,feature_names = computeFeatures(train_bids);
	del train_bids;

	train_X = [];
	train_Y = [];
	for key in train_bidder_features.keys():
		train_X.append(train_bidder_features[key]);
		train_Y.append(label_train[key]);

	best_model, imputer, one_hot_encoder = model.train(train_X, train_Y,feature_names);

	del train_bidder_features;

	print("Test Set Features");
	test_bidder_features,feature_names = computeFeatures(test_bids);
	del test_bids;

	test_X   = [];
	test_ids = [];
	for key in test_bidder_features.keys():
		test_ids.append(key);
		test_X.append(test_bidder_features[key]);

	model.predict_and_write(best_model, test_X, test_ids, test_bidders_ids_without_bids, imputer, one_hot_encoder);	

