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


def computeFeatures(data_bids):
	#Convert into a matrix
	data_bids = data_bids.groupby('bidder_id')

	bidder_features  = dict();

	for name,group in data_bids:
		auctions  			    = group['auction'].value_counts(ascending=False,sort=True,dropna=True);

		#Auction Features
		no_of_distinct_auctions      = len(auctions.keys());
		min_bids_in_an_auction       = np.min(auctions.values);
		max_bids_in_an_auction       = np.max(auctions.values);
		avg_bids_in_an_auction       = np.mean(auctions.values);
		median_bids_in_an_auction    = np.median(auctions.values);
		std_bids_in_an_auction       = np.std(auctions.values);
		no_auctions_with_single_bid  = len(auctions.values[auctions.values==1]);
		per_auctions_with_single_bid = no_auctions_with_single_bid / float(len(auctions.values));




		device      = len(group['device'].value_counts().keys());
		merchandise = len(group['merchandise'].value_counts().keys());
		country 	= len(group['country'].value_counts().keys());

		print((country, auction, device, merchandise))

		break;









if __name__ == '__main__':
	warnings.filterwarnings("ignore");
	train, test, train_bids, test_bids = read("./data/train.csv", "./data/test.csv", "./data/bids.csv");

	computeFeatures(train_bids);