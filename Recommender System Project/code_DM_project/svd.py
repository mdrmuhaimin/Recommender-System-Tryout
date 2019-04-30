from surprise import Dataset, SVD, Reader
import pandas as pd

train_rating_df = pd.read_csv("train_rating.txt", header=0, index_col=0)
test = pd.read_csv('test_rating.txt', header=0, index_col=0)
test['dummy_rating'] = '-1'
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_rating_df[['user_id', 'business_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD(lr_all=0.0035, reg_all=0.04, n_factors=200, lr_bu=0.01, lr_bi=0.01)
algo.train(trainset)
testdata = Dataset.load_from_df(test[['user_id', 'business_id', 'dummy_rating']], reader)
predictions = algo.test(testdata.construct_testset(raw_testset=testdata.raw_ratings))
df = pd.DataFrame(predictions)
newdf = df['est']
newdf.rename('rating', inplace=True)
newdf.to_csv('submission.csv',header='rating',index_label='test_id')