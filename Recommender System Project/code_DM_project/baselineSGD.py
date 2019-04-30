from surprise import Dataset, Reader, BaselineOnly
import pandas as pd

train_rating_df = pd.read_csv("train_rating.txt", header=0, index_col=0)
test = pd.read_csv('test_rating.txt', header=0, index_col=0)
test['dummy_rating'] = '-1' 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_rating_df[['user_id', 'business_id', 'rating']], reader)
trainset = data.build_full_trainset()
bsl_options = {'method':'sgd','reg':0.08,'n_epochs':50,'learning_rate':0.0035, 'reg_u':0.07, 'reg_i':0.2}
algo=BaselineOnly(bsl_options=bsl_options)
algo.train(trainset)
testdata = Dataset.load_from_df(test[['user_id', 'business_id', 'dummy_rating']], reader)
predictions = algo.test(testdata.construct_testset(raw_testset=testdata.raw_ratings))
df = pd.DataFrame(predictions)
newdf = df['est']
newdf.rename('rating', inplace=True)
newdf.to_csv('submission.csv',header='rating',index_label='test_id')