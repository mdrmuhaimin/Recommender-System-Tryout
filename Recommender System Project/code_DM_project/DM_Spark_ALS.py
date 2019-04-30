# Based on documentation provided my Spark MLLib official documentation

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions, types
from datetime import datetime

spark = SparkSession.builder.appName('DM_Coll_Fil').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
def main():
    convert_to_int = functions.udf(lambda value: float(value), types.FloatType())
    get_timestamp = functions.udf(lambda datestring: datetime.strptime(datestring, '%Y-%m-%d').timestamp(), types.LongType())
    fit_in_range_lower =  functions.udf(lambda value: 1.0 if value < 1 else value, types.FloatType())
    fit_in_range_higher =  functions.udf(lambda value: 5.0 if value > 5 else value, types.FloatType())

    training_data = spark.read.csv('train_rating.txt', header=True)
    test_data = spark.read.csv('test_rating.txt', header=True).drop('date')

    for column in training_data.columns:
        if(column != 'date'):
            training_data = training_data.withColumn(column, convert_to_int(training_data[column]))
        else:
            training_data = training_data.withColumn(column, get_timestamp(training_data[column]))

    for column in test_data.columns:
        if (column != 'date'):
            test_data = test_data.withColumn(column, convert_to_int(test_data[column]))
        else:
            test_data = test_data.withColumn(column, get_timestamp(test_data[column]))

    (training, test) = training_data.randomSplit([0.8, 0.2])

    als = ALS(maxIter=10, regParam=0.5, userCol="user_id", itemCol="business_id", ratingCol="rating",
              coldStartStrategy="drop", alpha=1, rank=40)
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    test_predictions = model.transform(test)
    test_predictions = test_predictions.withColumn('prediction', fit_in_range_lower(test_predictions['prediction']))
    test_predictions = test_predictions.withColumn('prediction', fit_in_range_higher(test_predictions['prediction']))


    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(test_predictions)
    print("Root-mean-square error = " + str(rmse))
    rmse_string = "Root-mean-square error = {}\\n".format(str(rmse))
    with open("result_{}.txt".format(datetime.now().isoformat()), "a") as myfile:
        myfile.write(rmse_string)
    #Uncomment when you are satisfied with training
    # final_pred = model.transform(test_data)
    # final_pred = final_pred.withColumnRenamed('prediction', 'rating')
    # final_pred.select("test_id","rating").toPandas().to_csv('submission.csv', sep=',', encoding='utf-8',index=False)
    return

if __name__=='__main__':
    # Note: in current version output is only used for debugging
    main()
