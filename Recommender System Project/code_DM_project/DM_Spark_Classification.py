#
# Original authour Mr. Greg Baker, Senior Lecturer, School of Computing Science, Simon Fraser University
# Modified by, Muhammad Raihan Muhaimin, mmuhaimi@sfu.ca
#


from pyspark.sql import SparkSession, functions, types
from datetime import datetime
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from datetime import datetime
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier


spark = SparkSession.builder.appName('Spark_Neural').getOrCreate()
sc = spark.sparkContext



def get_sq_error(actual, prediction):
    return (actual - prediction) ** 2



def main():
    get_pred_error = functions.udf(get_sq_error, types.FloatType())
    convert_to_int = functions.udf(lambda value: int(value), types.IntegerType())
    get_day_of_week = functions.udf(lambda timestamp: datetime.strptime(timestamp, '%Y-%m-%d').weekday(), types.StringType())
    get_month = functions.udf(lambda timestamp: datetime.strptime(timestamp, '%Y-%m-%d').month,
                                    types.StringType())
    training_data = spark.read.csv('train_rating.txt', header=True)
    test_data = spark.read.csv('test_rating.txt', header=True)

    for column in training_data.columns:
        if(column != 'date'):
            training_data = training_data.withColumn(column, convert_to_int(training_data[column]))
        else:
            training_data = training_data.withColumn('dow', get_day_of_week(training_data[column]))
            training_data = training_data.withColumn('month', get_month(training_data[column]))

    for column in test_data.columns:
        if (column != 'date'):
            test_data = test_data.withColumn(column, convert_to_int(test_data[column]))
        else:
            test_data = test_data.withColumn('dow', get_day_of_week(test_data[column]))
            test_data = test_data.withColumn('month', get_month(test_data[column]))

    training_data = training_data.drop('date')
    test_data = test_data.drop('date')


    discreete_columns = ['dow', 'month']
    string_indexer = [StringIndexer(inputCol='{}'.format(column), outputCol='{}_ind'.format(column)) for column in discreete_columns]
    hot_encoders = [OneHotEncoder(inputCol='{}_ind'.format(column), outputCol='{}_he'.format(column)) for column in discreete_columns]
    vector_assembler = VectorAssembler(inputCols=["user_id", "business_id", "dow_he", "month_he"], outputCol="features")

    rf = RandomForestClassifier(numTrees=25, maxDepth=10, labelCol="rating", seed=42)


    models = [
        ('Rand-forest', Pipeline(stages=string_indexer + hot_encoders + [vector_assembler, rf]))
    ]

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol='rating')

    # split data into training and testing
    train, test = training_data.randomSplit([0.8, 0.2])
    train = train.cache()
    test = test.cache()

    for label, pipeline in models:

        model = pipeline.fit(train)
        predictions = model.transform(test)
        predictions = predictions.withColumn('sq_error', get_pred_error(predictions['rating'], predictions['prediction']))
        rmse_score = predictions.groupBy().avg('sq_error').head()[0]
        # calculate a score
        score = evaluator.evaluate(predictions)
        print(label, rmse_score ** 0.5)

    #Uncomment when you are satisfied with training
    # final_pred = model.transform(test_data)
    # final_pred = final_pred.withColumnRenamed('prediction', 'rating')
    # final_pred.select("test_id","rating").toPandas().to_csv('submission.csv', sep=',', encoding='utf-8',index=False)
    return



if __name__=='__main__':
    main()
