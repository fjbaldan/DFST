# DFST algorithm

This algorithm implements DFST (Distributed FastShapelet Transform). DFST is the first time series classification algorithm developed for distributed environments (Spark). This algorithm performs a shapelet transform on a data set, trains a Random Forest model on that data, and uses this model to classify a test set that has also been transformed. DFST leverages the low computational complexity of the FastShapelet algorithm to extract these features in distributed environments efficiently.

## Example (ml)

```scala
import org.apache.spark.mllib.classification._

var a= new DFST(train_file_name, // hdfs route to training file
                min_len, // minimum length for FastShapelet search algorithm
                max_len, // maximum length for FastShapelet search algorithm
                step, // amount of increase in the length of subsequences processed between iterations
                Num_partitions, // number of partitions in the training dataset (RDD).
                seed, // random seed
                tree_file_name, // output tree file of FastShapelet search algorithm
                time_file_name, // output time file  FastShapelet search algorithm
                R, // number of random projections
                top_k, // number of best candidate shapelets evaluated by iteration
                Classificator, // 0 for shapelet search only / 1 for DFST
                shapelet_file_name, //  output file with shapelets selected for shapelet transformation
                test_file_name, // hdfs route to test file
                model_file_name, // output file with Random Forest Model
                sc) // input SparkContext

val output = DFST.run() // generates the file f_scala_results. txt with the runtime records and results on the test dataset.
```
