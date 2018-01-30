package org.apache.spark.mllib.classification

/**
  * @author
  * @version 1.0
  */

//Import libraries needed
// Need to import RDD, which is not included in the default shell scala
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/* How to store data contain values or characteristics thereof and the
corresponding label.*/

import org.apache.spark.mllib.regression.LabeledPoint

/* Library that allows the creation of dense vectors (used in the labeled
points) and sparse vectors.*/

import org.apache.spark.mllib.linalg.Vectors

import java.io._

import org.apache.spark.mllib.tree.RandomForest


class FastShapelet(train_file_name: String, min_len: Int, max_len: Int, step: Int, Num_partitions: Int, seed: Int,
                   tree_file_name: String, time_file_name: String, R: Int, top_k: Int, Classificator: Int, shapelet_file_name: String, test_file_name: String, model_file_name: String) extends Serializable {


  /** It shows the possible known bugs own class to consider, when they occur.
    * It lets the user know that problems may be occurring in the program and
    * some possible solutions.
    *
    * @param typ Type of error occurred.
    */
  def error(typ: Int): Unit = {
    typ match {
      case 0 =>
        printf(" Please check the input parameters.\n\n")
        printf(" == Syntax ==\n")
        printf(" shapelet.exe  Train_Data   Number_of_Class  " +
          "Number_of_Instance  Max_Length  Min_Length  Step [R] [top_k] " +
          "[output_tree_file] [output_time_file]\n\n")
        printf(" == Examples ==\n")
        printf(" shapelet.exe  ECG200_TRAIN  2  100  96  1  1  \n")
        printf(" shapelet.exe  ECG200_TRAIN  2  100  96  1  1 10  \n")
        printf(" shapelet.exe  ECG200_TRAIN  2  10   30  20 1 10  " +
          "tree_ECG.txt  time_ECG.txt\n")
        printf("\n")

      case 1 =>
        printf("Error!! Input file  not found.")

      case 2 =>
        printf("Error!! The input Dataset has more class than define. Please" +
          " check.\n")

      case 3 =>
        printf("Error!! Memory can't be allocated!!!\n\n")

      case 4 =>
        printf("Error!! bsf_shapelet is empty.\n")

      case 5 =>
        printf("Error!! Max_Length < Min_Length.\n")

      case 6 =>
        printf("Error!! The input Dataset has more instances than define." +
          " Please check.\n")

      case 7 =>
        printf("Error!! The minimum number of partitions is one." +
          "Please check.\n")
      case 8 =>
        printf("Error!! It is necessary that part of the training set is" +
          " treated as a set of validation (Must be configured with" +
          " Part_Train < 1). Please check.\n")
    }
    sys.exit() //exit(0) //Deprecation warning
  }


  /* Type  Declaration */

  /** It contained and uniquely identifies each word SAX found. It also stores
    * data relevant to that word, useful for further processing.
    */
  //type SAX_id_type = Vector[Tuple2[Int,Int]]


  /** Mapping between the identifier of the time series and the collision
    * counter.
    */
  //type Obj_count_type = scala.collection.mutable.Map[Int,Int]

  type USAX_elm_type = (Vector[Tuple2[Int, Int]], Set[Int], scala.collection.mutable.Map[Int, Int])

  /** Format container classification nodes.
    */
  type Classify_type = scala.collection.mutable.MutableList[Int]

  /** Concatenation of integer values that make a symbolic representation of a
    * SAX word.
    */
  type SAX_word_type = Int

  /** List that contains all pairs of SAX words with their respective updated
    * information.
    */
  type USAX_Map_type = scala.collection.mutable.HashMap[SAX_word_type,
    USAX_elm_type]

  /** Mapping each SAX words with their respective updated information. */
  //var USAX_Map : USAX_Map_type = new collection.mutable.HashMap[SAX_word_type,
  //USAX_elm_type]

  /** It contains the identifier of the time series which has been obtained
    * word SAX and values masked SAX words.
    */
  type Obj_set_type = Set[Int]

  /** Pair consisting of a SAX word and the set of identifiers time series
    * which is obtained.
    */
  type Hash_Mark_type = scala.collection.mutable.HashMap[SAX_word_type,
    Obj_set_type]


  /** Distancia, Indicador
    */
  type NN_ED = (Array[Double], Array[Int])

  /* Global vars  Declaration */
  /** Number of classes present in the dataset introduced. */
  var num_class: Int = 0

  /** Number of objects (time series) that exist in the dataset introduced. */
  var num_obj: Int = 0

  /** Entropy for the given class. */
  var class_entropy = 0.0

  /** List of processed elements in each node. */
  //var Node_Obj_List = new Node_Obj_set_type with Serializable
  var Node_Obj_List: Array[RDD[Long]] = new Array(0)

  /** Identifies a non-leaf nodes (-1) and contains labels leaf nodes. */
  var Classify_list = new Classify_type

  /** It contains the mapping from the new labels to the original. */
  var Real_Class_Label = Map[Int, Int]()

  /** It contains the mapping from the original labels to the new labels. */
  var Relabel = Map[Int, Int]()

  /** It contains the list of Shapelets selected so far. */
  var Final_Sh = new scala.collection.mutable.MutableList[Shapelet]

  /** Contiene un listado con la mejor shapelet de cada longitud, en caso de que se vayan a usar como trasnformadores. */
  var Final_Sh_aux = new scala.collection.mutable.MutableList[Shapelet]

  /** Minimum number of objects per partition (internally in the algorithm,
    * not with respect to parallelization).
    */
  var MIN_OBJ_SPLIT: Int = 0

  /** Start Time algorithm. */
  var t0_0: Double = 0

  /** End Time algorithm. */
  var t0_1: Double = 0

  /** Start Time local algorithm. */
  //var t0:Double = 0

  /** The time of the end of the algorithm execution. */
  //var t6:Double = 0

  /** Runtime CreateSAXList. */
  var s1: Double = 0

  /** Runtime RandomProjection. */
  var s2: Double = 0

  /** Runtime ScoreAllSAX. */
  var s3: Double = 0

  /** Runtime de FindBestSAX. */
  var s4: Double = 0

  /** Random number generator used in the environment. */
  var rand: scala.util.Random with java.io.Serializable = new scala.util.Random(seed: Int) with Serializable

  //Default Parameters
  /** Minimum length of Shapelet. */
  val SH_MIN_LEN = 5
  /** Variable control errors cleaning mode. */
  val DEBUG = 0
  /** Maximum allowable purity division. */
  val MAX_PURITY_SPLIT = 0.90
  /** Minimum percentage of total division objects. */
  val MIN_PERCENT_OBJ_SPLIT = 0.1
  /** Extra Depth allowed for the tree created. */
  val EXTRA_TREE_DEPTH = 2

  //  Shapelet Selection for Shapelet Transform
  var list_ST: scala.collection.mutable.ListBuffer[(Int, Array[Double], Int, Int, Int, Double)] = scala.collection.mutable.ListBuffer.empty
  var list_ST_selected: scala.collection.mutable.ListBuffer[(Int, Array[Double], Int, Int, Int, Double)] = scala.collection.mutable.ListBuffer.empty

  /** Calculates the distance between the candidate (Q) and the points of the
    * sequence (T) listed in "order" sequence. Discarding the current
    * calculation for exceeding the threshold specified in best_so_far (avoid
    * unnecessary calculations)
    *
    * @param Q           Sequence candidate.
    * @param order       List of positions (points) of the original time series to
    *                    compare with the candidate subsequence. They should indicate the time
    *                    series subsequence of equal size to the candidate subsequence.
    * @param T           Time series with the candidate subsequence is compared. The time
    *                    series is concatenated with itself once, that is, T has a length twice as
    *                    the original time series.
    * @param j           Subscript scroll to consider to take the values of the time
    *                    series subsequence of the dataset on which the candidate will be compared.
    * @param m           Candidate subsequence length.
    * @param mean        Average values of the T time series.
    * @param std         Standard deviation of the values of the T sequence.
    * @param best_so_far Best distance value obtained so far.
    * @return The Euclidean distance value.
    * @note Euclidean distance between the candidate and the subsequence of
    *       equal size to the candidate of the time series processed is calculated.
    */
  def distance(Q: Array[Double], order: Array[Int], T: Array[Double], j: Int, m: Int,
               mean: Double, std: Double, best_so_far: Double = 1E20): Double = {

    var i: Int = 0
    var sum: Double = 0
    val bsf2: Double = best_so_far * best_so_far
    var x: Double = 0.0

    while (i < m && sum < bsf2) {
      x = (T(order(i) + j) - mean) / std
      val a = (x - Q(i))
      sum += a * a
      i += 1
    }
    Math.sqrt(sum)
  }

  /** Implementation of the algorithm "quicksort" on lists.
    *
    * @param coll     List containing the items to order.
    * @param lessThan Tuple of two items on the list to compare.
    * @return A list containing the of organization changes on the entries in
    *         the input tuple. The values greater than the pivot go to one side thereof
    *         and the other children.
    * @note The use of this function is fully recursive.
    */
  //http://rosettacode.org/wiki/Sorting_algorithms/Quicksort#Scala
  def quicksortFunc[T](coll: List[T], lessThan: (T, T) => Boolean): List[T] =
    if (coll.isEmpty) {
      coll
    } else {
      val (smaller, bigger) = coll.tail partition (lessThan(_, coll.head))
      quicksortFunc(smaller, lessThan) ::: coll.head :: quicksortFunc(bigger,
        lessThan)
    }

  /** Calculating the shortest distance between the candidate subsequence and
    * the time series of the given dataset.
    *
    * @param query  ListBuffer that contains the values of the data of the
    *               candidate.
    * @param data   Vector that contains the values of the time series data taken
    *               from the dataset on which will compare the candidate.
    * @param obj_id Variable used as a flag to initialize and enter the values
    *               of the candidate in Q.
    * @return A 3-tuple where the first element is the shortest distance
    *         between the candidate subsequence and the time series of the given
    *         dataset, the second element contains the ListBuffer Q_aux which is an
    *         updated version of the Q input and the third element contains the
    *         ListBuffer order_aux which is an updated version of the input order.
    * @note The shortest distance between the candidate subsequence and the
    *       processed time series is obtained. Including Z-normalized data of the
    *       candidate in Q and indexes data from the candidate in order. It is
    *       necessary to return the updated Q values and order to properly dispose of
    *       the normalized input data values. In the second and subsequent execution
    *       must enter the values of Q and order date and algorithm parameters.
    */
  def NearestNeighborSearch(query: Array[Double], data: org.apache.spark.mllib.linalg.Vector,
                            obj_id: Int, nn_ed: NN_ED, control: Int): (Double, Array[Double], Array[Int]) = {
    var d: Double = 0
    var i: Int = 0
    var j: Int = 0
    var ex: Double = 0
    var ex2: Double = 0
    var mean: Double = 0
    var std: Double = 0
    var loc: Int = 0

    /** < Length Shapelet. */
    var m: Int = query.size
    /** < Length of each time series. */
    var M: Int = data.size

    var bsf: Double = 1E20

    var Q_obj = nn_ed._1
    var order_obj = nn_ed._2

    /*Variable auxiliary local necessary for normalization and calculation
    nearest neighbor, because the input parameters are not modifiable.*/
    var Q_aux: Array[Double] = Q_obj
    var order_aux: Array[Int] = order_obj
    /*The standardization is only done the first time, making it necessary to
    change the Q value outside the function.*/

    //Obtaining candidate with Z-normalized values.
    if (control == 0) {
      if (obj_id == 0) {
        while (i < m) {
          d = query(i)
          ex += d
          ex2 += d * d
          Q_aux(i) = d
          i += 1
        }

        mean = ex / m;
        std = ex2 / m;
        std = math.sqrt(std - mean * mean);

        i = 0
        while (i < m) {
          Q_aux(i) = ((Q_aux(i) - mean) / std)
          i += 1
        }

        var Q_tmp: Array[(Int, Double)] = Array.fill(m)((0, 0.0))

        if (Q_tmp.size == 0)
          error(3)

        i = 0
        while (i < m) {
          Q_tmp(i) = ((i, Q_aux(i)))
          i += 1
        }

        Q_tmp = Q_tmp.sortBy(_._2)

        /*Obtain the values of the candidate and their indexes.*/
        i = 0
        while (i < m) {
          Q_aux(i) = Q_tmp(i)._2
          order_aux(i) = Q_tmp(i)._1
          i += 1
        }
        Q_obj = Q_aux
        order_obj = order_aux
      }
    }

    i = 0
    j = 0
    ex = 0
    ex2 = 0
    var T: Array[Double] = Array.fill(2 * m)(0.0)
    if (T.isEmpty == true) error(3)
    var dist: Double = 0

    while (i < M) {
      /*Generates substring of m size in T, repeated 2 times, with time
      series data contained in the dataset (Each execution of this function
      takes a time series). */
      d = data(i)
      ex += d
      ex2 += d * d
      T(i % m) = d
      T((i % m) + m) = d
      /*When each substring is completed it passes to obtain the distance
      between it and the candidate. Saving should get a far better case the
      value and location of the case (starting point). */
      if (i >= m - 1) {
        mean = ex / m
        std = ex2 / m
        std = math.sqrt(std - mean * mean)
        j = (i + 1) % m
        dist = distance(Q_aux, order_aux, T, j, m, mean, std, bsf)
        if (dist < bsf) {
          bsf = dist
          loc = i - m + 1
        }
        ex -= T(j)
        ex2 -= T(j) * T(j)
      }
      i += 1
    }
    return (bsf, Q_obj, order_obj);
  }

  /** Gets a SAX word of given length for a given subsequence.
    *
    * @param sum_segment ListBuffer that contains the sum of the values of the
    *                    time series at each point, these values are within the window at all
    *                    times.
    * @param elm_segment ListBuffer that contains the values of the window size
    *                    at each point.
    * @param mean        Average value of sum_segment ListBuffer.
    * @param std         Standard deviation of sum_segment ListBuffer.
    * @param sax_len     Desired length for the resulting SAX word.
    * @return An object of type SAX_word_type which is an integer (concatenation
    *         of integers that represent the symbolic approach achieved).
    */
  def CreateSAXWord(sum_segment: Array[Double], elm_segment: Array[Int],
                    mean: Double, std: Double, sax_len: Int): Int = {
    var word: SAX_word_type = 0
    var value = 0
    var i = 0
    while (i < sax_len) {
      var d = (sum_segment(i) / elm_segment(i) - mean) / std
      /*Obtaining different symbols based on the values for each segment.*/
      if (d < 0)
        if (d < -0.67) value = 0
        else value = 1
      else if (d < 0.67) value = 2
      else value = 3
      word = (word << 2) | (value)
      i += 1
    }
    word
  }


  /** Creates a mask to apply to SAX words, in order to reduce the quadratic
    * growth in the number of words SAX and the possibility of false discards.
    *
    * @param num_mask Number of values to mask.
    * @param word_len Length of SAX words on which applies.
    * @return The mask created.
    */
  def CreateMaskWord(num_mask: Int, word_len: Int):
  SAX_word_type = {
    //Create mask word (two random may give same position, we ignore it)
    var a, b: SAX_word_type = 0
    var i = 0
    while (i < num_mask) {
      b = 1 << (rand.nextInt(word_len))
      a = a | b
      i += 1
    }
    a
  }

  def CreateSAXList(ts_input: (Long, org.apache.spark.mllib.regression.LabeledPoint),
                    subseq_len_aux: Int, sax_len: Int, w: Int): USAX_Map_type = {
    var ex, ex2, mean, std: Double = 0.0
    var sum_segment: Array[Double] = Array.fill(sax_len)(0.0)
    var obj_id, j, j_st, k, slot: Int = 0
    var d: Double = 0
    var word, prev_word: SAX_word_type = 0

    var USAX_Map_local: USAX_Map_type = new collection.mutable.HashMap[SAX_word_type,
      USAX_elm_type]

    /*The size of the sliding window is saved to be considered at each
    point of the subsequence studied.*/
    var elm_segment: Array[Int] = Array.fill(sax_len)(w)
    elm_segment(sax_len - 1) = subseq_len_aux - (sax_len - 1) * w

    ex = 0
    ex2 = 0
    prev_word = -1
    sum_segment = Array.fill(sax_len)(0.0)
    // For any time series:
    // Case 1: Initial (first subsequence)
    var j_aux = 0

    while (j < ts_input._2.features.size && j < subseq_len_aux) {
      d = ts_input._2.features(j)
      ex += d //Sum of all the values considered within the subsequence.
      ex2 += d * d

      //Lower integer value, integer part and rounded down.
      slot = Math.floor((j) / w).toInt

      /*Sum of the values of the time series contained within a single
      segment. Each window has its own slot.*/
      sum_segment(slot) += d

      /*Save the index of the iteration which produces the output of the loop,
      and from which the following is started.*/
      j_aux = j + 1
      j += 1
    }
    // Case 2: Slightly Update (For the rest of subsequences contained in
    //the time series (least first) )
    j = j_aux
    while (j <= ts_input._2.features.size) {
      //j is the position in time series and j_st is the inicial
      //position of the subsequence processed
      j_st = j - subseq_len_aux
      mean = ex / subseq_len_aux
      std = Math.sqrt(ex2 / subseq_len_aux - mean * mean)

      word = 0
      var value = 0
      var i = 0
      while (i < sax_len) {
        var d = (sum_segment(i) / elm_segment(i) - mean) / std
        /*Obtaining different symbols based on the values for each segment.*/
        if (d < 0)
          if (d < -0.67) value = 0
          else value = 1
        else if (d < 0.67) value = 2
        else value = 3
        word = (word << 2) | (value)
        i += 1
      }
      word

      if (word != prev_word) {
        prev_word = word
        /* In this direction new words are added, with their respective
        characteristic structure. The USAX_elm_type object is created and
        stored in the HashMap that contains them. If the desired entry does
        not exist create a new one. If there is the desired values are added.
        */
        if (USAX_Map_local.contains(word) == false) {
          USAX_Map_local += ((word, (Vector(), Set(), scala.collection.mutable.Map[Int, Int]())))
        }
        /*Structures "Set" no duplicate values are stored, so that the
        operation is the desired.*/
        var aux = (word, USAX_Map_local(word))

        var usax_aux: USAX_elm_type = (USAX_Map_local(word))
        usax_aux = (aux._2._1 :+ (ts_input._1.toInt, j_st), aux._2._2 + ts_input._1.toInt, aux._2._3)
        USAX_Map_local += ((word, usax_aux))
      }

      // For next update
      if (j < ts_input._2.features.size) {
        /* It is removed from the accumulated values corresponding to
        the first element of the subsequence, now removed by advancing
        the window. The values for the new item (last item) of the
        subsequence to be processed in the next iteration are added.*/
        ex -= ts_input._2.features(j_st)
        ex2 -= ts_input._2.features(j_st) * ts_input._2.features(j_st)
        var k = 0
        while (k < (sax_len - 1)) {
          sum_segment(k) -= ts_input._2.features(j_st + (k) * w)
          sum_segment(k) += ts_input._2.features(j_st + (k + 1) * w)
          k += 1
        }
        sum_segment(k) -= ts_input._2.features(j_st + (k) * w)
        sum_segment(k) += ts_input._2.features(j_st + Math.min((k + 1) * w, subseq_len_aux))

        d = ts_input._2.features(j)
        ex += d
        ex2 += d * d
      }
      j += 1
    }
    USAX_Map_local
  }

  //Combinar USAX_Map_type
  def Combine_USAX_elm_type(USAX_1: USAX_elm_type,
                            USAX_2: USAX_elm_type): USAX_elm_type = {

    if (USAX_1._1(0)._1 < USAX_2._1(0)._1) {
      (USAX_1._1, USAX_1._2 ++ USAX_2._2, USAX_1._3 ++ USAX_2._3.map { case (k, v) => k -> (v + USAX_1._3.getOrElse(k, 0)) })
    } else {
      (USAX_2._1, USAX_1._2 ++ USAX_2._2, USAX_1._3 ++ USAX_2._3.map { case (k, v) => k -> (v + USAX_1._3.getOrElse(k, 0)) })
    }
  }

  def Combine_USAX_elm_type_last(USAX_1: USAX_elm_type,
                                 USAX_2: USAX_elm_type): USAX_elm_type = {

    (USAX_1._1, USAX_1._2, USAX_1._3 ++ USAX_2._3.map { case (k, v) => k -> (v + USAX_1._3.getOrElse(k, 0)) })
  }


  def Create_HashMark_full(SAX_word: SAX_word_type, obj_set: Obj_set_type,
                           mask_word: SAX_word_type, USAX: USAX_elm_type):
  (SAX_word_type, (Obj_set_type, Array[(SAX_word_type, USAX_elm_type)])) = {

    var word = SAX_word
    //The masked SAX word is added to the SAX object that comes.
    var new_word = word | mask_word

    (new_word, (obj_set, Array((word, USAX))))
  }

  def Create_HashMark_full_vector(SAX_word: SAX_word_type, obj_set: Obj_set_type,
                                  mask_words: Array[(Int, SAX_word_type)], USAX: USAX_elm_type):
  Array[((Int, Int), (Obj_set_type, Array[(Int, USAX_elm_type)]))] = {

    var word = SAX_word
    //The masked SAX word is added to the SAX object that comes.
    var out = mask_words.map(x => ((x._1, word | x._2), (obj_set, Array((word, USAX)))))

    return out
  }

  def Count_projection_issues(USAX: Array[(SAX_word_type, USAX_elm_type)],
                              obj_set: Set[Int]):
  Array[(SAX_word_type, USAX_elm_type)] = {

    return USAX.map(kv1 => {
      val word = kv1._1
      var USAX_elm = kv1._2
      //The masked SAX word is added to the SAX object that comes.
      //var new_word = word | mask_word
      obj_set.foreach(kv2 => {
        if (USAX_elm._3.contains(kv2) == false)
          USAX_elm._3 += ((kv2, 0))

        USAX_elm._3(kv2) += 1
      })

      (word, USAX_elm)
    })
  }

  /** Using random projections (random masking)
    *
    * @param R            Number of repetitions of the process of random projections.
    * @param percent_mask Masking desired percentage.
    * @param sax_len      SAX word length.
    * @note Modifies the original SAX words obtained by randomly projection
    *       process.
    */
  def RandomProjection(USAX_Map_local: org.apache.spark.rdd.RDD[(SAX_word_type,
    USAX_elm_type)], R: Int, percent_mask: Double, sax_len: Int):
  org.apache.spark.rdd.RDD[(SAX_word_type, USAX_elm_type)] = {

    var USAX_Map_aux = USAX_Map_local
    var mask_word: SAX_word_type = 0
    var num_mask: Int = Math.ceil(percent_mask * sax_len).toInt //Rounding up
    var r = 0

    while (r < R) {
      /*Masking some of the values of the SAX words. (The masked values are
        not taken into account).*/
      mask_word = CreateMaskWord(num_mask, sax_len)

      USAX_Map_aux = USAX_Map_aux.
        map(x => Create_HashMark_full(x._1, x._2._2, mask_word, x._2)).
        reduceByKey((x, y) => (x._1 ++ y._1, x._2 ++ y._2)).
        flatMap(x => Count_projection_issues(x._2._2, x._2._1)).
        reduceByKey((x, y) => Combine_USAX_elm_type_last(x, y)).
        map(x => {
          var aux = Vector(x._2._1.sortBy(_._1).take(1)(0))
          (x._1, (aux, x._2._2, x._2._3))
        })

      USAX_Map_aux.cache
      USAX_Map_aux.count
      USAX_Map_aux.unpersist()
      r += 1
    }

    println(USAX_Map_aux.count)
    return USAX_Map_aux
  }

  //Calculo de score de las palabas SAX
  /** Calculate the score for a SAX word. Score each sax in the matrix.
    *
    * @param c_in  Number of positive occurrences (as a time series is in the
    *              correct area given class).
    * @param c_out No occurrences number (when a time series is not in the
    *              correct class given area).
    * @return Score for a SAX word.
    */
  def CalScoreFromObjCount(c_in: Array[Double], c_out: Array[Double]): Double = {
    // For 2 classes only
    // return Math.abs((c_in(0)+c_out(1)) - (c_out(0)+c_in(1)))

    // Multi-class
    var diff, sum: Double = 0.0
    var max_val = -1E8
    var min_val = 1E8

    var i = 0
    while (i < num_class) {
      diff = (c_in(i) - c_out(i))
      if (diff > max_val) max_val = diff
      if (diff < min_val) min_val = diff
      sum += Math.abs(diff)
      i += 1
    }
    (sum - Math.abs(max_val) - Math.abs(min_val)) + Math.abs(max_val - min_val)
  }

  /** Calculate the score for a SAX word.
    *
    * @param usax Structure defining a SAX word.
    * @return Score for a SAX word.
    */
  def CalScore(usax: USAX_elm_type, Local_Label: Vector[Int]): Double = {
    var score: Double = -1
    var cid, count: Int = 0
    /* Fixed size and not particularly high therefore to vectors that are more
    efficient is used.*/
    // Count object inside hash bucket
    var c_in: Array[Double] = Array.fill(num_class)(0.0)
    // Count object outside hash bucket
    var c_out: Array[Double] = Array.fill(num_class)(0.0)

    // Note that if no c_in, then no c_out of that object
    usax._3.foreach(kv => {
      cid = Local_Label(kv._1)
      count = kv._2
      c_in(cid) += count
      c_out(cid) += (R - count)
    })
    score = CalScoreFromObjCount(c_in, c_out)
    score
  }

  /** Calculates a score for all SAX words available for a certain specified
    * number of repetitions.
    */
  def ScoreAllSAX(USAX_Map_local: org.apache.spark.rdd.RDD[(SAX_word_type,
    USAX_elm_type)], Local_Label: Vector[Int]): org.apache.spark.rdd.RDD[(SAX_word_type, Double)] = {
    var word: SAX_word_type = 0
    var score: Double = 0.0
    var usax: USAX_elm_type = (Vector(), Set(), scala.collection.mutable.Map[Int, Int]())

    return USAX_Map_local.map { kv =>
      word = kv._1
      usax = kv._2
      score = CalScore(usax, Local_Label)
      (word, score)
    }
  }

  /** Calculate the value of entropy based on the number of occurrences and the
    * total possible number of occurrences.
    *
    * @param A     Number of occurrences.
    * @param total Total possible number of occurrences.
    * @return The entropy value calculated for the specified parameters.
    */
  def EntropyArray(A: Array[Int], total: Int): Double = {
    // new function still in doubt (as in Mueen's paper)
    var en, a: Double = 0.0
    var i = 0
    while (i < num_class) {
      a = A(i).toDouble / total
      if (a > 0)
        en -= a * Math.log(a)
      i += 1
    }
    en
  }


  /** Getting the information gain for a SAX word determined.
    *
    * @param c_in        Number of positive occurrences.
    * @param c_out       No occurrences number.
    * @param total_c_in  Cumulative total number of positive occurrences.
    * @param total_c_out Cumulative total number of no occurrences.
    * @return Gain information obtained for the word "SAX" with those counts
    *         occurrences and not occurrences.
    */
  def CalInfoGain2(c_in: Array[Int], c_out: Array[Int], total_c_in: Int,
                   total_c_out: Int, class_entropy: Double, num_obj_local: Int): Double = {
    // new function as in in riva paper)
    class_entropy - ((total_c_in).toDouble / num_obj_local *
      EntropyArray(c_in, total_c_in) + (total_c_out).toDouble / num_obj_local *
      EntropyArray(c_out, total_c_out))
  }


  /** Get the best shapelet obtained, of the top-k-score SAX. Calculate Real
    * Infomation Gain.
    *
    * @param top_k Number of better SAX words to consider.
    * @return Best Shapelet found based on information obtained gain.
    */
  def FindBestSAX(top_k: Int, subseq_len: Int, USAX_Map_local: org.apache.spark.rdd.RDD[(SAX_word_type,
    USAX_elm_type)], Score_List_local: org.apache.spark.rdd.RDD[(SAX_word_type, Double)],
                  Data: org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)],
                  RDD_Label: org.apache.spark.rdd.RDD[(Long, Int)], Org_Data_first: org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)]
                 ): Shapelet = {
    var word: SAX_word_type = 0
    var gain, dist_th, gap: Double = 0
    var q_obj, q_pos: Int = 0
    var usax: USAX_elm_type = (Vector(), Set(), scala.collection.mutable.Map[Int, Int]())
    var label, kk, total_c_in, num_diff: Int = 0
    var sh, bsf_sh: Shapelet = new Shapelet
    /*As access to the following variables it is random, and is required to be
    fast, structures used ListBuffer*/
    var c_in = Array.fill(num_class)(0)
    var c_out = Array.fill(num_class)(0)
    var query = Array.fill(subseq_len)(0.0)

    var Score_List_sorted = Score_List_local

    /* We need an auxiliary variable to override the value of top_k. The input
    parameters of the function are taken as "val" (constant).*/
    var top_k_aux: Int = top_k
    top_k_aux = Math.abs(top_k_aux)

    var k = 0
    val Score_List_aux = Score_List_local.sortBy(_._2, false).take(top_k_aux)
    while (k < Math.min(top_k_aux, Score_List_local.count).toInt) {

      word = Score_List_aux(k)._1
      usax = USAX_Map_local.lookup(word)(0)

      var kk = 0

      while (kk < Math.min(usax._1.size.toInt, 1)) {
        q_obj = usax._1(kk)._1
        q_pos = usax._1(kk)._2

        var ts = Data.lookup(q_obj)(0).features
        query = ts.toArray.slice(q_pos, q_pos + subseq_len)

        var m: Int = query.size

        var nn_ed: NN_ED = (Array.fill(m)(0.0), Array.fill(m)(0))

        /* For each candidate a list containing the pair of values (time series
          identifier, distance from the candidate to the series) is created */
        var NN_aux = NearestNeighborSearch(query, Org_Data_first.lookup(0)(0).features, 0, nn_ed, 0)
        nn_ed = (NN_aux._2, NN_aux._3)

        var shapelet_calculated = Data.mapPartitionsWithIndex((index: Int, it: Iterator[(Long, LabeledPoint)]) => {
          if (it.isEmpty) {
            Array((-1, bsf_sh)).iterator
          } else {
            var part = it.toArray

            val labels = part.map(x => x._2.label.toInt)
            var data = part.map(x => x._2.features)

            var sh, bsf_sh: Shapelet = new Shapelet

            var num_obj_local = labels.size

            var Class_Freq_local = Array.fill(num_class)(0)
            var aux = labels.groupBy(identity).mapValues(_.size).toArray //.sortBy(_._1)//.map(_._2)

            var i = 0
            while (i < aux.size) {
              Class_Freq_local(aux(i)._1) = aux(i)._2
              i = i + 1
            }

            i = 0
            while (i < num_class) {
              c_in(i) = 0
              c_out(i) = Class_Freq_local(i)
              i += 1
            }

            var Dist: Array[(Int, Double)] = Array.fill(num_obj_local)((0, 0.0))
            var obj = 0
            while (obj < num_obj_local) {
              Dist(obj) = (obj, NearestNeighborSearch(query, data(obj), obj, nn_ed, 1)._1)
              obj += 1
            }
            Dist = Dist.sortBy(_._2)

            var class_entropy_local = EntropyArray(Class_Freq_local, num_obj_local)

            var total_c_in = 0
            i = 0
            while (i < (num_obj_local - 1)) {
              dist_th = (Dist(i)._2 + Dist(i + 1)._2) / 2.0
              gap = (Dist(i + 1)._2 - dist_th).toDouble / Math.sqrt(subseq_len)
              label = labels(Dist(i)._1)
              c_in(label) += 1
              c_out(label) -= 1
              total_c_in += 1
              num_diff = Math.abs(num_obj_local - 2 * total_c_in)
              gain = CalInfoGain2(c_in, c_out, total_c_in,
                num_obj_local - total_c_in, class_entropy_local, num_obj_local)
              sh.setValueFew(gain, gap, dist_th)
              if (bsf_sh.<(sh)) {
                bsf_sh.setValueAll(gain, gap, dist_th, q_obj, q_pos, subseq_len,
                  num_diff, c_in.toVector, c_out.toVector)
              }
              i += 1
            }
            Array((index, bsf_sh)).iterator
          }
        }).filter(x => x._1 != (-1)).map(_._2)
        val gain_RDD = shapelet_calculated.map(x => (x.gain, x))
        val max_gain = gain_RDD.max()(Ordering[Double].on(_._1))._1
        val max_gain_instances = gain_RDD.filter(x => x._1 == max_gain)
        var out = new Shapelet

        if (max_gain_instances.count > 1) {

          val num_diff_RDD = max_gain_instances.map(_._2).map(x => (x.num_diff, x))
          val max_num_diff = num_diff_RDD.max()(Ordering[Double].on(_._1))._1
          val max_num_diff_instances = num_diff_RDD.filter(x => x._1 == max_num_diff)

          if (max_num_diff_instances.count > 1) {
            val gap_RDD = max_num_diff_instances.map(_._2).map(x => (x.gap, x))

            val max_gap = gap_RDD.max()(Ordering[Double].on(_._1))._1
            val max_gap_instances = gap_RDD.filter(x => x._1 == max_gap)
            out = max_gap_instances.first()._2
          } else {
            out = max_num_diff_instances.first()._2
          }
        } else {
          out = max_gain_instances.first()._2
        }
        sh = out

        if (Classificator == 1) {
          list_ST.append((top_k, query, sh.obj, sh.pos, sh.len, sh.gain))
        }

        if (bsf_sh.<(sh)) {
          bsf_sh = sh
        }
        kk += 1
      }
      k += 1
    }
    return bsf_sh
  }


  /** Create a list of words SAX, adding them to the appropriate variable.
    *
    * @param node_id        Node identifier.
    * @param Org_Data       It contains data per node. It is MODIFIED.
    * @param Org_Label      It contains labels per node. It is MODIFIED.
    * @param Org_Class_Freq It contains the number of occurrences of each class per
    * node. It is MODIFIED.
    * @note The modification of the parameters refer to its continuous updating
    *       based on the processing node. Set variables for next node. They are Data,
    *       Label, Class_Freq, num_obj
    */
  def SetCurData(node_id: Int, Org_Data: org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)],
                 Org_Label: Vector[Int],
                 Org_Class_Freq: Vector[Int],
                 sc: SparkContext):
  (org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)], Vector[Int], Vector[Int]) = {
    println("SetCurData NODE:")
    println(node_id)
    var Data = Org_Data
    var Label = Org_Label
    var Class_Freq = Org_Class_Freq

    if (node_id == 1) {
      Data = Org_Data
      Label = Org_Label
      Class_Freq = Org_Class_Freq

      Data.cache
    } else {

      println("SetCurData: 1")
      num_obj = Node_Obj_List(node_id).count.toInt //Se dobla el num object ERROR
      println("SetCurData: 2")

      val Tokens = sc.broadcast(Node_Obj_List(node_id).collect.toSet)
      Data = Org_Data.filter { case (x, y) => Tokens.value.contains(x) }
      Data.coalesce(Data.getNumPartitions, true)
      Data.count

      Label = Data.map(x => x._2.label.toInt).collect.toVector
      println("SetCurData: 11")
      Class_Freq = Relabel.map(x => Label.count(_ == x._2)).toVector
    }

    print(Class_Freq.toArray)
    print(num_obj)
    class_entropy = EntropyArray(Class_Freq.toArray, num_obj)

    printf("\n=== Node %d ===\n", node_id)

    if (DEBUG >= 1) {
      // Global Entropy for the original state
      printf("max entropy = %.4f  <-- ", class_entropy)
      for (i <- 0 until num_class)
        printf(" %d ", Class_Freq(i))
      printf("  [num_obj=%d][num_class=%d]\n", num_obj, num_class)
    }

    Data.unpersist()
    (Data, Label, Class_Freq)
  }

  /** Screen printing for own node information contained in the list generated
    * by the built tree. (Identification, classification and size)
    */
  def printNodeSetList(Node_Obj_List_local: Array[RDD[Long]]): Unit = {
    printf("\n Node Set List ==> \n")
    for (i <- 0 until Node_Obj_List_local.size) {
      if (Node_Obj_List_local(i).count.toInt == 0) {

      } else {
        printf(" Node %2d : [c:%2d] : [%2d] ", i, Classify_list(i),
          Node_Obj_List_local(i).count.toInt)
        Node_Obj_List_local(i).foreach(kv => {
          printf("%d ", kv)
        })
        printf("\n")
      }
    }
  }

  /** Create List of Object on next level in the decision tree. If the accuracy
    * is high, not create node.
    *
    * @param node_id        Node identifier.
    * @param sh             Shapelet item itself node.
    * @param num_obj        The number of items in node. It is MODIFIED.
    * @param Data           Data's node.
    * @param sc             SparkContext.
    * @param Org_Data_first Original data.
    */
  def SetNextNodeObj(node_id: Int, sh: Shapelet, num_obj: Int,
                     Node_Obj_List_in: Array[RDD[Long]],
                     Data: org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)],
                     sc: SparkContext,
                     Org_Data_first: org.apache.spark.rdd.RDD[(Long, org.apache.spark.mllib.regression.LabeledPoint)]
                    ): Array[RDD[Long]] = {

    println("SetNextNodeObj")
    print("NODEID ")
    print(node_id)

    var q_len: Int = sh.len
    var dist_th: Double = sh.dist_th
    var query: Array[Double] = Array.fill(q_len)(0.0)
    var left_node_id: Int = node_id * 2
    var right_node_id: Int = node_id * 2 + 1

    var tmp: RDD[Long] = sc.emptyRDD
    var tmp_sh = new Shapelet

    var Node_Obj_List_local = Node_Obj_List_in

    while (Node_Obj_List_local.size <= right_node_id) {
      println("Node_Obj_List_local.size")
      println(Node_Obj_List_local.size)
      Node_Obj_List_local = Node_Obj_List_local ++ Array(tmp)
      Classify_list += -2
      Final_Sh += tmp_sh
      if (Node_Obj_List_local.size == 2) {
        var aux = Data.map(x => x._1)
        aux.cache
        aux.count
        Node_Obj_List_local(1) = aux
      }
    }
    Final_Sh(node_id) = sh
    query = sh.ts.toArray
    var m: Int = query.size

    var nn_ed = (Array.fill(m)(0.0), Array.fill(m)(0))

    if (nn_ed._1.isEmpty == true) error(3)
    if (nn_ed._1.isEmpty == true) error(3)

    var NN_aux = NearestNeighborSearch(query, Org_Data_first.lookup(0)(0).features, 0, nn_ed, 0)

    nn_ed = (NN_aux._2, NN_aux._3)

    var dist = Data.map(x => (x._1, NearestNeighborSearch(query, x._2.features, x._1.toInt, nn_ed, 1)._1, x._2.label))

    var c_in_dist = 0

    var aux1 = dist.filter(x => x._2 <= dist_th)
    aux1.count

    var c_in_rdd_final = aux1.map(x => {
      (x._3.toInt, 1)
    }).reduceByKey((x, y) => x + y).collect

    c_in_rdd_final.foreach(x => print(x))

    var c_in_rdd = Array.fill(num_class)(0)

    c_in_rdd_final.foreach(x => c_in_rdd(x._1) = x._2)

    var aux = aux1.map(_._1)
    aux.cache
    aux.count
    Node_Obj_List_local(left_node_id) = aux //dist.filter(x=>x._2<=dist_th).map(_._1)//.collect

    Node_Obj_List_local(left_node_id).count

    var aux2 = dist.filter(x => x._2 > dist_th)
    aux2.count

    var c_out_rdd_final = aux2.map(x => {
      (x._3.toInt, 1)
    }).reduceByKey((x, y) => x + y).collect

    var c_out_rdd = Array.fill(num_class)(0)

    c_out_rdd_final.foreach(x => c_out_rdd(x._1) = x._2)

    aux = aux2.map(_._1)
    aux.cache
    aux.count
    Node_Obj_List_local(right_node_id) = aux //dist.filter(x=>x._2>dist_th).map(_._1)//.collect


    Node_Obj_List_local(right_node_id).count

    var obj = 0

    nn_ed = (Array.fill(m)(0.0), Array.fill(m)(0))

    if (nn_ed._1.isEmpty == true) error(3)
    if (nn_ed._1.isEmpty == true) error(3)

    // If left/right is pure, or so small, stop spliting
    var max_c_in: Int = -1
    var sum_c_in: Int = 0
    var max_c_out: Int = -1
    var sum_c_out: Int = 0
    var max_ind_c_in: Int = -1
    var max_ind_c_out: Int = -1

    sh.c_in = c_in_rdd.toVector
    sh.c_out = c_out_rdd.toVector
    Final_Sh(node_id) = sh
    var i = 0

    while (i < sh.c_in.length) {
      sum_c_in += sh.c_in(i)
      if (max_c_in < sh.c_in(i)) {
        max_c_in = sh.c_in(i)
        max_ind_c_in = i
      }
      sum_c_out += sh.c_out(i)
      if (max_c_out < sh.c_out(i)) {
        max_c_out = sh.c_out(i)
        max_ind_c_out = i
      }
      i += 1
    }

    var left_is_leaf, right_is_leaf: Boolean = false

    MIN_OBJ_SPLIT = Math.ceil((MIN_PERCENT_OBJ_SPLIT * num_obj).toDouble /
      num_class).toInt

    if ((sum_c_in <= MIN_OBJ_SPLIT) || (max_c_in.toDouble / sum_c_in >=
      MAX_PURITY_SPLIT))
      left_is_leaf = true
    if ((sum_c_out <= MIN_OBJ_SPLIT) || (max_c_out.toDouble / sum_c_out >=
      MAX_PURITY_SPLIT))
      right_is_leaf = true

    var max_tree_dept: Int = (EXTRA_TREE_DEPTH + Math.ceil(Math.log(num_class) /
      Math.log(2)).toInt)

    if (node_id >= Math.pow(2, max_tree_dept)) {
      left_is_leaf = true
      right_is_leaf = true
    }

    //Classify_list(node_id) = -1
    Classify_list(node_id) = -1
    if (left_is_leaf) {
      Classify_list(left_node_id) = max_ind_c_in
    } else {
      Classify_list(left_node_id) = -1
    }
    if (right_is_leaf) {
      Classify_list(right_node_id) = max_ind_c_out
    } else {
      Classify_list(right_node_id) = -1
    }

    if (DEBUG >= 2)
      printNodeSetList(Node_Obj_List_local)

    return (Node_Obj_List_local)
  }

  /** Print in the output file the tree Shapelets built.
    *
    * @param fo   Name the file that contains the tree Shapelet.
    * @param argc Number of parameters expected in each Shapelet.
    * @param seed The random seed value.
    * @param argv Values of parameters for each Shapelet.
    */
  def printTree(fo: java.io.PrintStream, argc: Int, seed: Int, argv: Seq[String])
  : Unit = {
    var rand = new scala.util.Random(seed)

    fo.printf(";Shapelet Tree \n")
    fo.printf("\n")
    fo.printf(";random seed:%d, rand()=%d, rand()=%d\n", seed: java.lang.Integer,
      rand.nextInt: java.lang.Integer, rand.nextInt: java.lang.Integer)

    fo.printf(";command = ")
    for (i <- 0 until argc)
      fo.printf("%s ", argv(i).toString)
    fo.printf("\n")
    fo.printf("\n")

    fo.printf(";Local Class:  ")
    for (i <- 0 until num_class)
      fo.printf("%2d ", i: java.lang.Integer)
    fo.printf("\n")

    fo.printf(";Real  Class:  ")
    for (i <- 0 until num_class)
      fo.printf("%2d ", Real_Class_Label(i): java.lang.Integer)
    fo.printf("\n")
    fo.printf("\n")
    fo.printf(";Node ID class   obj  pos  len  dist_th \n")

    fo.printf("TreeSize: %d \n", Node_Obj_List.size: java.lang.Integer)
    var real_label: Int = 0

    var sh = new Shapelet()

    for (i <- 1 until Node_Obj_List.size) {
      if (Node_Obj_List(i).count.toInt == 0) {

      } else {
        sh = Final_Sh(i)
        if (Classify_list(i) < 0) { // non-leaf
          fo.printf("NonL %2d    %2s    %3d  %3d  %3d  %7.4f ",
            i: java.lang.Integer, "--", sh.obj: java.lang.Integer,
            sh.pos: java.lang.Integer, sh.len: java.lang.Integer,
            sh.dist_th: java.lang.Double)

          fo.printf("==> ")
          for (j <- 0 until num_class)
            fo.printf("%3d ", sh.c_in(j): java.lang.Integer)
          fo.printf(" / ")
          for (j <- 0 until num_class)
            fo.printf("%3d ", sh.c_out(j): java.lang.Integer)
          fo.printf("\n")
        } else {
          real_label = Real_Class_Label(Classify_list(i))
          fo.printf("Leaf %2d    %2d    \n", i: java.lang.Integer,
            real_label: java.lang.Integer)
        }
      }
    }

    if (fo == System.out) {
      fo.printf("\n")
      return
    }

    fo.printf("\n")
    fo.printf(";Shapelet id   <data>\n")
    for (i <- 1 until Node_Obj_List.size) {
      if (Node_Obj_List(i).count.toInt == 0) {

      } else {
        sh = Final_Sh(i)
        if (Classify_list(i) < 0) // non-leaf
        {
          fo.printf("Shapelet %3d  ", i: java.lang.Integer)
          for (j <- 0 until sh.ts.size)
            fo.printf("%9.6f ", sh.ts(j): java.lang.Double)
          fo.printf("\n")
        }
      }
    }
    fo.printf("\n\n")
  }


  /** Print in the output file the time used execute the main parts of the
    * algorithm.
    *
    * @param fo       Filename of interest. Is is MODIFIED.
    * @param time_arr Vector that contains the execute times used in these
    *                 parts.
    */
  def printTime(fo: PrintStream, time_arr: Vector[Double]): Unit = {
    fo.printf("; time =>  ")
    for (i <- 0 until 5)
      fo.printf("%7.3f ", time_arr(i): java.lang.Double)
    fo.printf("\n")
  }

  /** Prints in the output file the desired command.
    *
    * @param fo   name of the file that will contain the print command. Is is
    *             MODIFIED.
    * @param argc number of arguments present on the command.
    * @param argv argument values present in the command.
    */
  def printCommand(fo: PrintStream, argc: Int, argv: Seq[String]): Unit = {
    fo.printf("; cmd =")
    for (i <- 0 until argc)
      fo.printf(" %s", argv(i).toString)
    fo.printf("\n")
  }

  def ReadTrainData_BigData(train_file_name: String, Num_partitions: Int, sc: SparkContext): (RDD[(Long, LabeledPoint)], Vector[Int], Vector[Int],
    RDD[(Long, Int)]) = {
    //Data readout:
    val data = sc.textFile(train_file_name, Num_partitions)
    /*All spaces are replaced concantenaciones, like single spaces, by a single
    space (separation problems are solved between positive and negative
    numbers). Input format csv file so that a separation of the elements is done
    by commas is expected. Spaces at the beginning and end of the string, for
    entry, resulting from these transactions are eliminated. At this point you
    have a string per entry (line) in which the elements are separated by single
    spaces. The space-separated elements are identified by selecting the first
    element as a label and the rest is transformed into an Array [Double] for
    subsequent conversion to Vector Dense and inclusion in LabeledPoint object
    with its corresponding label (data with its label).*/
    var Org_Data = data.map(line => line.replaceAll(" +", " ").split(",").
      map(_.trim)).map(x => x(0).split(" ")).map(x => LabeledPoint(x(0).toDouble,
      Vectors.dense(x.drop(1).map(_.toDouble)))).zipWithIndex.map(x => (x._2, x._1)).coalesce(Num_partitions, true)

    Org_Data.cache
    Org_Data.count

    //Counting classes.
    val aux = Org_Data.map(x => (x._2.label, 1)).reduceByKey((x, y) => x + y).collect()

    //Frequency of appearance of the classes found.
    var Org_Class_Freq = aux.map(x => x._2).toVector

    //Original Labels
    val aux2 = aux.map(x => (x._1).toInt)

    //Number of classes found.
    num_class = Org_Class_Freq.size

    /* Keep the real labels which are read from data, i.e.,
    Real_Class_Label(new_label) = real_label*/
    Relabel = (0 until num_class).map(x => (aux2(x), x)).toMap
    Real_Class_Label = (0 until num_class).map(x => (x, aux2(x))).toMap

    Org_Data = Org_Data.map(x => (x._1, LabeledPoint(Relabel(x._2.label.toInt), x._2.features)))

    val Org_Label = Org_Data.map(x => x._2.label.toInt).collect.toVector

    Org_Class_Freq = Relabel.map(x => Org_Label.count(_ == x._2)).toVector

    //Number of instances.
    num_obj = Org_Label.size

    //Number of read instances.
    val num_ser = aux.map(x => x._2).reduce((x, y) => x + y)

    printf("relabel [new,real] => ")
    Real_Class_Label.foreach(kv => {
      printf("[%d,%d]  ", kv._1: java.lang.Integer, kv._2: java.lang.Integer)
    })
    printf("\n")

    var RDD_label = Org_Data.map(x => (x._1, x._2.label.toInt))

    Org_Data.unpersist()
    (Org_Data, Org_Label, Org_Class_Freq, RDD_label)
  }


  def main(train_file_name: String, min_len: Int, max_len: Int, step: Int, Num_partitions_in: Int, seed: Int, tree_file_name: String,
           time_file_name: String, R: Int, top_k: Int, Classificator: Int, shapelet_file_name: String, test_file_name: String, model_file_name: String) {
    val conf = new SparkConf().set("spark.default.parallelism", Num_partitions_in.toString)
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    var Num_partitions = Num_partitions_in

    var read = ReadTrainData_BigData(train_file_name, Num_partitions, sc)

    var Org_Data = read._1
    var Org_Label = read._2
    var Org_Class_Freq = read._3
    var Org_RDD_Label = read._4

    Org_Data.cache
    Org_RDD_Label.cache

    if (Classificator == 1) {
      if (shapelet_file_name == "") {
        print("Introduce a shapelet file name")
        return 0
      }
    }

    t0_0 = (System.nanoTime / 1000000000.0).toDouble

    //Local vars
    var t0, t1, t2, t3, t4, t5, t6: Double = 0.0
    var sax_max_len = 12 //15
    var sax_len = 0
    var sax_min_len = 0
    var w = 0

    var percent_mask = 0.25
    var subseq_len = 0
    var max_tree_depth = (EXTRA_TREE_DEPTH +
      Math.ceil(Math.log(num_class) / Math.log(2)).toInt + 2).toInt

    var flag: Int = 0
    var node_id = 1
    var sh: Shapelet = new Shapelet

    t0 = (System.nanoTime() / 1000000000).toDouble

    var bsf_sh = new Shapelet
    while ((node_id == 1) || (node_id < Node_Obj_List.size.toInt)) {
      var Data = Org_Data
      var Label = Org_Label
      var Class_Freq = read._3
      var RDD_Label = read._4

      flag = 0
      bsf_sh = new Shapelet

      if (node_id <= 1) {
        val aux = SetCurData(node_id, Org_Data, Org_Label, Org_Class_Freq, sc)
        Data = aux._1
        Label = aux._2
        Class_Freq = aux._3
      } else if (Classify_list(node_id) == -1) {
        // non-leaf node (-1:body node, -2:unused node)
        val aux = SetCurData(node_id, Org_Data, Org_Label, Org_Class_Freq, sc)
        Data = aux._1
        Class_Freq = aux._3
        RDD_Label = Data.map(x => (x._1, x._2.label.toInt))
      } else {
        flag = 1
      }

      Data.cache
      RDD_Label.cache

      if (flag == 0) {
        println("Flag 0")
        subseq_len = min_len
        while (subseq_len <= max_len) {
          if (subseq_len < SH_MIN_LEN) {

          } else {
            print("Cantidad de instancias:")
            println(RDD_Label.count)
            sax_len = sax_max_len
            // Make w and sax_len both integer
            w = Math.ceil(1.0 * subseq_len / sax_len).toInt
            sax_len = Math.ceil(1.0 * subseq_len / w).toInt

            t1 = (System.nanoTime / 1000000000.0).toDouble
            println(sax_len)
            println(subseq_len)
            println(w)

            var USAX_Map = Data.flatMap(x => CreateSAXList(x, subseq_len, sax_len, w)).reduceByKey((x, y) => Combine_USAX_elm_type(x, y))

            USAX_Map.cache
            t2 = (System.nanoTime / 1000000000.0).toDouble
            s1 += t2 - t1

            var USAX_Map_2 = RandomProjection(USAX_Map, R, percent_mask, sax_len)
            USAX_Map_2.cache
            USAX_Map_2.count

            USAX_Map.unpersist()

            t3 = (System.nanoTime / 1000000000.0).toDouble
            s2 += t3 - t2

            var Score_List = ScoreAllSAX(USAX_Map_2, Label)
            Score_List.cache
            Score_List.count
            t4 = (System.nanoTime / 1000000000.0).toDouble
            s3 += t4 - t3

            sh = FindBestSAX(top_k, subseq_len, USAX_Map_2, Score_List, Data, RDD_Label, Org_Data)
            t5 = (System.nanoTime / 1000000000.0).toDouble
            s4 += t5 - t4

            if (Classificator == 1) {
              if (sh.len > 0) {
                var query = Data.lookup(sh.obj)(0).features.toArray.slice(sh.pos, sh.pos + sh.len)
                sh.setTS(query.toVector)
                Final_Sh_aux += sh
              }
            }
            if (bsf_sh.<(sh)) {
              bsf_sh = sh
            }

            USAX_Map_2.unpersist()
            Score_List.unpersist()
          }
          subseq_len += step
        } //End while subseq_len

        if (bsf_sh.len <= 0)
          printf("Error!! bsf_sh is empty  at node_id=%d \n",
            node_id: java.lang.Integer)
        if (bsf_sh.len > 0) {
          var query = Data.lookup(bsf_sh.obj)(0).features.toArray.slice(bsf_sh.pos,
            bsf_sh.pos + bsf_sh.len)
          bsf_sh.setTS(query.toVector)

          Final_Sh += bsf_sh
          print("NODEID ")
          print(node_id)
          print(" ")
          println(num_obj)
          Node_Obj_List = SetNextNodeObj(node_id, bsf_sh, num_obj, Node_Obj_List, Data, sc, Org_Data)
        }
        Data.unpersist()
        RDD_Label.unpersist()
      } //End if flag
      println("Fin_node:")
      println(node_id)
      node_id += 1

      if (Classificator == 1 && list_ST.length >= 1) {
        list_ST_selected.appendAll((list_ST.sortWith(_._6 > _._6).take(top_k)))
        list_ST = scala.collection.mutable.ListBuffer.empty
      }

    } //End while nodes

    val oos = new ObjectOutputStream(new FileOutputStream(shapelet_file_name))
    oos.writeObject((list_ST_selected.toArray, Real_Class_Label, Relabel))
    oos.close

    t6 = (System.nanoTime / 1000000000.0).toDouble
    println("Bloqueo salida")
    println("SALIDA")

    if (DEBUG >= 0) {
      printf("\n")
      printf("num_class=%d, num_obj=%d, R=%d, top_k=%d, percent_mask=%.2f \n",
        num_class: java.lang.Integer,
        Org_Data.count.toInt: java.lang.Integer,
        //num_obj:java.lang.Integer,
        R: java.lang.Integer,
        top_k: java.lang.Integer,
        percent_mask: java.lang.Double)
      printf("  Time Create SAX   = %8.2f sec\n", s1: java.lang.Double)
      printf("  Time Rand Project = %8.2f sec\n", s2: java.lang.Double)
      printf("  Time Score Matrix = %8.2f sec\n", s3: java.lang.Double)
      printf("  Time Compute Gain = %8.2f sec\n", s4: java.lang.Double)
      printf("  Time environment preparation per iteration = %8.2f sec\n", (t6 - t0) - (s1 + s2 + s3 + s4): java.lang.Double)

      printf("Running  Time for this Shapelet= %8.2f sec <<\n",
        (t6 - t0): java.lang.Double)
    }
    printf("\n\nTotal Running  Time = %8.2f sec <<\n",
      (t6 - t0_0): java.lang.Double)

    val f_tree = new PrintStream(new FileOutputStream(tree_file_name))
    val f_time = new PrintStream(new FileOutputStream(time_file_name, true))

    printTree(f_tree, 11, seed, Seq(train_file_name, min_len.toString(), max_len.toString(), step.toString(), Num_partitions.toString(), seed.toString(), tree_file_name, time_file_name, R.toString(), top_k.toString(), Classificator.toString()))

    var time_arr: Vector[Double] = Vector(t6 - t0,
      s1, s2, s3, s4)
    printTime(f_time, time_arr)
    printTime(f_tree, time_arr)

    val f_scala_results = new PrintStream(new FileOutputStream("f_scala_results_ST.txt", true))

    f_scala_results.printf("\n")
    val cmd_input: Seq[String] = Seq(train_file_name, min_len.toString, max_len.toString, step.toString, Num_partitions.toString, seed.toString,
      tree_file_name, time_file_name, R.toString, top_k.toString, Classificator.toString, shapelet_file_name, test_file_name, model_file_name)
    printCommand(f_scala_results, cmd_input.length, cmd_input)
    f_scala_results.printf("num_class=%d, num_obj=%d, R=%d, top_k=%d, percent_mask=%.2f \n",
      num_class: java.lang.Integer,
      Org_Data.count.toInt: java.lang.Integer,
      //num_obj:java.lang.Integer,
      R: java.lang.Integer,
      top_k: java.lang.Integer,
      percent_mask: java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Time Create SAX\n", s1: java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Time Rand Project\n", s2: java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Time Score Matrix\n", s3: java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Time Compute Gain\n", s4: java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Time environment preparation per iteration\n", ((t6 - t0) - (s1 + s2 + s3 + s4)): java.lang.Double)
    f_scala_results.printf("%8.2f     sec << Total Training Time\n", (t6 - t0): java.lang.Double)


    val f_tree_scala_results = new PrintStream(new FileOutputStream("f_tree_scala_results.txt", true))

    printTime(f_tree_scala_results, time_arr)


    var st_t0, st_t1, st_t2, st_t3, st_t4: Double = 0.0

    st_t0 = (System.nanoTime / 1000000000.0).toDouble
    if (Classificator == 1) {
      var list_ST_Array = list_ST_selected.toArray
      var first_data2 = Org_Data.filter(x => x._1 == 0).collect()
      var first_data = first_data2(0)._2.features

      var Dist_ST_RDD = Org_Data.map(ins => {
        val out = list_ST_Array.map(sh => {
          val query = sh._2
          val m: Int = query.size
          var nn_ed = (Array.fill(m)(0.0), Array.fill(m)(0))
          val NN_aux = NearestNeighborSearch(query, first_data, 0, nn_ed, 0)
          nn_ed = (NN_aux._2, NN_aux._3)
          val dist = NearestNeighborSearch(query, ins._2.features, ins._1.toInt, nn_ed, 1)._1
          dist
        })
        LabeledPoint(ins._2.label, Vectors.dense(out))
        //  })
      }).coalesce(Num_partitions, true)
      Dist_ST_RDD.cache
      Dist_ST_RDD.count

      st_t1 = (System.nanoTime / 1000000000.0).toDouble

      // Train a RandomForest model.
      // Empty categoricalFeaturesInfo indicates all features are continuous.
      val numClasses = num_class
      val categoricalFeaturesInfo = Map[Int, Int]()
      val numTrees = 1000 // Use more in practice.
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val impurity = "gini"
      val maxDepth = 4
      val maxBins = 32

      val model = RandomForest.trainClassifier(Dist_ST_RDD, numClasses, categoricalFeaturesInfo,
        numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

      st_t2 = (System.nanoTime / 1000000000.0).toDouble


      val testdata = sc.textFile(test_file_name, Num_partitions)

      var Test_Data = testdata.map(line => line.replaceAll(" +", " ").split(",").
        map(_.trim)).map(x => x(0).split(" ")).map(x => LabeledPoint(x(0).toDouble,
        Vectors.dense(x.drop(1).map(_.toDouble)))).zipWithIndex.map(x => (x._2, x._1)).
        map(x => (x._1, LabeledPoint(Relabel(x._2.label.toInt), x._2.features)))

      var Dist_Test_ST_RDD = Test_Data.map(ins => {
        val out = list_ST_Array.map(sh => {
          val query = sh._2
          val m: Int = query.size
          var nn_ed = (Array.fill(m)(0.0), Array.fill(m)(0))
          val NN_aux = NearestNeighborSearch(query, first_data, 0, nn_ed, 0)
          nn_ed = (NN_aux._2, NN_aux._3)
          val dist = NearestNeighborSearch(query, ins._2.features, ins._1.toInt, nn_ed, 1)._1
          dist
        })
        LabeledPoint(ins._2.label, Vectors.dense(out))
      })
      Dist_Test_ST_RDD.count

      st_t3 = (System.nanoTime / 1000000000.0).toDouble


      // Evaluate model on test instances and compute test error
      val labelAndPreds = Dist_Test_ST_RDD.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      val testAcc = (labelAndPreds.filter(r => r._1 == r._2).count.toDouble / Test_Data.count()) * 100
      println("Accuracy = " + testAcc)

      st_t4 = (System.nanoTime / 1000000000.0).toDouble

      //  Save label modifications:
      val aaa = new ObjectOutputStream(new FileOutputStream(model_file_name))
      var model_labels_info = (model, Real_Class_Label, Relabel)
      aaa.writeObject(model_labels_info)
      aaa.close

      //Test time
      f_scala_results.printf("%8.2f     sec << Time for Train Shapelets Transform\n", (st_t1 - st_t0): java.lang.Double)
      f_scala_results.printf("%8.2f     sec << Time to train RandomForestModel\n", (st_t2 - st_t1): java.lang.Double)
      f_scala_results.printf("%8.2f     sec << Time for Test Shapelets Transform\n", (st_t3 - st_t2): java.lang.Double)
      f_scala_results.printf("%8.2f     sec << Time for Test Validation\n", (st_t4 - st_t3): java.lang.Double)
      f_scala_results.printf("%8.3f 		 %%  	<< Test Accuracy\n", testAcc: java.lang.Double)
    }

    f_tree_scala_results.close()
    f_scala_results.close()
    f_tree.close()
    f_time.close()
    return 0
  }
}

object FastShapelet extends Serializable {
  def run(train_file_name: String, min_len: Int, max_len: Int, step: Int, Num_partitions: Int, seed: Int, tree_file_name: String, time_file_name: String, R: Int, top_k: Int, Classificator: Int, shapelet_file_name: String, test_file_name: String, model_file_name: String) = {
    var a = new FastShapelet(train_file_name, min_len, max_len, step, Num_partitions, seed, tree_file_name, time_file_name, R, top_k, Classificator, shapelet_file_name, test_file_name, model_file_name)
    a.main(train_file_name, min_len, max_len, step, Num_partitions, seed, tree_file_name, time_file_name, R, top_k, Classificator, shapelet_file_name, test_file_name, model_file_name)
  }
}
