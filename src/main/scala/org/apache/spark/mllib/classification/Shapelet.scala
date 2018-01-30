package org.apache.spark.mllib.classification
/**
  * @author
  * @version 1.0
  */

/** It contains the value of the infinite contemplated in this program. */
object Shapelet extends Serializable {
  val INF = Double.MaxValue
}

/** Each object in the TDA Shapelet, models a tuple consisting of a subsequence
  * of data (start point and length) and the distance threshold achieved. The
  * distance between a subsequence and a time series make reference to the
  * minimum distance between itself and any subsequence of equal size, part of
  * the time series in question. A time series is an ordered list of numbers.
  *
  * A Shapelet is a division that separates the dataset in two (binary case)
  * smaller sets with maximum gain of information data; the gap is maximized.
  *  
  * They are mutable objects.
  *
  * @constructor The default constructor initializes all the variables to
  *              default values.
  */
//"case" is used to solve ambiguous problems
case class Shapelet() extends java.io.Serializable {
  //Mandatory variable initialization
  /** Gain information obtained by the Shapelet. */
  //var gain: Double= -1.7976931348623157E308
  var gain: Double = -1E8

  /** Achieved distance between the two positions (classes). */
  //var gap: Double= -1.7976931348623157E308
  var gap: Double = -1E8

  /** Minimum distance between Shapelet and time series. */
  //var dist_th: Double= 1.7976931348623157E308
  var dist_th: Double = 1E8

  /** Identifier of the time series. */
  var obj: Int = -1

  /** Shapelet start position on time series. */
  var pos: Int = -1

  /** Shapelet length from the initial position (pos) in the time series to
    * the end of it.
    */
  var len: Int = -1

  /** Number of time series after the division of the original dataset in
    * different areas (corresponding to the different classes present in the
    * dataset) are in the wrong area.
    */
  var num_diff: Int = -1

  /** Number of positive occurrences (as a time series is in the correct area
    * given class).
    */
  var c_in: Vector[Int] = Vector()

  /** No occurrences number (when a time series is not in the correct class
    * given area).
    */
  var c_out: Vector[Int] = Vector()

  /** Time series (list sorted numbers) from Shapelet object. */
  var ts: Vector[Double] = Vector()

  /**
    * @constructor Creates a Shapelet with the specified values for the
    *              parameters: gain, gap, dist_th, obj, pos, len and num_diff.
    * @param gain     : initialization value of information gain.
    * @param gap      : initialization value gap (separation between areas/possible
    *                 classes).
    * @param dist_th  : initialization value threshold distance.
    * @param obj      : index initialization value of the time series of the Shapelet
    *                 is obtained.
    * @param pos      : initialization value of the starting position in the time
    *                 series, from which begins Shapelet.
    * @param len      : initialization value Shapelet length.
    * @param num_diff : initialization value of the number of time series after
    *                 the division of the original dataset in different areas (corresponding to
    *                 the different classes present) are in the wrong area.
    *
    */
  def this(gain: Double = -1, gap: Double = 0, dist_th: Double = 0, obj: Int = 0,
           pos: Int = 0, len: Int = 0, num_diff: Int = 0) {
    this()
    this.gain = gain
    this.gap = gap
    this.dist_th = dist_th
    this.obj = obj
    this.pos = pos
    this.len = len
    this.num_diff = num_diff
  }

  /** Make a simple modification of the gain, gap and dist_th attributes of the
    * Shapelet object on which it is applied.
    *
    * @param gain    Value of information gain desired.
    * @param gap     Value gap (separation between areas / possible classes)
    *                desired.
    * @param dist_th Value threshold distance desired.
    *
    */
  def setValueFew(gain: Double, gap: Double, dist_th: Double): Unit = {
    //This method pointers accessed using "this->"
    this.gain = gain
    this.gap = gap
    this.dist_th = dist_th
  }


  /** Make a simple modification of the gain, cap, dist_th, obj, pos, len,
    * numd_diff, c_in and c_out attributes of the Shapelet object on which it
    * is applied.
    *
    * @param gain     Value of information gain desired.
    * @param gap      Value gap (separation between areas / possible classes)
    *                 desired.
    * @param dist_th  Value threshold distance desired.
    * @param obj      Identifier of the time series desired.
    * @param pos      Shapelet start position on time series desired.
    * @param len      Shapelet length from the initial position (pos) in the time
    *                 series to the end of it desired.
    * @param num_diff Number of time series after the division of the original
    *                 dataset in different areas (corresponding to the different classes
    *                 present in the dataset) are in the wrong area desired.
    * @param c_in     Number of positive occurrences (as a time series is in the
    *                 correct area given class) desired.
    * @param c_out    No occurrences number (when a time series is not in the
    *                 correct class given area) desired.
    */
  def setValueAll(gain: Double, gap: Double, dist_th: Double, obj: Int,
                  pos: Int, len: Int, num_diff: Int, c_in: Vector[Int], c_out: Vector[Int]): Unit = {
    //This method pointers accessed using "this->"
    this.gain = gain
    this.gap = gap
    this.dist_th = dist_th
    this.obj = obj
    this.pos = pos
    this.len = len
    this.num_diff = num_diff
    this.c_in = c_in
    this.c_out = c_out
  }


  /** Make a simple modification on dist_th attributes, obj, pos and len
    * attributes of Shapelet object in the list of Shapelets, on which it is
    * applied.
    *
    * @param dist_th Value threshold distance desired.
    * @param obj     Identifier of the time series desired.
    * @param pos     Shapelet start position on time series desired.
    * @param len     Shapelet length from the initial position (pos) in the time
    *                series to the end of it desired.
    */
  def setValueTree(obj: Int, pos: Int, len: Int, dist_th: Double): Unit = {
    //This method pointers accessed using "this->"
    this.dist_th = dist_th
    this.obj = obj
    this.pos = pos
    this.len = len
  }

  /**
    * Performs a simple modification to the attribute ts of the Shapelet object
    * on which it is applied, corresponding to the introduction of a new time
    * series.
    *
    * @param ts Time series (list sorted numbers) desired for the Shapelet
    *           object.
    */
  def setTS(ts: Vector[Double]): Unit = {
    this.ts = ts
  }

  def <(other: Shapelet): Boolean = {
    if (gain > other.gain) return false
    return ((gain < other.gain) || //\
      ((gain == other.gain) && (num_diff > other.num_diff)) || //\
      ((gain == other.gain) && (num_diff == other.num_diff) && (gap < other.gap))
      )
  }

  /**
    * Comparison operator greater than.
    *
    * @param other Shapelet with the receiver is compared.
    * @return true: if the receiver has more gain than other or if in case of
    *         equality of gain, the receiver has fewer misplaced time series after
    *         splitting the dataset or in case of *equality of gain and number of
    *         misplaced time series, the receiver has a gap (distance between different
    *         areas) greater than other.
    *             * false: in any case contrary to the above.
    */
  def >(other: Shapelet): Boolean = {
    return (other < this);
  }

  /**
    * Prints out the value of attributes len, obj, pos, dist_th, gap, gain,
    * num_diff with vectors c_in and c_out  and gain for the Shapelet object on
    * which it is applied.
    */
  def printShort(): Unit = {
    printf("\n")
    printf("obj pos len dist_th   gain\n")
    printf("%3d %3d %3d %7.4f %6.4f\n", obj, pos, len, dist_th, gain)
  }

  /**
    * Prints out the value of attributes obj, pos, len, dist_th and gain for
    * the Shapelet object on which it is applied.
    */
  def printLong(): Unit = {
    printf("\nLen=%-3d  @[%3d,%3d]  d_th=%5.2f gap=%.6f gain=%8.5f (%-3d)", len,
      obj, pos, dist_th, gap, gain, num_diff)
    printf("==> ")
    for (i <- 0 until c_in.size) printf("%3d ", c_in(i))
    printf(" / ")
    for (i <- 0 until c_in.size) printf("%3d ", c_out(i))
    printf("\n")
  }

}

