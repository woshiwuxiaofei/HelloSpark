package org.demos.wuxiaofei.spark.ml.algorithm.classification

import breeze.linalg
import breeze.linalg.{DenseVector => BreezeDenseVector}
import breeze.numerics._
import org.apache.spark.annotation.Since
import org.apache.spark.ml.awaken.platform.baseclazz.ml.predictor.classification.ProbabilisticClassifier
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.predictor.PredictorParams
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.shared.HasWeightCol
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.awaken.platform.component.ml.model.algorithm.classification.NaiveBayesModel
import org.apache.spark.ml.awaken.platform.component.persistent.serialize.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.ml.awaken.platform.util.ml.Identifiable
import org.apache.spark.ml.linalg.{BLAS, DenseMatrix, DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, lit}

import scala.math.Pi



/**
  * author: wuxiaofei
  * date: 2018/9/12 16:05
  * description: { 完备朴素贝叶斯分类器,支持高斯模型 }
  * version: altering based on {Spark2.10 ml_sc}
  */

@Since("1.6.0")
class NaiveBayes @Since("1.5.0") (
                                   @Since("1.5.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, NaiveBayes, NaiveBayesModel]
    with NaiveBayesParams with DefaultParamsWritable {

  import NaiveBayes._

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("nb"))

  /**
    * Set the smoothing parameter.
    * Default is 1.0.
    * @group setParam
    */
  @Since("1.5.0")
  //防止极大似然估计概率值为0的情况，采用贝叶斯估计，当smoothing = 1时称为拉普拉斯平滑
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  /**
    * Set the model type using a string (case-sensitive).
    * Supported options: "multinomial" and "bernoulli".
    * Default is "multinomial"
    * @group setParam
    */
  @Since("1.5.0")
  //计算条件概率的模型，当前只支持伯努利和多项式（默认）
  def setModelType(value: String): this.type = set(modelType, value)
  setDefault(modelType -> Multinomial)

  /**
    * Sets the value of param [[weightCol]].
    * If this is not set or empty, we treat all instance weights as 1.0.
    * Default is not set, so all instances have weight one.
    *
    * @group setParam
    */
  @Since("2.1.0")
  //给定的权重，默认值1.0
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
    * support Multinomial and Bernoulli training
    * @param dataset  Training dataset
    * @return  Fitted model
    */
  override protected def train(dataset: Dataset[_]): NaiveBayesModel = {
    trainWithLabelCheck(dataset, positiveLabel = true)
  }

  /**
    * ml assumes input labels in range [0, numClasses). But this implementation
    * is also called by mllib NaiveBayes which allows other kinds of input labels
    * such as {-1, +1}. `positiveLabel` is used to determine whether the label
    * should be checked and it should be removed when we remove mllib NaiveBayes.
    */
  private[spark] def trainWithLabelCheck(dataset: Dataset[_], positiveLabel: Boolean): NaiveBayesModel = {

    if (positiveLabel) {
      val numClasses = getNumClasses(dataset)
      if (isDefined(thresholds)) {
        require($(thresholds).length == numClasses, this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
      }
    }

    //特征维度：特征属性数量
    val numFeatures = dataset.select(col($(featuresCol))).head().getAs[Vector](0).size
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    // TODO: Calling aggregateByKey and collect creates two stages, we can implement something
    // TODO: similar to reduceByKeyLocally to save one stage.

    //对每个标签进行聚合操作，得到每个标签对应特征的频数与特征之和
    val aggregated = dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd
      .map { row => (row.getDouble(0), (row.getDouble(1), row.getAs[Vector](2)))
      }.aggregateByKey[(Double, DenseVector)]((0.0, Vectors.zeros(numFeatures).toDense))(
      //返回：（label,(count，sumFeatures))
      seqOp = {
        //同一个分区求(count，sumFeatures)
        case ((weightSum: Double, featureSum: DenseVector), (weight, features)) =>
          valuesCheck(features)
          BLAS.axpy(weight, features, featureSum) //featureSum += weight * features
          (weightSum + weight, featureSum)
      },
      combOp = {
        //不同分区合并(count，sumFeatures)
        case ((weightSum1, featureSum1), (weightSum2, featureSum2)) =>
          BLAS.axpy(1.0, featureSum2, featureSum1)
          (weightSum1 + weightSum2, featureSum1)
      }).collect().sortBy(_._1)

    //类别标签数
    val numLabels = aggregated.length
    //对于每个类别的特征频数求和
    val numDocuments = aggregated.map(_._2._1).sum
    //类别标签列表
    val labelArray = new Array[Double](numLabels)
    //pi类别的先验概率
    val piArray = new Array[Double](numLabels)
    //theta各个特征在各个类别下的条件概率
    val thetaArray = new Array[Double](numLabels * numFeatures)
    val lambda = $(smoothing)
    //p(i)=log(p(yi )=log((i类别的特征数+平滑因子)/(所有类别总特征数+类别数*平滑因子))
    val piLogDenom = math.log(numDocuments + numLabels * lambda)

    var gauThetaFlag = false
    //所有类别的所有特征的均值方差
    val gauThetaFun : Array[Array[Double => Double]] = new Array(numLabels)
    //每个类别的所有特征的均值方差
    var gauThetaArr : Array[Double => Double] = Array.empty
    //所有类别分组后的所有特征向量
    val grouped = dataset.select(col($(labelCol)), col($(featuresCol))).rdd
      .map { row => (row.getDouble(0), row.getAs[Vector](1))}.groupByKey().sortBy(_._1).cache()

    var i = 0
    aggregated.foreach { case (label, (n, sumTermFreqs)) =>
      labelArray(i) = label
      piArray(i) = math.log(n + lambda) - piLogDenom //各类别的先验概率p(i)
      gauThetaArr = new Array(numFeatures) //放在里面new对象防止gauThetaFun指向同一个对象
      val thetaLogDenom = $(modelType) match {
        case Multinomial => math.log(sumTermFreqs.values.sum + numFeatures * lambda)
        case Bernoulli => math.log(n + 2.0 * lambda)
        case Gaussian => {
          gauThetaFlag = true
          //计算每个类别下各特征的均值和方差
          val meanVec : BreezeDenseVector[Double] = BreezeDenseVector(sumTermFreqs.values) / (n * 1.0)
          var varianceVec : linalg.Vector[Double] = Vectors.zeros(numFeatures).asBreeze
          val variVec = grouped.mapValues( feaVecs => {
            //特征属性相互独立，不求协方差矩阵，针对每个属性分开求方差
            varianceVec = feaVecs.map(feaVec => {
              (feaVec.toDense.asBreeze - meanVec) :* (feaVec.toDense.asBreeze - meanVec)
            }).reduce((v1, v2) => v1 + v2) / (n * 1.0)
            varianceVec
          }
          ).filter(_._1 == label).map(v => v._2).collect

          val meanVarZip : Array[(Double, Double)] = meanVec.toArray.zip(variVec(0).toArray)
          for (index <- 0 until numFeatures) {
            //返回(类别,[特征属性1的分布函数, ..., 特征属性n的分布函数])
            gauThetaArr(index) = gaussianTheta(meanVarZip(index)._1, meanVarZip(index)._2)
          }
          1.0//得到gauThetaArr，为了赋值所以随便返回值
        }
        case _ =>
          // This should never happen.
          throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
      }
      var j = 0
      while (j < numFeatures) {
        //各类别下各个特征属性的条件概率估计
        if (gauThetaFlag) {
          gauThetaFun(i) = gauThetaArr
          j = numFeatures //break
        }
        else
          thetaArray(i * numFeatures + j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
        j += 1
      }
      i += 1
    }

    val pi = Vectors.dense(piArray)
    val theta = new DenseMatrix(numLabels, numFeatures, thetaArray, true)
    //训练生成模型（包括类别标签列表，类别先验概率，各类别下每个特征的条件概率以及高斯参数）
    new NaiveBayesModel(uid, pi, theta).setOldLabels(labelArray).setNumLabels(numLabels).setGauFunc(gauThetaFun)
  }

  private[spark] def valuesCheck = {
    val modelTypeValue = $(modelType)
    val requireValues: Vector => Unit = {
      modelTypeValue match {
        case Multinomial =>
          requireNonnegativeValues
        case Bernoulli =>
          requireZeroOneBernoulliValues
        //加入高斯分布模型
        case Gaussian =>
          requireNumericValues
        case _ =>
          // This should never happen.
          throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
      }
    }
    requireValues
  }

  private[spark] def gaussianTheta(mean: Double, variance: Double)(x: Double): Double = {
    var theta = 1.0
    if (variance == 0.0) {
        1.0
    } else {
      theta = 1.0 / sqrt(2 * Pi * variance) * exp(-pow(x - mean, 2.0) / (2 * variance * variance))
      if (theta != 0.0)
        theta
      else
        1.0
    }
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): NaiveBayes = defaultCopy(extra)
}

@Since("1.6.0")
object NaiveBayes extends DefaultParamsReadable[NaiveBayes] {


  private[NaiveBayes] def requireNonnegativeValues(v: Vector): Unit = {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }

    require(values.forall(_ >= 0.0),
      s"Naive Bayes requires nonnegative feature values but found $v.")
  }

  private[NaiveBayes] def requireZeroOneBernoulliValues(v: Vector): Unit = {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }

    require(values.forall(v => v == 0.0 || v == 1.0),
      s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
  }

  private[NaiveBayes] def requireNumericValues(v: Vector): Unit = {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }

    var bCheckResult = true
    var errV = ""
    values.foreach(value => {
      try {
        val dCheckValue = java.lang.Double.parseDouble(value.toString)
        if (dCheckValue.isInstanceOf[Double] == false) {
          bCheckResult = false
          errV = value.toString
        }
      } catch {
        case e: Exception => {
          bCheckResult = false
          errV = value.toString
        }
      }
    })
    require(bCheckResult, s"Naive Bayes requires numeric feature values but found $errV.")
  }

  @Since("1.6.0")
  override def load(path: String): NaiveBayes = super.load(path)
}

private[spark] trait NaiveBayesParams extends PredictorParams with HasWeightCol {


  /** String name for multinomial model type. */
  private[ml] val Multinomial: String = "multinomial"

  /** String name for Bernoulli model type. */
  private[ml] val Bernoulli: String = "bernoulli"

  /** String name for Gaussian model type. */
  private[ml] val Gaussian: String = "gaussian"

  /* Set of modelTypes that NaiveBayes supports */
  private[ml] val supportedModelTypes = Set(Multinomial, Bernoulli, Gaussian)

  /**
    * The smoothing parameter.
    * (default = 1.0).
    * @group param
    */
  final val smoothing: DoubleParam = new DoubleParam(this, "smoothing", "The smoothing parameter.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSmoothing: Double = $(smoothing)

  /**
    * The model type which is a string (case-sensitive).
    * Supported options: "multinomial" and "bernoulli".
    * (default = multinomial)
    * @group param
    */
  final val modelType: Param[String] = new Param[String](this, "modelType", "The model type " +
    "which is a string (case-sensitive). Supported options: multinomial(default) ,bernoulli and gaussian.",
    ParamValidators.inArray[String](supportedModelTypes.toArray))

  /** @group getParam */
  final def getModelType: String = $(modelType)
}