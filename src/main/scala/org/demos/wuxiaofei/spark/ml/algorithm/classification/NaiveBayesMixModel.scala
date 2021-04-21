package org.demos.wuxiaofei.spark.ml.algorithm.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.awaken.platform.baseclazz.ml.predictor.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.predictor.PredictorParams
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.shared.HasWeightCol
import org.apache.spark.ml.awaken.platform.basicdependency.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.awaken.platform.component.persistent.serialize._
import org.apache.spark.ml.awaken.platform.util.ml.MLUtils
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Matrix, SparseVector, Vector}
import org.apache.spark.sql.Row


/**
  * author: wuxiaofei
  * date: 2018/9/29 11:22
  * description: {NaiveBayesMixModel: GaussianNBModel and Raw NaiveBayesModel}
  */

@Since("0.1")
trait BaseNaiveBayesModel {
  protected def tagBaseNBModel() : Unit = {}
}

/**
  * Model produced by [["NaiveBayes"]]
  * @param pi log of class priors, whose dimension is C (number of classes)
  * @param theta log of class conditional probabilities, whose dimension is C (number of classes)
  *              by D (number of features)
  */
@Since("1.5.0")
class NaiveBayesModel private[ml] (
                                    @Since("1.5.0") override val uid: String,
                                    @Since("2.0.0") val pi: Vector,
                                    @Since("2.0.0") val theta: Matrix)
  extends ProbabilisticClassificationModel[Vector, NaiveBayesModel] with BaseNaiveBayesModel
    with NaiveBayesParams with MLWritable{

  /**
    * mllib NaiveBayes is a wrapper of ml implementation currently.
    * Input labels of mllib could be {-1, +1} and mllib NaiveBayesModel exposes labels,
    * both of which are different from ml, so we should store the labels sequentially
    * to be called by mllib. This should be removed when we remove mllib NaiveBayes.
    */
  private[spark] var oldLabels: Array[Double] = null

  private[spark] def setOldLabels(labels: Array[Double]): this.type = {
    this.oldLabels = labels
    this
  }

  private[spark] var numLabels : Int = 0

  private[spark] def setNumLabels(numLabels : Int) : this.type = {
    this.numLabels = numLabels
    this
  }

  /**
    * gaussian function
    */
  private[spark] var gauThetaFun : Array[Array[Double => Double]] = new Array(numLabels)

  private[spark] def setGauFunc(func :Array[Array[Double => Double]]): this.type = {
    this.gauThetaFun = func
    this
  }

  /**
    * Bernoulli scoring requires log(condprob) if 1, log(1-condprob) if 0.
    * This precomputes log(1.0 - exp(theta)) and its sum which are used for the linear algebra
    * application of this condition (in predict function).
    */
  private lazy val (thetaMinusNegTheta, negThetaSum) = $(modelType) match {
    case Multinomial => (None, None)
    case Bernoulli =>
      //事件失败的概率
      val negTheta = theta.map(value => math.log(1.0 - math.exp(value)))
      val ones = new DenseVector(Array.fill(theta.numCols) {1.0})
      //事件成功的概率
      val thetaMinusNegTheta = theta.map { value =>
        value - math.log(1.0 - math.exp(value))
      }
      (Option(thetaMinusNegTheta), Option(negTheta.multiply(ones)))
    case _ =>
      // This should never happen.
      throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
  }

  @Since("1.6.0")
  override val numFeatures: Int = theta.numCols

  @Since("1.5.0")
  override val numClasses: Int = pi.size

  //log(p(x│y_i )p(y_i ))=log(p(x│y_i )) + log(p(y_i ))
  private def multinomialCalculation(features: Vector) = {
    val prob = theta.multiply(features)
    BLAS.axpy(1.0, pi, prob)
    prob
  }

  private def bernoulliCalculation(features: Vector) = {
    features.foreachActive((_, value) =>
      require(value == 0.0 || value == 1.0,
        s"Bernoulli naive Bayes requires 0 or 1 feature values but found $features.")
    )
    val prob = thetaMinusNegTheta.get.multiply(features)
    BLAS.axpy(1.0, pi, prob)
    BLAS.axpy(1.0, negThetaSum.get, prob)
    prob
  }

  private def gaussianCalculation(features: Vector) = {
    val probTheta : Array[Double] = new Array[Double](numLabels)
    for (i <- (0 until numLabels)) {
      probTheta(i) = gauThetaFun(i).zip(features.toArray).map {
        case (gauFunc, x) =>
          math.log(gauFunc(x))
      }.sum
      probTheta(i) += pi(i)
    }
    val prob = new DenseVector(probTheta)
    prob
  }

  override protected def predictRaw(features: Vector): Vector = {
    $(modelType) match {
      case Multinomial =>
        multinomialCalculation(features)
      case Bernoulli =>
        bernoulliCalculation(features)
      case Gaussian =>
        gaussianCalculation(features)
      case _ =>
        // This should never happen.
        throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
    }
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        val maxLog = dv.values.max
        while (i < size) {
          dv.values(i) = math.exp(dv.values(i) - maxLog)
          i += 1
        }
        val probSum = dv.values.sum
        i = 0
        while (i < size) {
          dv.values(i) = dv.values(i) / probSum
          i += 1
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in NaiveBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): NaiveBayesModel = {
    copyValues(new NaiveBayesModel(uid, pi, theta).setParent(this.parent), extra)
  }

  @Since("1.5.0")
  override def toString: String = {
    s"NaiveBayesModel (uid=$uid) with ${pi.size} classes"
  }

  @Since("1.6.0")
  override def write: MLWriter = new NaiveBayesModel.NaiveBayesModelWriter(this)
}


@Since("1.6.0")
object NaiveBayesModel extends MLReadable[NaiveBayesModel] {

  @Since("1.6.0")
  override def read: MLReader[NaiveBayesModel] = new NaiveBayesModelReader

  @Since("1.6.0")
  override def load(path: String): NaiveBayesModel = super.load(path)

  /** [[MLWriter]] instance for [[NaiveBayesModel]] */
  private[NaiveBayesModel] class NaiveBayesModelWriter(instance: NaiveBayesModel) extends MLWriter {

    private case class Data(pi: Vector, theta: Matrix)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: pi, theta
      val data = Data(instance.pi, instance.theta)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class NaiveBayesModelReader extends MLReader[NaiveBayesModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[NaiveBayesModel].getName

    override def load(path: String): NaiveBayesModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val vecConverted = MLUtils.convertVectorColumnsToML(data, "pi")
      val Row(pi: Vector, theta: Matrix) = MLUtils.convertMatrixColumnsToML(vecConverted, "theta")
        .select("pi", "theta")
        .head()
      val model = new NaiveBayesModel(metadata.uid, pi, theta)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

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
    "which is a string (case-sensitive). Supported options: multinomial (default) and bernoulli.",
    ParamValidators.inArray[String](supportedModelTypes.toArray))

  /** @group getParam */
  final def getModelType: String = $(modelType)
}