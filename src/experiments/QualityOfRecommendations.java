package experiments;

import baselines.Nmf;
import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.Bmf;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.Precision;
import cf4j.qualityMeasures.Recall;
import cf4j.utils.PrintableQualityMeasure;
import cf4j.utils.Range;
import qualityMeasures.Ndcg;

public class QualityOfRecommendations {

	private static String CF4J_FILE = "datasets/ml100k.cf4j";
	private static int NUM_TOPICS = 9;

//	private static String CF4J_FILE = "datasets/ml1m.cf4j";
//	private static int NUM_TOPICS = 12;

//	private static String CF4J_FILE = "datasets/nf3m.cf4j";
//	private static int NUM_TOPICS = 11;

	private static int NUM_ITERS = 50;
	private static double LAMBDA = 0.055;
	private static double GAMMA = 0.01;

	private static int [] NUM_RECOMMENDATIONS = Range.ofIntegers(1,1,15);
	private static double THRESHOLD = 4;

	public static void main (String [] args) {
		
		Kernel.getInstance().readKernel(CF4J_FILE);

		String [] series = {"ProposedMethod", "PMF", "NMF", "BNMF"};

		PrintableQualityMeasure precision = new PrintableQualityMeasure("Precision", NUM_RECOMMENDATIONS, series);
		PrintableQualityMeasure ndcg = new PrintableQualityMeasure("nDCG", NUM_RECOMMENDATIONS, series);


		ProposedMethod pm = new ProposedMethod(NUM_TOPICS, NUM_ITERS, LAMBDA, GAMMA);
		pm.train();
		Processor.getInstance().testUsersProcess(new FactorizationPrediction(pm));

		for (int numRecommendations : NUM_RECOMMENDATIONS) {
			Processor.getInstance().testUsersProcess(new Precision(numRecommendations, THRESHOLD));
			precision.putError(numRecommendations, "ProposedMethod", Kernel.getInstance().getQualityMeasure("Precision"));

			Processor.getInstance().testUsersProcess(new Ndcg(numRecommendations));
			ndcg.putError(numRecommendations, "ProposedMethod", Kernel.getInstance().getQualityMeasure("NDCG"));
		}


		Pmf pmf = new Pmf(NUM_TOPICS, NUM_ITERS, LAMBDA, GAMMA, false);
		pmf.train();
		Processor.getInstance().testUsersProcess(new FactorizationPrediction(pmf));

		for (int numRecommendations : NUM_RECOMMENDATIONS) {
			Processor.getInstance().testUsersProcess(new Precision(numRecommendations, THRESHOLD));
			precision.putError(numRecommendations, "PMF", Kernel.getInstance().getQualityMeasure("Precision"));

			Processor.getInstance().testUsersProcess(new Ndcg(numRecommendations));
			ndcg.putError(numRecommendations, "PMF", Kernel.getInstance().getQualityMeasure("NDCG"));
		}



		Nmf nmf = new Nmf(NUM_TOPICS, NUM_ITERS);
		nmf.train();
		Processor.getInstance().testUsersProcess(new FactorizationPrediction(nmf));

		for (int numRecommendations : NUM_RECOMMENDATIONS) {
			Processor.getInstance().testUsersProcess(new Precision(numRecommendations, THRESHOLD));
			precision.putError(numRecommendations, "NMF", Kernel.getInstance().getQualityMeasure("Precision"));

			Processor.getInstance().testUsersProcess(new Ndcg(numRecommendations));
			ndcg.putError(numRecommendations, "NMF", Kernel.getInstance().getQualityMeasure("NDCG"));
		}



		Bmf bnmf = new Bmf(NUM_TOPICS, NUM_ITERS, 0.7, 5);
		bnmf.train();
		Processor.getInstance().testUsersProcess(new FactorizationPrediction(bnmf));

		for (int numRecommendations : NUM_RECOMMENDATIONS) {
			Processor.getInstance().testUsersProcess(new Precision(numRecommendations, THRESHOLD));
			precision.putError(numRecommendations, "BNMF", Kernel.getInstance().getQualityMeasure("Precision"));

			Processor.getInstance().testUsersProcess(new Ndcg(numRecommendations));
			ndcg.putError(numRecommendations, "BNMF", Kernel.getInstance().getQualityMeasure("NDCG"));
		}

		precision.print();
		ndcg.print();
	}
}
