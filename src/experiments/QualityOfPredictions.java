package experiments;

import baselines.Nmf;
import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.*;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import cf4j.utils.PrintableQualityMeasure;
import cf4j.utils.Range;

public class QualityOfPredictions {

	private static String CF4J_FILE = "datasets/ml100k.cf4j";
//	private static String CF4J_FILE = "datasets/ml1m.cf4j";
//	private static String CF4J_FILE = "datasets/nf3m.cf4j";

	private static int [] NUM_TOPICS = Range.ofIntegers(7, 1, 8);

	private static int NUM_ITERS = 50;
	private static double LAMBDA = 0.055;
	private static double GAMMA = 0.01;

	
	public static void main (String [] args) {
		
		Kernel.getInstance().readKernel(CF4J_FILE);

		String [] series = {"ProposedMethod", "PMF", "NMF", "BNMF"};

		PrintableQualityMeasure mae = new PrintableQualityMeasure("MAE", NUM_TOPICS, series);

		for (int numTopics : NUM_TOPICS) {

			ProposedMethod pm = new ProposedMethod(numTopics, NUM_ITERS, LAMBDA, GAMMA);
			pm.train();

			Processor.getInstance().testUsersProcess(new FactorizationPrediction(pm));

			Processor.getInstance().testUsersProcess(new MAE());
			mae.putError(numTopics, "ProposedMethod", Kernel.getInstance().getQualityMeasure("MAE"));


			Pmf pmf = new Pmf(numTopics, NUM_ITERS, LAMBDA, GAMMA, false);
			pmf.train();

			Processor.getInstance().testUsersProcess(new FactorizationPrediction(pmf));

			Processor.getInstance().testUsersProcess(new MAE());
			mae.putError(numTopics, "PMF", Kernel.getInstance().getQualityMeasure("MAE"));


			Nmf nmf = new Nmf(numTopics, NUM_ITERS);
			nmf.train();

			Processor.getInstance().testUsersProcess(new FactorizationPrediction(nmf));

			Processor.getInstance().testUsersProcess(new MAE());
			mae.putError(numTopics, "NMF", Kernel.getInstance().getQualityMeasure("MAE"));


			Bmf bnmf = new Bmf(numTopics, NUM_ITERS, 0.7, 5);
			bnmf.train();

			Processor.getInstance().testUsersProcess(new FactorizationPrediction(bnmf));

			Processor.getInstance().testUsersProcess(new MAE());
			mae.putError(numTopics, "BNMF", Kernel.getInstance().getQualityMeasure("MAE"));
		}

		mae.print();
	}
}
