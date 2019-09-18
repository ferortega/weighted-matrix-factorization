package experiments;

import cf4j.Item;
import cf4j.ItemsPartible;
import cf4j.Kernel;
import cf4j.Processor;
import cf4j.User;
import cf4j.UsersPartible;
import cf4j.model.matrixFactorization.FactorizationModel;
import cf4j.utils.Methods;

public class ProposedMethod implements FactorizationModel {

	private final static String USER_FACTORS_KEY = "pm-user-factors";
	private final static String ITEM_FACTORS_KEY = "pm-item-factors";

	private double gamma;
	private double lambda;
	private int numFactors;
	private int numIters;


	public ProposedMethod(int numFactors, int numIters, double lambda, double gamma) {

		this.numFactors = numFactors;
		this.numIters = numIters;
		this.lambda = lambda;
		this.gamma = gamma;

		for (int u = 0; u < Kernel.gi().getNumberOfUsers(); u++) {
			this.setUserFactors(u, this.random(this.numFactors, -1, 1));
		}

		for (int i = 0; i < Kernel.gi().getNumberOfItems(); i++) {
			this.setItemFactors(i, this.random(this.numFactors, -1, 1));
		}
	}

	public int getNumberOfTopics () {
		return this.numFactors;
	}

	
	public double getLambda () {
		return this.lambda;
	}

	public double getGamma () {
		return this.gamma;
	}

	public void train () {

		System.out.println("\nProcessing Propsed Method...");

		for (int iter = 1; iter <= this.numIters; iter++) {

			// ALS: fix q_i and update p_u -> fix p_u and update q_i
			Processor.getInstance().usersProcess(new UpdateUsersFactors(), false);
			Processor.getInstance().itemsProcess(new UpdateItemsFactors(), false);

			if ((iter % 10) == 0) System.out.print(".");
			if ((iter % 100) == 0) System.out.println(iter + " iterations");
		}
	}


	public double [] getUserFactors (int userIndex) {
		User user = Kernel.gi().getUsers()[userIndex];
		return (double []) user.get(USER_FACTORS_KEY);
	}

	private void setUserFactors (int userIndex, double [] factors) {
		User user = Kernel.gi().getUsers()[userIndex];
		user.put(USER_FACTORS_KEY, factors);
	}

	public double [] getItemFactors (int itemIndex) {
		Item item = Kernel.gi().getItems()[itemIndex];
		return (double []) item.get(ITEM_FACTORS_KEY);
	}

	private void setItemFactors (int itemIndex, double [] factors) {
		Item item = Kernel.gi().getItems()[itemIndex];
		item.put(ITEM_FACTORS_KEY, factors);
	}


	public double getPrediction (int userIndex, int itemIndex) {
		double [] pu = this.getUserFactors(userIndex);
		double [] qi = this.getItemFactors(itemIndex);
		return Methods.dotProduct(pu, qi);
	}

	
	private class UpdateUsersFactors implements UsersPartible {

		@Override
		public void beforeRun() { }

		@Override
		public void run (int userIndex) {

			User user = Kernel.gi().getUsers()[userIndex];
			
			double [] pu = ProposedMethod.this.getUserFactors(userIndex);

			int itemIndex = 0;

			for (int j = 0; j < user.getNumberOfRatings(); j++) {

				while (Kernel.gi().getItems()[itemIndex].getItemCode() < user.getItems()[j]) itemIndex++;

				double [] qi = ProposedMethod.this.getItemFactors(itemIndex);

				double rating = user.getRatingAt(j);
				double prediction = ProposedMethod.this.getPrediction(userIndex, itemIndex);

				double error = rating - prediction;
				
				double weight = Math.sqrt(rating / Kernel.getInstance().getMaxRating());

				for (int k = 0; k < ProposedMethod.this.numFactors; k++)	{
					pu[k] += ProposedMethod.this.gamma * (weight * error * qi[k] - ProposedMethod.this.lambda * pu[k]);
				}
			}
		}

		@Override
		public void afterRun() { }
	}

	
	private class UpdateItemsFactors implements ItemsPartible {

		@Override
		public void beforeRun() { }

		@Override
		public void afterRun() { }

		@Override
		public void run(int itemIndex) {

			Item item = Kernel.gi().getItems()[itemIndex];
			
			double [] qi = ProposedMethod.this.getItemFactors(itemIndex);

			int userIndex = 0;

			for (int v = 0; v < item.getNumberOfRatings(); v++)
			{
				while (Kernel.gi().getUsers()[userIndex].getUserCode() < item.getUsers()[v]) userIndex++;

				double [] pu = ProposedMethod.this.getUserFactors(userIndex);


				double rating = item.getRatingAt(v);
				double prediction = ProposedMethod.this.getPrediction(userIndex, itemIndex);

				double error = rating - prediction;

				double weight = Math.sqrt(rating / Kernel.getInstance().getMaxRating());
				
				for (int k = 0; k < ProposedMethod.this.numFactors; k++) {
					qi[k] += ProposedMethod.this.gamma * (weight * error * pu[k] - ProposedMethod.this.lambda * qi[k]);
				}
			}
		}
	}


	private double random (double min, double max) {
		return Math.random() * (max - min) + min;
	}

	private double [] random (int size, double min, double max) {
		double [] d = new double [size];
		for (int i = 0; i < size; i++) d[i] = this.random(min, max);
		return d;
	}	
}
