
# pandas is a Python library used for data manipulation and analysis. 
# It offers data structures like DataFrame which are efficient for data manipulation.
import pandas as pd

# numpy is a Python library used for working with arrays. 
# It also has functions for working in the domain of linear algebra, fourier transform, and matrices.
import numpy as np

# Import the concurrent execution handling library
from concurrent.futures import ProcessPoolExecutor

# Import modules related to trading operations and dataset preparation
from strategy_operation import *
from generate_dataset import *





# Define a class to optimize trading strategies using genetic algorithms
class TradeStraOpti:

    def __init__(self,
                 
            amount_stock,
            starting_funds,
            commission,
            data_frame,
            stop_lose,
            numb_generations,
            size_generation,
            numb_genes,
            range_gene,
            prob_mutation,
            gene_prob_mutation,
            numb_best_select,
            fitness_fn,
            elite_gene= 0
            ):
        
        """
        Initialize the Trade Strategy Optimizer class with necessary parameters for both trading and 
        running a genetic algorithm. This setup includes parameters for trading conditions and settings 
        for genetic operations like mutation and selection.

        Parameters:
        - amount_stock: Initial stock quantity for trading simulations.
        - starting_funds: Initial amount of funds available for trading.
        - commission: Commission cost per trade, as a percentage.
        - data_frame: DataFrame containing historical market data used for backtesting.
        - stop_lose: The stop loss limit to control maximum loss per trade.
        - numb_generations: Number of generations for which the genetic algorithm will run.
        - size_generation: Number of individuals (trading strategies) in each generation.
        - numb_genes: Number of parameters (genes) in each trading strategy.
        - range_gene: List of tuples specifying the allowable range for each gene.
        - prob_mutation: Probability of mutating a gene during the genetic process.
        - gene_prob_mutation: Probability of a mutation occurring at each gene.
        - numb_best_select: Number of top individuals to select for reproduction.
        - fitness_fn: Function used to evaluate the fitness of each strategy.
        - elite_gene: Optional elite gene that can be carried over to new generations.

        """
        
        # Initialize trading related parameters
        self.amount_stock =amount_stock
        self.starting_funds =starting_funds
        self.commission = commission
        self.data_frame = data_frame
        self.stop_lose =stop_lose

        # Initialize genetic algorithm parameters
        self.numb_genes = numb_genes
        self.prob_mutation= prob_mutation
        self.numb_best_select= numb_best_select
        self.gene_prob_mutation=gene_prob_mutation
        self.numb_generations = numb_generations
        self.size_generation = size_generation
        self.range_gene =range_gene
        self.fitness_fn = fitness_fn
        self.elite_gene = elite_gene
        

    def population_create(self, size_population):
        """
        Generate an initial population of trading strategies. This function creates a list of trading strategies,
        where each strategy is defined by its genes. It includes the option to carry over an elite gene from
        the previous generation to ensure high-quality solutions are not lost.
        """
        population = []  # Initialize an empty list to hold the population of strategies
        # If an elite gene is defined, include it in the initial population and reduce the required new individuals by one
        if self.elite_gene:
            size_population_ = size_population - 1 # Reduce the population size to make room for the elite gene
            population.append(self.elite_gene) # Add the elite gene to the population
        else:
            size_population_ = size_population # Set the effective population size without an elite gene

        # Fill the rest of the population with new individuals created via the individual_create method
        for i in range(size_population_):
            population.append(self.individual_create())  # Append a new individual to the population
        
        return population # Return the complete population of strategies


    def individual_create(self):
        """
        Create an individual trading strategy with random genes. This method initializes 
        an individual's genetic makeup based on predefined gene range settings. It generates 
        either floating-point or integer genes depending on the specified range.
        """
        individual = []

        # Loop through the gene range definitions
        for i in range(len(self.range_gene)):  # Use the length of `range_gene` list instead of `numb_genes`
            if self.range_gene[i][1] == 1:  # Check if the upper limit of the gene range is 1 (for floating-point genes)
                # Generate a floating-point number within the specified range
                gene = np.random.uniform(self.range_gene[i][0], self.range_gene[i][1])
            else:
                # Generate an integer number within the specified range
                gene = np.random.randint(self.range_gene[i][0], self.range_gene[i][1] + 1)
            individual.append(gene)  # Append the gene to the individual's genetic sequence
        return individual



    def population_mutate(self, population):
        """
        Apply mutation to an entire population of strategies. This function iterates over each strategy
        in the population and probabilistically decides whether to apply mutations to it. This process
        introduces variability into the population, helping to explore new areas of the solution space.
        """
        list_mutated = []  # Initialize an empty list to store the mutated population

        # Check if mutation should occur for this individual based on the mutation probability
        for individual in population:
            mutated_individual = individual # Start with the individual unchanged

        # Check if mutation should occur for this individual based on the mutation probability
            if np.random.random() < self.prob_mutation:
                # Call mutate_individual to apply mutations to the individual's parameters
                mutated_individual = self.mutate_individual(individual)
            
            # Add the potentially mutated individual to the new population list
            list_mutated.append(mutated_individual)

        return list_mutated # Return the new population, which includes the mutated individuals



    def mate_parents(self, parents, numb_offspring): #input = parents pool, select 2 parents randomly, mate for n_list_offsprings count
        """
        Generate offspring by combining genes of two parent strategies. This method performs crossover between pairs
        of parent strategies to create new offspring. It uses a random mask to decide which genes to inherit from
        each parent.
        """
        numb_parent = len(parents)   # Determine the number of available parents
        list_offsprings = [] # Initialize an empty list to store offspring strategies

        for i in range(numb_offspring):
            # Select two random parents from the pool
            selected_parent = parents[np.random.randint(0, numb_parent)]
            selected_parent2 = parents[np.random.randint(0, numb_parent)]
        
            # Create a random mask to apply crossover between two parents
            random_mask1 = np.random.randint(0,2,size=np.array(selected_parent).shape) # Mask for genes to take from the first parent
            random_mask2 = np.logical_not(random_mask1) # Inverse mask for genes to take from the second parent

            # Create the child by combining genes from both parents according to the masks
            child = np.add(np.multiply(selected_parent, random_mask1), np.multiply(selected_parent2, random_mask2))
            list_offsprings.append(child) # Add the newly created offspring to the list

        return list_offsprings # Return the list of offspring
        


    def mutate_individual(self, individual):  
        """
        Mutate an individual strategy by randomly altering its genes according to predefined probabilities.
        This introduces variations which can potentially lead to finding better solutions in the genetic algorithm.
        """
        mutated_individual = [] # Initialize a list to store the mutated genes

        # Iterate through each gene in the individual's strategy
        for i in range(self.numb_genes):
            single_gene = individual[i] # Current gene value
            # Decide randomly if this gene should be mutated, based on the gene mutation probability
            if np.random.random() < self.gene_prob_mutation:
                # Choose the type of mutation: brute force or fine-tuning
                if np.random.random() < 0.5:
                    # Brute force mutation: randomly set the gene within its allowed range
                    single_gene = np.random.randint(self.range_gene[i][0], self.range_gene[i][1]+1)
                
                else: 
                    # Fine-tuning mutation: adjust the gene by a small random factor within its range
                    range_right = self.range_gene[i][1]
                    range_left = self.range_gene[i][0]
                    gene_dist = range_right - range_left # Distance between the maximum and minimum allowable values
                    if gene_dist == 1:
                        single_gene = np.random.randint(0,2)

                    else: 
                        # Calculate new gene value by randomly adjusting within +/- 33% of its range
                        x = individual[i] + gene_dist / 3 * (2*np.random.random()-1) #move randomly between +/- 33% from where the gene is
                        # Ensure the new gene value does not exceed its range; wrap around if necessary
                        if x > range_right:
                            x = (x - range_right) + range_left # Wrap around to stay within the upper limit
                        elif x < range_left:
                            x = range_right - (range_left - x) # Wrap around to stay within the lower limit

                        single_gene = int(x) # Convert the gene to an integer if needed
                
            # Add the (mutated or not) gene to the new individual
            mutated_individual.append(single_gene)

        return mutated_individual # Return the newly created individual with possibly mutated genes




    def best_gene(self, population, n_best):
        """
        Select the best performing strategies from the population. This method evaluates the fitness of each
        strategy and selects the top performers to potentially serve as parents for the next generation.
        """
        fitness = [] # Initialize a list to store fitness results

        # Use ProcessPoolExecutor for parallel execution to speed up the fitness evaluation
        with ProcessPoolExecutor(max_workers=10) as executor:
        # Map the fitness_evaluate_function function over all individuals in the population
        # Each individual is enumerated to maintain their indices for later reference
            for r in executor.map(self.fitness_evaluate_function, enumerate(population)):
                fitness.append(r) # Append each result to the fitness list
                print("  >",r) # Print each fitness result
        
        # Create a DataFrame from the fitness results and sort it by the fitness score, descending
        cost_tmp = pd.DataFrame(fitness).sort_values(by=1, ascending=False).reset_index(drop=True)
        # Extract indices of the top n_best individuals
        selected_parents_idx = list(cost_tmp.iloc[:n_best, 0])
        # Retrieve the actual individuals from the population using their indices
        selected_parents = [parent for idx,parent in enumerate(population) if idx in selected_parents_idx] 

        # Print summary statistics of the population's performance
        print('\n','>Best Score: {}\n >Average Score: {}\n >Worst Score: {}\n'.format(cost_tmp[1].max(), cost_tmp[1].mean(), cost_tmp[1].min()))
        print('Best genes is ', cost_tmp.iloc[0][2],'\n')


        return cost_tmp, selected_parents  #Return the DataFrame of scores and the list of selected parent strategies


    def excute_algo(self):
        """
        Execute the genetic algorithm over multiple generations to optimize strategies.
        This function handles the loop through each generation, selecting the best candidates,
        mating them to create new generations, and mutating the new generations to explore new possibilities.
        """
        # Create an initial population of trading strategies
        parent_gen = self.population_create(self.size_generation)

        # Loop through the specified number of generations
        for i in range(self.numb_generations):
            print('>> Generation - ',i) # Display the current generation


            # Select the best strategies based on fitness to act as parents for the next generation
            _, top_parent_pool = self.best_gene(parent_gen, self.numb_best_select)


            # Mate the selected parents to create a new generation of strategies
            new_gen = self.mate_parents(top_parent_pool, self.size_generation)


            # Mutate the new generation to introduce variability
            parent_gen = self.population_mutate(new_gen)


        # At the end of all generations, select the best strategies from the final generation   
        record_mutation, best_child = self.best_gene(parent_gen, 10)
        return record_mutation, best_child # Return the performance of the best strategies and their genetic makeup




    def fitness_evaluate_function(self, input):
        """
        Evaluate the fitness of an individual strategy using multiprocessing. This function is designed
        to be called in parallel for each member of a population, assessing each strategy's performance 
        based on simulated trading results.
        """
        index, individual = input # Unpack the tuple input which contains an index and an individual's strategy

        # Simulate trading using the individual's strategy parameters to calculate net profit
        # The backtesting function uses historical data to evaluate the profitability of the strategy
        netprofit = backtesting(
            self.data_frame, # Historical market data
            individual, # Strategy parameters
            self.stop_lose,  # Stop loss threshold
            wallet=self.starting_funds, # Initial funds available for trading
            commission=self.commission,  # Trading commission rate
            trade_on_close=True # Specifies whether trading takes place at the closing of a trading period
        )

        
        return (index, netprofit, individual)



if __name__ == '__main__':
    # Set initial conditions for the trading simulation and genetic algorithm optimization.
    starting_funds = 19980605  # Initial capital for the trading account
    amount_stock = 12345678    # Initial amount of stock in the portfolio
    stop_lose = 0.01           # Stop loss threshold as a decimal

    # Check if the market prices data file exists; if not, retrieve and preprocess data.
    if not os.path.isfile('data/market_prices.csv'):
        data_frame = Read_data() # Retrieve market data from an external source or database
        data_frame = generate_new_frame(data_frame) # Generate necessary trading features like indicators
        complete_data(data_frame) # Save the processed dataframe to a CSV file
        data_frame = data_frame.set_index(pd.to_datetime(data_frame['date'].apply(lambda x: unix_to_date_convert(x/1000))))
        print('Done. File saved!')
    
    else: 
        # If data exists, load from the CSV file and set the date column as the index
        data_frame = pd.read_csv('data/market_prices.csv')
        data_frame = data_frame.set_index(pd.to_datetime(data_frame['date'].apply(lambda x: unix_to_date_convert(x/1000))))

    # Define the gene range for the genetic algorithm parameters:
    # These ranges define the initial settings for each gene in an individual strategy.


    
    range_gene = [(10,20), (60,80), (20,40),   #rsi length, overbought, oversold
            (0,1),      #fibo level
            (10,20), (-30,-10), (-90,-70), #williams length, overbought, oversold
            (0,1), (0,1),  #hammer shooting # haamer # hammer
            (0,1), (0,1), #bull bear engulf # bear  # nothing
            (0,1), (0,1), #two green red bars # nothing # both
            (0,1), (0,1), #simple rsi, simple williams
            ]


    commission = 0.011 # plot and output testing = 0.01 /for testing purpose

    # Instantiate the TradeStraOpti class with initial settings
    SO = TradeStraOpti(
        fitness_fn=0, 
        numb_generations=1,  # Number of generations to evolve
        size_generation=10,   # Number of individuals per generation
        numb_genes=15,        # Number of genes in each individual # 7
        range_gene=range_gene, # Range of values each gene can take
        prob_mutation=0.4,   # Pro5ability of mutation per individual
        gene_prob_mutation=0.6, # Probability of mutation per gene
        numb_best_select=8,  # Number of top performers to select for breeding
        data_frame=data_frame, # DataFrame containing market data
        stop_lose=stop_lose,  # Stop loss threshold
        amount_stock = amount_stock, # Amount of stock in the initial portfolio
        starting_funds=starting_funds, # Starting funds in the trading account
        commission = commission, # Commission rate for trades
        elite_gene = 0   # Initial elite gene to seed the population
        )

    # Run the genetic algorithm to optimize trading strategies
    record_mutation, best_child = SO.excute_algo()
    # Save the results of the genetic algorithm to a CSV file
    record_mutation.to_csv('data/result.csv')
    # print(record_mutation)
    #print(best_child)
    