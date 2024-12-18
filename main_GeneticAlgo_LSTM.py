import numpy as np
import tensorflow as tf
from tensorflow import keras
from deap import base, creator, tools, algorithms
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class AdvancedLSTMGeneticOptimizer:
    def __init__(self,
                 problem_dimension=10,
                 population_size=100,
                 max_generations=50,
                 lower_bound=-10,
                 upper_bound=10):
        """
        Initialize Advanced Genetic Algorithm with LSTM Enhancement
        """
        # Reset potential creator conflicts
        try:
            del creator.FitnessMin
            del creator.Individual
        except Exception:
            pass

        # Problem parameters
        self.problem_dimension = problem_dimension
        self.population_size = population_size
        self.max_generations = max_generations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Tracking variables
        self.fitness_history = []

        # Random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # Prepare genetic algorithm components
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialize toolbox and LSTM model
        self.toolbox = self._setup_genetic_algorithm()
        self.lstm_model, self.X_test, self.y_test = self._build_and_evaluate_lstm_model()

    def _generate_simulated_data(self):
        """
        Generate simulated training data for LSTM
        """
        X = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(1000, self.problem_dimension)
        )

        # Synthetic fitness function
        y = np.sum(X ** 2, axis=1)

        return X, y

    def _build_and_evaluate_lstm_model(self):
        """
        Build and evaluate LSTM model with comprehensive visualization
        """
        # Generate simulated data
        X, y = self._generate_simulated_data()

        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Build LSTM model
        model = keras.Sequential([
            keras.layers.LSTM(50, activation='relu', input_shape=(1, self.problem_dimension)),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )

        # Model evaluation
        y_pred = model.predict(X_test)

        # Inverse transform predictions
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_test_inv = scaler_y.inverse_transform(y_test)

        # Plot only required visualizations
        self._plot_lstm_training_history(history)
        self._plot_lstm_prediction_error(y_test_inv, y_pred_inv)

        return model, X_test, y_test

    def _setup_genetic_algorithm(self):
        """
        Setup genetic algorithm toolbox
        """
        toolbox = base.Toolbox()

        toolbox.register(
            "attr_float",
            random.uniform,
            self.lower_bound,
            self.upper_bound
        )

        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=self.problem_dimension
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
        )

        toolbox.register("evaluate", self._fitness_function)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=1,
            indpb=0.2
        )
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def _fitness_function(self, individual):
        """
        Evaluate fitness of an individual
        """
        standard_fitness = sum(x ** 2 for x in individual)

        individual_array = np.array(individual).reshape(1, 1, -1)
        lstm_prediction = self.lstm_model.predict(individual_array)[0][0]

        combined_fitness = standard_fitness * 0.7 + lstm_prediction * 0.3

        return (combined_fitness,)

    def optimize(self):
        """
        Run genetic algorithm optimization with comprehensive evaluation
        """
        population = self.toolbox.population(n=self.population_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution process
        for gen in range(self.max_generations):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            # Track fitness history
            current_fitness = [ind.fitness.values[0] for ind in population]
            self.fitness_history.append(current_fitness)

        # Generate fitness convergence plot
        self._plot_fitness_convergence(logbook)

        # Get best solution
        best_ind = tools.selBest(population, k=1)[0]

        # Generate the optimized schedule
        optimized_schedule = self._generate_schedule_table(best_ind)

        # Generate the original schedule
        original_schedule = self._generate_original_schedule()

        # Display both schedules side by side
        print("\nOriginal Schedule and Optimized Schedule (Side by Side):")
        combined_schedule = pd.concat([original_schedule.set_index('Nurse'), optimized_schedule.set_index('Nurse')],
                                      axis=1)
        combined_schedule.columns = ['Original Shift', 'Optimized Shift']
        print(combined_schedule)

        return {
            'best_solution': best_ind,
            'best_fitness': best_ind.fitness.values[0],
            'logbook': logbook
        }

    def _generate_original_schedule(self):
        """
        Generates a basic schedule (this could be a random or heuristic schedule).
        """
        # Generate a random schedule (for demonstration purposes, you can adjust as needed)
        shifts = ['Morning', 'Afternoon', 'Night']  # Example shifts
        nurses = [f'Nurse {i + 1}' for i in range(self.problem_dimension)]  # Example nurse names

        # Create an empty DataFrame with 'Nurse' and 'Shift' columns
        schedule = pd.DataFrame(columns=['Nurse', 'Shift'])

        # Generate the schedule randomly
        schedule_rows = []  # List to accumulate the rows
        for i, nurse in enumerate(nurses):
            shift = random.choice(shifts)  # Randomly assign a shift to each nurse
            schedule_rows.append({'Nurse': nurse, 'Shift': shift})

        # Use pd.concat to add the rows to the schedule DataFrame
        schedule = pd.concat([schedule, pd.DataFrame(schedule_rows)], ignore_index=True)

        return schedule

    def _generate_schedule_table(self, best_individual):
        """
        Generate schedule table from the best individual
        """
        shifts = ['Morning', 'Afternoon', 'Night']
        nurses = [f'Nurse {i + 1}' for i in range(self.problem_dimension)]
        schedule = pd.DataFrame(columns=['Nurse', 'Shift'])

        # Generate schedule based on the best individual (this part depends on how you interpret the individual)
        schedule_rows = []
        for i, nurse in enumerate(nurses):
            shift_index = int(best_individual[i] % len(shifts))  # Wrap around shifts
            schedule_rows.append({'Nurse': nurse, 'Shift': shifts[shift_index]})

        schedule = pd.concat([schedule, pd.DataFrame(schedule_rows)], ignore_index=True)
        return schedule

    def _plot_fitness_convergence(self, logbook):
        """
        Plot fitness convergence graph
        """
        plt.figure(figsize=(10, 5))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("avg")
        plt.plot(gen, fit_mins, label='Best Fitness', color='green')
        plt.plot(gen, fit_maxs, label='Worst Fitness', color='red')
        plt.plot(gen, fit_avgs, label='Average Fitness', color='blue')
        plt.title('Fitness Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('fitness_convergence.png')
        plt.close()

    def _plot_lstm_training_history(self, history):
        """
        Plot LSTM model training history
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lstm_training_history.png')
        plt.close()

    def _plot_lstm_prediction_error(self, y_true, y_pred):
        """
        Plot LSTM prediction error
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', lw=2)
        plt.title('LSTM Prediction vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.tight_layout()
        plt.savefig('lstm_prediction_scatter.png')
        plt.close()

    def display_generated_data(self):
        """
        Display some of the generated data
        """
        X, y = self._generate_simulated_data()
        print("\nSample of Generated Data:")
        print("Feature Data (X) Sample:")
        print(X[:5])  # Display the first 5 rows of X
        print("\nTarget Data (y) Sample:")
        print(y[:5])  # Display the first 5 rows of y


def main():
    """
    Main function to demonstrate advanced optimization
    """
    # Initialize optimizer
    optimizer = AdvancedLSTMGeneticOptimizer(
        problem_dimension=10,  # Optimization problem dimension
        population_size=100,  # Number of individuals in population
        max_generations=20,  # Maximum number of generations
        lower_bound=-10,  # Lower bound of search space
        upper_bound=10  # Upper bound of search space
    )

    # Display generated data
    optimizer.display_generated_data()

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print("\nOptimization Results:")
    print("Best Solution:", result['best_solution'])
    print("Best Fitness:", result['best_fitness'])


# Ensure the script can be run directly
if __name__ == "__main__":
    main()
