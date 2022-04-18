import json, copy, math, sys

import numpy as np
import random as rnd
import networkx as nx

from plotter import *
from reporter import *

from scipy.stats import boltzmann
from scipy.spatial import distance
from geopy.distance import geodesic

protected_constituencies = ['W07000041', # Ynys Môn (Anglesey)
                            'S14000051', # Orkney and Shetland
                            'S14000027', # Na h-Eileanan an Iar (Western Isles)
                            'E14000762'] # Isle of Wight

class Redistricter:

    def __init__(self, wards_cons_file, create_plotter=False, show_progress=True, results_folder='results/', log_file_location='data/logs/', boundary_files_location='../datasets/wards/boundaries/', verbose=True):
        """
        :param wards_cons_file: Location of ward and constituency JSON data
        :param create_plotter: Denotes whether to create a plotter object
        :param show_progress: Denotes whether to show a progress bar
        :param results_folder: Folder for results/graphs output
        :param log_file_location: Folder to write logs to, if none, no logs are created
        :param boundary_files_location: Directory containing boundary files
        """

        with open(wards_cons_file, 'r') as wards_cons_file:
            data = json.load(wards_cons_file)
            self.wards = data['wards']
            self.constituencies = data['constituencies']

        self.constituency_ids = list(self.constituencies.keys())
        self.ward_ids = list(self.wards.keys())

        self.k = len(self.constituency_ids)
        self.n = len(self.ward_ids)

        self.generate_matrices()

        self.stats_reporter = Reporter(progress_bar=show_progress, log_file_location=log_file_location)
        if create_plotter: self.stats_reporter.plotter = Plotter(verbose=verbose, boundary_files_location=boundary_files_location)

        self.results_folder = results_folder

    def generate_map(self, kmax, f_alpha=1, f_beta=1, improvements=100, f_n=1, reward_factor=0.8, penalisation_factor=0.6, compensation_factor=0.8, compactness_stage_length=0, electorate_deviation=0.05, video_filename=None, plot_random_colours=False, make_final_plots=True, show_plots=True, final_map_location=None, save_solution_location=None, verbose=False):
        """
        Generates a new constituency map using the outlined method.
        :param kmax: Number of stage 1 iterations
        :param f_alpha: Stage 1 fairness priority (Set to 0 in stage 2)
        :param f_beta: Stage 1 compactness priority (Set to 1 in stage 2)
        :param improvements: Number of steps made in the DBLS
        :param f_n: Number of random swaps per step in the DBLS
        :param reward_factor: Constituency reward factor for RL
        :param penalisation_factor: Constituency penalisation factor for RL
        :param compensation_factor: Constituency compensation factor for RL
        :param compactness_stage_length: Number of iterations for stage 2
        :param electorate_deviation: Allowed electorate deviation for proposed constituency
        :param video_filename: If given, used as the filename for video creation
        :param plot_random_colours: Denotes whether maps are plotted using random constituency colours
        :param make_final_plots: Denotes whether to draw final plots
        :param show_plots: Denotes whether to show plots (blocking)
        :param final_map_location: If given, the final map is plotted
        :param save_solution_location: If given, the final ward and constituency JSON are saved
        """

        if self.stats_reporter != None: self.stats_reporter.save_parameters(f_alpha, f_beta, improvements, reward_factor, penalisation_factor, compensation_factor, compactness_stage_length)
        if video_filename != None and self.stats_reporter.plotter == None:
            print('Class REDISTRICTER WARNING: Video filename given, but \'stats_reporter\' has no \'Plotter\'.')

        # Calculate electoral quota (average constituency electorate),
        # and record protected constituencies

        # First, we calculate the electoral quota (average constituency electorate),
        # record any protected constituencies and generate random colours if necessary
        if plot_random_colours: random_colours = {}
        else: random_colours = None

        total_electorate = 0
        self.protected_constituencies = {}
        for constituency_key in self.constituencies.keys():
            constituency = self.constituencies[constituency_key]
            if constituency['is_protected']: self.protected_constituencies[constituency_key] = constituency['wards']
            total_electorate += constituency['election_results']['electorate']

            if plot_random_colours: random_colours[constituency] = "#%06x" % rnd.randint(0, 0xFFFFFF)

        electoral_quota = int(total_electorate / len(self.constituencies.keys()))

        # Create initial solution, performing default selection, which does not change
        # ward assignments.
        initial_solution = Solution(self.wards, self.constituencies,
                                    [self.constituency_ids, self.ward_ids],
                                    self.probability_matrix, self.distance_matrix,
                                    protected_constituencies=self.protected_constituencies,
                                    selection_method='default',
                                    electoral_quota=electoral_quota,
                                    electorate_deviation=electorate_deviation,
                                    reporter=self.stats_reporter)
       
        # Record the initial fitness metric values, results and proportional vote shares
        self.f_alpha, self.f_beta = f_alpha, f_beta
        initial_fitness, initial_fairness, initial_compactness, initial_results = initial_solution.evaluate(self.f_alpha, self.f_beta, run_election=True)
        self.proportional_vote = initial_solution.proportional_votes

        if verbose:
            print('Initial solution:\n  Fitness: {0}\n  Fairness: {1}\n  Compactness: {2}\n'.format(round(initial_fitness, 6), round(initial_fairness, 6), round(initial_compactness, 6)))
            print_results(initial_results['national_votes'])
        
        if video_filename != None and self.stats_reporter.plotter != None:
            self.stats_reporter.record_video_frame(self.wards, self.constituencies, 0, text='Initial Solution', random_colours=random_colours, frame_repeats=3)
    
        # The total number of iterations is the 1st stage + 2nd stage
        k = 0
        kmax += compactness_stage_length
        no_changes = math.inf

        if compactness_stage_length > 0: stage_text = "(1st Stage)"
        else: stage_text = ""
        
        # Use the stats reporter to record the run statistics
        self.stats_reporter.kmax = kmax
        self.stats_reporter.start_time = time.time()
        self.stats_reporter.update_stats([initial_fitness, initial_fairness, initial_compactness], initial_results['national_votes'], k=k)

        while k < kmax and no_changes != 0:

            # Solutions are generated using the acceptance selection method
            current_solution = Solution(self.wards,
                                        self.constituencies,
                                        [self.constituency_ids, self.ward_ids],
                                        self.probability_matrix,
                                        self.distance_matrix,
                                        protected_constituencies=self.protected_constituencies,
                                        selection_method='acceptance-selection',
                                        electoral_quota=electoral_quota,
                                        electorate_deviation=electorate_deviation,
                                        selection_threshold=0.9,
                                        proportional_votes=self.proportional_vote,
                                        reporter=self.stats_reporter)

            current_solution.run_election(verbose=False)

            # The descent-based local search is then run to improve the solution
            current_fitness, current_wvs, current_lwr, current_results = current_solution.improve_solution(f_n, self.f_alpha, self.f_beta, improvements, hillclimb=True, verbose=False)

            # The probability matrix is then updated, recording the number of changes made
            no_changes = self.update_probabilities(current_solution, alpha=reward_factor,
                                                                     beta=penalisation_factor,
                                                                     gamma=compensation_factor)

            if no_changes == 0: print('\nNo changes were made during local search!')
            
            self.perform_probability_smoothing(0.8, 0.995)

            k += 1

            self.stats_reporter.update_stats([current_fitness, current_wvs, current_lwr], current_results['national_votes'], no_changes=no_changes, k=k, stage_text=stage_text)
            self.wards, self.constituencies = current_solution.wards, current_solution.constituencies

            if video_filename != None and self.stats_reporter.plotter != None:
                self.stats_reporter.record_video_frame(self.wards, self.constituencies, k, text='Iteration: {0}, Fitness: {1} {2}'.format(k, round(current_fitness, 4), stage_text), random_colours=random_colours)

            if compactness_stage_length != 0 and k == kmax - compactness_stage_length:
                self.f_alpha, self.f_beta = 0, 1
                stage_text = "(2nd stage)"

        self.f_alpha, self.f_beta = f_alpha, f_beta
        
        if self.stats_reporter.plotter != None:
            if video_filename != None:
                self.stats_reporter.generate_image_video(video_filename, delete_frames=False)
            if final_map_location != None:
                self.stats_reporter.plotter.plot_ward_party_support_map(self.wards, self.constituencies, metric='winner', value_type='constituency', image_savefile=final_map_location, random_colours=random_colours)

        if make_final_plots:
            self.stats_reporter.close(show_plot=show_plots, save_plot=self.results_folder+'performance.png', plot_title='Solution Fitness (ɑ='+str(f_alpha)+', β='+str(f_beta)+')', verbose=verbose)
            plot_results_comparison_bchart([initial_results['national_votes'], current_results['national_votes']], 'Party Seat Share Comparison: 2017 General Election and Model-generated Results', save_plot=self.results_folder+'prop_vote.png', show_plot=show_plots)
            plot_seats_grid(current_results['national_votes'], 'horizontal', title='Solution: Parliament Seat Shares', save_image=self.results_folder+'seat_share.png', show_image=show_plots)

        current_solution.run_election(verbose=verbose)

        if save_solution_location != None:
            current_solution.save_wards_constituencies(save_solution_location)

    def update_probabilities(self, new_solution, alpha, beta, gamma, clip_vals=True):
        """
        Updates probabilities by rewarding, penalising and compensating
        constituencies based on changes in ward assignments
        :param alpha: Reward factor
        :param beta: Penalisation factor
        :param gamma: Compensation factor
        :return int: Number of ward assignment changes 
        """

        if self.constituency_ids != new_solution.constituency_ids or self.ward_ids != new_solution.ward_ids:
            print('Class REDISTRICTER ERROR: Original and new solutions have different ward and constituency lists:\n  ↳ Ensure both solutions use the same id lists.')
            exit()

        no_changes = 0
        for ward_index, ward_id in enumerate(self.ward_ids):
            
            orig_constituency_id = self.wards[ward_id]['constituency_id']
            new_constituency_id = new_solution.wards[ward_id]['constituency_id']

            if orig_constituency_id != new_constituency_id: no_changes += 1

            for constituency_index, constituency_id in enumerate(self.constituency_ids):
                pij = self.probability_matrix[ward_index][constituency_index]
                
                # If the ward's original constituency is the same as its new constituency,
                # reward that constituency by a factor alpha.
                if orig_constituency_id == constituency_id and new_constituency_id == constituency_id:
                    pij = alpha + ((1 - alpha) * pij)
                
                #else:
                    #pij = (1 - alpha) * pij


                # Else, if the ward has been assigned differently, we penalise the original
                # constituency by a factor beta.
                if orig_constituency_id == constituency_id and new_constituency_id != constituency_id:
                    pij = (1 - gamma) * (1 - beta) * pij

                # We also compensate the new constituency assignment by a factor gamma.
                elif orig_constituency_id != constituency_id and new_constituency_id == constituency_id:
                    pij = gamma + ((1 - gamma) * (beta / (self.k - 1))) + ((1 - gamma) * (1 - beta) * pij)

                #else:
                    #pij = ((1 - gamma) * (beta / (self.k - 1))) + ((1 - gamma) * (1 - beta) * pij)

                if clip_vals and pij > 1: pij = 1
                elif clip_vals and pij < 0: pij = 0

                self.probability_matrix[ward_index][constituency_index] = pij

        return no_changes

    def perform_probability_smoothing(self, smoothing_probability, smoothing_coefficient):
        """
        Reduces values in the probability matrix to forget old decisions
        :param smoothing_probability: Smoothing threshold (0 <= x <= 1)
        :param smoothing_coefficient: Smoothing value (0 <= x <= 1)
        """
        for i in range(self.n):
            
            probability_row = self.probability_matrix[i]
            max_probability_index = list(probability_row).index(max(list(probability_row)))

            if self.probability_matrix[i][max_probability_index] > smoothing_probability:

                for j in range(self.k):
                    if j == max_probability_index:
                        pij = smoothing_coefficient * self.probability_matrix[i][j]
                    
                    else:
                        pij = (((1 - smoothing_coefficient) / (self.k - 1)) * self.probability_matrix[i][j]) + self.probability_matrix[i][j]

                    if pij > 1: pij = 1
                    elif pij < 0: pij = 0
                    self.probability_matrix[i][j] = pij

    def generate_matrices(self, distance_matrix_file="data/dst.npy", load_distances=True, euclidean_distance=False, save_txt=True, verbose=False):
        """
        Generates the probability and distance matrices for group selection.
        :param distance_matrix_file: Location of distance matrix file - .npy format
        :param load_distances: Denotes whether to load or create the distance matrix
        :param euclidean_distance: Denotes whether to calculate euclidean or geodesic distances
        :param save_txt: Denotes whether to save readable txt version of the distance matrix
        """

        self.probability_matrix = np.zeros((self.n, self.k), dtype=np.float64)
        self.distance_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        val = 0
        if verbose:
            print('Generating Matrices...')
            if not load_distances: print('  ↳ Creating distance matrix...')

        for i in range(len(self.ward_ids)):
            ward = self.ward_ids[i]
            constituency_id = self.wards[ward]['constituency_id']
            if constituency_id in self.constituency_ids:
                constituency_index = self.constituency_ids.index(constituency_id)
                self.probability_matrix[i][constituency_index] = 1
            else:
                print('Class REDISTRICTER ERROR: Could not generate probability matrix:\n  ↳ Constituency \''+self.wards[ward]['constituency_id']+'\' not found.')
                exit()

            if not load_distances:
                for j in range(len(self.ward_ids)):
                    if self.distance_matrix[i][j] == 0:
                        if i != j and self.ward_ids[j][0] == ward[0]:
                            val += 1
                            ward_i_centroid = self.wards[ward]['geography']['centroid']
                            ward_j_centroid = self.wards[self.ward_ids[j]]['geography']['centroid']
                            if euclidean_distance: dst = distance.euclidean(ward_i_centroid, ward_j_centroid)
                            else: dst = geodesic((ward_i_centroid[1], ward_i_centroid[0]), (ward_j_centroid[1], ward_j_centroid[0])).km
                            if verbose: print(val, '-', dst)
                            self.distance_matrix[i][j] = dst
                            if self.distance_matrix[j][i] == 0: self.distance_matrix[j][i] = dst


        if load_distances:
            if verbose: print('  ↳ Loading distance matrix...')
            self.distance_matrix = np.load(distance_matrix_file)
        else:
            np.save(distance_matrix_file, self.distance_matrix)
            if save_txt: np.savetxt(distance_matrix_file.split('.')[0]+'.txt', self.distance_matrix)
        
        if verbose: print('Matrices complete!')
    
class Solution:

    def __init__(self, wards, constituencies, matrix_ids, probability_matrix, distance_matrix, selection_method, electoral_quota, protected_constituencies=None, electorate_deviation=0.05, max_constituency_area=13000, area_electorate_threshold=12000, proportional_votes=None, aim_seat_share=None, selection_threshold=None, countries=['England', 'Scotland', 'Wales', 'Northern Ireland'], reporter=None):
        """
        :param wards: Wards dictionary
        :param constituencies: Constituencies dictionary
        :param matrix_ids: 2D array containing constituency & ward ids in the probability matrix
        :param probability_matrix: 2D numpy array containing ward-constituency assignment probabilities
        :param distance_matrix: 2D numpy array containing the distances between each ward
        :param selection_method: Selection method used to assign constituencies
        :param selection_threshold: Threshold value for selection method (when required)
        :param electoral_quota: Constituency average electorate
        :param protected_constituencies: List of protected constituencies
        :param electorate_deviation: Allowed percent deviation for constituency population
        :param max_constituency_area: Maximum allowed constituency area
        :param area_electorate_threshold: Area threshold to allow electorates below limit
        :param proportional_votes: Aim proportional votes (Can be calculated using run_election())
        :param aim_seat_share: Aim seat share (Can be calculated using run_election())
        :param countries: Countries to use for model
        :param reporter: Statistics reporter
        """

        self.reporter = reporter
        
        self.protected_constituencies = protected_constituencies
        self.proportional_votes = proportional_votes
        self.aim_seat_share = aim_seat_share

        self.countries = countries
        with open('data/dict_templates.json', 'r') as templates_file:
            self.templates = json.load(templates_file)

        self.wards, self.constituencies = copy.deepcopy(wards), copy.deepcopy(constituencies)

        self.probability_matrix = probability_matrix
        self.distance_matrix = distance_matrix
        self.constituency_ids = matrix_ids[0]
        self.ward_ids = matrix_ids[1]
            
        self.electoral_quota = electoral_quota
        self.electorate_deviation = electorate_deviation
        self.max_constituency_area = max_constituency_area
        self.area_electorate_threshold = area_electorate_threshold

        self.assign_ward_constituencies(selection_method, selection_threshold)

    def assign_ward_constituencies(self, method='default', threshold=None):
        """
        Starts group selection method, assigning wards to constituencies:
        :param method 'default': Wards are not changed from dictionary assignments
        :param method 'acceptance-selection': Assign wards using an acceptance selection method
        :param method 'random-selection': Assign wards completely at random
        :param method 'voronoi-selection': Assign wards using a voronoi tessellation
        :return bool: Denotes whether a valid method was given
        """

        if method.upper() not in ['DEFAULT', 'ACCEPTANCE-SELECTION', 'RANDOM-SELECTION', 'VORONOI-SELECTION']:
            print("\nClass SOLUTION ERROR: Invalid selection method:\n  ↳ Must be 'default', 'acceptance-selection', 'random-selection' or 'voronoi-selection', not '"+method.upper()+"'.")
            exit()
        elif method.upper() in ['ACCEPTANCE-SELECTION', 'VORONOI-SELECTION']:
            if threshold is None or not isinstance(threshold, float) or threshold > 1 or threshold < 0:
                print("\nClass SOLUTION ERROR: Invalid selection threshold:\n  ↳ The '"+method.upper()+"' method requires a threshold value as a float (0 <= x <= 1), not type "+str(type(threshold))+".")
                exit()
            
        if method.upper() == 'ACCEPTANCE-SELECTION':
            self.acceptance_selection(threshold)
        elif method.upper() == 'RANDOM-SELECTION':
            self.random_selection()
        elif method.upper() == 'VORONOI-SELECTION':
            self.voronoi_selection(threshold)

    
    def random_selection(self):
        """
        Randomly assigns wards to constituencies, ignoring the values in the
        probability matrix. Contiguity, population and protected constituencies
        are also ignored.
        """

        for constituency in self.constituencies.values(): constituency['wards'] = []
        for i in range(len(self.ward_ids)):
            ward = self.wards[self.ward_ids[i]]
            valid_found = False

            while not valid_found:
                assigned_constituency = rnd.choice(self.constituency_ids)
                if assigned_constituency[0] == self.ward_ids[i][0]: valid_found = True

            ward['constituency_id'] = assigned_constituency
            ward['constituency_name'] = self.constituencies[assigned_constituency]['name']
            self.constituencies[assigned_constituency]['wards'].append(self.ward_ids[i])

        self.merge_constituency_wards()

    def acceptance_selection(self, hybrid_threshold):
        """
        Assigns wards by accepting their neighbouring wards' constituency
        according to a probability given by the probability matrix. Initial
        wards for each constituency are assigned using hybrid roulette
        wheel / greedy selection.
        :param hybrid_threshold: Probability of using greedy selection
        """
        assigned_wards = []
        for constituency_id in self.protected_constituencies:
            assigned_wards += self.protected_constituencies[constituency_id]
            self.constituencies[constituency_id]['wards'] = self.protected_constituencies[constituency_id]
        
        # First, we have to assign 1 ward to each constituency to ensure
        # that all constituencies have at least one ward
        for constituency_index, constituency_id in enumerate(self.constituency_ids):
            constituency = self.constituencies[constituency_id]
            if not constituency['is_protected']:
                constituency['wards'] = []

                # Assign the intial ward based on a hybrid selection method
                ward_probabilities = list(self.probability_matrix[:,constituency_index])
                masked_ward_ids, masked_ward_probabilities = get_country_constituency_probabilities(self.ward_ids, ward_probabilities, constituency['country'])
                starting_ward = hybrid_select(masked_ward_ids, masked_ward_probabilities, hybrid_threshold)
                assigned_wards.append(starting_ward)
                self.wards[starting_ward]['constituency_id'] = constituency_id
                self.wards[starting_ward]['constituency_name'] = constituency['name']
                constituency['wards'].append(starting_ward)

        attempts = [0] * len(self.ward_ids)
        prev_length = len(assigned_wards)
        no_assignments = prev_length
        while len(assigned_wards) < len(self.ward_ids):
            
            for ward_index, ward_id in enumerate(self.ward_ids):
                if ward_id in assigned_wards: continue
                ward = self.wards[ward_id]

                neighbours = []
                for neighbour_id in ward['geography']['adjacencies']:
                    if self.wards[neighbour_id]['country'] == ward['country']:
                        neighbours.append(neighbour_id)

                if len(neighbours) == 0:
                    assigned_wards.append(ward_id)
                    continue
                
                # Fetch the constituencies & probability of any neighbouring ward that has already
                # been assigned a constituency.
                bordering_constituencies, constituency_probabilities = [], []
                for neighbour_id in neighbours:
                    neighbour = self.wards[neighbour_id]
                    neighbouring_con = neighbour['constituency_id']
                    if neighbour['country'] == ward['country'] and neighbouring_con not in self.protected_constituencies.keys() and neighbour_id in assigned_wards:
                        if neighbouring_con not in bordering_constituencies:
                            con_index = self.constituency_ids.index(neighbouring_con)
                            constituency_probability = self.probability_matrix[ward_index][con_index]
                            if constituency_probability > 0:
                                bordering_constituencies.append(neighbouring_con)
                                constituency_probabilities.append(self.probability_matrix[ward_index][con_index])
                
                assigned_constituency = None
                if len(bordering_constituencies) > 0:
                    if len(bordering_constituencies) == 1:
                        assigned_constituency = bordering_constituencies[0]
                    else:
                        possible_cons = [(con, prob) for prob, con in sorted(zip(constituency_probabilities, bordering_constituencies), reverse=True)]
                        
                        for constituency, probability in possible_cons:
                            if rnd.random() < probability:
                                assigned_constituency = constituency
                                break

                        if assigned_constituency == None:
                            assigned_constituency = possible_cons[0][0]

                elif no_assignments == 0:
                    
                    closest_wards = get_country_constituency_probabilities(self.ward_ids, self.distance_matrix[ward_index], ward['country'], order_by_val=True)
                    
                    for close_ward_id, distance in closest_wards:
                        if close_ward_id != ward_id:
                            close_ward = self.wards[close_ward_id]
                            close_ward_con = close_ward['constituency_id']
                            if close_ward['country'] == ward['country'] and close_ward_con not in self.protected_constituencies.keys():
                                if close_ward_id in assigned_wards: assigned_constituency = close_ward_con
                

                if assigned_constituency != None:
                    assigned_wards.append(ward_id)
                    ward['constituency_id'] = assigned_constituency
                    ward['constituency_name'] = self.constituencies[assigned_constituency]['name']
                    self.constituencies[assigned_constituency]['wards'].append(ward_id)
                else:
                    attempts[ward_index] += 1
            
            no_assignments = len(assigned_wards) - prev_length
            prev_length = len(assigned_wards)

        for constituency_id, constituency in self.constituencies.items():
            if len(constituency['wards']) == 0:
                print("\nClass SOLUTION ERROR: Constituency has not been assigned any wards.\n  ↳ Constituency: "+constituency['name']+" ("+constituency_id+").")
                exit()
        self.merge_constituency_wards()

    def voronoi_selection(self, hybrid_threshold, voronoi_threshold=6, contiguity_threshold=10):
        """
        Assigns wards using a voronoi tessellation, using central points
        found through either a roulette wheel or greedy selection method
        :param hybrid_threshold: Probability of using greedy selection
        :param voronoi_threshold: Attempt threshold to stop trying tessellation and
                                  assign wards to the closest assigned ward
        :param contiguity_threshold: The number of attempts after which contiguity is ignored
        """
        central_wards = []
        central_ward_ids = []
        for i in range(len(self.constituency_ids)):

            constituency_id = self.constituency_ids[i]
            if constituency_id in self.protected_constituencies.keys(): continue

            constituency = self.constituencies[constituency_id]
            constituency['wards'] = []

            ward_probabilities = list(self.probability_matrix[:,i])
            # Choose central ward according to either a roulette wheel or greedy selection
            valid_found = False
            while not valid_found:
                if rnd.random() > hybrid_threshold:
                    # Roulette wheel selection
                    central_ward_index = rnd.choices(list(range(len(ward_probabilities))),
                                               weights=ward_probabilities,
                                               k=1)[0]

                else:
                    # Should try with multiple max values!
                    #Greedy selection
                    central_ward_index = ward_probabilities.index(max(ward_probabilities))

                if self.ward_ids[central_ward_index][0] == constituency_id[0]: valid_found = True
            
            central_wards.append(central_ward_index)
            central_ward_id = self.ward_ids[central_ward_index]
            central_ward_ids.append(central_ward_id)


            self.wards[central_ward_id]['constituency_id'] = constituency_id
            self.wards[central_ward_id]['constituency_name'] = constituency['name']
            constituency['wards'].append(central_ward_id)


        central_ward_distances = []
        for central_ward_index in central_wards:
            central_ward_distances.append(list(self.distance_matrix[central_ward_index,:]))
        
        central_ward_distances = np.array(central_ward_distances)

        assigned_wards = central_ward_ids
        for constituency_id in self.protected_constituencies:
            assigned_wards += self.protected_constituencies[constituency_id]
            self.constituencies[constituency_id]['wards'] = self.protected_constituencies[constituency_id]

        assignment_attempts = [0] * len(self.ward_ids)
        ignore_contiguity = [False] * len(self.ward_ids)
        while len(assigned_wards) < len(self.ward_ids):
            for i in range(len(self.ward_ids)):

                ward_id = self.ward_ids[i]
                if ward_id not in central_ward_ids and ward_id not in assigned_wards:
                    ward = self.wards[ward_id]
                    if assignment_attempts[i] < voronoi_threshold or assignment_attempts[i] >= contiguity_threshold:
                        
                        constituency_distances = list(central_ward_distances[:,i])
                        valid_constituencies = np.nonzero(constituency_distances)[0]
                        valid_con_distances = [constituency_distances[index] for index in valid_constituencies]
                        assigned_constituency_index = constituency_distances.index(min(valid_con_distances))
                        assigned_constituency = self.constituency_ids[assigned_constituency_index]

                        add = False
                        if len(ward['geography']['adjacencies']) == 0 or ignore_contiguity[i]: add = True
                        else:
                            for constituency_wards in self.constituencies[assigned_constituency]['wards']:
                                if ward_id in self.wards[constituency_wards]['geography']['adjacencies']:
                                    add = True
                                    break
                
                        if add:
                            assigned_wards.append(ward_id)
                            ward['constituency_id'] = assigned_constituency
                            ward['constituency_name'] = self.constituencies[assigned_constituency]['name']
                            self.constituencies[assigned_constituency]['wards'].append(ward_id)
                        else: assignment_attempts[i] += 1
                    else:
                        for adjacent_ward in ward['geography']['adjacencies']:
                            if adjacent_ward in assigned_wards:
                                assigned_constituency = self.wards[adjacent_ward]['constituency_id']
                                ward['constituency_id'] = assigned_constituency
                                ward['constituency_name'] = self.constituencies[assigned_constituency]['name']
                                self.constituencies[assigned_constituency]['wards'].append(ward_id)
                                assigned_wards.append(ward_id)
                                break
                        assignment_attempts[i] += 1
                        if assignment_attempts[i] >= contiguity_threshold: ignore_contiguity[i] = True

        self.merge_constituency_wards()
        
        notcontiguous = []
        for constituency_id in self.constituencies.keys():
            constituency = self.constituencies[constituency_id]
            if not self.is_contiguous(constituency):
                notcontiguous.append(constituency_id)
                
        return True

    def improve_solution(self, n_steps, n, alpha, beta, failed_attempts_multiplier=2, hillclimb=True, verbose=False):
        """
        Perform simulated annealing algorithm on constituency assignments.
        :param n_steps: Number of steps made during the improvement
        :param n: Number of "mutations" or ward swaps per step
        :param alpha: Fairness importance (0 - 1)
        :param beta: Compactness importance (0 - 1)
        :param failed_attempts_multiplier: Defines the max number of failed attempts
                                           (ie. rejected swaps) as n_steps * failed_attempts_multiplier
        :param hillclimb: Denotes whether only good swaps are made (if false, uses simulated annealing)
        :return float: Current fitness (0 <= x <= 1)
        :return float: Current fairness (0 <= x <= 1)
        :return float: Current compactness (0 <= x <= 1)
        :return dict: Current results
        """

        self.merge_constituency_wards()
        current_fitness, current_fairness, current_compactness, current_results = self.evaluate(alpha, beta, run_election=True, verbose=False)

        steps = 0
        failed_attempts = 0
        max_failed_attempts = n_steps * failed_attempts_multiplier

        while steps < n_steps and failed_attempts < max_failed_attempts:

            # Randomly swap n wards
            swapped_wards, affected_countries = self.randomly_swap_n_wards(n)
            
            fitness, fairness, compactness, results = self.evaluate(alpha, beta, verbose=False)

            # Evaluate whether an improvement was made (specifically ensuring an increase in fairness
            # happened in any affected country)
            if fitness > current_fitness and self.calculate_avg_sv_diff(results, affected_countries) > self.calculate_avg_sv_diff(current_results, affected_countries):
                accepted = True
            elif not hillclimb and rnd.random() <= boltzmann.pmf(steps, 1, n_steps):
                accepted = True
            else: accepted = False

            if accepted:
                steps += 1
                current_fitness, current_fairness, current_compactness, current_results = fitness, fairness, compactness, results
            else:
                # If the solution was made worse, undo the changes
                failed_attempts += 1
                for swap in swapped_wards:
                    self.make_swap(swap[0], swap[1])
                    self.merge_constituency_wards(swap[1:])

            if self.reporter != None: self.reporter.update_stats([current_fitness, current_fairness, current_compactness], current_results['national_votes'])
        
        if verbose: print(current_fitness)

        return current_fitness, current_fairness, current_compactness, current_results

    def randomly_swap_n_wards(self, n, verbose=False):
        """
        Randomly swaps n wards to adjacent constituencies. Wards will stay in
        the same country and region, and contiguity should always be preserved.
        :param n: Number of ward swaps
        :return []: List containing all swaps made
        :return []: List containing all affected countries (where swaps were made)
        """
        swapped = []
        invalid = []
        affected_countries = []

        while len(swapped) < n:
            ward_id = rnd.choice(list(self.wards.keys()))

            already_swapped = False
            for swap in swapped:
                if swap[0] == ward_id: 
                    already_swapped = True
                    break

            if already_swapped or ward_id in invalid:
                continue

            ward = self.wards[ward_id]

            original_constituency_id = ward['constituency_id']
            original_constituency = self.constituencies[original_constituency_id]

            if original_constituency['is_protected']:
                invalid.append(ward_id)
                continue

            # Check if the ward is the only ward in the constituency / has no bordering wards
            if len(original_constituency['wards']) == 1 or len(ward['geography']['adjacencies']) == 0:
                invalid.append(ward_id)
                continue
            
            possible_swaps = []
            for neighbour_id in ward['geography']['adjacencies']:
                neighbour = self.wards[neighbour_id]
                if neighbour['country'] == ward['country'] and neighbour['region'] == ward['region']:
                    if neighbour['constituency_id'] != original_constituency_id:
                        if not self.constituencies[neighbour['constituency_id']]['is_protected']:
                            possible_swaps.append(neighbour['constituency_id'])

            # Ignore if no constituencies to swap to
            if len(possible_swaps) == 0:
                invalid.append(ward_id)
                continue

            valid_swap_found = False
            while not valid_swap_found and len(possible_swaps) > 0:
                constituency_index = rnd.randint(0, len(possible_swaps) - 1)
                new_constituency_id = possible_swaps[constituency_index]

                valid_swap_found = self.is_legal_swap(ward_id, original_constituency_id, new_constituency_id)

                possible_swaps.pop(constituency_index)

            if not valid_swap_found:
                invalid.append(ward_id)
                continue

            self.make_swap(ward_id, new_constituency_id)
            swapped.append([ward_id, original_constituency_id, new_constituency_id])
            if ward['country'] not in affected_countries: affected_countries.append(ward['country'])

        changed_constituencies = []

        if verbose: print("SWAPS:")
        for swap in swapped:
            if verbose: print("  {0}: {1} -> {2}".format(swap[0], swap[1], swap[2]))
            changed_constituencies.append(swap[1])
            changed_constituencies.append(swap[2])
        
        changed_constituencies = list(set(changed_constituencies))
        #self.merge_constituency_wards(changed_constituencies)

        return swapped, affected_countries

    def make_swap(self, ward_id, new_constituency_id):
        """
        Performs a ward swap to a new constituency.
        :param ward_id: Ward to swap
        :param new_constituency_id: ID of the ward's new constituency
        """

        ward = self.wards[ward_id]
        orig_constituency_id = ward['constituency_id']
        orig_constituency = self.constituencies[orig_constituency_id]
        new_constituency = self.constituencies[new_constituency_id]

        if ward_id in orig_constituency['wards'] or ward_id in new_constituency['wards']:
            ward['constituency_id'] = new_constituency_id
            ward['constituency_name'] = new_constituency['name']

            orig_constituency['wards'].remove(ward_id)
            new_constituency['wards'].append(ward_id)

            self.merge_constituency_wards([orig_constituency_id, new_constituency_id])

        else:
            print('\nClass SOLUTION ERROR: Invalid swap attempted:\n  ↳ Ward \'{0}\' not in constituency \'{1}\'.'.format(ward_id, new_constituency_id))
            exit()

    def compactness_score(self, geography, metric='REOCK', verbose=False):
        """
        Returns the compactness score of a constituency's geography (0 <= x <= 1)
        :param geography: Constituency geography dictionary
        :param metric 'REOCK': Scores compactness as the constituency area / its bounding circle
        :param metric 'LWR': Scores compactness as the length to width ratio
        :param metric 'SQUAREST': Scores compactness as the constituency area / its bounding square
        :return float: Compactness score (0 <= x <= 1)
        """
        
        bounds = geography['bounds']
        centroid = geography['centroid']

        length = geodesic((bounds['north'], centroid[0]), (bounds['south'], centroid[0])).km
        width = geodesic((centroid[1], bounds['east']), (centroid[1], bounds['west'])).km

        if metric.upper() == 'LWR':
            compactness = length / width
        elif metric.upper() == 'SQUAREST':
            # Squarest constituency (Bristol South)
            compactness = (length * width) / geography['area']
        elif metric.upper() == 'REOCK':
            # Isn't 100%, but will do for now
            compactness = math.pi * (max([length, width]) / 2)**2 / geography['area']
        else:
            print('\nClass SOLUTION ERROR: Invalid compactness metric:\n  ↳ Metric must be \'LWR\', \'SQUARES\' or \'REOCK\', not '+metric)
            exit()

        if compactness > 1: compactness = 1 / compactness

        if verbose: print(length, ':', width, '=', compactness)
 
        return compactness

    def is_legal_swap(self, ward_id, original_constituency_id, new_constituency_id):
        """
        Determines whether a swap fits the population, area
        and contiguity requirements for a constituency.
        :param ward_id: Ward to swap
        :param original_constituency_id: The ward's original constituency
        :param new_constituency_id: The potential new constituency
        :return bool: Denotes whether a swap would produce legal constituencies
        """

        ward = self.wards[ward_id]
        orig_constituency = self.constituencies[original_constituency_id]
        new_constituency = self.constituencies[new_constituency_id]

        ward_electorate = ward['election_results']['electorate']
        ward_area = ward['geography']['area']

        orig_consituency_electorate = orig_constituency['election_results']['electorate']
        orig_constituency_area = orig_constituency['geography']['area']

        new_consituency_electorate = new_constituency['election_results']['electorate']
        new_constituency_area = new_constituency['geography']['area']

        # Test if electorate is smaller than allowed
        if orig_consituency_electorate - ward_electorate < (self.electoral_quota * (1 - self.electorate_deviation)):
            if orig_constituency_area - ward_area < self.area_electorate_threshold:
                return False

        if new_consituency_electorate + ward_electorate < (self.electoral_quota * (1 - self.electorate_deviation)):
            if new_constituency_area + ward_area < self.area_electorate_threshold:
                return False
        elif new_consituency_electorate + ward_electorate > (self.electoral_quota * (1 + self.electorate_deviation)):
            return False
        elif new_constituency_area + ward_area > self.max_constituency_area:
            return False

        return self.is_contiguous(orig_constituency, ward_id)

    def is_contiguous(self, constituency, ward_id=None):
        """
        Checks whether a constituency is contiguous by checking
        the eigenvalues of the laplacian matrix of the adjacency
        matrix.
        :param wards_list: List of constituency wards
        """

        wards_list = constituency['wards']

        if len(wards_list) == 1 and ward_id != None:
            print("\nClass SOLUTION ERROR: Contiguity test attempted on empty ward list.")
            exit()

        adjacency_matrix = []
        for i in range(len(wards_list)):

            if wards_list[i] == ward_id: continue
            matrix_row = []
            for j in range(len(wards_list)):

                if wards_list[j] == ward_id: continue

                if i == j: matrix_row.append(1)
                elif wards_list[i] in self.wards[wards_list[j]]['geography']['adjacencies']: matrix_row.append(1)
                elif wards_list[j] in self.wards[wards_list[i]]['geography']['adjacencies']: matrix_row.append(1)
                else: matrix_row.append(0)

            adjacency_matrix.append(matrix_row)

        adjacency_matrix = np.array(adjacency_matrix)
        constituency_graph = nx.from_numpy_array(adjacency_matrix)
        return nx.is_connected(constituency_graph)

    def calculate_avg_sv_diff(self, results, countries=['national']):
        result_keys = []
        for country in countries:
            result_keys.append(country.lower().replace(' ', '_')+'_votes')

        country_sv_differences = []
        for country in result_keys:
            sv_differences = []
            for results_vals in results[country].values():
                sv_differences.append(abs(results_vals[1] - results_vals[3])/results_vals[3])

            sv_difference = 1 - (sum(sv_differences) / len(sv_differences))
            country_sv_differences.append(sv_difference)
            
        return sum(country_sv_differences) / len(country_sv_differences)

    def evaluate(self, alpha, beta, run_election=True, countries=['national'], verbose=False):
        """
        Calculate wasted vote score and compactness score.
        :param results: Election results
        :param alpha: Fairness importance (0 - 1)
        :param beta: Compactness importance (0 - 1)
        """

        if run_election: results = self.run_election(countries=self.countries, verbose=verbose)
        else: results = None

        if verbose: print_results(results['national_votes'])

        if self.reporter != None: self.reporter.no_func_evals += 1

        sv_difference = self.calculate_avg_sv_diff(results, countries=countries)

        compactnesses = []
        least_compact = [1, '']
        most_compact = [0, '']
        
        for constituency_key in self.constituencies.keys():

            constituency = self.constituencies[constituency_key]

            if 'national' in countries or constituency['country'] in countries:
                compactness = self.compactness_score(constituency['geography'], metric='reock')
                if compactness >= most_compact[0]: most_compact = [compactness, constituency_key]
                elif compactness <= least_compact[0]: least_compact = [compactness, constituency_key]
                compactnesses.append(compactness)

        average_compactness = sum(compactnesses) / len(compactnesses)

        fitness_score = ((sv_difference*alpha) + (average_compactness*beta))/(alpha+beta)

        return fitness_score, sv_difference, average_compactness, results

    def merge_constituency_wards(self, merge_list=None, ignore_demographics=True):
        """
        Merges a given set of constituency wards. If no list is given,
        all constituencies are merged.
        :param merge_list: List of constituencies to merge
        :param ignore_demographics: Denotes whether to skip calculating constituency demographics from wards
        """

        if merge_list == None: merge_list = list(self.constituencies.keys())

        for constituency_key in merge_list:
            constituency = reset_constituency_values(self.constituencies[constituency_key], self.templates)
            if 'E05002599' in constituency['wards']: constituency['is_speaker_seat'] = True

            north, south, east, west = -math.inf, math.inf, -math.inf, math.inf
            centroid = [0, 0]

            for ward_id in constituency['wards']:
                ward = self.wards[ward_id]

                constituency['population'] += ward['population']
                if not ignore_demographics:
                    for demographic in ['ages', 'education', 'economic_activity', 'households', 'ethnicity']:
                        for data_key in ward[demographic].keys():
                            constituency[demographic][data_key] += int(ward['population'] * ward[demographic][data_key])

                constituency['election_results']['electorate'] += ward['election_results']['electorate']
                constituency['election_results']['turnout'] += ward['election_results']['turnout']
                if not constituency['is_speaker_seat']:
                    for party in ward['election_results']['vote_share'].keys():
                        if party in constituency['election_results']['vote_share']:
                            constituency['election_results']['vote_share'][party] += int(ward['election_results']['vote_share'][party] * ward['election_results']['electorate'] * ward['election_results']['turnout'])
                        else:
                            print("\nClass REDISTRICTER ERROR: Invalid ward assignment found!")
                            print("  ↳ A ward has likely been assigned to the wrong country as '"+party+"' is not in its constituency's party list.")
                            print("  ↳ Ward: "+ward['name']+" ("+ward_id+", "+ward['country']+"), Constituency: "+constituency['name']+" ("+constituency_key+", "+constituency['country']+")")
                            exit()
                else:
                    for party in constituency['election_results']['vote_share'].keys():
                        if party == 'other_parties': constituency['election_results']['vote_share'][party] = 1
                        else: constituency['election_results']['vote_share'][party] = 0
                        
                
                if ward['geography']['bounds']['north'] > north: north = ward['geography']['bounds']['north']
                if ward['geography']['bounds']['south'] < south: south = ward['geography']['bounds']['south']
                if ward['geography']['bounds']['east'] > east:   east  = ward['geography']['bounds']['east']
                if ward['geography']['bounds']['west'] < west:   west  = ward['geography']['bounds']['west']

                constituency['geography']['area'] += ward['geography']['area']
                centroid = [sum(coor) for coor in zip(centroid, ward['geography']['centroid'])]

            constituency['geography']['bounds']['north'] = north
            constituency['geography']['bounds']['south'] = south
            constituency['geography']['bounds']['east']  = east
            constituency['geography']['bounds']['west']  = west
            
            no_wards = len(constituency['wards'])
            constituency['geography']['centroid'] = [coor / no_wards for coor in centroid]
            constituency['election_results']['turnout'] /= no_wards

            try:
                factor = 1.0 / (sum(constituency['election_results']['vote_share'].values()))
            except ZeroDivisionError:
                print('Zero division error!')
                print(constituency)
                exit()
            constituency['election_results']['vote_share'] = {key: value * factor for key, value in constituency['election_results']['vote_share'].items()}

            if not ignore_demographics:
                for demographic in ['ages', 'education', 'economic_activity', 'ethnicity']:
                    dem_total = sum(constituency[demographic].values())
                    constituency[demographic] = {k: v / dem_total for k, v in constituency[demographic].items()}

                num_households = sum(constituency['households'].values()) - constituency['households']['total_no']
                constituency['households'] = {k: v / num_households for k, v in constituency['households'].items()}
                constituency['households']['total_no'] = num_households

            self.constituencies[constituency_key] = constituency

    def run_election(self, countries=['England', 'Scotland', 'Wales', 'Northern Ireland'], verbose=True):
        """
        Runs an election
        """
        results = {
                    'countries': countries,
                    'national_votes': {
                        'conservatives': [0, 0, 0, 0],
                        'labour': [0, 0, 0, 0],
                        'liberal_democrats': [0, 0, 0, 0],
                        'ukip': [0, 0, 0, 0],
                        'green': [0, 0, 0, 0],
                        'plaid_cymru': [0, 0, 0, 0],
                        'snp': [0, 0, 0, 0],
                        'dup': [0, 0, 0, 0],
                        'sinn_fein': [0, 0, 0, 0],
                        'sdlp': [0, 0, 0, 0],
                        'uup': [0, 0, 0, 0],
                        'alliance': [0, 0, 0, 0],
                        'other_parties': [0, 0, 0, 0]
                        }
                    }
        
        for country in countries:
            results_dict = copy.deepcopy(self.templates[country])
            for party in results_dict.keys():
                results_dict[party] = [0, 0, 0, 0]
            results[country.lower().replace(' ', '_')+'_votes'] = results_dict
            
        for constituency in self.constituencies.values():

            if constituency['country'] not in countries: continue
            else: country = constituency['country'].lower().replace(' ', '_')

            constituency_results = constituency['election_results']['vote_share']

            electorate = constituency['election_results']['electorate']
            turnout = constituency['election_results']['turnout']

            winner = max(constituency_results, key=constituency_results.get)
            # Seats at index 0
            results['national_votes'][winner][0] += 1
            results[country+'_votes'][winner][0] += 1
            
            for party in constituency_results.keys():
                votes = int(electorate * turnout * constituency_results[party])
                # Total votes at index 2
                results['national_votes'][party][2] += votes
                results[country+'_votes'][party][2] += votes

        for country in [country.lower().replace(' ', '_')+'_votes' for country in ['national']+countries]:
            total_seats = sum([values[0] for values in results[country].values()])
            total_votes = sum([values[2] for values in results[country].values()])
        
            for party in results[country]:
                # Seat share at index 1
                results[country][party][1] = results[country][party][0] / total_seats
                # Vote share at index 3
                results[country][party][3] = results[country][party][2] / total_votes
            
            sorted_tuples = sorted(results[country].items(), key=lambda item: item[1])
            sorted_tuples.reverse()
            results[country] = {k: v for k, v in sorted_tuples}
        
        if self.proportional_votes == None:
            self.proportional_votes = {party: stats[3] for party, stats in copy.deepcopy(results['national_votes']).items()}
            self.aim_seat_share = {party: stats[1] for party, stats in copy.deepcopy(results['national_votes']).items()}

        if verbose: print_results(results['national_votes'])
        return results

    def save_wards_constituencies(self, save_file):
        """
        Saves the solution's ward and constituency assignments. Ward assignment
        changes must first be made to the solution ward & constituency dicts.
        :param save_file: JSON filename
        """
        with open(save_file, 'w') as fp:
            json.dump({'wards': self.wards, 'constituencies': self.constituencies}, fp, indent=4)

    def save_election_results(self, results, save_file):
        """
        Saves an election result dictionary as a JSON file.
        :param save_file: JSON filename
        """
        with open(save_file, 'w') as fp:
            json.dump(results, fp, indent=4)

def hybrid_select(options, weights, rw_threshold):
    if rnd.random() > rw_threshold:
        # Roulette wheel selection
        assigned_constituency = rnd.choices(options,
                                            weights=weights,
                                            k=1)[0]
    else:
        # Greedy selection
        assigned_constituency = options[list(weights).index(np.max(weights))]

    return assigned_constituency

def get_country_constituency_probabilities(constituency_ids, probabilities, country, order_by_val=False, reverse=False):
    country_mask = [constituency_id.startswith(country[0].upper()) for constituency_id in constituency_ids]
    masked_constituency_ids = np.extract(country_mask, constituency_ids)
    masked_probabilities = np.extract(country_mask, probabilities)
    if order_by_val: return [(con, prob) for prob, con in sorted(zip(masked_probabilities, masked_constituency_ids), reverse=reverse)]
    else: return masked_constituency_ids, masked_probabilities
    
def reset_constituency_values(constituency, templates):
    new_constituency = copy.deepcopy(templates['constituencies'])
    for key in ['name', 'country', 'region', 'is_protected', 'wards']:
        new_constituency[key] = constituency[key]
    new_constituency['election_results']['vote_share'] = copy.deepcopy(templates[new_constituency['country']])
    return new_constituency

def gps_to_ecef(coors, alt=0):
    lat = coors[1]
    lon = coors[0]
    rad_lat = lat * (math.pi / 180.0)
    rad_lon = lon * (math.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(rad_lat) * math.sin(rad_lat))

    x = (v + alt) * math.cos(rad_lat) * math.cos(rad_lon)
    y = (v + alt) * math.cos(rad_lat) * math.sin(rad_lon)
    z = (v * (1 - e2) + alt) * math.sin(rad_lat)

    return [x, y, z]

def get_video_filename(flag):
    if flag in sys.argv:
        args_index = sys.argv.index(flag)
        if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
            print('Class REDISTRICTER ERROR: No filename value given:\n  ↳ \''+flag+'\' argument must be followed by a valid video filename')
            exit()
        else:
            if sys.argv[args_index + 1].endswith('.mp4'):
                return sys.argv[args_index + 1]
            else:
                print('Class REDISTRICTER ERROR: Invalid video filename given:\n  ↳ Video files are output as \'.mp4\' and \''+sys.argv[args_index + 1]+'\' is invalid.')
                exit()
    else:
        return None

def get_int_arguments(flags, defaults):
    vals = []
    for i in range(len(flags)):
        if flags[i] in sys.argv:
            args_index = sys.argv.index(flags[i])
            if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
                print('Class REDISTRICTER ERROR: No K value given:\n  ↳ \''+flags[i]+'\' argument must be followed by an int')
                exit()
            else:
                if sys.argv[args_index + 1].isdigit():
                    vals.append(int(sys.argv[args_index + 1]))
                else:
                    print('Class REDISTRICTER ERROR: Invalid K value given:\n  ↳ K must be an int, not \''+sys.argv[args_index + 1]+"'")
                    exit()
        else:
            vals.append(defaults[i])
    return vals

def get_float_arguments(flags, defaults):
    vals = []
    for i in range(len(flags)):
        flag = flags[i]
        default_val = defaults[i]
        if flag in sys.argv:
            args_index = sys.argv.index(flag)
            if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
                print('Class REDISTRICTER ERROR: No value given:\n  ↳ \''+flag+'\' argument must be followed by a float (0 <= x <= 1)')
                exit()
            else:
                if float(sys.argv[args_index + 1]) >= 0 and float(sys.argv[args_index + 1]) <= 1:
                    vals.append(float(sys.argv[args_index + 1]))
                else:
                    print(sys.argv[args_index + 1].isnumeric())
                    print('Class REDISTRICTER ERROR: Invalid \''+flag[1:]+'\' value given:\n  ↳ Value must be a float (0 <= x <= 1), not \''+sys.argv[args_index + 1]+"'")
                    exit()
        else:
            vals.append(default_val)
    return vals

if __name__ == "__main__":
    """
    Command Line arguments:
    -m:        Skip creating a plotter object
    -p:        Force no progress bar
    -sr:       Show election results with each function evaluation
    -v:        Run the program verbose
    -rcolours: Use random constituency colours in plots
    
    -vf <filename>      Create a video with the given filename
    -k <iterations>     Number of iterations for the first optimisation phase (main phase)
    -c <iterations>     Number of iterations for the second optimisation phase, where alpha=0 and beta=1 (default 0)
    -falpha <alpha>     Set the alpha value for fitness calculations (fairness)
    -fbeta <beta>       Set the beta value for fitness calculations (compactness)
    -ims <improvements> Set the number of improvements during the local search phase
    -seed <rnd_seed>    Sets the random seed for the model
    """
    verbose = '-v' in sys.argv
    skip_plotter = '-m' not in sys.argv
    skip_progress = '-p' not in sys.argv
    random_colours = '-rcolours' in sys.argv

    save_final_json = 'final_map.json'

    default_iterations = 10
    iterations, improvements, compactness_stage_length, rnd_seed = get_int_arguments(['-k', '-ims', '-c', '-seed'], [default_iterations, 100, 0, None])

    rnd.seed(rnd_seed)
    video_filename = get_video_filename('-vf')

    f_alpha, f_beta = get_float_arguments(['-falpha', '-fbeta'], [1, 1])

    redistricter = Redistricter('data/wards_constituencies.json', create_plotter=skip_plotter, show_progress=skip_progress)
    redistricter.generate_map(iterations, f_alpha=f_alpha, f_beta=f_beta,
                              improvements=improvements,
                              compactness_stage_length=compactness_stage_length,
                              video_filename=video_filename,
                              save_solution_location=save_final_json, plot_random_colours=random_colours,
                              verbose=verbose)