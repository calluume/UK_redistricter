import json, os, sys, math, cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

party_colours = {'conservatives': ['#0087DC', 'Blues'],
                 'labour': ['#DC241f', 'Reds'],
                 'liberal_democrats': ['#FDBB30', 'YlOrBr'],
                 'ukip': ['#33114B', 'Purples'],
                 'green': ['#6AB123', 'YlGn'],
                 'plaid_cymru': ['#338736', 'YlGn'],
                 'snp': ['#FFFF00', 'YlOrBr'],
                 'dup': ['#D4694B', 'OrRd'],
                 'sinn_fein': ['#00635E', 'Greens'],
                 'sdlp': ['#4DA165', 'YlGn'],
                 'uup': ['#48A6EE', 'Blues'],
                 'alliance': ['#CCAE2C', 'YlOrBr'],
                 'other_parties': ['#909090', 'Greys']          
                }

class Plotter:
    def __init__(self, countries=['England', 'Wales', 'Scotland', 'Northern Ireland'], regions=None, verbose=False):
        country_dataframes = []
        if verbose: print('PLOTTER:')
        for country in countries:
            if verbose: print('  ↳ Reading', country, 'boundaries file...')

            if country == 'England': country_file = '../datasets/wards/boundaries/eng_wards.gpkg'
            if country == 'Scotland': country_file = '../datasets/wards/boundaries/sco_wards.gpkg'
            if country == 'Wales': country_file = '../datasets/wards/boundaries/wa_wards.gpkg'
            if country == 'Northern Ireland': country_file = '../datasets/wards/boundaries/ni_wards.gpkg'

            if len(countries) != 1:
                country_dataframe = gpd.read_file(country_file)
                country_dataframes.append(country_dataframe)
            else:
                uk = gpd.read_file(country_file)
        if verbose: print()
        if len(country_dataframes) > 1:
            uk = country_dataframes[0]
            for i in range(1, len(country_dataframes)):
                next_country = pd.DataFrame(country_dataframes[i])
                uk = pd.DataFrame(uk)
                uk = pd.concat([uk, next_country], ignore_index=True)
                uk = gpd.GeoDataFrame(uk)
                country_dataframes[0] = uk
        
            uk = country_dataframes[0]
        
        self.countries = countries
        self.regions = regions
        self.uk_map = uk

    def plot_ward_party_support_map(self, wards, constituencies, metric, value_type, plot_cities=False, city_population_threshold=150000, image_savefile=None, image_size=(8, 6), dpi=600, random_colours=None, verbose=False):
        """
        Plots party support from a given ward/constituency dataset. This can be used
        to either plot the winning party at the ward/constituency level, or to plot
        the support of specific parties.
        :param wards: Wards dictionary
        :param constituencies: Constituencies dictionary
        :param metric: Given metric to plot, either 'winner' or specific party
        :param value_type: If metric is 'winner', use either 'ward' or 'constituency
                        to choose which level to plot. If metric is a party,
                        use either 'percent' or 'total' to plot either given party
                        percent support in the region (default), or the total number
                        of votes in that region.
        :param countries: Countries to plot data for, defaults to all (must intersect with regions)
        :param regions: Regions to plot data for, defaults to all (must intersect with countries)
        :param plot_cities: Denotes whether cities are annotated on the map
        :param city_population_threshold: Minimum population threshold for annotated cities. Some
                                        major cities like Manchester or London are drawn by default
        :param image_savefile:
        """

        headers = ['wd11cd', 'wd11nm', 'pcon11cd', 'region']

        # If the metric is a party, add the party column to the table
        if metric in party_colours.keys(): headers.append(metric)
        # Otherwise, plotting the winner, so only add colour
        elif metric == 'winner': headers.append('colour')
        
        rows = []

        if verbose: print('PLOTTER: Reading ward data...')
        for key in wards.keys():
            ward = wards[key]
            if ward['country'] in self.countries:
                if random_colours == None:
                    if metric in ward['election_results']['vote_share'].keys() or metric == 'winner':
                        
                        party_colour = np.nan

                        if metric == 'winner':
                            # Find the winning party, whether for ward/constituency level
                            if value_type == 'ward':
                                winner = max(ward['election_results']['vote_share'], key=ward['election_results']['vote_share'].get)
                            elif value_type == 'constituency':
                                constituency_results = constituencies[ward['constituency_id']]['election_results']['vote_share']
                                winner = max(constituency_results, key=constituency_results.get)
                            else:
                                print('Class PLOTTER ERROR: Invlaid value_type')
                                exit()
                            
                            # Get party colour
                            party_colour = party_colours[winner][0]

                        if metric in ward['election_results']['vote_share'].keys():
                            # If plotting party support, get percent of vote
                            value = ward['election_results']['vote_share'][metric]
                            if value_type == 'total':
                                # If plotting total votes for the party, multiply by electorate
                                value *= ward['election_results']['electorate']
                    else:
                        value = np.nan
                        party_colour = np.nan
                        
                    if metric == 'winner':
                        rows.append([key, ward['name'].upper(), ward['constituency_id'], ward['region'], party_colour])
                    elif metric in party_colours.keys():
                        rows.append([key, ward['name'].upper(), ward['constituency_id'], ward['region'], value])

                else:
                    rows.append([key, ward['name'].upper(), ward['constituency_id'], ward['region'], random_colours[ward['constituency_id']]])

        map_data = pd.DataFrame(rows, columns=headers)

        wards_map = self.uk_map.merge(map_data, left_on="wd11cd", right_on="wd11cd")
        if verbose:
            print('\nData collected!')
            print(wards_map.head())
        if self.regions != None:
            region_string = ''.join('"'+str(x)+'", ' for x in self.regions)
            region_string = "["+region_string[:-2]+"]"
            wards_map = wards_map.query('region in '+region_string)

        if verbose and self.regions != None: print(wards_map.head())
        if verbose: print('Plotting map...')
        
        if metric == 'winner' and value_type == 'constituency': constituency_map = wards_map.dissolve(by='pcon11cd')

        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()

        if metric in party_colours.keys():
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1) 
            wards_map.plot(column=metric, ax=ax, legend=True, cax=cax, cmap=party_colours[metric][1], missing_kwds={'color': 'lightgrey'}).set_axis_off()
        elif metric == 'winner':
            if value_type == 'ward': wards_map.plot(ax=ax, color=wards_map['colour'], missing_kwds={'color': 'lightgrey'}).set_axis_off()
            elif value_type == 'constituency': constituency_map.plot(ax=ax, color=constituency_map['colour'], missing_kwds={'color': 'lightgrey'}).set_axis_off()
            
        if plot_cities:
            uk_cities = gpd.read_file('data/uk_cities.geojson') 
            if city_population_threshold != None: uk_cities = uk_cities[uk_cities['population'] >= city_population_threshold]
            uk_cities.plot(ax=ax, marker='o', color='black', markersize=7)
            for x, y, label in zip(uk_cities.geometry.x, uk_cities.geometry.y, uk_cities.city):
                ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", horizontalalignment='right', size='small', weight='bold')
            
        if image_savefile != None:
            height, width = image_size[1], image_size[0]
            fig.set_size_inches(width, height)
            plt.savefig(image_savefile, dpi=dpi)
            plt.close()
        else:
            plt.show(block=True)
            plt.close()

        if verbose: print('Done!')

def plot_ward_statistic_map(wards, category, statistic=None, countries=['England', 'Wales', 'Scotland', 'Northern Ireland'], regions=None, verbose=True):
    """
    Plots the demographic data for each ward onto a map.
    :param wards: Wards dictionary
    :param category: Desired statistic category (ie. age)
    :param statistic: Desired statistic (ie. 18_to_24)
    :param countries: Countries to plot data for, defaults to all
                      (must intersect with regions)
    :param regions: Regions to plot data for, defaults to all
                    (must intersect with countries)
    """

    rows = []
    if verbose: print('Reading '+category.replace("_", " ")+' data...')
    for key in wards.keys():
        ward = wards[key]
        if ward['country'] in countries:

            value = ward[category]
            if statistic != None:
                if type(value) is dict and statistic in value.keys():
                    value = value[statistic]
                else:
                    print('Invalid statistic:', statistic)
                    exit()

            rows.append([key, value, ward['region']])
    
    map_data = pd.DataFrame(rows, columns=["wd11cd", category, 'region'])

    country_dataframes = []
    for country in countries:
        if verbose: print('Reading', country, 'boundaries file...')

        if country == 'England': country_file = '../datasets/wards/boundaries/eng_wards.gpkg'
        if country == 'Scotland': country_file = '../datasets/wards/boundaries/sco_wards.gpkg'
        if country == 'Wales': country_file = '../datasets/wards/boundaries/wa_wards.gpkg'
        if country == 'Northern Ireland': country_file = '../datasets/wards/boundaries/ni_wards.gpkg'

        if len(countries) != 1:
            country_dataframe = gpd.read_file(country_file)
            country_dataframes.append(country_dataframe)
        else:
            uk = gpd.read_file(country_file)

    if len(country_dataframes) > 1:
        uk = country_dataframes[0]
        for i in range(1, len(country_dataframes)):
            next_country = pd.DataFrame(country_dataframes[i])
            uk = pd.DataFrame(uk)
            uk = pd.concat([uk, next_country], ignore_index=True)
            uk = gpd.GeoDataFrame(uk)
            country_dataframes[0] = uk
    
        uk = country_dataframes[0]

    wards_map = uk.merge(map_data, left_on="wd11cd", right_on="wd11cd")
    if verbose:
        print('\nData collected!')
        print(wards_map.head())
    
    if regions != None:
        region_string = ''.join('"'+str(x)+'", ' for x in regions)
        region_string = "["+region_string[:-2]+"]"
        wards_map = wards_map.query('region in '+region_string)

    if verbose and regions != None: print(wards_map.head())
    if verbose: print('Plotting map...')
    
    _, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    wards_map.plot(column=category, ax=ax, legend=True, cax=cax)
    plt.show(block=True)

    if verbose: print('Done!')

def plot_results_comparison_bchart(results_dicts, title, series_labels=['Model Seat Share (%)', '2017 Seat Share (%)', 'Proportional Vote (%)'], show_plot=True, save_plot=None):

    parties = list(results_dicts[0].keys())
    bchart_series = []
    
    bchart_series.append([results_dicts[1][party][0] for party in parties])
    bchart_series.append([results_dicts[0][party][0] for party in parties])
    bchart_series.append([results_dicts[0][party][3] for party in parties])

    for i in range(len(bchart_series)):
        factor = 1 / sum(bchart_series[i])
        bchart_series[i] = [val * factor * 100 for val in bchart_series[i]]
    
    vals = []
    colours = []

    for index, party in enumerate(parties):
        party_vals = [series[index] for series in bchart_series]
        vals.append(party_vals)
        colours.append(party_colours[party][0])
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(axis='x', linestyle = '--', linewidth=0.6, zorder=0)
    left = [0] * len(bchart_series)
    for i in range(len(vals)):

        if parties[i] in ['ukip', 'sdlp', 'dup', 'snp', 'uup']:
            party_label = parties[i].upper()
        else: party_label = parties[i].replace('_', ' ').title()

        ax.barh(series_labels, vals[i], left=left, label=party_label, color=colours[i], zorder=3)
        left = [sum(x) for x in zip(vals[i], left)]

    party_labels = []
    for party in parties:
        if party in ['ukip', 'sdlp', 'dup', 'snp', 'uup']:
            party_labels.append(party.upper())
        else: party_labels.append(party.replace('_', ' ').title())

    ax.legend(title="Parties", bbox_to_anchor=(0.5, -0.7), loc='lower center', ncol=6)
    ax.set_title(title, y=1.05)
    ticks = [0, 25, 50, 75, 100]
    ax.set_xticks(ticks, ticks)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.4)

    if show_plot: plt.show(block=True)
    if save_plot != None and save_plot.endswith('png'):
        fig.savefig(save_plot, dpi=400)

def plot_seats_grid(results, rotation='horizontal', cols=13, seat_radius=60, seat_spacing=60, majority_spacing=1000, margin=(300, 300, 300, 300), title=None, show_image=True, save_image='images/graphs/parliament_seats.png'):
    no_seats = sum([results[party][0] for party in results.keys()])
    
    n_cols, n_rows = cols, math.ceil(no_seats / cols)

    if title != None:
        font = ImageFont.truetype('misc/texgyreheros-regular.otf', 200)
        title_width, title_height = get_text_dimensions(title, font)
        margin = (margin[0], margin[1] + math.floor(title_height * 1.2), margin[2], margin[3])

    if rotation == 'vertical':
        image_width = margin[0] + margin[2]+ ((seat_radius * 2) * n_cols) + (seat_spacing * (n_cols - 1))
        image_height = margin[1] + margin[3] + ((seat_radius * 2) * n_rows) + (seat_spacing * (n_rows - 1)) + majority_spacing
    elif rotation == 'horizontal':
        image_width = margin[0] + margin[2] + ((seat_radius * 2) * n_rows) + (seat_spacing * (n_rows - 1)) + majority_spacing
        image_height = margin[1] + margin[3] + ((seat_radius * 2) * n_cols) + (seat_spacing * (n_cols - 1))
    
    image_dimensions = (image_width, image_height)
    image = Image.new('RGBA', image_dimensions, "WHITE")
    draw = ImageDraw.Draw(image)
    
    if title != None:
        title_location = ((image_width - title_width) / 2, title_height / 2)
        draw.text(title_location, title, (0, 0, 0), font=font)

    coors = margin

    curr_row = 1
    curr_column = 1
    seats_drawn = 0

    parties = list(results.keys())
    party_no = 0
    party_seats = 0

    while seats_drawn < no_seats:

        if party_seats >= results[parties[party_no]][0]:
            party_seats = 0
            party_no += 1
            if party_no == 1:
                if majority_spacing > 0:
                    
                    if rotation == 'vertical': coors = (coors[0], coors[1] + majority_spacing)
                    elif rotation == 'horizontal': coors = (coors[0] + majority_spacing, coors[1])
        seats_drawn += 1
        party_seats += 1
        colour = party_colours[parties[party_no]][0]

        draw.ellipse((coors[0], coors[1], coors[0] + (seat_radius * 2), coors[1] + (seat_radius * 2)), fill=colour)
        
        if curr_column == n_cols:
            curr_row += 1
            curr_column = 1

            if rotation == 'vertical': coors = (margin[0], coors[1] + (seat_radius * 2) + seat_spacing)
            elif rotation == 'horizontal': coors = (coors[0] + (seat_radius * 2) + seat_spacing, margin[1])
        
        else:
            curr_column += 1
            if rotation == 'vertical': coors = (coors[0] + (seat_radius * 2) + seat_spacing, coors[1])
            elif rotation == 'horizontal': coors = (coors[0], coors[1] + (seat_radius * 2) + seat_spacing)

    if show_image: image.show()
    if save_image != None:
        Image.save(save_image)

def get_text_dimensions(text_string, font):
    _, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)

def plot_performance(logs, show_plot=True, title=None, save_plot=None, plot_for_k=False):
    
    x_vals = []
    fitnesses = []
    fairnesses = []
    compactnesses = []
    changes = []

    if plot_for_k: log_index = 0
    else: log_index = 1

    for i in range(len(logs)):
        log = logs[i]
        
        if int(log[log_index]) not in x_vals:
            x_vals.append(int(log[1]))
            fitnesses.append(float(log[2]))
            fairnesses.append(float(log[3]))
            compactnesses.append(float(log[4]))

        if log[5] != 'NULL':
            changes.append(int(log[5]))

    max_y = max([max(fitnesses), max(fairnesses), max(compactnesses)]) * 1.05
    min_y = min([min(fitnesses), min(fairnesses), min(compactnesses)]) * 0.95
    
    if max_y > 1: max_y = 1
    if min_y < 0: min_y = 0
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,3.5))
    fig.tight_layout(w_pad=2)
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.07, right=0.96)

    if title == None: title = 'Algorithm Performance'
    fig.suptitle(title, fontsize=13, y=0.95)

    ax1.plot(x_vals, fitnesses, color='red', linewidth=0.95)
    ax1.set_title("Fitness")
    ax1.set_xlabel("Function Evaluations")
    ax1.set_ylabel("Fitness Score")
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, max(x_vals)])
    
    ax2.plot(x_vals, fairnesses, color='green', linewidth=1)
    ax2.set_title("Fairness")
    ax2.set_xlabel("Function Evaluations")
    ax2.set_ylabel("Avg. SV Difference")
    ax2.set_ylim([0, 1])
    ax2.set_xlim([0, max(x_vals)])

    ax3.plot(x_vals, compactnesses, color='blue', linewidth=1)
    ax3.set_title("Compactness")
    ax3.set_xlabel("Function Evaluations")
    ax3.set_ylabel("Avg. Reock-Score")
    ax3.set_ylim([0, 1])
    ax3.set_xlim([0, max(x_vals)])

    ax4.plot(range(1, len(changes) + 1), changes, color='purple', linewidth=1)
    ax4.set_title("No. of Assignment Changes")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Changes")
    ax4.set_xlim([1, len(changes)])
    ax4.grid(axis='y', linestyle = '--', linewidth=0.3)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in [ax1, ax2, ax3]:
        ax.grid(axis='both', linestyle = '--', linewidth=0.3)

    if show_plot: plt.show(block=True)
    if save_plot != None and save_plot.endswith('png'):
        fig.savefig(save_plot, dpi=200)

def print_results(results):
    rows = []
    for key in results.keys():
        if len(key) <= 4: party = key.upper()
        else: party = key.replace('_', ' ').title()
        rows.append([party, results[key][0],
                     round(results[key][1]*100, 6),
                     results[key][2],
                     round(results[key][3]*100, 6),
                     round((results[key][1] - results[key][3])*100, 6)])
    df_results = pd.DataFrame(np.array(rows), columns=['Party', 'Seats', 'Seat Share (%)', 'Total Votes', 'Proportional Vote (%)', 'S/V Difference (%)'])
    print("                                  ** ELECTION RESULTS **\n")
    print(df_results.to_string(index=False), '\n')
    return df_results

def generate_progress_bar (cur_val, max_val, state='', bar_length = 20, fill = '█', empty='-', prefix=''):
    """
    Prints a progress bar to the console.
    :param cur_val:    current value
    :param max_val:    maximum value
    :param state:      message printed next to the progress bar
    :param bar_length: bar dimension
    :param fill:       filled bar characters
    :param empty:      empty bar characters
    :param prefix:     message printed before the progress bar
    """

    if state == '':
        state = '{0}/{1}'.format(cur_val, max_val)

    filledLength = int(bar_length * cur_val // max_val)
    bar = fill * filledLength + empty * (bar_length - filledLength)
    bar = (f'| {bar} |')

    sys.stdout.write("\r"+prefix+"{0} {1}".format(bar, state))
    sys.stdout.flush()