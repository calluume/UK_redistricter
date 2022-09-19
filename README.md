
# UK Electoral Redistricter

<p align="center" width="100%">
    <img width="600" alt="south" src="https://user-images.githubusercontent.com/62618224/161809829-1a0d43fb-a570-40de-a3ea-a2b02b6a2361.png">
</p>

This redistricter was developed as part of my dissertation for the final year of my MSci in Computer Science at the University of Exeter with the goal of developing a new method for electoral districting using a reinforcement-learning approach, based on a graph grouping algorithm proposed by Zhou et al. (2016) [1]. The project has been applied to the UK and aims to directly optimise boundaries for fairness and the solution's ability to best represent the electorate, unlike many papers within the literature which mostly focus on factors such as population distribution. This is done through a newly developed metric for electoral fairness, the "Seats-votes difference," which is able to make direct comparisons towards party seat and vote shares and, importantly, can be applied to parliamentary systems with any number of parties.

A paper detailing the proposed algorithm and the seats-votes difference will be published as part of "[SoGood 2022: The 7th Workshop on Data Science for Social Good](https://sites.google.com/view/ecmlpkddsogood2022/)," including a poster presentation at the conference in Grenoble on the 23<sup>rd</sup> of September.

If you have any questions, feel free to contact me on my website at [callum-evans.co.uk](https://callum-evans.co.uk/)

## Installation

The project has been developed using Python 3.8.12, and requires a number of extra dependencies, although some are only required to create maps and videos to show the model's generated solutions. For all functionalities, the required packages are:
- NumPy
- Pandas
- GeoPandas
- NetworkX
- SciPy
- GeoPy
- OpenCV
- Matplotlib
- Gdown

## Setup

In order to run the redistricter, a few files are required:

- `dst.npy`: The distance matrix containing the distance between all wards
- `eng_wards.gpkg`: England ward boundaries
- `wa_wards.gpkg`: Wales ward boundaries
- `sco_wards.gpkg`: Scotland ward boundaries
- `ni_wards.gpkg`: Northern Ireland ward boundaries

The boundary files are available online and distance matrix can be generated using `redistricter.py` (although this takes a long time), but all of these can be downloaded with Gdown by running `setup.py`.

## Datasets

The dataset used in the model is available within this repository or on Kaggle [2], and covers the over 9,500 wards in England, Wales, Scotland and Northern Ireland, with socio-demographic data collected from the 2011 census. More detailed information on the dataset and its creation can be found on Kaggle, but ward-level election results have then been generated using this data and a linear regression model trained on constituency party support from the 2017 general election.

However, once re-combined to the constituency-level, there is a small difference between the real results and the model results, as shown in the table below:

<p align="center" width="100%">
    <img width="500" alt="diff_table" src="https://user-images.githubusercontent.com/62618224/161809783-ac13c34f-ce49-4381-bb4b-b08fcf13165b.png">
</p>

GeoPackage files containing the boundaries for each country are also required in order to map solutions. In which case, England and Wales use the 2011 boundaries available from the ONS, Scotland uses the 2014 boundaries from Boundaries Scotland and Northern Ireland uses the 1993 boundaries from OpenDataNI (distributed under the [Open Government License](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)).

All wards and constituencies are referred to using their ONS code, (England E05 / E14, Wales W05 / W06, Scotland S13 / S14, Northern Ireland N08 / N06), except for the City of London, which uses the London borough boundaries (E09) due to data availability from the 2011 census.

The selection methods used within the algorithm also require a distance matrix storing the distances between each ward, which has to be created when the model is first run if it has not already been downloaded on setup, done using the `generate_matrices(load_distances=False)` function.

## Usage

The redistricter object can be defined and used within code by itself, but command line arguments have also been added for ease of use. These include:

    -m:        Skip creating a plotter object
    -p:        Force no progress bar
    -v:        Run the program verbose
    -rcolours: Use random constituency colours in plots
    
    -vf <filename>      Create a video with the given filename
    -k <iterations>     Number of iterations for the first optimisation phase (main phase)
    -c <iterations>     Number of iterations for the second optimisation phase, where alpha=0 and beta=1 (default 0)
    -falpha <alpha>     Set the alpha value for fitness calculations (fairness)
    -fbeta <beta>       Set the beta value for fitness calculations (compactness)
    -ims <improvements> Set the number of improvements during the local search phase
    -seed <rnd_seed>    Sets the random seed for the model
    
For example, the following command:

```python3 redistricter.py -k 10 -c 2 -ims 100 -vf test.mp4 -falpha 0.8 -fbeta 0.2 -v```

will run for 12 iterations (10 for the first stage and 2 more for the second stage, optimising only for fairness), with 100 steps during the local search. The model will be run during the first stage with an alpha value of 0.8, and a beta value of 0.2, and the maps at each stage will be collected into a video called "test.mp4" (saved in the video directory).

Using the `-h` flag will print a manual with all possible command line arguments, although all the default values can be tweaked in the `data/default_parameters.json` file.

## Examples

Below is a video of the algorithm generating boundaries from scratch, done by setting `generate_new_constituencies=True` in the `generate_matrices` function.

<p align="center" width="100%">
    <img width="600" alt="south" src="https://user-images.githubusercontent.com/62618224/165603799-d1cb578b-2fe5-4aec-b0fd-06a55f849087.gif">
</p>

## References

[1] Zhou, Yangming, Jin-Kao Hao and Béatrice Duval (Dec. 2016). ‘Reinforcement learning based local search for grouping problems: A case study on graph coloring’. In: Expert Systems with Applications 64, pp. 412–422. doi: 10.1016/j.eswa.2016.07.047.

[2] Evans, C.: UK Ward Demographic Data and 2017 Election Results Dataset (Jul 2022), https://doi.org/10.34740/kaggle/dsv/3706511