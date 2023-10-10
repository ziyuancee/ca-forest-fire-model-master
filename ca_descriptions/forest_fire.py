# Name: Forest Fire Modelling
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

from capyle.ca import Grid2D, Neighbourhood, CAConfig, randomise2d
import capyle.utils as utils
import numpy as np
from math import ceil, sin, pi


class States():
    CHAPARRAL = 0
    SCRUBLAND = 1
    FOREST = 2
    WATER = 3
    FIRE_BARRIER = 4
    BURNING = 5
    BURNT_OUT = 6
    TOWN = 7


# RGB state colours
class StateColours():
    CHAPARRAL = (88, 138, 14)
    SCRUBLAND = (179, 165, 16)
    FOREST = (3, 61, 21)
    WATER = (11, 160, 230)
    FIRE_BARRIER = (112, 112, 112)
    BURNING = (226, 88, 34)
    BURNT_OUT = (0, 0, 0)
    TOWN = (128, 0, 106)


class WindDirection():
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7
    NONE = 8

# CUSTOMIZABLE PARAMETERS

# Format: bool
# True = Power Plant ignition enabled, False = Power Plant ignition enabled
POWER_PLANT = True

# Format: bool
# True = Proposed incineratior ignition enabled, False = Proposed incineratior ignition disabled
PROPOSED_INCINERATOR = False

# Format: (int, int), Range: 0-100
# X, Y coordinates for power plant
POWER_PLANT_IGNITION_SOURCE_COORDINATES_X, POWER_PLANT_IGNITION_SOURCE_COORDINATES_Y = (
    0, 0)

# Format: (int, int), Range: 0-100
# X, Y coordinates for proposed incineratior
PROPOSED_INCINERATOR_IGNITION_SOURCE_COORDINATES_X, PROPOSED_INCINERATOR_IGNITION_SOURCE_COORDINATES_Y = (
    99, 0)

# Format: ((int, int), (int, int)), Range: 0-100
# (X0 - X1, Y0 - Y1) town coordinates
TOWN_COORDINATES_X, TOWN_COORDINATES_Y = (36, 41), (86, 91)

# Format: [((int, int), (int, int))], Range: 0-100
# List of ((X0, Y0), (X1, Y1)) coordinates for water drop
WATER_DROP_LOCATION = []
# Format: int, Range: 0-NUMBER_OF_GENERATIONS
# Generation during which to drop water
WATER_DROP_GENERATION = 0

# Format: [((int, int), (int, int))], Range: 0-100
# List of ((X0, Y0), (X1, Y1)) coordinates for fire break
FIRE_BARRIER_LOCATION = []

# Format: WindDirection
# Wind direction to be used for simulation, WindDirection.N = wind coming from north direction
WIND_DIRECTION = WindDirection.N

# Format: int, Range: 1-INF
# Burning duration for each terrain type once it catches fire
CHAPARRAL_BURNING_GENERATIONS = 40
SCRUBLAND_BURNING_GENERATIONS = 3
FOREST_BURNING_GENERATIONS = 120

# Format: int, Range: 1-INF
# Generations per single day and length of simulation in number of generations
GENERATIONS_PER_DAY = 4
NUMBER_OF_GENERATIONS = 1500

# Format: float, Range: 0-1
# a(i,j) ignition probability mentioned in the equation in the report
CHAPARRAL_IGNITION_PROBABILITY_RAW = 0.4
SCRUBLAND_IGNITION_PROBABILITY_RAW = 0.8
FOREST_IGNITION_PROBABILITY_RAW = 0.1

# Format: float, Range 0-INF
# D coefficient mentioned in the equation in the report
# Value is used in the exponent used during the exponential scaling of the wind
WIND_SPEED = 2

# Format: float, Range 1-INF for positive wind direction, 1 for no wind, -INF-1 for negative wind direction
# E coefficient mentioned in the equation in the report
# Value is used in the base used during the exponential scaling of the wind
# The closer the value to INF or -INF the stronger the wind scaling
EXPONENTIAL_COMPONENT = 5

# Format: float, Range 1-INF
# Z coefficient mentioned in the equation in the report
# The higher the parameter value, the higher the impact of burning neighbouring squares and the wind on the final square ignition probability
NTH_ROOT_NEIGHBOUR_BURNING_IMPORTANCE = 2

# Normalizing parameters, best not to modify, they are calculated on the basis of the values of other parameters
UPWIND_COEFFICIENT = EXPONENTIAL_COMPONENT**WIND_SPEED
UPWIND_CORNER_COEFFICIENT = EXPONENTIAL_COMPONENT**(WIND_SPEED*sin(pi/4))
PARALLEL_COEFFICIENT = 1
DOWNWIND_COEFFICIENT = 1
DOWNWIND_CORNER_COEFFICIENT = 1

WIND_NORMALIZING_FACTOR = UPWIND_COEFFICIENT + 2*UPWIND_CORNER_COEFFICIENT + \
    2*PARALLEL_COEFFICIENT + 2*DOWNWIND_CORNER_COEFFICIENT + DOWNWIND_COEFFICIENT

NO_WIND_CORNER_COEFFICIENT = 1
NO_WIND_SIDE_COEFFICIENT = 1
NO_WIND_NORMALIZING_COEFFICIENT = 4 * \
    NO_WIND_CORNER_COEFFICIENT + 4*NO_WIND_SIDE_COEFFICIENT

def transition_func(grid, neighbourstates, neighbourcounts, burning_generations_left, generations_counter, fire_reached_town):

    generations_counter[0] += 1

    _, _, _, _, _, burning_neighbours, _, _ = neighbourcounts

    if np.any(burning_neighbours[TOWN_COORDINATES_Y[0]:TOWN_COORDINATES_Y[1]+1, TOWN_COORDINATES_X[0]:TOWN_COORDINATES_X[1]+1] > 0) and not fire_reached_town[0]:
        fire_reached_town[0] = True
        print(
            f"Fire reached town! Days elapsed since fire started: {ceil(generations_counter[0]/GENERATIONS_PER_DAY)-1}")

    burning_generations_left[burning_generations_left >= 0] -= 1
    grid[burning_generations_left == 0] = States.BURNT_OUT

    nw_burning, n_burning, ne_burning, w_burning, e_burning, sw_burning, s_burning, se_burning = (
        neighbourstates == States.BURNING).astype(np.float64)

    if WIND_DIRECTION == WindDirection.NONE:
        wind_ignition_probability_coefficient = NO_WIND_CORNER_COEFFICIENT * \
            (nw_burning + ne_burning + sw_burning + se_burning) + \
            NO_WIND_SIDE_COEFFICIENT * \
            (n_burning + w_burning + e_burning + s_burning)
        wind_ignition_probability_coefficient /= NO_WIND_NORMALIZING_COEFFICIENT
    else:
        wind_ignition_probability_coefficient = calculate_wind_ignition_coefficient(
            nw_burning, n_burning, ne_burning, w_burning, e_burning, sw_burning, s_burning, se_burning)

    wind_ignition_probability_coefficient**(1/NTH_ROOT_NEIGHBOUR_BURNING_IMPORTANCE)
    ignition_probability_grid = np.zeros(grid.shape, dtype=np.float64)

    ignition_probability_grid[grid ==
                              States.CHAPARRAL] = CHAPARRAL_IGNITION_PROBABILITY_RAW
    ignition_probability_grid[grid ==
                              States.SCRUBLAND] = SCRUBLAND_IGNITION_PROBABILITY_RAW
    ignition_probability_grid[grid ==
                              States.FOREST] = FOREST_IGNITION_PROBABILITY_RAW

    ignition_probability_grid *= wind_ignition_probability_coefficient

    generate_uniform_probabilities = np.random.random(grid.shape)

    ignition = generate_uniform_probabilities < ignition_probability_grid

    chaparral_ignited = ignition & (grid == States.CHAPARRAL)
    scrubland_ignited = ignition & (grid == States.SCRUBLAND)
    forest_ignited = ignition & (grid == States.FOREST)

    burning_generations_left[chaparral_ignited] = CHAPARRAL_BURNING_GENERATIONS
    burning_generations_left[scrubland_ignited] = SCRUBLAND_BURNING_GENERATIONS
    burning_generations_left[forest_ignited] = FOREST_BURNING_GENERATIONS

    grid[ignition] = States.BURNING

    if generations_counter[0] == WATER_DROP_GENERATION:
        for (water_drop_x0, water_drop_y0), (water_drop_x1, water_drop_y1) in WATER_DROP_LOCATION:
            grid[water_drop_y0:water_drop_y1+1, water_drop_x0: water_drop_x1+1] = States.WATER
            burning_generations_left[water_drop_y0:water_drop_y1+1, water_drop_x0: water_drop_x1+1] = 0

    return grid


def calculate_wind_ignition_coefficient(nw_burning, n_burning, ne_burning, w_burning, e_burning, sw_burning, s_burning, se_burning):
    if WIND_DIRECTION == WindDirection.N:
        nw_burning *= UPWIND_CORNER_COEFFICIENT
        n_burning *= UPWIND_COEFFICIENT
        ne_burning *= UPWIND_CORNER_COEFFICIENT
        w_burning *= PARALLEL_COEFFICIENT
        e_burning *= PARALLEL_COEFFICIENT
        sw_burning *= DOWNWIND_CORNER_COEFFICIENT
        s_burning *= DOWNWIND_COEFFICIENT
        se_burning *= DOWNWIND_CORNER_COEFFICIENT

    if WIND_DIRECTION == WindDirection.S:
        nw_burning *= DOWNWIND_CORNER_COEFFICIENT
        n_burning *= DOWNWIND_COEFFICIENT
        ne_burning *= DOWNWIND_CORNER_COEFFICIENT
        w_burning *= PARALLEL_COEFFICIENT
        e_burning *= PARALLEL_COEFFICIENT
        sw_burning *= UPWIND_CORNER_COEFFICIENT
        s_burning *= UPWIND_COEFFICIENT
        se_burning *= UPWIND_CORNER_COEFFICIENT

    if WIND_DIRECTION == WindDirection.W:
        nw_burning *= UPWIND_CORNER_COEFFICIENT
        n_burning *= PARALLEL_COEFFICIENT
        ne_burning *= DOWNWIND_CORNER_COEFFICIENT
        w_burning *= UPWIND_COEFFICIENT
        e_burning *= DOWNWIND_COEFFICIENT
        sw_burning *= UPWIND_CORNER_COEFFICIENT
        s_burning *= PARALLEL_COEFFICIENT
        se_burning *= DOWNWIND_CORNER_COEFFICIENT

    if WIND_DIRECTION == WindDirection.E:
        nw_burning *= DOWNWIND_CORNER_COEFFICIENT
        n_burning *= PARALLEL_COEFFICIENT
        ne_burning *= UPWIND_CORNER_COEFFICIENT
        w_burning *= DOWNWIND_COEFFICIENT
        e_burning *= UPWIND_COEFFICIENT
        sw_burning *= DOWNWIND_CORNER_COEFFICIENT
        s_burning *= PARALLEL_COEFFICIENT
        se_burning *= UPWIND_CORNER_COEFFICIENT

    if WIND_DIRECTION == WindDirection.NW:
        nw_burning *= UPWIND_COEFFICIENT
        n_burning *= UPWIND_CORNER_COEFFICIENT
        ne_burning *= PARALLEL_COEFFICIENT
        w_burning *= UPWIND_CORNER_COEFFICIENT
        e_burning *= DOWNWIND_CORNER_COEFFICIENT
        sw_burning *= PARALLEL_COEFFICIENT
        s_burning *= DOWNWIND_CORNER_COEFFICIENT
        se_burning *= DOWNWIND_COEFFICIENT

    if WIND_DIRECTION == WindDirection.NE:
        nw_burning *= PARALLEL_COEFFICIENT
        n_burning *= UPWIND_CORNER_COEFFICIENT
        ne_burning *= UPWIND_COEFFICIENT
        w_burning *= DOWNWIND_CORNER_COEFFICIENT
        e_burning *= UPWIND_CORNER_COEFFICIENT
        sw_burning *= DOWNWIND_COEFFICIENT
        s_burning *= DOWNWIND_CORNER_COEFFICIENT
        se_burning *= PARALLEL_COEFFICIENT

    if WIND_DIRECTION == WindDirection.SW:
        nw_burning *= PARALLEL_COEFFICIENT
        n_burning *= DOWNWIND_CORNER_COEFFICIENT
        ne_burning *= DOWNWIND_COEFFICIENT
        w_burning *= UPWIND_CORNER_COEFFICIENT
        e_burning *= DOWNWIND_CORNER_COEFFICIENT
        sw_burning *= UPWIND_COEFFICIENT
        s_burning *= UPWIND_CORNER_COEFFICIENT
        se_burning *= PARALLEL_COEFFICIENT

    if WIND_DIRECTION == WindDirection.SE:
        nw_burning *= DOWNWIND_COEFFICIENT
        n_burning *= DOWNWIND_CORNER_COEFFICIENT
        ne_burning *= PARALLEL_COEFFICIENT
        w_burning *= DOWNWIND_CORNER_COEFFICIENT
        e_burning *= UPWIND_CORNER_COEFFICIENT
        sw_burning *= PARALLEL_COEFFICIENT
        s_burning *= UPWIND_CORNER_COEFFICIENT
        se_burning *= UPWIND_COEFFICIENT

    wind_ignition_coefficient = nw_burning + n_burning + ne_burning + \
        w_burning + e_burning + sw_burning + s_burning + se_burning
    wind_ignition_coefficient /= WIND_NORMALIZING_FACTOR

    return wind_ignition_coefficient


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    # ---THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED---
    config.title = "Forest Fire Model"
    config.dimensions = 2

    # ---- Override the defaults below (these may be changed at anytime) ----
    tuple_of_state_values = tuple([state_value for state_name, state_value in vars(
        States).items() if not state_name.startswith("__")])
    config.states = tuple_of_state_values

    list_of_state_colour_tuples = [tuple([component_value/255 for component_value in state_colour_value]) for state_colour, state_colour_value in vars(
        StateColours).items() if not state_colour.startswith("__")]
    config.state_colors = list_of_state_colour_tuples

    config.num_generations = NUMBER_OF_GENERATIONS

    config.initial_grid = np.zeros((100, 100), dtype=np.uint8)
    config.grid_dims = config.initial_grid.shape

    config.wrap = States.BURNT_OUT

    # Forest section north
    config.initial_grid[10:36, 30:51].fill(States.FOREST)
    # Water
    config.initial_grid[35:41, 10:51].fill(States.WATER)
    # Forest section south
    config.initial_grid[40:71, 0:51].fill(States.FOREST)
    # Canyon scrubland
    config.initial_grid[10:81, 60:66].fill(States.SCRUBLAND)
    # Town
    config.initial_grid[TOWN_COORDINATES_Y[0]:TOWN_COORDINATES_Y[1]+1,
                        TOWN_COORDINATES_X[0]:TOWN_COORDINATES_X[1]+1].fill(States.TOWN)
    # Ignition source
    if POWER_PLANT:
        config.initial_grid[POWER_PLANT_IGNITION_SOURCE_COORDINATES_Y,
                            POWER_PLANT_IGNITION_SOURCE_COORDINATES_X] = States.BURNING

    if PROPOSED_INCINERATOR:
        config.initial_grid[PROPOSED_INCINERATOR_IGNITION_SOURCE_COORDINATES_Y,
                            PROPOSED_INCINERATOR_IGNITION_SOURCE_COORDINATES_X] = States.BURNING
    
    for (fire_barrier_x0, fire_barrier_y0), (fire_barrier_x1, fire_barrier_y1) in FIRE_BARRIER_LOCATION:
        config.initial_grid[fire_barrier_y0:fire_barrier_y1+1, fire_barrier_x0:fire_barrier_x1+1] = States.FIRE_BARRIER

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def main():
    # Open the config object
    config = setup(sys.argv[1:])

    burning_generations_left = np.zeros(config.grid_dims, dtype=np.int8)
    generations_counter = [0]
    fire_reached_town = [False]

    # Create grid object
    grid = Grid2D(config, (transition_func, burning_generations_left,
                  generations_counter, fire_reached_town))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
