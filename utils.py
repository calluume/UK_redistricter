import sys, math, copy

import numpy as np
import random as rnd

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